"""PyTorch models, repair/APR layers and checkpointed contingency losses."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def repair(g,d,lower,upper):
    total=g.sum(1,keepdim=True); demand=d.sum(1,keepdim=True)
    tiny=torch.finfo(g.dtype).eps
    up=(demand-total)/(upper.sum(1,keepdim=True)-total).clamp_min(tiny)
    down=(total-demand)/(total-lower.sum()).clamp_min(tiny)
    # Inputs are prechecked for aggregate nominal capacity feasibility.
    return torch.where(total<demand,g+up*(upper-g),g-down*(g-lower))


def apr(g,d,upper,lower,gamma,outages,steps=40):
    r=gamma*(upper-lower)
    mask=F.one_hot(outages,num_classes=g.shape[1]).bool()[None,:,:]
    with torch.no_grad():
        lo=g.new_zeros((len(g),len(outages),1)); hi=torch.ones_like(lo)
        for _ in range(steps):
            mid=(lo+hi)/2
            q=torch.minimum(g[:,None,:]+mid*r[:,None,:],upper[:,None,:]).masked_fill(mask,0)
            short=q.sum(2,keepdim=True)<d.sum(1)[:,None,None]
            lo=torch.where(short,mid,lo); hi=torch.where(short,hi,mid)
        signal=(lo+hi)/2
    # Expression derivative holding the binary-search signal fixed. This is a
    # deliberate backward convention, not the implicit derivative of the root.
    q=torch.minimum(g[:,None,:]+signal*r[:,None,:],upper[:,None,:]).masked_fill(mask,0)
    return q,signal.squeeze(-1)


class Physics(nn.Module):
    def __init__(self,grid,matrices,gamma,chunk=16,bisection=40,penalty=1500,slack_unit='pu'):
        super().__init__()
        P,offset,L,kg,ke=matrices
        for name,value in [('P',P),('Pg',P[:,grid.gen_bus]),('offset',offset),('L',L),
                           ('lower',grid.lower),('limit',grid.limit)]:
            self.register_buffer(name,torch.as_tensor(value))
        self.register_buffer('kg',torch.as_tensor(kg,dtype=torch.long))
        self.register_buffer('ke',torch.as_tensor(ke,dtype=torch.long))
        self.gamma=gamma; self.chunk=chunk; self.steps=bisection
        self.base=grid.base; self.penalty=penalty*(grid.base if slack_unit=='mw' else 1)

    def flows(self,g,d): return g@self.Pg.T-d@self.P.T+self.offset

    def residuals(self,g,d,u):
        hs=[]
        for k in self.kg.split(self.chunk):
            q,_=apr(g,d,u,self.lower,self.gamma,k,self.steps)
            hs.append(q.sum(2)-d.sum(1,keepdim=True))
        return torch.cat(hs,1)

    def terms(self,g,d,c,u):
        f=self.flows(g,d)
        eta=F.relu(f.abs()-self.limit).sum(1)
        hs=[]
        # Checkpoint only the expensive loss blocks, so all scenarios contribute
        # while their dense activations need not coexist in memory.
        for k in self.kg.split(self.chunk):
            def gen_block(g,d,u,k=k):
                q,_=apr(g,d,u,self.lower,self.gamma,k,self.steps)
                fk=q@self.Pg.T-(d@self.P.T)[:,None,:]+self.offset
                return F.relu(fk.abs()-self.limit).sum((1,2)),q.sum(2)-d.sum(1,keepdim=True)
            if torch.is_grad_enabled() and g.requires_grad:
                s,h=checkpoint(gen_block,g,d,u,use_reentrant=False)
            else: s,h=gen_block(g,d,u)
            eta=eta+s; hs.append(h)
        for a in range(0,len(self.ke),self.chunk):
            k=self.ke[a:a+self.chunk]; L=self.L[:,a:a+self.chunk]
            def line_block(f,k=k,L=L):
                fk=f[:,:,None]+L[None,:,:]*f[:,None,k]
                eta_k=F.relu(fk.abs()-self.limit[None,:,None])
                # Ignore the failed branch explicitly (also has L[k,k]=-1).
                keep=torch.ones_like(L)
                keep[k,torch.arange(len(k),device=k.device)]=0
                return (eta_k*keep).sum((1,2))
            s=checkpoint(line_block,f,use_reentrant=False) if torch.is_grad_enabled() and f.requires_grad else line_block(f)
            eta=eta+s
        objective=(c*g).sum(1)*self.base+self.penalty*eta
        return objective,torch.cat(hs,1),eta


class MLP(nn.Module):
    """Four ReLU hidden layers, width round(1.5*dim(x)), affine output head.

    The paper does not disambiguate whether its four-layer count includes the
    output head; this implementation records this interpretation explicitly.
    """
    def __init__(self,grid,kg,dual=False,hidden=0,depth=4):
        super().__init__()
        load=np.flatnonzero(grid.demand!=0)
        self.register_buffer('load',torch.as_tensor(load,dtype=torch.long))
        self.register_buffer('lower',torch.as_tensor(grid.lower))
        dim=len(load)+2*len(grid.gen_ids); hidden=hidden or round(1.5*dim)
        layers=[nn.LayerNorm(dim)] if not dual else []
        for i in range(depth): layers += [nn.Linear(dim if i==0 else hidden,hidden),nn.ReLU()]
        layers += [nn.Linear(hidden,len(kg) if dual else len(grid.gen_ids))]
        self.net=nn.Sequential(*layers); self.dual=dual

    def forward(self,d,c,u):
        x=torch.cat([d[:,self.load],c,u],1)
        y=self.net(x)
        return y if self.dual else repair(self.lower+torch.sigmoid(y)*(u-self.lower),d,self.lower,u)


class GIN(nn.Module):
    """Experimental edge-aware GIN with generator-local outputs; not paper MLP.

    Implemented with PyTorch index_add, so torch_geometric is not required.
    Multiple generators at a bus retain separate metadata and output heads.
    """
    def __init__(self,grid,kg,dual=False,hidden=64,depth=4):
        super().__init__(); hidden=hidden or 64
        for name,value in [('bus',grid.gen_bus),('source',np.r_[grid.source,grid.target]),
                           ('target',np.r_[grid.target,grid.source]),('kg',kg)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.long))
        self.register_buffer('lower',torch.as_tensor(grid.lower))
        ef=np.stack([grid.reactance,np.where(np.isfinite(grid.limit),grid.limit,0),
                     np.isfinite(grid.limit).astype(float),grid.tap,grid.shift],axis=1)
        self.register_buffer('edges',torch.as_tensor(np.r_[ef,ef]))
        self.nbus=len(grid.bus_ids); self.dual=dual
        self.embed=nn.Linear(5,hidden)
        self.edge_embed=nn.ModuleList(nn.Linear(5,hidden) for _ in range(depth))
        self.blocks=nn.ModuleList(nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.ReLU(),
                                              nn.Linear(hidden,hidden),nn.ReLU()) for _ in range(depth))
        self.eps=nn.Parameter(torch.zeros(depth))
        self.head=nn.Sequential(nn.Linear(2*hidden+3,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,d,c,u):
        x=d.new_zeros((len(d),self.nbus,5)); x[:,:,0]=d
        values=torch.stack([u,self.lower.expand_as(u),c,torch.ones_like(u)],2)
        agg=d.new_zeros((len(d),self.nbus,4)).index_add(1,self.bus,values)
        x[:,:,1:]=agg
        h=self.embed(x)
        for eps,edge,block in zip(self.eps,self.edge_embed,self.blocks):
            msg=F.relu(h[:,self.source,:]+edge(self.edges)[None,:,:])
            neigh=torch.zeros_like(h).index_add(1,self.target,msg)
            h=block((1+eps)*h+neigh)
        common=h.mean(1,keepdim=True).expand(-1,len(self.bus),-1)
        features=torch.cat([h[:,self.bus,:],common,torch.stack([c,u,self.lower.expand_as(u)],2)],2)
        y=self.head(features).squeeze(-1)
        return y[:,self.kg] if self.dual else repair(self.lower+torch.sigmoid(y)*(u-self.lower),d,self.lower,u)


def make_models(grid,kg,config):
    cls=MLP if config['arch']=='mlp' else GIN
    kwargs=dict(hidden=config['hidden'],depth=config['depth'])
    return cls(grid,kg,False,**kwargs),cls(grid,kg,True,**kwargs)
