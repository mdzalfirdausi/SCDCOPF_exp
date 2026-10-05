"""Case ingestion, affine DC matrices, independent NumPy physics, and sampling."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import ndtr, ndtri, log_ndtr, logsumexp
from scipy.integrate import cumulative_trapezoid

PAPER_COUNTS = {
 '300_ieee': [300,69,201,411,57,322],
 '1354_pegase': [1354,260,673,1991,193,1430],
 '1888_rte': [1888,290,1000,2531,290,1567],
 '3022_goc': [3022,327,1574,4135,327,3180],
 '4917_goc': [4917,567,2619,6726,567,5066],
 '6515_rte': [6515,684,3673,9037,657,6474],
}

@dataclass
class Grid:
    name: str
    base: float
    bus_ids: np.ndarray
    gen_ids: np.ndarray
    gen_bus: np.ndarray
    demand: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    cost: np.ndarray
    source: np.ndarray
    target: np.ndarray
    reactance: np.ndarray
    tap: np.ndarray
    shift: np.ndarray
    limit: np.ndarray
    reference: int
    fingerprint: str
    dc_mode: str = 'matpower'

    def matrices(self):
        n, m = len(self.bus_ids), len(self.source)
        A = sparse.coo_matrix((np.r_[np.ones(m),-np.ones(m)],
            (np.r_[np.arange(m),np.arange(m)],np.r_[self.source,self.target])),shape=(m,n)).tocsr()
        b = 1 / (self.reactance * (self.tap if self.dc_mode=='matpower' else 1))
        Bf = sparse.diags(b) @ A
        Bbus = A.T @ Bf
        keep = np.delete(np.arange(n),self.reference)
        try:
            lu = splu(Bbus[keep][:,keep].tocsc())
        except RuntimeError as e:
            raise ValueError('The in-service grid must be connected; check islands and bus status.') from e
        P = np.zeros((m,n))
        for a in range(0,len(keep),128):
            columns = np.arange(a,min(a+128,len(keep)))
            rhs = np.zeros((len(keep),len(columns)))
            rhs[columns,np.arange(len(columns))]=1
            P[:,keep[columns]] = Bf[:,keep] @ lu.solve(rhs)
        pf_shift = -b*self.shift if self.dc_mode=='matpower' else np.zeros(m)
        offset = pf_shift - P @ (A.T @ pf_shift)
        denom = 1-(P[np.arange(m),self.source]-P[np.arange(m),self.target])
        ke = np.flatnonzero(np.abs(denom)>1e-8)
        L = np.empty((m,len(ke)))
        for a in range(0,len(ke),128):
            k=ke[a:a+128]
            L[:,a:a+len(k)]=(P[:,self.source[k]]-P[:,self.target[k]])/denom[k]
        L[ke,np.arange(len(ke))]=-1
        kg=np.flatnonzero((self.upper-self.lower>1e-12)&(self.lower>=0))
        if not len(kg): raise ValueError('No eligible generator outages.')
        return P,offset,L,kg,ke

    def audit(self,kg,ke):
        counts=[len(self.bus_ids),len(self.gen_ids),int(np.count_nonzero(self.demand)),len(self.source),len(kg),len(ke)]
        expected=next((v for key,v in PAPER_COUNTS.items() if self.name.endswith(key)),None)
        return dict(case=self.name, counts_N_G_loads_E_Kg_Ke=counts,
                    paper_counts=expected, matches_paper_counts=(counts==expected if expected else None),
                    sha256=self.fingerprint, dc_mode=self.dc_mode)


def _numeric_matrix(text,name):
    match=re.search(r'mpc\.'+name+r'\s*=\s*\[(.*?)\]\s*;',text,re.S)
    if not match: raise ValueError(f'Missing numeric mpc.{name} matrix')
    rows=[]
    for row in match.group(1).split(';'):
        if row.strip():
            try: rows.append([float(v) for v in row.split()])
            except ValueError as e: raise ValueError('Only literal numeric MATPOWER matrices are supported; no MATLAB execution.') from e
    if len({len(r) for r in rows}) != 1: raise ValueError(f'Ragged {name} matrix')
    return np.array(rows)


def load_case(path,dc_mode='matpower'):
    path=Path(path)
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix.lower()=='.m':
        text=re.sub(r'%[^\n]*','',path.read_text()).replace('...','')
        bm=re.search(r'mpc\.baseMVA\s*=\s*([\d.eE+-]+)',text)
        if not bm: raise ValueError('Missing baseMVA')
        base=float(bm.group(1))
        bus=_numeric_matrix(text,'bus'); gen=_numeric_matrix(text,'gen')
        branch=_numeric_matrix(text,'branch'); gc=_numeric_matrix(text,'gencost')
        if len(gc) not in (len(gen),2*len(gen)): raise ValueError('gencost/gen row mismatch')
        gc=gc[:len(gen)]
        if np.any(gc[:,0]!=2): raise ValueError('Only polynomial gencost is supported')
        cost=np.array([row[4+int(row[3])-2] if row[3]>=2 else 0 for row in gc])
        if np.any(gc[:,3]>2): print('Using c1 only: paper has linear cost; c2 and c0 are excluded.')
        bus_ids=bus[:,0].astype(int); demand=bus[:,2]; types=bus[:,1]
        gen_ids=np.arange(len(gen)); gen_buses=gen[:,0].astype(int)
        lower=gen[:,9]; upper=gen[:,8]; gs=gen[:,7]>0
        source=branch[:,0].astype(int); target=branch[:,1].astype(int)
        x=branch[:,3]; limit=branch[:,5]; tap=branch[:,8]; shift=np.deg2rad(branch[:,9]); bs=branch[:,10]>0
    elif path.suffix.lower()=='.xlsx':
        import pandas as pd
        sheets=pd.read_excel(path,sheet_name=['baseMVA','bus','gen','gencost','branch'])
        bus=sheets['bus']; gen=sheets['gen']; branch=sheets['branch']; gc=sheets['gencost']
        base=float(sheets['baseMVA']['baseMVA'].iloc[0])
        if 'gen_ID' in gen and 'gen_ID' in gc:
            keys=['bus_i','gen_ID'] if 'bus_i' in gc else ['gen_ID']
            gc=gen[keys].merge(gc,on=keys,how='left',validate='one_to_one')
        if len(gc)!=len(gen): raise ValueError('gencost/gen row mismatch')
        if 'model' in gc and np.any(gc['model']!=2): raise ValueError('Only polynomial costs supported')
        cost=gc['c1'].to_numpy(float)
        if 'c2' in gc and np.any(gc['c2']!=0): print('Using c1 only, as required by the paper linear objective.')
        bus_ids=bus['bus_i'].to_numpy(int); demand=bus['Pd'].to_numpy(float); types=bus['type'].to_numpy(int)
        gen_ids=gen['gen_ID'].to_numpy() if 'gen_ID' in gen else np.arange(len(gen))
        gen_buses=gen['bus_i'].to_numpy(int); lower=gen['Pmin'].to_numpy(float); upper=gen['Pmax'].to_numpy(float)
        gs=gen['status'].to_numpy()>0 if 'status' in gen else np.ones(len(gen),bool)
        source=branch['bus_i'].to_numpy(int); target=branch['bus_j'].to_numpy(int)
        x=branch['x'].to_numpy(float); limit=branch['rateA'].to_numpy(float)
        tap=branch['ratio'].to_numpy(float); shift=np.deg2rad(branch['angle'].to_numpy(float))
        bs=branch['status'].to_numpy()>0 if 'status' in branch else np.ones(len(branch),bool)
    else: raise ValueError('Use .m (numeric MATPOWER) or your .xlsx schema.')
    active=types!=4
    gs=gs&np.isin(gen_buses,bus_ids[active]); bs=bs&np.isin(source,bus_ids[active])&np.isin(target,bus_ids[active])
    order=np.argsort(bus_ids[active]); bus_ids=bus_ids[active][order]; types=types[active][order]; demand=demand[active][order]
    if len(np.unique(bus_ids))!=len(bus_ids): raise ValueError('Duplicate bus IDs')
    if not np.any(types==3): raise ValueError('Missing reference bus')
    lookup={int(b):i for i,b in enumerate(bus_ids)}
    if base<=0 or np.any(x[bs]==0): raise ValueError('Invalid baseMVA or zero branch reactance')
    if np.any(upper[gs]<lower[gs]): raise ValueError('Pmax below Pmin')
    arrays=[demand,lower[gs],upper[gs],cost[gs],x[bs],limit[bs],tap[bs],shift[bs]]
    if any(not np.isfinite(a).all() for a in arrays): raise ValueError('Nonfinite case data')
    return Grid(path.stem,base,bus_ids,gen_ids[gs],np.array([lookup[b] for b in gen_buses[gs]]),
        demand/base,lower[gs]/base,upper[gs]/base,cost[gs],
        np.array([lookup[b] for b in source[bs]]),np.array([lookup[b] for b in target[bs]]),
        x[bs],np.where(tap[bs]==0,1,tap[bs]),shift[bs],
        np.where(limit[bs]>0,limit[bs]/base,np.inf),int(np.flatnonzero(types==3)[0]),fingerprint,dc_mode)


def capacity_deficits(grid,d,u,kg):
    D=d.sum(-1)
    base=np.maximum(np.maximum(grid.lower.sum()-D,D-u.sum(-1)),0)
    shortage=np.maximum(D[:,None]-(u.sum(-1)[:,None]-u[:,kg]),0)
    excess=np.maximum((grid.lower.sum()-grid.lower[kg])[None,:]-D[:,None],0)
    return base,np.maximum(shortage,excess)


def apr_numpy(g,d,u,lower,gamma,kg,steps=60):
    """Independent float64 reference, including bounds on the system signal."""
    g=np.atleast_2d(g); d=np.atleast_2d(d); u=np.atleast_2d(u)
    lo=np.zeros((len(g),len(kg),1)); hi=np.ones_like(lo)
    response=gamma*(u-lower)
    mask=np.eye(g.shape[1],dtype=bool)[kg][None,:,:]
    for _ in range(steps):
        signal=(lo+hi)/2
        q=np.where(mask,0,np.minimum(g[:,None,:]+signal*response[:,None,:],u[:,None,:]))
        short=q.sum(2,keepdims=True)<d.sum(1)[:,None,None]
        lo=np.where(short,signal,lo); hi=np.where(short,hi,signal)
    signal=(lo+hi)/2
    q=np.where(mask,0,np.minimum(g[:,None,:]+signal*response[:,None,:],u[:,None,:]))
    return q,signal[:,:,0]


def assess_numpy(grid,matrices,g,d,c,u,gamma,penalty=1500,slack_unit='pu',steps=60):
    """One-sample independent full-contingency check, no neural graph or autograd."""
    P,offset,L,kg,ke=matrices
    g=np.asarray(g); d=np.asarray(d); u=np.asarray(u)
    injection=-d.copy(); np.add.at(injection,grid.gen_bus,g)
    f=P@injection+offset
    eta=np.maximum(np.abs(f)-grid.limit,0); sums=[eta.sum()]; peaks=[eta.max(initial=0)]
    gk,n=apr_numpy(g,d,u,grid.lower,gamma,kg,steps)
    gk=gk[0]; n=n[0]; h=gk.sum(1)-d.sum()
    bounds=max(np.maximum(grid.lower-g,0).max(initial=0),np.maximum(g-u,0).max(initial=0))
    survivor=~np.eye(len(g),dtype=bool)[kg]
    bounds=max(bounds,np.where(survivor,np.maximum(grid.lower-gk,0),0).max(initial=0),
               np.where(survivor,np.maximum(gk-u,0),0).max(initial=0))
    expected=np.minimum(g[None,:]+n[:,None]*gamma*(u-grid.lower),u)
    expected[np.arange(len(kg)),kg]=0
    residual=np.abs(expected-gk).max(initial=0)
    A=P[:,grid.gen_bus]; fixed=offset-P@d
    for a in range(0,len(kg),32):
        fk=gk[a:a+32]@A.T+fixed
        e=np.maximum(np.abs(fk)-grid.limit,0)
        sums.append(e.sum()); peaks.append(e.max(initial=0))
    for a in range(0,len(ke),32):
        k=ke[a:a+32]
        fk=f[:,None]+L[:,a:a+32]*f[k]
        fk[k,np.arange(len(k))]=0
        e=np.maximum(np.abs(fk)-grid.limit[:,None],0)
        sums.append(e.sum()); peaks.append(e.max(initial=0))
    slack=float(sum(sums)); gen_cost=float(np.dot(c,g)*grid.base)
    obj=gen_cost+penalty*slack*(grid.base if slack_unit=='mw' else 1)
    return dict(objective=obj,generation_cost=gen_cost,total_slack_pu=slack,
        max_line_overload_pu=float(max(peaks)),base_balance_pu=float(abs(g.sum()-d.sum())),
        contingency_balance_max_pu=float(abs(h).max(initial=0)),generator_bounds_pu=float(bounds),
        apr_formula_pu=float(residual),contingency_balance=h,signals=n)


class Sampler:
    """Independent joint-truncated demand samples via a 1-D latent Gaussian.

    Equicorrelation admits z_i=sqrt(r)*a+sqrt(1-r)*e_i. Conditional on
    the truncation box, a has density proportional to phi(a)*q(a)**n.
    A numerical inverse CDF samples that density, then independent truncated
    e_i are drawn conditional on a. This is not coordinate clipping or a
    finite-burn-in Gibbs chain. Quadrature is the only approximation.
    """
    def __init__(self,grid,seed,generator_std,mu=.5,zscore=1.959963984540054):
        self.grid=grid; self.rng=np.random.default_rng(seed); self.std=generator_std
        self.mu=mu; self.z=zscore
        self.loads=np.flatnonzero(grid.demand!=0)
        self.latent=np.linspace(-12.,12.,24001)
        r=.5; lo=(-zscore-np.sqrt(r)*self.latent)/np.sqrt(1-r)
        hi=(zscore-np.sqrt(r)*self.latent)/np.sqrt(1-r)
        # Stable subtraction of normal CDF values in either tail.
        a=np.where(lo>0,log_ndtr(-lo),log_ndtr(hi))
        b=np.where(lo>0,log_ndtr(-hi),log_ndtr(lo))
        logq=a+np.log(-np.expm1(np.minimum(b-a,-np.finfo(float).eps)))
        logpdf=-self.latent**2/2+len(self.loads)*logq
        pdf=np.exp(logpdf-logpdf.max())
        self.cdf=np.r_[0,cumulative_trapezoid(pdf,self.latent)]; self.cdf/=self.cdf[-1]
        unique=np.r_[True,np.diff(self.cdf)>0]
        self.latent=self.latent[unique]; self.cdf=self.cdf[unique]
        self.drawn=0

    def sample(self,batch):
        grid=self.grid; rng=self.rng; self.drawn+=batch
        a=np.interp(rng.random(batch),self.cdf,self.latent)[:,None]
        r=.5; lo=(-self.z-np.sqrt(r)*a)/np.sqrt(1-r); hi=(self.z-np.sqrt(r)*a)/np.sqrt(1-r)
        uniform=rng.random((batch,len(self.loads)))
        q=ndtr(lo)+(ndtr(hi)-ndtr(lo))*uniform
        z=np.sqrt(r)*a+np.sqrt(1-r)*ndtri(np.clip(q,1e-15,1-1e-15))
        d=np.broadcast_to(grid.demand,(batch,len(grid.demand))).copy()
        d[:,self.loads]+=self.mu*np.abs(grid.demand[self.loads])/self.z*z
        def factors():
            return 1+self.std*(np.sqrt(.8)*rng.normal(size=(batch,1))+
                               np.sqrt(.2)*rng.normal(size=(batch,len(grid.upper))))
        u=np.maximum(grid.upper*factors(),grid.lower+.01*(grid.upper-grid.lower))
        c=np.maximum(grid.cost*factors(),0)
        return d,c,u
