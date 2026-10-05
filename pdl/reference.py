"""Independent MILP/CCGA-style reference for the same soft-thermal SCOPF.

The master is rebuilt each iteration. This is an objective/feasibility
reference, not a reproduction of the authors' CCGA runtime implementation.
SciPy/HiGHS supports small-case tests without Gurobi; Gurobi is recommended
for paper-size reference solves. No incumbent is called an optimum without
checking the full-contingency upper bound against the solver lower bound.
"""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import milp, Bounds, LinearConstraint
from case_data import apr_numpy, assess_numpy


class LinearModel:
    def __init__(self):
        self.lo=[]; self.hi=[]; self.cost=[]; self.integer=[]
        self.ri=[]; self.ci=[]; self.v=[]; self.lb=[]; self.ub=[]
    def variables(self,lo,hi,cost=0,integer=0):
        lo=np.atleast_1d(lo); hi=np.broadcast_to(hi,lo.shape)
        ids=np.arange(len(self.lo),len(self.lo)+len(lo))
        self.lo.extend(lo); self.hi.extend(hi)
        self.cost.extend(np.broadcast_to(cost,lo.shape)); self.integer.extend(np.broadcast_to(integer,lo.shape))
        return ids
    def row(self,ids,values,lb=-np.inf,ub=np.inf):
        ids=np.asarray(ids); values=np.asarray(values)
        nz=values!=0; row=len(self.lb)
        self.ri.extend([row]*int(nz.sum())); self.ci.extend(ids[nz]); self.v.extend(values[nz])
        self.lb.append(lb); self.ub.append(ub)
    def solve(self,backend,seconds,gap,threads):
        A=sparse.coo_matrix((self.v,(self.ri,self.ci)),shape=(len(self.lb),len(self.lo))).tocsr()
        if backend=='scipy':
            r=milp(np.array(self.cost),integrality=np.array(self.integer),
                bounds=Bounds(self.lo,self.hi),constraints=LinearConstraint(A,self.lb,self.ub),
                options=dict(time_limit=seconds,mip_rel_gap=gap))
            bound=getattr(r,'mip_dual_bound',None)
            if bound is None and r.status==0: bound=r.fun
            return r.x,r.fun,bound,str(r.message),r.status==0
        import gurobipy as gp
        m=gp.Model(); m.Params.OutputFlag=0; m.Params.TimeLimit=seconds
        m.Params.MIPGap=gap; m.Params.Threads=threads
        m.Params.FeasibilityTol=1e-8
        x=m.addMVar(len(self.lo),lb=self.lo,ub=self.hi,obj=self.cost,
            vtype=np.where(np.array(self.integer)>0,gp.GRB.BINARY,gp.GRB.CONTINUOUS))
        lb=np.array(self.lb); ub=np.array(self.ub)
        equality=np.isfinite(lb)&(lb==ub)
        m.addMConstr(A[equality],x,'=',lb[equality])
        lower=np.isfinite(lb)&~equality; upper=np.isfinite(ub)&~equality
        m.addMConstr(A[lower],x,'>',lb[lower]); m.addMConstr(A[upper],x,'<',ub[upper])
        m.optimize()
        result=(x.X.copy() if m.SolCount else None,float(m.ObjVal) if m.SolCount else None,
                float(m.ObjBound),f'Gurobi status {m.Status}',m.Status==gp.GRB.OPTIMAL)
        m.dispose(); return result


def build_master(grid,matrices,d,c,u,gamma,Ug,Ue,penalty,slack_unit):
    P,offset,L,kg,ke=matrices
    A=P[:,grid.gen_bus]; b=offset-P@d; G=len(u)
    model=LinearModel(); g=model.variables(grid.lower,u,c*grid.base)
    model.row(g,np.ones(G),lb=d.sum(),ub=d.sum())
    Q={}; response=gamma*(u-grid.lower)
    for k in kg:
        low=grid.lower.copy(); high=u.copy(); low[k]=high[k]=0
        q=model.variables(low,high); Q[int(k)]=q
        model.row(q,np.ones(G),lb=d.sum(),ub=d.sum())
        for i in range(G):
            if i!=k: model.row([q[i],g[i]],[1,-1],ub=response[i])
    active=sorted({k for k,l in Ug})
    for k in active:
        q=Q[k]; n=model.variables([0],[1])[0]
        for i in range(G):
            if i==k: continue
            M=u[i]-grid.lower[i]
            if M<=1e-12: continue
            z=model.variables([0],[1],integer=1)[0]
            # q=min(g+n*r,u), using generator-range M (gamma<=1).
            model.row([q[i],g[i],n],[1,-1,-response[i]],ub=0)
            model.row([q[i],g[i],n,z],[1,-1,-response[i],M],lb=0)
            model.row([q[i],z],[1,-M],lb=u[i]-M)
    slack_cost=penalty*(grid.base if slack_unit=='mw' else 1)
    def thermal(ids,coeff,constant,limit):
        e=model.variables([0],[np.inf],cost=slack_cost)[0]
        model.row(np.r_[ids,e],np.r_[coeff,-1],ub=limit-constant)
        model.row(np.r_[ids,e],np.r_[coeff,1],lb=-limit-constant)
    for l in np.flatnonzero(np.isfinite(grid.limit)):
        thermal(g,A[l],b[l],grid.limit[l])
    for k,l in sorted(Ug): thermal(Q[k],A[l],b[l],grid.limit[l])
    kcol={int(k):j for j,k in enumerate(ke)}
    for k,l in sorted(Ue):
        a=L[l,kcol[k]]
        thermal(g,A[l]+a*A[k],b[l]+a*b[k],grid.limit[l])
    return model,g


def solve_reference(grid,matrices,d,c,u,gamma,backend='gurobi',seconds=3600,
                    gap=1e-4,threads=24,penalty=1500,slack_unit='pu',extensive=False,max_rounds=200):
    P,offset,L,kg,ke=matrices; finite=np.flatnonzero(np.isfinite(grid.limit))
    Ug={(int(k),int(l)) for k in kg for l in finite} if extensive else set()
    Ue={(int(k),int(l)) for k in ke for l in finite if k!=l} if extensive else set()
    start=time.perf_counter(); best=None; best_obj=np.inf; lower=-np.inf; history=[]
    for iteration in range(max_rounds):
        remaining=seconds-(time.perf_counter()-start)
        if remaining<=0: break
        model,ids=build_master(grid,matrices,d,c,u,gamma,Ug,Ue,penalty,slack_unit)
        x,master_obj,bound,status,optimal=model.solve(backend,remaining,gap/10,threads)
        if bound is not None and np.isfinite(bound): lower=max(lower,float(bound))
        if x is None:
            history.append(dict(iteration=iteration,status=status)); break
        g=x[ids]
        report=assess_numpy(grid,matrices,g,d,c,u,gamma,penalty,slack_unit)
        feasible=max(report['base_balance_pu'],report['contingency_balance_max_pu'],report['generator_bounds_pu'])<=1e-4
        if feasible and report['objective']<best_obj: best=g.copy(); best_obj=report['objective']
        certificate=max(0,best_obj-lower)/max(abs(best_obj),1) if np.isfinite(best_obj) and np.isfinite(lower) else np.inf
        history.append(dict(iteration=iteration,status=status,master_objective=master_obj,
                            full_objective=report['objective'],bound=lower,relative_certificate=certificate,
                            generator_pairs=len(Ug),line_pairs=len(Ue)))
        if certificate<=gap and best is not None: break
        if extensive: break
        # Add omitted positive thermal slacks. Existing positive soft slacks are
        # allowed; repeatedly treating them as hard violations cannot converge.
        f=P[:,grid.gen_bus]@g-P@d+offset
        q,_=apr_numpy(g,d,u,grid.lower,gamma,kg); candidates=[]
        for a,k in enumerate(kg):
            fk=P[:,grid.gen_bus]@q[0,a]-P@d+offset
            e=np.maximum(abs(fk)-grid.limit,0)
            candidates.extend(('g',int(k),int(l),float(e[l])) for l in finite if e[l]>1e-10 and (int(k),int(l)) not in Ug)
        for a,k in enumerate(ke):
            e=np.maximum(abs(f+L[:,a]*f[k])-grid.limit,0)
            candidates.extend(('e',int(k),int(l),float(e[l])) for l in finite if l!=k and e[l]>1e-10 and (int(k),int(l)) not in Ue)
        if not candidates: break
        threshold=max(t[3] for t in candidates)/5
        for kind,k,l,e in candidates:
            if e>=threshold: (Ug if kind=='g' else Ue).add((k,l))
    elapsed=time.perf_counter()-start
    certificate=max(0,best_obj-lower)/max(abs(best_obj),1) if best is not None and np.isfinite(lower) else np.inf
    return dict(dispatch=best,objective=float(best_obj),lower_bound=float(lower),
                relative_certificate=float(certificate),certified=bool(certificate<=gap),
                elapsed_seconds=elapsed,iterations=len(history),history=history,
                solver_backend=backend,algorithm='extensive' if extensive else 'CCGA-style')
