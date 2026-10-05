"""CLI for a transparent reconstruction of Park & Van Hentenryck PDL-SCOPF."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from case_data import load_case, Sampler, capacity_deficits, assess_numpy


def clean_args(args):
    return {k:v for k,v in vars(args).items() if k != "func"}


def jsonable(x):
    if isinstance(x,dict): return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray): return jsonable(x.tolist())
    if isinstance(x,np.generic): return jsonable(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    if isinstance(x,Path): return str(x)
    return x


def save_json(path,value):
    Path(path).write_text(json.dumps(jsonable(value),indent=2),encoding='utf-8')


def check_samples(grid,matrices,data,fail=False):
    d,c,u=data; base,cont=capacity_deficits(grid,d,u,matrices[3])
    invalid=(base>1e-8)|(cont.max(1)>1e-8)
    report=dict(samples=len(d),capacity_invalid=int(invalid.sum()),max_nominal_deficit_pu=float(base.max()),
                max_contingency_capacity_deficit_pu=float(cont.max()),demand_min_MW=float(d.sum(1).min()*grid.base),
                demand_max_MW=float(d.sum(1).max()*grid.base))
    if fail and invalid.any():
        row,col=np.unravel_index(cont.argmax(),cont.shape); k=matrices[3][col]
        raise ValueError(f'Capacity-infeasible input: {report}; worst CSV/sample row={row}, '
                         f'outaged generator={grid.gen_ids[k]}. Fix data/model assumptions; '
                         'no sample was discarded and demand was not rescaled.')
    return report


def init(args):
    grid=load_case(args.case,args.dc_mode); matrices=grid.matrices()
    print(json.dumps(jsonable(grid.audit(matrices[3],matrices[4])),indent=2))
    return grid,matrices


def sampler(grid,args,seed):
    return Sampler(grid,seed,args.generator_std,args.mu,args.zscore)


def sample_data(args):
    grid,matrices=init(args); data=sampler(grid,args,args.seed).sample(args.samples)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'instances.npz',d=data[0],c=data[1],u=data[2],case_sha256=grid.fingerprint)
    report=check_samples(grid,matrices,data)
    save_json(out/'data_manifest.json',dict(config=clean_args(args),grid=grid.audit(matrices[3],matrices[4]),capacity=report))
    print(json.dumps(report,indent=2))


def audit(args):
    grid,matrices=init(args); data=sampler(grid,args,args.seed).sample(args.samples)
    print(json.dumps(check_samples(grid,matrices,data),indent=2))
    print('Capacity checks are necessary; APR feasibility for gamma<1 additionally depends on dispatch.')
    print('Paper numerical matches require the matching case revision, gamma, variance, units, and settings.')


def get_torch(args):
    import torch
    from learning import Physics,make_models
    dtype=torch.float64 if args.dtype=='float64' else torch.float32
    device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    return torch,Physics,make_models,dtype,device


def train(args):
    grid,matrices=init(args)
    torch,Physics,make_models,dtype,device=get_torch(args)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.deterministic: torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(dtype)
    netP,netD=make_models(grid,matrices[3],clean_args(args)); netP=netP.to(device=device,dtype=dtype); netD=netD.to(device=device,dtype=dtype)
    physics=Physics(grid,matrices,args.gamma,args.chunk,args.bisection,args.penalty,args.slack_unit).to(device=device,dtype=dtype)
    optP=torch.optim.Adam(netP.parameters(),lr=args.lr); optD=torch.optim.Adam(netD.parameters(),lr=args.lr)
    stream=sampler(grid,args,args.seed+1000)
    monitor=sampler(grid,args,args.seed+2000).sample(args.monitor_samples)
    check_samples(grid,matrices,monitor,fail=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    save_json(out/'manifest.json',dict(config=clean_args(args),grid=grid.audit(matrices[3],matrices[4]),
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        device=str(device),cuda=torch.version.cuda,parameters_primal=sum(p.numel() for p in netP.parameters()),
        notes=['Independent reconstruction; not author code or a guarantee of table values.',
               'Four hidden layers plus affine head; norm dual loss; fixed-signal APR backward.',
               'gamma, generator_std and zscore are explicit reconstruction choices.',
               'Monitoring pool is separate from training and final test samples.']))
    rho=.1; previous=np.inf; global_step=0; first_outer=0
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if state['case_sha256']!=grid.fingerprint: raise ValueError('Checkpoint case differs')
        for key in ('arch','hidden','depth','gamma','generator_std','mu','zscore','dc_mode','slack_unit','penalty','dtype',
                    'inner','outer','batch','lr','bisection','monitor_samples','seed','dual_loss'):
            if state['config'][key]!=getattr(args,key): raise ValueError(f'Resume config mismatch: {key}')
        netP.load_state_dict(state['primal']); netD.load_state_dict(state['dual'])
        optP.load_state_dict(state['optP']); optD.load_state_dict(state['optD'])
        rho=state['rho']; previous=state['previous']; global_step=state['global_step']; first_outer=state['outer_completed']
        stream.rng.bit_generator.state=state['sampler_state']; stream.drawn=state['drawn']
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    def tensors(data): return tuple(torch.as_tensor(x,dtype=dtype,device=device) for x in data)
    total_steps=2*args.outer*args.inner
    def rate():
        lr=args.lr*(.1 if global_step>=.9*total_steps else 1)
        for opt in (optP,optD):
            for group in opt.param_groups: group['lr']=lr
    started=time.perf_counter()
    for outer in range(first_outer,args.outer):
        netP.train(); netD.eval(); sum_loss=0.
        for step in range(args.inner):
            data=stream.sample(args.batch); check_samples(grid,matrices,data,fail=True)
            d,c,u=tensors(data); rate(); optP.zero_grad(set_to_none=True)
            g=netP(d,c,u); objective,h,_=physics.terms(g,d,c,u)
            with torch.no_grad(): lambdas=netD(d,c,u)
            loss=(objective/1e5+(lambdas*h).sum(1)+rho/2*(h*h).sum(1)).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite primal loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in netP.parameters()):
                raise FloatingPointError('Nonfinite primal gradient')
            optP.step(); global_step+=1; sum_loss+=loss.item()
            if (step+1)%args.log_every==0: print(f'Outer {outer+1}/{args.outer}, primal update {step+1}/{args.inner}, loss={loss.item():.6g}',flush=True)
        # Algorithm 2 evaluates violation after primal learning, before dual learning.
        netP.eval(); vk=0.
        with torch.no_grad():
            for a in range(0,len(monitor[0]),args.batch):
                d,c,u=tensors(tuple(x[a:a+args.batch] for x in monitor))
                vk=max(vk,physics.residuals(netP(d,c,u),d,u).abs().max().item())
        frozen=copy.deepcopy(netD).eval(); frozen.requires_grad_(False); netD.train()
        for step in range(args.inner):
            data=stream.sample(args.batch); check_samples(grid,matrices,data,fail=True)
            d,c,u=tensors(data); rate(); optD.zero_grad(set_to_none=True)
            with torch.no_grad():
                h=physics.residuals(netP(d,c,u),d,u)
                target=frozen(d,c,u)+.1*h  # VI.A.7 deliberately fixes this step.
            error=netD(d,c,u)-target
            loss=error.norm(dim=1).mean() if args.dual_loss=='l2' else (error*error).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite dual loss')
            loss.backward(); optD.step(); global_step+=1
            if (step+1)%args.log_every==0: print(f'Outer {outer+1}/{args.outer}, dual update {step+1}/{args.inner}, loss={loss.item():.6g}',flush=True)
        used_rho=rho
        if vk>.9*previous: rho=min(2*rho,1e8)
        previous=vk
        record=dict(outer=outer+1,rho_used=used_rho,rho_next=rho,max_monitor_balance_pu=vk,
            primal_loss=sum_loss/args.inner,global_updates=global_step,training_samples_drawn=stream.drawn,
            elapsed_seconds=time.perf_counter()-started)
        print(json.dumps(record),flush=True)
        with (out/'training.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        state=dict(config=clean_args(args),case_sha256=grid.fingerprint,primal=netP.state_dict(),dual=netD.state_dict(),
                   optP=optP.state_dict(),optD=optD.state_dict(),rho=rho,previous=previous,global_step=global_step,
                   outer_completed=outer+1,sampler_state=stream.rng.bit_generator.state,drawn=stream.drawn,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        temp=out/'checkpoint.tmp'; torch.save(state,temp); temp.replace(out/'checkpoint.pt')
    print('Saved checkpoint. Run evaluate on separately generated test samples; completion is not a feasibility claim.')


def load_instances(path,grid):
    with np.load(path,allow_pickle=False) as z:
        if str(z['case_sha256'])!=grid.fingerprint: raise ValueError('Instances were generated for a different case file')
        data=tuple(z[key].copy() for key in ('d','c','u'))
    if any(not np.isfinite(x).all() for x in data): raise ValueError('Nonfinite test inputs')
    if data[0].shape[1]!=len(grid.bus_ids) or data[1].shape!=data[2].shape or data[1].shape[1]!=len(grid.gen_ids):
        raise ValueError('Test input dimensions differ')
    return data


def evaluate(args):
    # Load only your own local checkpoints: torch checkpoints use pickle metadata.
    import torch
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=argparse.Namespace(**state['config']); config.device=args.device
    grid=load_case(args.case,config.dc_mode); matrices=grid.matrices()
    if grid.fingerprint!=state['case_sha256']: raise ValueError('Checkpoint/case fingerprint mismatch')
    torch,Physics,make_models,dtype,device=get_torch(config); torch.set_default_dtype(dtype)
    net,_=make_models(grid,matrices[3],vars(config)); net.load_state_dict(state['primal']); net=net.to(device=device,dtype=dtype).eval()
    physics=Physics(grid,matrices,config.gamma,config.chunk,config.bisection,config.penalty,config.slack_unit).to(device=device,dtype=dtype)
    data=load_instances(args.data,grid)
    capacity=check_samples(grid,matrices,data,fail=True)
    predictions=[]; timing=[]
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    with torch.no_grad():
        for a in range(0,len(data[0]),args.batch):
            d,c,u=(torch.as_tensor(x[a:a+args.batch],device=device,dtype=dtype) for x in data)
            if a==0:
                for _ in range(3): physics.terms(net(d,c,u),d,c,u)
                sync()
            sync(); start=time.perf_counter(); g=net(d,c,u); obj,h,eta=physics.terms(g,d,c,u); sync()
            elapsed=time.perf_counter()-start
            timing.append(dict(batch_size=len(d),full_pipeline_ms=1000*elapsed,amortized_ms=1000*elapsed/len(d)))
            predictions.extend(g.cpu().numpy())
    # Independent NumPy reconstruction and full-contingency feasibility check.
    records=[]
    for i,g in enumerate(predictions):
        r=assess_numpy(grid,matrices,g,data[0][i],data[1][i],data[2][i],config.gamma,config.penalty,config.slack_unit,steps=config.bisection)
        h=r.pop('contingency_balance'); r.pop('signals')
        r['index']=i; r['violating_generator_contingencies']=int((abs(h)>1e-4).sum())
        r['hard_feasible']=max(r['base_balance_pu'],r['generator_bounds_pu'],r['contingency_balance_max_pu'])<=1e-4
        records.append(r)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',g=np.array(predictions),case_sha256=grid.fingerprint)
    write_csv(out/'instances.csv',records)
    summary=dict(samples=len(records),mean_instance_max_balance_pu=float(np.mean([r['contingency_balance_max_pu'] for r in records])),
        worst_balance_pu=max(r['contingency_balance_max_pu'] for r in records),
        violating_generator_contingencies=sum(r['violating_generator_contingencies'] for r in records),
        total_generator_contingencies=len(records)*len(matrices[3]),hard_feasible_instances=sum(r['hard_feasible'] for r in records),
        mean_objective=float(np.mean([r['objective'] for r in records])),
        mean_total_slack_pu=float(np.mean([r['total_slack_pu'] for r in records])),capacity=capacity,
        mean_full_pipeline_batch_ms=float(np.mean([t['full_pipeline_ms'] for t in timing])),
        mean_amortized_instance_ms=float(np.mean([t['amortized_ms'] for t in timing])),timing=timing,
        case=grid.name,case_sha256=grid.fingerprint,
        checkpoint=str(args.checkpoint),data=str(args.data),data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        physics_config={key:getattr(config,key) for key in ('gamma','dc_mode','penalty','slack_unit')},
        device=str(device),hardware=(torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor()),
        architecture=config.arch,seed=config.seed,
        note='No optimality gap is available until independently certified reference solves are compared.')
    save_json(out/'metrics.json',summary); print(json.dumps(jsonable({k:v for k,v in summary.items() if k!='timing'}),indent=2))


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def reference(args):
    from reference import solve_reference
    grid,matrices=init(args); data=load_instances(args.data,grid)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    meta=dict(case_sha256=grid.fingerprint,data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              gamma=args.gamma,dc_mode=args.dc_mode,penalty=args.penalty,slack_unit=args.slack_unit)
    meta_path=out/'reference_manifest.json'
    if meta_path.exists() and json.loads(meta_path.read_text())!=meta: raise ValueError('Existing reference run has different data or physics')
    save_json(meta_path,meta)
    for i in range(args.start,min(args.stop if args.stop is not None else len(data[0]),len(data[0]))):
        file=out/f'{i:06d}.json'
        if file.exists(): continue
        r=solve_reference(grid,matrices,data[0][i],data[1][i],data[2][i],args.gamma,args.backend,
                          args.time_limit,args.gap,args.threads,args.penalty,args.slack_unit,args.extensive)
        r['index']=i; save_json(file,r)
        print(f'Instance {i}: certified={r["certified"]}, objective={r["objective"]}, bound={r["lower_bound"]}, seconds={r["elapsed_seconds"]:.3f}',flush=True)


def compare(args):
    metrics=json.loads((Path(args.evaluation)/'metrics.json').read_text())
    meta=json.loads((Path(args.reference)/'reference_manifest.json').read_text())
    if meta['data_sha256']!=metrics['data_sha256']: raise ValueError('Evaluation/reference test datasets differ')
    for k,v in metrics['physics_config'].items():
        if meta[k]!=v: raise ValueError(f'Evaluation/reference physics mismatch: {k}')
    records=list(csv.DictReader((Path(args.evaluation)/'instances.csv').open()))
    rows=[]
    for r in records:
        p=Path(args.reference)/f'{int(r["index"]):06d}.json'
        if not p.exists(): continue
        ref=json.loads(p.read_text()); feasible=r['hard_feasible']=='True'
        J=float(r['objective']); optimum=ref['objective']
        gap=100*abs(J-optimum)/max(abs(optimum),1e-12) if feasible and ref['certified'] and optimum is not None else None
        rows.append(dict(index=int(r['index']),hard_feasible=feasible,reference_certified=ref['certified'],
                         predicted_objective=J,reference_objective=optimum,reference_bound=ref['lower_bound'],gap_percent=gap))
    valid=[r['gap_percent'] for r in rows if r['gap_percent'] is not None]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); write_csv(out/'gaps.csv',rows)
    summary=dict(total_evaluation_instances=len(records),reference_instances=len(rows),
                 feasible_certified_comparisons=len(valid),mean_gap_percent=float(np.mean(valid)) if valid else None,
                 complete_comparison=len(valid)==len(records),
                 note='A subset mean must not be reported as the full test-set paper result.')
    save_json(out/'comparison.json',summary); print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    def common(p,sampling=False,physics=False):
        p.add_argument('--case',required=True); p.add_argument('--dc-mode',choices=['matpower','reactance'],default='matpower')
        if sampling:
            p.add_argument('--generator-std',type=float,required=True,help='Not specified unambiguously in paper; record the chosen std.')
            p.add_argument('--mu',type=float,default=.5); p.add_argument('--zscore',type=float,default=1.959963984540054)
            p.add_argument('--seed',type=int,default=0)
        if physics:
            p.add_argument('--gamma',type=float,required=True,help='APR coefficient; match author setting when available.')
            p.add_argument('--penalty',type=float,default=1500); p.add_argument('--slack-unit',choices=['pu','mw'],default='pu')
    p=sub.add_parser('audit'); common(p,True); p.add_argument('--samples',type=int,default=1000); p.set_defaults(func=audit)
    p=sub.add_parser('sample'); common(p,True); p.add_argument('--samples',type=int,default=1000); p.add_argument('--out',required=True); p.set_defaults(func=sample_data)
    p=sub.add_parser('train'); common(p,True,True)
    p.add_argument('--arch',choices=['mlp','gin'],default='mlp'); p.add_argument('--hidden',type=int,default=0); p.add_argument('--depth',type=int,default=4)
    p.add_argument('--outer',type=int,default=20); p.add_argument('--inner',type=int,default=2000,help='Minibatch updates per phase, NOT dataset epochs.')
    p.add_argument('--batch',type=int,default=8); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--dual-loss',choices=['l2','mse'],default='l2'); p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--bisection',type=int,default=40); p.add_argument('--monitor-samples',type=int,default=1000)
    p.add_argument('--dtype',choices=['float64','float32'],default='float64'); p.add_argument('--device',default='auto')
    p.add_argument('--deterministic',action='store_true'); p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--resume'); p.add_argument('--out',required=True); p.set_defaults(func=train)
    p=sub.add_parser('evaluate'); p.add_argument('--case',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True); p.add_argument('--batch',type=int,default=1); p.add_argument('--device',default='auto'); p.add_argument('--out',required=True); p.set_defaults(func=evaluate)
    p=sub.add_parser('reference'); common(p,False,True); p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--backend',choices=['gurobi','scipy'],default='gurobi'); p.add_argument('--time-limit',type=float,default=3600)
    p.add_argument('--gap',type=float,default=1e-4); p.add_argument('--threads',type=int,default=24)
    p.add_argument('--start',type=int,default=0); p.add_argument('--stop',type=int); p.add_argument('--extensive',action='store_true'); p.set_defaults(func=reference)
    p=sub.add_parser('compare'); p.add_argument('--evaluation',required=True); p.add_argument('--reference',required=True); p.add_argument('--out',required=True); p.set_defaults(func=compare)
    args=parser.parse_args()
    for key in ('samples','outer','inner','batch','chunk','bisection','monitor_samples','depth','log_every'):
        if hasattr(args,key) and getattr(args,key)<=0: parser.error(f'--{key} must be positive')
    if hasattr(args,'gamma') and not 0<args.gamma<=1: parser.error('--gamma must lie in (0,1] for this implementation')
    if hasattr(args,'generator_std') and args.generator_std<0: parser.error('--generator-std must be nonnegative')
    args.func(args)

if __name__=='__main__': main()
