"""Small end-to-end integration check; NOT a paper benchmark."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(); p.add_argument('--out',default='smoke_results'); p.add_argument('--arch',choices=['mlp','gin'],default='mlp')
a=p.parse_args()
if importlib.util.find_spec('torch') is None:
    raise SystemExit('Activate your working PyTorch environment first. This check requires torch.')
root=Path(__file__).resolve().parent; out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
case=str(root/'examples'/'synthetic3.m')
def run(*args): subprocess.run([sys.executable,str(root/'run.py'),*map(str,args)],check=True,cwd=root)
subprocess.run([sys.executable,'-m','unittest','-v','test_reproduction.py'],check=True,cwd=root)
run('train','--case',case,'--arch',a.arch,'--gamma',1,'--generator-std',0,'--mu',0,
    '--outer',1,'--inner',3,'--hidden',16,'--monitor-samples',16,'--batch',4,'--log-every',1,'--out',out/'train')
run('sample','--case',case,'--generator-std',0,'--mu',0,'--seed',100000,'--samples',4,'--out',out/'test')
run('evaluate','--case',case,'--checkpoint',out/'train'/'checkpoint.pt','--data',out/'test'/'instances.npz','--out',out/'evaluation')
run('reference','--case',case,'--data',out/'test'/'instances.npz','--gamma',1,'--backend','scipy','--time-limit',30,'--out',out/'reference')
run('compare','--evaluation',out/'evaluation','--reference',out/'reference','--out',out/'comparison')
print(f'Smoke pipeline completed: {out}. Three training updates are not an optimality experiment.')
