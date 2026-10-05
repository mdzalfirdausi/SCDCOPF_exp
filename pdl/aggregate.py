"""Aggregate evaluated seeds without inventing absent/certified reference gaps."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('evaluations',nargs='+',help='Directories containing metrics.json')
p.add_argument('--out',default='aggregate.csv')
a=p.parse_args()
rows=[]
signature=None
for directory in a.evaluations:
    path=Path(directory); r=json.loads((path/'metrics.json').read_text())
    current=(r['case_sha256'],r['data_sha256'],r['physics_config'],r['architecture'])
    if signature is not None and current!=signature: raise ValueError('Aggregate only identical cases, test data, physics and architecture')
    signature=current
    rows.append(dict(directory=str(path),seed=r['seed'],architecture=r['architecture'],
        worst_balance_pu=r['worst_balance_pu'],mean_instance_max_balance_pu=r['mean_instance_max_balance_pu'],
        violating_contingencies=r['violating_generator_contingencies'],
        violation_percent=100*r['violating_generator_contingencies']/r['total_generator_contingencies'],
        feasible_instance_percent=100*r['hard_feasible_instances']/r['samples'],
        mean_objective=r['mean_objective'],mean_total_slack_pu=r['mean_total_slack_pu'],
        mean_batch_ms=r['mean_full_pipeline_batch_ms'],mean_amortized_ms=r['mean_amortized_instance_ms']))
if len({r['architecture'] for r in rows})>1: raise ValueError('Aggregate MLP and GNN separately')
if len({r['seed'] for r in rows})!=len(rows): raise ValueError('Duplicate seeds')
with Path(a.out).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
summary={'seeds':len(rows),'five_seed_protocol_complete':len(rows)==5}
for k in rows[0]:
    if k not in ('directory','seed','architecture'):
        values=[r[k] for r in rows]
        summary[k]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else 0.}
Path(a.out).with_suffix('.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
