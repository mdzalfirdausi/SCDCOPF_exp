# Validation actually performed

Creation environment: Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0.

- All Python source files pass `py_compile`.
- Seven numerical tests passed:
  1. Affine PTDF and LODF match direct angle-based solves, including a non-unit
     transformer tap and nonzero phase shift, before/after each eligible outage.
  2. APR handles feasible balance, saturation, the failed generator and infeasible
     surviving capacity without letting the system signal exceed one.
  3. The previous leaky-APR dispatch-independent shortage residual is reproduced.
  4. Joint truncated-normal sampling matches an independent rejection-sampling
     mean/covariance check and does not create point masses at clipping bounds.
  5. The CCGA-style reference agrees with the extensive MILP on a synthetic case;
     returned dispatches pass independent full-contingency checking.
  6. Infeasible reference inputs are not marked certified.
  7. MATPOWER parsing retains zero-capacity generators while excluding their outages.
- CLI `sample` and `reference --backend scipy` completed on three synthetic inputs.
  Each reference was certified, with objective and lower bound both
  3561.1111111111113, in two master iterations. These are synthetic validation
  results, not results for a PGLib case or any published table.
- `run.py --help` and all source syntax checks passed.

## Not executed here

PyTorch is not installed in the creation environment. An attempt to install its
CPU package was unsuccessful. The PyTorch unit test is included but was skipped;
MLP/GNN training, autograd integration, checkpoint resume and neural evaluation
have not been executed here. The Gurobi backend was not executed. Full paper-scale
cases, their original revisions and author's exact sampled data were not supplied.
No paper table, neural feasibility, gap or speedup has been reproduced here.

Run the following in your existing working PyTorch environment:

```powershell
python smoke.py --arch mlp --out smoke_mlp
python smoke.py --arch gin --out smoke_gin
```

These execute the neural tests and a tiny synthetic training/evaluation/reference
pipeline. Only proceed to the long case300 experiments once both complete. The
smoke dataset and settings are deliberately synthetic and are not paper settings.
