# PDL-SCOPF: paper-method reconstruction and GNN comparison

This package implements the main PDL-SCOPF method from Seonho Park and Pascal
Van Hentenryck, *Self-Supervised Learning for Large-Scale Preventive Security
Constrained DC Optimal Power Flow*, IEEE Transactions on Power Systems 40(3),
2025, DOI https://doi.org/10.1109/TPWRS.2024.3498705.

**This is an independent reconstruction, not the authors' code or a verified
reproduction of their numerical tables.** It supplies the training and evaluation
pipeline needed to attempt that reproduction. It also supplies an experimental
GIN architecture under the same physics. Published hardware timings, original
random streams, case revisions, and unspecified settings have not been recovered.
The supervised Naive/LD and penalty-only comparison networks are not implemented;
the target here is the PDL-SCOPF column and an additional GNN comparison.

## What is implemented

- Numeric MATPOWER/PGLib `.m` input, or the user's `.xlsx` schema.
- Retains online generators that are ineligible as outage scenarios, including
  zero-capacity units and negative-Pmin dispatchable loads. Removes offline units.
- Explicit bus ID mapping and generator-to-bus association, including multiple
  generators at one bus. Cost rows align on generator IDs for Excel input.
- Sparse-factorized affine DC PTDF with tap ratios and phase shifts, or a selectable
  simplified reactance-only model. LODF excludes bridge/islanding contingencies.
- Online sampling of joint truncated correlated demands and correlated generator
  cost/upper-bound perturbations. No optimal solutions are used for training.
- Sigmoid generator bounds, nominal power balance repair, bounded APR bisection,
  surviving-generator saturation, failed generator masking, and both contingency types.
- Base/generator/line thermal overload slacks, linear generation cost, AL primal
  loss, instance-dependent dual network and frozen dual update targets.
- Four-hidden-layer MLP baseline and an optional edge-aware GIN implemented directly
  in PyTorch. The GNN has generator-local outputs rather than one pooled vector
  that loses generator location information. It is an extension, not the paper model.
- Checkpointed contingency chunks: all eligible contingencies contribute to loss;
  chunks avoid retaining every dense flow activation during backward.
- Independent NumPy checking and a CCGA-style MILP reference, with SciPy/HiGHS for
  small cases or Gurobi for larger cases. Full-contingency objective upper bounds
  are compared with master lower bounds before a reference is called certified.
- Fixed test datasets, case/data fingerprints, checkpoint/resume, run metadata,
  feasibility/slack reports, objective-gap comparisons, and seed aggregation.

## Paper settings and disclosed choices

| Item | This implementation | Status |
|---|---|---|
| MLP/GNN default | MLP | Paper uses fully connected nets |
| Training inputs | Nonzero bus demands, all linear costs, all generator upper bounds | Matches stated x=(d,c,upper) |
| Number of outer iterations | 20 | Paper VI.A.6 |
| Inner iterations | 2,000 **minibatch updates per phase**, not epochs | Paper VI.A.6 |
| Batch size | 8 | Paper VI.A.6 |
| Total updates / training input draws | 80,000 / 640,000 | Both primal and dual phases counted |
| Adam learning rate | 1e-4, multiplied by 0.1 after 90% of total updates | Paper VI.A.6 |
| Primal penalty | 0.1 initially; double if v_k>0.9*v_previous; capped at 1e8 | Paper VI.A.7 |
| Dual update step | Fixed 0.1, independent of primal penalty | Explicit paper VI.A.7 choice |
| Objective divisor / thermal penalty | 1e5 / 1500 | Paper VI.A.7 / VI.A.3 |
| Test size / seeds | 1,000 / run five separate seeds | Paper VI.A |
| Hard-violation tolerance | 1e-4 p.u. | Paper VI.B |
| Demand correlation / bounds | 0.5 off diagonal, +/-50% | Paper VI.A.2 |
| Cost and capacity factor correlation | 0.8; independently drawn cost/capacity vectors | Paper VI.A.2 |
| APR gamma | **Required CLI argument** | Numeric value is not clearly specified in the supplied paper |
| Generator-factor standard deviation | **Required CLI argument** | Not clearly specified in the supplied paper |
| Demand z-score | 1.95996398454, configurable | Interprets central 95% interval; literal one-sided 95th percentile is 1.64485362695 |
| Demand upper endpoint | (1+mu)*d0 | Follows +/-50% prose; Eq.43 has an extra mu typographical inconsistency |
| Truncation sampler | 1D numerical latent CDF, then conditional truncated normal draws | Joint TN approximation; not clipping; quadrature error remains |
| Hidden layers | Four ReLU hidden layers plus affine output head; width round(1.5*dim(x)) | Layer-count wording ambiguous; `--depth` is configurable |
| Dual norm | L2 norm as written in Eq.34 | `--dual-loss mse` is available for comparison |
| APR backward | Selected bisection signal is held fixed; differentiate saturated Eq.16 expression | Explicit reconstruction choice; paper does not provide a detailed Jacobian |
| Bisection iterations | 40 | Not specified; configurable |
| Precision | float64 | Paper precision unspecified; `--dtype float32` for sensitivity/performance |
| Penalty slack units | p.u. by default | `--slack-unit mw` changes relative penalty by baseMVA; confirm author convention |
| Violation monitor | Separate fixed 1,000-input monitoring pool | Explicit interpretation of v_k for online training; not final test data |
| Reference algorithm | CCGA-style, rebuilt MILP masters | Same objective/constraints, **not** authors' runtime implementation |

Do not label any selected gamma/variance combination the "paper setting" without
verification from the authors. Record sensitivity runs when these values differ.
All settings and the case hash are saved. Constants and quadratic generation cost
terms are excluded because Model 1 uses c^T g, not a quadratic cost.

## Setup

Use your working `(pytorch)` environment. The neural code requires PyTorch 2.2+
and does not require torch_geometric. Dependency installation:

```powershell
python -m pip install -r requirements.txt
python -c "import torch, numpy, scipy; print(torch.__version__, torch.cuda.is_available())"
python -m unittest -v test_reproduction.py
```

If PyTorch is not installed, use the official selector at
https://docs.pytorch.org/get-started/locally/ for your CUDA version. Do not replace
a working installation just to match the file names in this package. Gurobi is
optional for training; install `gurobipy` with a suitable license for reference runs.

## Quick end-to-end check in your working environment

```powershell
python smoke.py --arch mlp --out smoke_mlp
python smoke.py --arch gin --out smoke_gin
```

These run the included tests plus a tiny synthetic training/evaluation/MILP
pipeline. They do not need any of your Excel files. They verify integration, not
paper performance. See VALIDATION.md for which tests were actually run here.

## First run: audit the paper's smallest case

Table I cases and counts are checked against the input data:

| Case | Buses | Generators | Nonzero loads | Lines | Generator outages | Line outages |
|---|---:|---:|---:|---:|---:|---:|
| 300_ieee | 300 | 69 | 201 | 411 | 57 | 322 |
| 1354_pegase | 1354 | 260 | 673 | 1991 | 193 | 1430 |
| 1888_rte | 1888 | 290 | 1000 | 2531 | 290 | 1567 |
| 3022_goc | 3022 | 327 | 1574 | 4135 | 327 | 3180 |
| 4917_goc | 4917 | 567 | 2619 | 6726 | 567 | 5066 |
| 6515_rte | 6515 | 684 | 3673 | 9037 | 657 | 6474 |

**Case14 is a debugging case, not one of the paper's benchmarks.** Counts alone
are not a case-version match: compare hashes and original author data if available.
The paper's `1354_peg` label is mapped to the usual `1354_pegase` filename here.

Commands below use **illustrative** gamma=0.05 and generator-std=0.1. These values
are NOT verified paper settings. Replace them consistently when the original
settings are known. The required CLI flags prevent silent assumptions.

```powershell
python run.py audit --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --generator-std 0.1 --samples 1000
```

If aggregate capacity checks fail, investigate case units, load definition,
status, and bounds. The program never discards samples, changes capacities,
rescales demand, relaxes generator balance, or reduces the contingency set to
manufacture feasibility. For gamma<1 these capacity checks are necessary but not
sufficient: APR headroom and the nominal dispatch matter too.

## Train the MLP baseline

```powershell
python run.py train --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --arch mlp --gamma 0.05 --generator-std 0.1 --seed 0 --out runs\mlp_seed0
```

Default schedule is the full 20 x (2000 primal + 2000 dual) update schedule.
Inputs are freshly sampled each update, including costs and upper bounds.
A quick *debug-only* run is:

```powershell
python run.py train --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --arch mlp --gamma 0.05 --generator-std 0.1 --outer 1 --inner 5 --hidden 32 --monitor-samples 16 --log-every 1 --out runs\smoke
```

A checkpoint can be resumed with the original arguments and
`--resume runs\mlp_seed0\checkpoint.pt`. Learning-rate state, optimizer state,
sampling state, and random-number states are saved. Resume is at outer-iteration
boundaries. Passing a different schedule or physical configuration is rejected.

## Compare the GNN

```powershell
python run.py train --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --arch gin --hidden 64 --gamma 0.05 --generator-std 0.1 --seed 0 --out runs\gin_seed0
```

The sampled inputs, constraints, loss, repair/APR layers and update counts match
the MLP run. Architecture/parameter counts differ and are recorded. Repeat seeds
0, 1, 2, 3, 4 for each architecture; do not infer GNN superiority from one seed.

## Generate held-out inputs AFTER training

The test generator uses a separate seed and saves actual arrays, so both networks
and the solver can evaluate exactly the same inputs. Keep the distribution options
identical to training. Do not use this dataset for tuning checkpoints.

```powershell
python run.py sample --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --generator-std 0.1 --seed 100000 --samples 1000 --out test300
python run.py evaluate --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --checkpoint runs\mlp_seed0\checkpoint.pt --data test300\instances.npz --out results\mlp_seed0
python run.py evaluate --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --checkpoint runs\gin_seed0\checkpoint.pt --data test300\instances.npz --out results\gin_seed0
```

Outputs include maximum contingency mismatch, mean per-instance maximum mismatch,
counts above 1e-4, hard-feasible instance counts, total thermal slacks, objectives,
and warmed-up inference times. The full network/repair/APR/thermal-loss pipeline
is timed with CUDA synchronization; disk I/O and independent NumPy auditing are
excluded. Report batch=1 or batch=8 explicitly. Amortized per-sample batch time is
not single-instance latency. Precision and GPU model can materially affect speed.

The NumPy checker independently reconstructs contingencies with the configured
neural bisection step count; it does not reuse the PyTorch layer. The separate
solver reference uses 60 steps. The unit test checks cross-implementation agreement.

## Solve reference instances and calculate objective gaps

```powershell
python run.py reference --case ..\excel_outputs\pglib_opf_case300_ieee.xlsx --data test300\instances.npz --gamma 0.05 --out reference300 --backend gurobi --time-limit 100000 --threads 24
python run.py compare --evaluation results\mlp_seed0 --reference reference300 --out comparisons\mlp_seed0
python run.py compare --evaluation results\gin_seed0 --reference reference300 --out comparisons\gin_seed0
```

`reference` defaults to all 1000 inputs, may take substantial compute time, and
resumes by skipping existing per-instance files. Use `--start 0 --stop 5` for an
initial solver check. Use a new output folder to retry an unfinished reference
with a different time limit. `--backend scipy` and `--extensive` are for small
verification cases; avoid the extensive model on large networks.

A reference result includes an incumbent, full-contingency objective, lower bound,
relative certificate, solve time, and master history. A time-limited incumbent is
not called ground truth. `compare` computes paper-style absolute relative objective
gaps only for hard-feasible predictions with certified reference solutions. It
reports coverage and refuses data/physical-setting mismatches. A subset mean is
not a complete test-set result. Runtime speedup against this rebuilt reference is
not a reproduction of the authors' Gurobi/CCGA timings.

## Aggregate five seeds

```powershell
python aggregate.py results\mlp_seed0 results\mlp_seed1 results\mlp_seed2 results\mlp_seed3 results\mlp_seed4 --out mlp_five_seeds.csv
python aggregate.py results\gin_seed0 results\gin_seed1 results\gin_seed2 results\gin_seed3 results\gin_seed4 --out gin_five_seeds.csv
```

Preserve all per-instance CSV/JSON files. The aggregator includes both the average
of seed-wise worst violations and average per-instance maxima, so they are not
silently conflated. Objective-gap comparisons remain in the `comparisons` folders.

## Meaning of feasibility

The paper uses **soft thermal constraints**: positive overload slacks may be
present even when all hard constraints pass. A saved checkpoint is not a feasible
solution certificate. The code fails on capacity-infeasible input batches rather
than hiding an irreducible training residual. No neural training method is given
a global feasibility or global optimality guarantee here.

## Implementation limits

- This code has no dependency on the previous `dcopf_model.py` or its simplified
  APR/PTDF behavior. It uses its own matrices and reference solver.
- Numeric MATPOWER `.m` matrices only: MATLAB expressions/functions are never
  executed. Excel columns follow the attached user's schema. Active-power costs
  only are read; reactive generator costs are irrelevant to DC-SCOPF.
- Single connected in-service network is required. Base-case islands raise an error.
- Fixed bus shunts are not added as extra demand because Model 1 uses the provided
  demand vector. Verify any author-specific preprocessing separately.
- Full PTDF/LODF arrays are still dense; large cases require substantial RAM/GPU
  memory. Activation checkpointing does not make those matrices sparse. Use
  `--chunk 8` to reduce flow activation memory, or `--dtype float32` only after
  checking tolerance sensitivity. Numerical timings have not been optimized to
  reproduce Table VIII.
- The MILP reference uses all provisional generator-response vectors, which can
  still be large; its master matrices are rebuilt, so author timing comparisons
  are not warranted. The Gurobi backend requires your license and installation.
- Exact table replication needs original case revision, gamma, perturbation
  variances, units, seeds, layer interpretation and backward convention. The
  paper alone does not establish all of these. No published results are bundled
  as if they had been produced by this implementation.

See `VALIDATION.md` for checks actually performed in the creation environment.
