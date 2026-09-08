"""Diagnostic revision of the uploaded GIN-PDL trainer.

Run --check_only first. Uses the supplied dcopf_model.py PTDF/LODF helpers.
This version is a fixed-cost, fixed-Pmax, load-only experiment, not a full
reproduction of the paper's on-the-fly demand/cost/capacity sampling.
APR forward uses bounded bisection; backward evaluates the saturated APR
expression with the selected signal held fixed (not an implicit derivative
of the exact root). Final metrics are training-set diagnostics, not a
held-out benchmark. Original checkpoints are incompatible (LayerNorm).
"""
import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from dcopf_model import build_ptdf, build_lodf

# ==============================================================================
# 1. DIFFERENTIABLE PHYSICS LAYERS
# ==============================================================================

def power_balance_repair_layer(g_raw, d_total, pmin, pmax):
    """Enforces nominal power balance 1^T g = 1^T d differentiably."""
    g_total = g_raw.sum(dim=1, keepdim=True)
    pmax_total = pmax.sum(dim=1, keepdim=True)
    pmin_total = pmin.sum(dim=1, keepdim=True)

    zeta_up = (d_total - g_total) / (pmax_total - g_total + 1e-9)
    zeta_down = (g_total - d_total) / (g_total - pmin_total + 1e-9)

    condition = g_total < d_total
    g_repaired = torch.where(
        condition,
        (1.0 - zeta_up) * g_raw + zeta_up * pmax,
        (1.0 - zeta_down) * g_raw + zeta_down * pmin
    )
    return g_repaired

def differentiable_apr_layer(g_star, d_total, pmax, gamma, pmin, kg, steps=40):
    """Eq. (16) forward with n in [0,1], saturation, and failed-unit masking."""
    response = gamma * (pmax - pmin)
    failed = F.one_hot(kg, num_classes=g_star.shape[1]).bool().unsqueeze(0)
    # Branch decisions are not differentiated. Re-evaluate APR below with
    # autograd enabled so the dispatch expression provides the backward path.
    with torch.no_grad():
        low = g_star.new_zeros((g_star.shape[0], len(kg), 1))
        high = torch.ones_like(low)
        for _ in range(steps):
            signal = (low + high) / 2
            trial = torch.minimum(
                g_star.unsqueeze(1) + signal * response.unsqueeze(1),
                pmax.unsqueeze(1),
            ).masked_fill(failed, 0.0)
            short = trial.sum(dim=2, keepdim=True) < d_total.unsqueeze(1)
            low = torch.where(short, signal, low)
            high = torch.where(short, high, signal)
        signal = (low + high) / 2
    return torch.minimum(
        g_star.unsqueeze(1) + signal * response.unsqueeze(1),
        pmax.unsqueeze(1),
    ).masked_fill(failed, 0.0)


def compute_physics_loss(g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map, pmin, kg, ke):
    """Computes operational generation cost, physical thermal slacks, and ALM mismatches."""
    batch_size, num_gens = g_star.shape
    d_total = d_bus.sum(dim=1, keepdim=True)
    g_bus = torch.matmul(g_star, bus_gen_map.T)

    # 1. Base Case Power Flows
    net_injections = g_bus - d_bus
    f_star = torch.matmul(net_injections, PTDF.T)
    eta_0 = F.relu(torch.abs(f_star) - f_max)

    # 2. Line Contingencies (Ke) via LODF
    f_star_outaged = f_star[:, ke].unsqueeze(1)
    f_k_e = f_star.unsqueeze(2) + LODF[:, ke].unsqueeze(0) * f_star_outaged
    eta_k_e = F.relu(torch.abs(f_k_e) - f_max.view(1, -1, 1))

    # 3. Generator Contingencies (Kg) via APR
    g_k = differentiable_apr_layer(g_star, d_total, pmax, gamma, pmin, kg)
    h_x = g_k.sum(dim=2) - d_total  # Power balance mismatch: (Batch, Kg)

    g_k_bus = torch.matmul(g_k, bus_gen_map.T)
    net_injections_k = g_k_bus - d_bus.unsqueeze(1)
    f_k_g = torch.matmul(net_injections_k, PTDF.T)
    eta_k_g = F.relu(torch.abs(f_k_g) - f_max)

    gen_cost = torch.sum(c1 * g_star + c0, dim=1)
    total_slack = eta_0.sum(dim=1) + eta_k_e.sum(dim=(1, 2)) + eta_k_g.sum(dim=(1, 2))

    return gen_cost, total_slack, h_x

# ==============================================================================
# 2. GRAPH ISOMORPHISM NETWORK (GIN) ARCHITECTURES (Xu et al., 2019)
# ==============================================================================

class GINPrimalNet(nn.Module):
    """
    GIN-based Primal Network using universal multiset sum-aggregation
    to predict nominal active generator dispatches.
    """
    def __init__(self, in_features, hidden_dim, num_gens):
        super(GINPrimalNet, self).__init__()
        
        # Layer 1: MLP over sum-aggregated neighbors
        mlp1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv1 = GINConv(mlp1, eps=0.0, train_eps=True)

        # Layer 2: Deeper representation extraction
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv2 = GINConv(mlp2, eps=0.0, train_eps=True)

        # Readout: Maps global graph embedding to per-generator dispatches
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gens)
        )

    def forward(self, x, edge_index, batch, pmin, pmax):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h_graph = global_mean_pool(h, batch)
        
        raw_output = torch.sigmoid(self.readout(h_graph))
        g_raw = pmin + raw_output * (pmax - pmin)
        return g_raw

class GINDualNet(nn.Module):
    """
    GIN-based Dual Network predicting Lagrangian Multipliers (lambda)
    for generator contingency power balance constraints.
    """
    def __init__(self, in_features, hidden_dim, num_kg):
        super(GINDualNet, self).__init__()
        
        mlp1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv1 = GINConv(mlp1, eps=0.0, train_eps=True)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv2 = GINConv(mlp2, eps=0.0, train_eps=True)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_kg)
        )

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h_graph = global_mean_pool(h, batch)
        return self.readout(h_graph)

# ==============================================================================
# 3. DATA PIPELINE
# ==============================================================================

def create_pyg_dataset(case, load_data_np, baseMVA):
    bus_df = case['bus']
    branch_df = case['branch']
    gen_df = case['gen']
    num_buses = len(bus_df)

    bus_idx_map = {bus_id: i for i, bus_id in enumerate(bus_df['bus_i'].values)}
    edge_source = [bus_idx_map[i] for i in branch_df['bus_i'].values]
    edge_target = [bus_idx_map[j] for j in branch_df['bus_j'].values]
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long)

    pmax_node = np.zeros(num_buses)
    pmin_node = np.zeros(num_buses)
    for _, row in gen_df.iterrows():
        b_idx = bus_idx_map[row['bus_i']]
        pmax_node[b_idx] += row['Pmax'] / baseMVA
        pmin_node[b_idx] += row['Pmin'] / baseMVA

    dataset = []
    for s in range(len(load_data_np)):
        pd_s = load_data_np[s]
        x_features = np.stack([pd_s, pmax_node, pmin_node], axis=1)
        x_tensor = torch.tensor(x_features, dtype=torch.float32)

        data = Data(x=x_tensor, edge_index=edge_index)
        data.Pd = torch.tensor(pd_s, dtype=torch.float32)
        dataset.append(data)
    return dataset

# ==============================================================================
# 4. PRIMAL-DUAL ALM TRAINING LOOP
# ==============================================================================

def preflight_capacity(load_data_np, pmin, pmax, kg, gens, baseMVA):
    """Necessary hard-feasibility checks, without solver labels or load shedding."""
    lower = pmin.detach().cpu().numpy().ravel()
    upper = pmax.detach().cpu().numpy().ravel()
    outages = kg.detach().cpu().numpy()
    if np.any(upper < lower):
        raise ValueError("Pmax below Pmin.")
    total = load_data_np.sum(axis=1)
    if not len(total):
        raise ValueError("Empty demand dataset.")
    base_bad = (total < lower.sum()-1e-4) | (total > upper.sum()+1e-4)
    survivor_upper = upper.sum()-upper[outages]
    survivor_lower = lower.sum()-lower[outages]
    deficit = np.maximum(total[:, None]-survivor_upper[None, :], 0)
    excess = np.maximum(survivor_lower[None, :]-total[:, None], 0)
    row, col = np.unravel_index(deficit.argmax(), deficit.shape)
    k = outages[col]
    identity = gens.iloc[k]['gen_ID'] if 'gen_ID' in gens else k
    print(f"Capacity check: demand range={total.min()*baseMVA:.6f}..{total.max()*baseMVA:.6f} MW; "
          f"nominal Pmax sum={upper.sum()*baseMVA:.6f} MW")
    print(f"Worst capacity deficit: CSV data row {row} (zero-based), generator {identity}, "
          f"demand={total[row]*baseMVA:.6f} MW, "
          f"surviving Pmax={survivor_upper[col]*baseMVA:.6f} MW, "
          f"deficit={deficit[row,col]:.6f} p.u. ({deficit[row,col]*baseMVA:.6f} MW)")
    print(f"Old APR's expected max shortage residual at gamma=1, "
          f"valid nominal balance and bounds: {0.9*deficit.max():.6f} p.u.")
    bad = base_bad | (deficit.max(axis=1)>1e-4) | (excess.max(axis=1)>1e-4)
    if np.any(bad):
        raise ValueError(f"{bad.sum()}/{len(total)} samples fail necessary capacity checks. "
                         "Zero hard violations are impossible for these inputs. "
                         "Verify units, generator status/bounds, outage set and demand sampling. "
                         "No samples were silently removed or rescaled.")
    print("Necessary capacity checks pass. For gamma<1, APR adequacy needs additional checking.")


def train_gin_pdl(case_name, outer_K=20, inner_L=50, batch_size=32, check_only=False, gamma_value=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing GIN-PDL Solver for {case_name.upper()} on {device} ---")

    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA', 'bus', 'gen', 'gencost', 'branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]

    # Keep dispatchable units even if they are not eligible outage scenarios.
    # Align all bus-indexed arrays with the PTDF helper's sorted bus order.
    gen_status = (case['gen']['status'].to_numpy() > 0
                  if 'status' in case['gen'] else np.ones(len(case['gen']), dtype=bool))
    if len(case['gencost']) != len(case['gen']):
        raise ValueError("Expected one aligned active-cost row per generator.")
    case['gen'] = case['gen'].iloc[np.flatnonzero(gen_status)].reset_index(drop=True)
    case['gencost'] = case['gencost'].iloc[np.flatnonzero(gen_status)].reset_index(drop=True)
    if 'status' in case['branch']:
        case['branch'] = case['branch'].loc[case['branch']['status'] > 0]
    case['branch'] = case['branch'].reset_index(drop=True)
    case['bus'] = case['bus'].sort_values('bus_i').reset_index(drop=True)
    # The supplied helper uses 1/x and has no phase-shift flow offset.
    # Refuse unsupported branches rather than silently evaluate another grid.
    if 'ratio' in case['branch']:
        taps = case['branch']['ratio'].to_numpy(dtype=float)
        if np.any(~np.isclose(taps, 0) & ~np.isclose(taps, 1)):
            print("NOTE: supplied build_ptdf ignores transformer tap ratios; "
                  "this run uses the original helper's simplified 1/x DC model.")
    if 'angle' in case['branch'] and np.any(~np.isclose(case['branch']['angle'], 0)):
        raise ValueError("Nonzero transformer phase shifts require affine DC flow offsets in dcopf_model.py.")

    bus_list = sorted(case['bus']['bus_i'].tolist())
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])

    PTDF_np, _ = build_ptdf(case['bus'], case['branch'], ref_bus)
    LODF_np = build_lodf(PTDF_np, case['branch'], bus_list)

    PTDF = torch.tensor(PTDF_np, dtype=torch.float32, device=device)
    LODF = torch.tensor(LODF_np, dtype=torch.float32, device=device)
    f_max = torch.tensor(case['branch']['rateA'].values / baseMVA, dtype=torch.float32, device=device)

    # MATPOWER rateA=0 means unconstrained; do not penalize it as zero capacity.
    f_max = torch.where(f_max > 0, f_max, torch.full_like(f_max, float('inf')))
    # Supplied build_lodf leaves islanding columns zero and valid diagonals -1.
    ke = torch.as_tensor(np.flatnonzero(np.isclose(np.diag(LODF_np), -1.0)),
                         dtype=torch.long, device=device)
    num_gens = len(case['gen'])
    pmax = torch.tensor(case['gen']['Pmax'].values / baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    pmin = torch.tensor(case['gen']['Pmin'].values / baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    gamma = torch.full_like(pmax, gamma_value)
    kg = torch.where(((pmax - pmin) > 0).squeeze(0) & (pmin >= 0).squeeze(0))[0]
    if not len(kg):
        raise ValueError("No eligible generator contingencies.")
    c1 = torch.tensor(case['gencost']['c1'].values * baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    c0 = torch.tensor(case['gencost']['c0'].values, dtype=torch.float32, device=device).unsqueeze(0)

    bus_idx_map = {bus_id: i for i, bus_id in enumerate(bus_list)}
    bus_gen_map_np = np.zeros((len(bus_list), num_gens))
    for j, bus_i in enumerate(case['gen']['bus_i']):
        bus_gen_map_np[bus_idx_map[bus_i], j] = 1.0
    bus_gen_map = torch.tensor(bus_gen_map_np, dtype=torch.float32, device=device)

    # Load dataset features
    csv_path = f"data/{case_name}_generated_data.csv"
    if not os.path.exists(csv_path):
        csv_path = f"data/{case_name}_generated_loads.csv"

    df_csv = pd.read_csv(csv_path)
    pd_cols = [f"Bus_{b}_Pd" for b in bus_list]
    if all(c in df_csv.columns for c in pd_cols):
        load_data_np = df_csv[pd_cols].values / baseMVA
    else:
        missing = [c for c in pd_cols if c not in df_csv.columns]
        raise ValueError(f"Missing named MW load columns: {missing}. Do not infer columns by position.")
    if not np.isfinite(load_data_np).all():
        raise ValueError("Demand data contains NaN or infinity.")
    if any(c.startswith('Gen_') for c in df_csv.columns):
        print("NOTE: load-only experiment: Gen_* CSV columns are not used. "
              "Costs and bounds come from the workbook; paper-wide input sampling is not reproduced.")
    preflight_capacity(load_data_np, pmin, pmax, kg, case['gen'], baseMVA)
    print(f"Samples={len(load_data_np)}, generators={num_gens}, "
          f"Kg={len(kg)}, non-islanding Ke={len(ke)}, gamma={gamma_value}")
    if check_only:
        return

    dataset = create_pyg_dataset(case, load_data_np, baseMVA)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate GIN Networks (3 input features: Pd, Pmax, Pmin)
    net_P = GINPrimalNet(in_features=3, hidden_dim=64, num_gens=num_gens).to(device)
    net_D = GINDualNet(in_features=3, hidden_dim=64, num_kg=len(kg)).to(device)

    opt_P = optim.Adam(net_P.parameters(), lr=1e-4)
    opt_D = optim.Adam(net_D.parameters(), lr=1e-4)

    # ALM Hyperparameters
    rho = 0.1
    rho_max = 1e8
    tau = 0.9
    alpha = 2.0
    v_prev = float('inf')

    print("Beginning Training Loop...")
    for k in range(outer_K):
        # -------------------------------------------------------------
        # Phase 1: Primal GIN Learning (Minimize Cost + Slacks + ALM)
        # -------------------------------------------------------------
        net_P.train()
        net_D.eval()
        for _ in range(inner_L):
            for batch in dataloader:
                batch = batch.to(device)
                opt_P.zero_grad()

                g_raw = net_P(batch.x, batch.edge_index, batch.batch, pmin, pmax)
                d_bus = batch.Pd.view(batch.num_graphs, -1)
                d_total = d_bus.sum(dim=1, keepdim=True)
                g_star = power_balance_repair_layer(g_raw, d_total, pmin, pmax)

                gen_cost, total_slack, h_x = compute_physics_loss(
                    g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map, pmin, kg, ke
                )

                lambdas = net_D(batch.x, batch.edge_index, batch.batch).detach()

                alm_linear = torch.sum(lambdas * h_x, dim=1)
                alm_quad = torch.sum((rho / 2.0) * (h_x ** 2), dim=1)
                loss_P = torch.mean((gen_cost + 1500.0 * total_slack) / 1e5 + alm_linear + alm_quad)

                loss_P.backward()
                opt_P.step()

        # -------------------------------------------------------------
        # Phase 2: Dual GIN Learning (Track Multipliers)
        # -------------------------------------------------------------
        net_P.eval()
        net_D.train()
        net_D_frozen = copy.deepcopy(net_D).eval()
        net_D_frozen.requires_grad_(False)

        max_mismatch = 0.0
        for _ in range(inner_L):
            for batch in dataloader:
                batch = batch.to(device)
                opt_D.zero_grad()

                with torch.no_grad():
                    g_raw = net_P(batch.x, batch.edge_index, batch.batch, pmin, pmax)
                    d_bus = batch.Pd.view(batch.num_graphs, -1)
                    d_total = d_bus.sum(dim=1, keepdim=True)
                    g_star = power_balance_repair_layer(g_raw, d_total, pmin, pmax)
                    _, _, h_x = compute_physics_loss(
                        g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map, pmin, kg, ke
                    )
                    lambda_k = net_D_frozen(batch.x, batch.edge_index, batch.batch)
                    max_mismatch = max(max_mismatch, torch.max(torch.abs(h_x)).item())

                lambda_est = net_D(batch.x, batch.edge_index, batch.batch)
                # Paper VI.A.7 deliberately fixes the dual step at 0.1.
                target = (lambda_k + 0.1 * h_x).detach()
                loss_D = F.mse_loss(lambda_est, target)

                loss_D.backward()
                opt_D.step()

        # -------------------------------------------------------------
        # Phase 3: Penalty Coefficient Update
        # -------------------------------------------------------------
        v_k = max_mismatch
        if v_k > tau * v_prev:
            rho = min(alpha * rho, rho_max)
        v_prev = v_k

        print(f"Outer Iter [{k+1:2d}/{outer_K}] | rho: {rho:.2e} | Max |h(x)|: {v_k:.4f} p.u.")

    # Explicitly evaluate hard feasibility; saving weights is not convergence.
    net_P.eval()
    max_base = max_bounds = max_cont = max_slack = 0.0
    violations = total_pairs = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            d_bus = batch.Pd.view(batch.num_graphs, -1)
            d_total = d_bus.sum(1, keepdim=True)
            g_raw = net_P(batch.x, batch.edge_index, batch.batch, pmin, pmax)
            g_star = power_balance_repair_layer(g_raw, d_total, pmin, pmax)
            _, slack, h = compute_physics_loss(
                g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma,
                bus_gen_map, pmin, kg, ke)
            gk = differentiable_apr_layer(g_star, d_total, pmax, gamma, pmin, kg)
            survivor = ~F.one_hot(kg, num_classes=num_gens).bool().unsqueeze(0)
            b0 = torch.maximum(F.relu(pmin-g_star), F.relu(g_star-pmax)).max()
            bk = torch.maximum(F.relu(pmin.unsqueeze(1)-gk),
                               F.relu(gk-pmax.unsqueeze(1))).masked_fill(~survivor, 0).max()
            max_bounds = max(max_bounds, b0.item(), bk.item())
            max_base = max(max_base, (g_star.sum(1, keepdim=True)-d_total).abs().max().item())
            max_cont = max(max_cont, h.abs().max().item())
            max_slack = max(max_slack, slack.max().item())
            violations += int((h.abs() > 1e-4).sum().item())
            total_pairs += h.numel()
    print(f"Training-set check: base balance={max_base:.3e} p.u.; "
          f"generator bounds={max_bounds:.3e} p.u.; contingency balance={max_cont:.3e} p.u. "
          f"({max_cont*baseMVA:.6f} MW); violations={violations}/{total_pairs}; "
          f"max total thermal slack per sample={max_slack:.6g} p.u.")
    if max(max_base, max_bounds, max_cont) > 1e-4:
        print("NOT FEASIBLE at 1e-4 p.u. tolerance. Checkpoint is diagnostic only.")
    else:
        print("Training-set hard checks pass at 1e-4 p.u.; held-out evaluation still required.")
    os.makedirs('data/models', exist_ok=True)
    torch.save(net_P.state_dict(), f"data/models/{case_name}_gin_revised_primal.pth")
    torch.save(net_D.state_dict(), f"data/models/{case_name}_gin_revised_dual.pth")
    print(f"\n[+] GIN-PDL training finished; revised weights saved for {case_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GIN-PDL for SCOPF")
    parser.add_argument('--case', type=str, default="pglib_opf_case300_ieee")
    parser.add_argument('--outer_K', type=int, default=20)
    parser.add_argument('--inner_L', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--check_only', action='store_true', help='Check data and capacity; do not train.')
    parser.add_argument('--gamma', type=float, default=1.0, help='APR coefficient; match the intended optimization benchmark.')
    args = parser.parse_args()
    if not (0 < args.gamma <= 1):
        parser.error('--gamma must be in (0,1] for this diagnostic version.')

    train_gin_pdl(args.case, args.outer_K, args.inner_L, args.batch_size, args.check_only, args.gamma)