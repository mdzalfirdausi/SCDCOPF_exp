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

def differentiable_apr_layer(g_star, d_total, pmax, gamma):
    """Algebraic APR evaluation with leaky relaxation avoiding zero gradients."""
    batch_size, num_gens = g_star.shape
    delta = gamma * (pmax - g_star)
    total_delta = delta.sum(dim=1, keepdim=True)
    reserve_k = total_delta - delta

    n_k_raw = g_star / (reserve_k + 1e-9)
    n_k = torch.where(n_k_raw > 1.0, 1.0 + 0.1 * (n_k_raw - 1.0), n_k_raw)

    g_prov = g_star.unsqueeze(1) + n_k.unsqueeze(2) * delta.unsqueeze(1)
    mask = torch.eye(num_gens, device=g_star.device).bool().unsqueeze(0)
    g_k = g_prov.masked_fill(mask, 0.0)

    return g_k

def compute_physics_loss(g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map):
    """Computes operational generation cost, physical thermal slacks, and ALM mismatches."""
    batch_size, num_gens = g_star.shape
    d_total = d_bus.sum(dim=1, keepdim=True)
    g_bus = torch.matmul(g_star, bus_gen_map.T)

    # 1. Base Case Power Flows
    net_injections = g_bus - d_bus
    f_star = torch.matmul(net_injections, PTDF.T)
    eta_0 = F.relu(torch.abs(f_star) - f_max)

    # 2. Line Contingencies (Ke) via LODF
    f_star_outaged = f_star.unsqueeze(1)
    f_k_e = f_star.unsqueeze(2) + LODF.unsqueeze(0) * f_star_outaged
    eta_k_e = F.relu(torch.abs(f_k_e) - f_max.view(1, -1, 1))

    # 3. Generator Contingencies (Kg) via APR
    g_k = differentiable_apr_layer(g_star, d_total, pmax, gamma)
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv1 = GINConv(mlp1, eps=0.0, train_eps=True)

        # Layer 2: Deeper representation extraction
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv1 = GINConv(mlp1, eps=0.0, train_eps=True)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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

def train_gin_pdl(case_name, outer_K=20, inner_L=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing GIN-PDL Solver for {case_name.upper()} on {device} ---")

    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA', 'bus', 'gen', 'gencost', 'branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]

    # Filter out inactive generators
    zero_gen_idx = [
        num for num, i in enumerate(case['gen'].Pmax.values / baseMVA)
        if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or
           (case['gen'].Pmin.values / baseMVA)[num] < 0
    ]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    bus_list = sorted(case['bus']['bus_i'].tolist())
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])

    PTDF_np, _ = build_ptdf(case['bus'], case['branch'], ref_bus)
    LODF_np = build_lodf(PTDF_np, case['branch'], bus_list)

    PTDF = torch.tensor(PTDF_np, dtype=torch.float32, device=device)
    LODF = torch.tensor(LODF_np, dtype=torch.float32, device=device)
    f_max = torch.tensor(case['branch']['rateA'].values / baseMVA, dtype=torch.float32, device=device)

    num_gens = len(case['gen'])
    pmax = torch.tensor(case['gen']['Pmax'].values / baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    pmin = torch.tensor(case['gen']['Pmin'].values / baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    gamma = torch.tensor([1.0] * num_gens, dtype=torch.float32, device=device).unsqueeze(0)
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
        load_data_np = df_csv.iloc[:, :len(bus_list)].values / baseMVA

    dataset = create_pyg_dataset(case, load_data_np, baseMVA)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate GIN Networks (3 input features: Pd, Pmax, Pmin)
    net_P = GINPrimalNet(in_features=3, hidden_dim=64, num_gens=num_gens).to(device)
    net_D = GINDualNet(in_features=3, hidden_dim=64, num_kg=num_gens).to(device)

    opt_P = optim.Adam(net_P.parameters(), lr=1e-3)
    opt_D = optim.Adam(net_D.parameters(), lr=1e-3)

    # ALM Hyperparameters
    rho = 0.1
    rho_max = 1e6
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
                    g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map
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
        net_D_frozen = copy.deepcopy(net_D)

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
                        g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map
                    )
                    lambda_k = net_D_frozen(batch.x, batch.edge_index, batch.batch)
                    max_mismatch = max(max_mismatch, torch.max(torch.abs(h_x)).item())

                lambda_est = net_D(batch.x, batch.edge_index, batch.batch)
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

    os.makedirs('data/models', exist_ok=True)
    torch.save(net_P.state_dict(), f"data/models/{case_name}_gin_primal.pth")
    torch.save(net_D.state_dict(), f"data/models/{case_name}_gin_dual.pth")
    print(f"\n[+] GIN-PDL model successfully trained and saved for {case_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GIN-PDL for SCOPF")
    parser.add_argument('--case', type=str, default="pglib_opf_case300_ieee")
    parser.add_argument('--outer_K', type=int, default=20)
    parser.add_argument('--inner_L', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    train_gin_pdl(args.case, args.outer_K, args.inner_L, args.batch_size)