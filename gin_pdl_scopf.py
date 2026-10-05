import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset
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

class ExactBinarySearchLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g_base, d_total, pmax, pmin, gamma, max_iter=20):
        # Shapes: g_base (B, G), pmax (1, G), gamma (1, G)
        B, G = g_base.shape
        device = g_base.device
        
        g_hat = pmax - g_base
        
        # Initialize tensors for binary search (B, G)
        n_k = torch.full((B, G), 0.5, device=device)
        n_min = torch.zeros((B, G), device=device)
        n_max = torch.ones((B, G), device=device)
        
        g_k = torch.zeros((B, G, G), device=device)
        mask = torch.eye(G, device=device).bool().unsqueeze(0).expand(B, -1, -1)
        
        for _ in range(max_iter):
            # Calculate provisional dispatch: g_base + n_k * gamma * g_hat
            prov_g = g_base.unsqueeze(1) + n_k.unsqueeze(2) * (gamma.unsqueeze(1) * g_hat.unsqueeze(1))
            g_k = torch.minimum(prov_g, pmax.unsqueeze(1))
            
            # g_{k,k} = 0 (Contingency generator drops to 0)
            g_k.masked_fill_(mask, 0.0)
            
            # Mismatch: e_k = 1^T g_k - 1^T d
            e_k = g_k.sum(dim=2) - d_total
            
            n_max = torch.where(e_k > 0, n_k, n_max)
            n_min = torch.where(e_k <= 0, n_k, n_min)
            n_k = 0.5 * (n_max + n_min)
            
        # Determine active sets
        rho = (g_k >= pmax.unsqueeze(1)).float() 
        rho.masked_fill_(mask, 1.0)
        
        ctx.save_for_backward(rho, gamma, g_hat)
        return g_k

    @staticmethod
    def backward(ctx, grad_output):
        rho, gamma, g_hat = ctx.saved_tensors
        
        active_mask = 1.0 - rho
        gamma_ghat = gamma.unsqueeze(1) * g_hat.unsqueeze(1)
        
        sum_active_gamma = (active_mask * gamma_ghat).sum(dim=2, keepdim=True) + 1e-9
        
        delta_grad = (grad_output * active_mask).sum(dim=1)
        ratio_term = (grad_output * active_mask * gamma_ghat).sum(dim=2, keepdim=True) / sum_active_gamma
        dist_grad = (ratio_term * active_mask).sum(dim=1)
        
        grad_g_base = delta_grad - dist_grad
        return grad_g_base, None, None, None, None, None

def exact_binary_search_layer(g_star, d_total, pmax, pmin, gamma):
    return ExactBinarySearchLayer.apply(g_star, d_total, pmax, pmin, gamma, 20)

def compute_physics_loss(g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, pmin, gamma, bus_gen_map):
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

    # 3. Generator Contingencies (Kg) via Exact APR
    g_k = exact_binary_search_layer(g_star, d_total, pmax, pmin, gamma)
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

class DynamicSCOPFDataset(IterableDataset):
    def __init__(self, base_pd, pmax_node, pmin_node, edge_index, mu=0.5):
        self.base_pd = torch.tensor(base_pd, dtype=torch.float32)
        self.pmax_node = torch.tensor(pmax_node, dtype=torch.float32)
        self.pmin_node = torch.tensor(pmin_node, dtype=torch.float32)
        self.edge_index = edge_index
        self.mu = mu
        self.sigma_pd = (self.mu * self.base_pd) / 1.96

    def __iter__(self):
        while True: # Infinite on-the-fly generation
            # Truncated Gaussian load perturbation
            noise = torch.randn_like(self.base_pd)
            pd_sample = self.base_pd + noise * self.sigma_pd
            pd_sample = torch.clamp(pd_sample, (1 - self.mu) * self.base_pd, (1 + self.mu) * self.base_pd)

            x_features = torch.stack([pd_sample, self.pmax_node, self.pmin_node], dim=1)
            data = Data(x=x_features, edge_index=self.edge_index)
            data.Pd = pd_sample
            yield data

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

    # Extract base load from the static case file
    pmax_node = np.zeros(len(bus_list))
    pmin_node = np.zeros(len(bus_list))
    base_load_np = np.zeros(len(bus_list))
    
    for i, bus_id in enumerate(bus_list):
        row = case['bus'][case['bus']['bus_i'] == bus_id]
        if not row.empty and 'Pd' in row.columns:
            base_load_np[i] = row['Pd'].values[0] / baseMVA

    for _, row in case['gen'].iterrows():
        b_idx = bus_idx_map[row['bus_i']]
        pmax_node[b_idx] += row['Pmax'] / baseMVA
        pmin_node[b_idx] += row['Pmin'] / baseMVA

    edge_source = [bus_idx_map[i] for i in case['branch']['bus_i'].values]
    edge_target = [bus_idx_map[j] for j in case['branch']['bus_j'].values]
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long)

    # Instantiate dynamic dataset (total batches * batch_size instances)
    total_samples = outer_K * inner_L * batch_size * 2 # *2 covers both Primal and Dual phases
    dataset = DynamicSCOPFDataset(base_load_np, pmax_node, pmin_node, edge_index)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    data_iter = iter(dataloader) # Create the iterator
    
    # Instantiate GIN Networks (3 input features: Pd, Pmax, Pmin)
    net_P = GINPrimalNet(in_features=3, hidden_dim=64, num_gens=num_gens).to(device)
    net_D = GINDualNet(in_features=3, hidden_dim=64, num_kg=num_gens).to(device)

    opt_P = optim.Adam(net_P.parameters(), lr=1e-4)
    opt_D = optim.Adam(net_D.parameters(), lr=1e-4)

    # ALM Hyperparameters
    rho = 0.1
    rho_max = 1e6
    tau = 0.9
    alpha = 2.0
    v_prev = float('inf')

    print("Beginning Training Loop...")
    for k in range(outer_K):
        # -------------------------------------------------------------
        # Phase 1: Primal GIN Learning
        # -------------------------------------------------------------
        net_P.train()
        net_D.eval()
        for _ in range(inner_L):
            batch = next(data_iter).to(device) # Pull exactly one batch
            opt_P.zero_grad()

            g_raw = net_P(batch.x, batch.edge_index, batch.batch, pmin, pmax)
            d_bus = batch.Pd.view(batch.num_graphs, -1)
            d_total = d_bus.sum(dim=1, keepdim=True)
            g_star = power_balance_repair_layer(g_raw, d_total, pmin, pmax)

            gen_cost, total_slack, h_x = compute_physics_loss(
                g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, pmin, gamma, bus_gen_map
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
            batch = next(data_iter).to(device) # Pull exactly one batch
            opt_D.zero_grad()

            with torch.no_grad():
                g_raw = net_P(batch.x, batch.edge_index, batch.batch, pmin, pmax)
                d_bus = batch.Pd.view(batch.num_graphs, -1)
                d_total = d_bus.sum(dim=1, keepdim=True)
                g_star = power_balance_repair_layer(g_raw, d_total, pmin, pmax)
                _, _, h_x = compute_physics_loss(
                    g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, pmin, gamma, bus_gen_map
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