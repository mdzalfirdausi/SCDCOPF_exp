import os
import time
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# Import your exact solver
from dcopf_model import build_ptdf, run_ccga_algorithm

# ==========================================
# 1. GNN ARCHITECTURE (Required for loading)
# ==========================================
class SupervisedContingencyGNN(nn.Module):
    def __init__(self, num_node_features, num_gen_targets, num_line_targets):
        super(SupervisedContingencyGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.gen_head = nn.Linear(128, num_gen_targets)
        self.line_head = nn.Linear(128, num_line_targets)

    def forward(self, x, edge_index, batch):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        h_graph = global_mean_pool(h, batch)
        return self.gen_head(h_graph), self.line_head(h_graph)

# ==========================================
# 2. PROFILING SCRIPT
# ==========================================
def profile_computational_speed(case_name, num_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Profiling {case_name} on {device} ---")

    # 1. Setup Exact Solver Environment
    print("Loading physical grid data for exact solver...")
    case_path = f'../excel_outputs/{case_name}.xlsx'
    bus_df = pd.read_excel(case_path, sheet_name='bus')
    gen_df = pd.read_excel(case_path, sheet_name='gen')
    branch_df = pd.read_excel(case_path, sheet_name='branch')
    cost_df = pd.read_excel(case_path, sheet_name='gencost')

    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    PTDF_matrix, bus_list = build_ptdf(bus_df, branch_df, ref_bus_id)

    # 2. Setup GNN Environment
    data_dir = f"data/pyg_dataset/{case_name}"
    sample_pt = torch.load(os.path.join(data_dir, "scenario_0.pt"), weights_only=False)
    
    model = SupervisedContingencyGNN(sample_pt.x.shape[1], sample_pt.y_gen.shape[0], sample_pt.y_line.shape[0]).to(device)
    model.load_state_dict(torch.load(f"data/models/{case_name}_supervised_model.pth", map_location=device, weights_only=True))
    model.eval()

    # GPU Warmup (Crucial for accurate PyTorch profiling)
    print("Warming up GPU...")
    dummy_batch = Batch.from_data_list([sample_pt.to(device)])
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_batch.x, dummy_batch.edge_index, dummy_batch.batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Load test loads
    loads_df = pd.read_csv(f"data/{case_name}_generated_data.csv")
    
    gnn_times = []
    solver_times = []
    active_contingencies_count = []

    print(f"\nRunning {num_samples} scenarios to compare times...")
    
    # Randomly select scenarios from the dataset
    test_indices = np.random.choice(len(loads_df), min(num_samples, len(loads_df)), replace=False)

    for idx in test_indices:
        row = loads_df.iloc[idx]
        scenario_id = int(row['Scenario_ID']) if 'Scenario_ID' in row else idx
        
        # --- TIMING THE GNN ---
        graph_data = torch.load(os.path.join(data_dir, f"scenario_{scenario_id}.pt"), weights_only=False).to(device)
        batch = Batch.from_data_list([graph_data])
        
        with torch.no_grad():
            if device.type == 'cuda': torch.cuda.synchronize()
            t0_gnn = time.perf_counter()
            
            _ = model(batch.x, batch.edge_index, batch.batch)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            t_gnn = time.perf_counter() - t0_gnn
            
        gnn_times.append(t_gnn)

        # --- TIMING THE EXACT SOLVER ---
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}
        
        t0_solver = time.perf_counter()
        try:
            _, _, _, active_Kg, active_Ke = run_ccga_algorithm(
                bus_df, gen_df, branch_df, cost_df, load_vector, PTDF_matrix
            )
            t_solver = time.perf_counter() - t0_solver
            total_active = len(active_Kg) + len(active_Ke)
        except Exception as e:
            print(f"Solver failed on scenario {scenario_id}: {e}")
            t_solver = np.nan
            total_active = 0
            
        solver_times.append(t_solver)
        active_contingencies_count.append(total_active)
        
        print(f"Scenario {scenario_id:4d} | GNN: {t_gnn:.5f}s | Solver: {t_solver:.3f}s | Speedup: {t_solver/t_gnn:.1f}x")

    # ==========================================
    # 3. GENERATE PAPER-READY CHARTS
    # ==========================================
    os.makedirs('data/plots', exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    df_results = pd.DataFrame({
        'Scenario': range(len(gnn_times)),
        'GNN_Time': gnn_times,
        'Solver_Time': solver_times,
        'Active_Count': active_contingencies_count
    }).dropna()

    avg_gnn = df_results['GNN_Time'].mean()
    avg_solver = df_results['Solver_Time'].mean()
    speedup = avg_solver / avg_gnn

    print("\n--- FINAL PROFILING RESULTS ---")
    print(f"Average GNN Inference Time:   {avg_gnn:.5f} seconds")
    print(f"Average Exact Solver Time:    {avg_solver:.3f} seconds")
    print(f"Average Speedup:              {speedup:,.1f}x faster")

    # Chart 1: Bar Chart Comparison (Log Scale)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['GNN', 'Gurobi'], [avg_gnn, avg_solver], color=['#2ca02c', '#1f77b4'])
    plt.yscale('log')
    plt.ylabel('Average Computation Time (Seconds) [Log Scale]')
    plt.title(f'Computational Speed: GNN vs Exact Solver\n({speedup:,.0f}x Speedup)')
    
    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', va='bottom', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'data/plots/{case_name}_time_bar_chart.pdf', dpi=600)
    plt.close()

    # Chart 2: Scatter plot showing variance (Time vs Active Contingencies)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_results, x='Active_Count', y='Solver_Time', label='Exact Solver', color='#1f77b4', s=100)
    sns.scatterplot(data=df_results, x='Active_Count', y='GNN_Time', label='GNN', color='#2ca02c', s=100)
    
    plt.xlabel('Number of Active Contingencies in Scenario')
    plt.ylabel('Computation Time (Seconds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'data/plots/{case_name}_time_scatter.pdf', dpi=600)
    plt.close()

    print(f"Charts saved to 'data/plots/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--samples', type=int, default=10, help="Number of scenarios to profile")
    args = parser.parse_args()
    
    profile_computational_speed(args.case, args.samples)