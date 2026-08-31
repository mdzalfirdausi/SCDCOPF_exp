import os
import sys

# OpenMP runtime conflict fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool

# Import baseline functions
from dcopf_model import build_ptdf, run_ccga_algorithm, check_contingency_violations, calculate_primary_response

# ==========================================
# 1. CENTRALIZED GNN ARCHITECTURE
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
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_generation_cost(pg_dict, case_data, baseMVA):
    cost = 0.0
    for i, pg_val in pg_dict.items():
        row = case_data['gencost'][case_data['gencost']['gen_ID'] == i].iloc[0]
        c2, c1, c0 = row['c2'], row['c1'], row['c0']
        pg_mw = pg_val * baseMVA
        cost += (c2 * (pg_mw**2)) + (c1 * pg_mw) + c0
    return cost

def evaluate_true_n1_security(pg_base_mw, case_data, PTDF_matrix, load_vector, baseMVA):
    """Simulates actual N-1 generator outages and primary frequency response."""
    gen_df = case_data['gen'].copy()
    branch_df = case_data['branch']
    bus_gen_map = gen_df.set_index('gen_ID')['bus_i'].to_dict()
    gen_df.attrs['gamma'] = case_data.get('gamma', 0.05)
    
    max_overall_violation = 0.0
    
    for k in gen_df['gen_ID']:
        try:
            n_s, g_s = calculate_primary_response(pg_base_mw, k, gen_df, load_vector, baseMVA)
            viol, worst_line = check_contingency_violations(g_s, PTDF_matrix, load_vector, branch_df, bus_gen_map, baseMVA)
            if viol > max_overall_violation:
                max_overall_violation = viol
        except Exception:
            return float('inf')
            
    return max_overall_violation

# ==========================================
# 3. MAIN BENCHMARK ENGINE
# ==========================================
def evaluate_benchmarks():
    parser = argparse.ArgumentParser(description="Evaluate ML-CCGA vs Exact CCGA Baseline")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case300_ieee")
    parser.add_argument('--num_instances', type=int, default=5, help="Number of scenarios to benchmark")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    case_name = args.case
    baseMVA = 100.0

    print(f"=======================================================")
    print(f" HYBRID BENCHMARKING: {case_name.upper()} on {device}")
    print(f"=======================================================")

    # Load Physical Data
    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    case['gen']['Pmax'] = case['gen']['Pmax'].astype(float)
    case['gencost']['c1'] = case['gencost']['c1'].astype(float)
    case['gencost']['c2'] = case['gencost']['c2'].astype(float)
    case['gamma'], case['M_eta'] = 1, 1500
    case['gen'].attrs['gamma'] = case['gamma'] 
    
    # Drop zero-capacity generators
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    bus_list = sorted(case['bus']['bus_i'].tolist())
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    
    # Define Targets (must match training order)
    global_kg = case['gen']['gen_ID'].tolist()
    global_ke = case['branch']['line_ID'].tolist() 
    
    print("Pre-computing exact PTDF Matrix for baseline solver...")
    PTDF_matrix, _ = build_ptdf(case['bus'], case['branch'], ref_bus)

    # Build PyG Graph Topology for inference
    bus_to_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    edge_source = [bus_to_idx[i] for i in case['branch']['bus_i'].values]
    edge_target = [bus_to_idx[j] for j in case['branch']['bus_j'].values]
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long).to(device)

    # Load Centralized ML Model
    print("Loading Trained Centralized GNN...")
    gnn_model = SupervisedContingencyGNN(num_node_features=1, num_gen_targets=len(global_kg), num_line_targets=len(global_ke)).to(device)
    gnn_model.load_state_dict(torch.load(f"data/models/{case_name}_supervised_model.pth", map_location=device, weights_only=True))
    gnn_model.eval()

    # Load Loads Data
    csv_path = f"data/{case_name}_generated_data.csv"
    if not os.path.exists(csv_path):
        csv_path = f"data/{case_name}_generated_loads.csv"
    
    load_profiles = pd.read_csv(csv_path)
    num_instances = min(args.num_instances, len(load_profiles))
    results = []

    # GPU Warmup
    print("Warming up GPU for accurate ML timing...")
    dummy_x = torch.zeros((len(bus_list), 1), device=device)
    dummy_batch = torch.zeros(len(bus_list), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = gnn_model(dummy_x, edge_index, dummy_batch)

    for s in range(num_instances):
        print(f"\n--- Scenario {s+1}/{num_instances} ---")
        row = load_profiles.iloc[s]
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}
        
        # Inject Perturbed Limits and Costs
        if f"Gen_{global_kg[0]}_Pmax" in row:
            for gen_id in global_kg:
                case['gen'].loc[case['gen']['gen_ID'] == gen_id, 'Pmax'] = row[f"Gen_{gen_id}_Pmax"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c1'] = row[f"Gen_{gen_id}_c1"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c2'] = row[f"Gen_{gen_id}_c2"]
        
        # =========================================================
        # 1. BASELINE: EXACT CCGA
        # =========================================================
        start_ccga = time.time()
        opt_g_baseline, status, ccga_iters, active_Kg, active_Ke = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], load_vector, PTDF_matrix
        )
        time_ccga = time.time() - start_ccga
        
        pg_base_pu = {k: v / baseMVA for k, v in opt_g_baseline.items()}
        cost_baseline = calculate_generation_cost(pg_base_pu, case, baseMVA)

        # =========================================================
        # 2. PROPOSED: HYBRID ML-CCGA
        # =========================================================
        # A. Neural Network Inference
        load_features = np.array([load_vector[b] for b in bus_list], dtype=np.float32)
        x_tensor = torch.tensor(load_features).view(-1, 1).to(device)
        batch = torch.zeros(x_tensor.size(0), dtype=torch.long, device=device)

        if device.type == 'cuda': torch.cuda.synchronize()
        start_ml_inf = time.perf_counter()
        
        with torch.no_grad():
            gen_logits, line_logits = gnn_model(x_tensor, edge_index, batch)
            preds_gen = (torch.sigmoid(gen_logits.squeeze()) > 0.5).cpu().numpy()
            preds_line = (torch.sigmoid(line_logits.squeeze()) > 0.5).cpu().numpy()
            
        if device.type == 'cuda': torch.cuda.synchronize()
        time_ml_inf = time.perf_counter() - start_ml_inf

        # Map binary predictions back to physical IDs
        pred_Kg = [global_kg[i] for i, val in enumerate(preds_gen) if val]
        pred_Ke = [global_ke[i] for i, val in enumerate(preds_line) if val]

        # B. Physical Solver (Warm-started with GNN predictions)
        start_ml_solve = time.perf_counter()
        opt_g_ml, status_ml, ml_iters, final_Kg, final_Ke = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], load_vector, PTDF_matrix,
            initial_active_Kg=pred_Kg, initial_active_Ke=pred_Ke  # <--- The magic happens here
        )
        time_ml_solve = time.perf_counter() - start_ml_solve
        
        time_ml_total = time_ml_inf + time_ml_solve
        
        pg_ml_pu = {k: v / baseMVA for k, v in opt_g_ml.items()}
        cost_ml = calculate_generation_cost(pg_ml_pu, case, baseMVA)

        # =========================================================
        # 3. METRICS
        # =========================================================
        print(f" -> ML-CCGA Done: {time_ml_total:.3f}s (Inference: {time_ml_inf:.4f}s) | Iterations: {ml_iters} | Cost: ${cost_ml:.2f}")

        pg_ml_mw = {k: v * baseMVA for k, v in pg_ml_pu.items()}
        max_viol = evaluate_true_n1_security(pg_ml_mw, case, PTDF_matrix, load_vector, baseMVA)
        
        speedup = time_ccga / time_ml_total if time_ml_total > 0 else 0.0
        opt_gap = ((cost_ml - cost_baseline) / cost_baseline) * 100.0 if cost_baseline > 0 else 0.0

        print(f" -> Metrics: Speedup: {speedup:.2f}x | Opt Gap: {opt_gap:.6f}% | Max Viol: {max_viol:.2f} MW")

        results.append({
            'Scenario': s + 1,
            'Time_Exact_s': round(time_ccga, 4),
            'Exact_Iters': ccga_iters,
            'Cost_Exact': round(cost_baseline, 2),
            'Time_ML_Total_s': round(time_ml_total, 4),
            'Time_ML_Inference_s': round(time_ml_inf, 4),
            'ML_CCGA_Iters': ml_iters,
            'Cost_ML_CCGA': round(cost_ml, 2),
            'Speedup_Ratio': round(speedup, 2),
            'Optimality_Gap_%': round(opt_gap, 6),
            'Max_Violation_MW': round(max_viol, 4)
        })

    # Save Results
    os.makedirs('data/paper_figs', exist_ok=True)
    df_results = pd.DataFrame(results)
    out_csv = f"data/{case_name}_hybrid_benchmark_summary.csv"
    df_results.to_csv(out_csv, index=False)
    
    print(f"\n=======================================================")
    print(f" BENCHMARK COMPLETE. Results saved to {out_csv}")
    print(f" Average Speedup: {df_results['Speedup_Ratio'].mean():.2f}x")
    print(f" Average Optimality Gap: {df_results['Optimality_Gap_%'].mean():.6f}%")
    print(f" Maximum Overall Violation: {df_results['Max_Violation_MW'].max():.4f} MW")
    print(f"=======================================================\n")

if __name__ == "__main__":
    evaluate_benchmarks()