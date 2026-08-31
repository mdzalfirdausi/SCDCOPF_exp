import os
import sys

# OpenMP runtime conflict fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import baseline functions and the standardized ML-ADMM runner
from dcopf_model import build_ptdf, run_ccga_algorithm, check_contingency_violations, calculate_primary_response
from gnn_erdos import Zone_ADMM_GNN, create_zonal_data
from run_ml_ccga import run_ml_ccga_scenario

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

def evaluate_benchmarks():
    parser = argparse.ArgumentParser(description="Evaluate ML-CCGA vs Monolithic CCGA Baseline")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case118_ieee")
    parser.add_argument('--num_instances', type=int, default=5, help="Number of scenarios to benchmark")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    case_name = args.case
    baseMVA = 100.0

    print(f"=======================================================")
    print(f" BENCHMARKING: {case_name.upper()} ")
    print(f"=======================================================")

    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    case['gen']['Pmax'] = case['gen']['Pmax'].astype(float)
    case['gencost']['c1'] = case['gencost']['c1'].astype(float)
    case['gencost']['c2'] = case['gencost']['c2'].astype(float)
    
    case['gamma'], case['M_eta'] = 1, 1500
    case['gen'].attrs['gamma'] = case['gamma'] 
    
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zone1_buses = total_buses[:midpoint]
    zone2_buses = total_buses[midpoint:]

    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    global_kg = zonal_data['global_Kg']
    global_ke = zonal_data['global_Ke']
    boundary_buses = sorted(list(zonal_data['boundary_buses']))
    
    num_boundaries = len(boundary_buses)
    num_global_kg = len(global_kg)
    num_global_ke = len(global_ke)
    bus_list = sorted(case['bus']['bus_i'].tolist())

    print("Pre-computing exact PTDF Matrix for baseline solver...")
    PTDF_matrix, _ = build_ptdf(case['bus'], case['branch'], ref_bus)

    print("Loading Trained Erdős-GNN Agents...")
    gnn_z1 = Zone_ADMM_GNN(num_boundaries, num_global_kg, num_global_ke).to(device)
    gnn_z1.load_state_dict(torch.load(f"data/ccga_models/zone1_gnn_oracle_{case_name}.pth", map_location=device))
    gnn_z1.eval()

    gnn_z2 = Zone_ADMM_GNN(num_boundaries, num_global_kg, num_global_ke).to(device)
    gnn_z2.load_state_dict(torch.load(f"data/ccga_models/zone2_gnn_oracle_{case_name}.pth", map_location=device))
    gnn_z2.eval()

    csv_path = f'data/{case_name}_generated_data.csv'
    if not os.path.exists(csv_path):
        csv_path = f'data/{case_name}_generated_loads.csv'
        
    load_profiles = pd.read_csv(csv_path)
    num_instances = min(args.num_instances, len(load_profiles))
    
    results = []

    for s in range(num_instances):
        print(f"\n--- Scenario {s+1}/{num_instances} ---")
        row = load_profiles.iloc[s]
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}
        
        # Inject Perturbed Limits and Costs (if using _generated_data.csv)
        if f"Gen_{global_kg[0]}_Pmax" in row:
            for gen_id in global_kg:
                case['gen'].loc[case['gen']['gen_ID'] == gen_id, 'Pmax'] = row[f"Gen_{gen_id}_Pmax"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c1'] = row[f"Gen_{gen_id}_c1"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c2'] = row[f"Gen_{gen_id}_c2"]
        
        # =========================================================
        # BASELINE: EXACT CCGA
        # =========================================================
        start_ccga = time.time()
        opt_g_baseline, status, ccga_iters, active_Kg, active_Ke = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], load_vector, PTDF_matrix
        )
        time_ccga = time.time() - start_ccga
        
        pg_base_pu = {k: v / baseMVA for k, v in opt_g_baseline.items()}
        cost_baseline = calculate_generation_cost(pg_base_pu, case, baseMVA)

        # =========================================================
        # PROPOSED: ML-CCGA
        # =========================================================
        pg_ml_pu, time_ml_total, ml_iters, pred_Kg, pred_Ke = run_ml_ccga_scenario(
            case, zonal_data, load_vector, PTDF_matrix, gnn_z1, gnn_z2, device, baseMVA, verbose=False
        )
        cost_ml = calculate_generation_cost(pg_ml_pu, case, baseMVA)

        print(f" -> ML-CCGA Done: {time_ml_total:.2f}s | Iterations: {ml_iters} | Cost: ${cost_ml:.2f}")

        pg_ml_mw = {k: v * baseMVA for k, v in pg_ml_pu.items()}
        max_viol = evaluate_true_n1_security(pg_ml_mw, case, PTDF_matrix, load_vector, baseMVA)
        
        speedup = time_ccga / time_ml_total if time_ml_total > 0 else 0.0
        opt_gap = ((cost_ml - cost_baseline) / cost_baseline) * 100.0 if cost_baseline > 0 else 0.0

        print(f" -> Metrics: Speedup: {speedup:.2f}x | Opt Gap: {opt_gap:.4f}% | Max Viol: {max_viol:.2f} MW")

        results.append({
            'Scenario': s + 1,
            'Time_Monolithic_s': round(time_ccga, 4),
            'Cost_Monolithic': round(cost_baseline, 2),
            'Time_ML_CCGA_s': round(time_ml_total, 4),
            'Cost_ML_CCGA': round(cost_ml, 2),
            'ML_CCGA_Iters': ml_iters,
            'Speedup_Ratio': round(speedup, 2),
            'Optimality_Gap_%': round(opt_gap, 6),
            'Max_Violation_MW': round(max_viol, 4)
        })

    os.makedirs('data/paper_figs', exist_ok=True)
    df_results = pd.DataFrame(results)
    out_csv = f"data/{case_name}_benchmark_summary.csv"
    df_results.to_csv(out_csv, index=False)
    
    print(f"\n=======================================================")
    print(f" BENCHMARK COMPLETE. Results saved to {out_csv}")
    print(f" Average Speedup: {df_results['Speedup_Ratio'].mean():.2f}x")
    print(f" Average Optimality Gap: {df_results['Optimality_Gap_%'].mean():.4f}%")
    print(f" Maximum Overall Violation: {df_results['Max_Violation_MW'].max():.4f} MW")
    print(f"=======================================================\n")

if __name__ == "__main__":
    evaluate_benchmarks()