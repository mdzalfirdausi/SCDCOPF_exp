import os
import time
import math
import argparse
import torch
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# Import from your existing scripts
from dcopf_model import build_ptdf, run_ccga_algorithm, check_contingency_violations
from gnn_erdos import Zone_ADMM_GNN, create_zonal_data, create_pyg_dataset
from run_ml_aladin import build_mi_aladin_zone, solve_aladin_master_qp, damped_bfgs_update

def calculate_generation_cost(pg_dict, case_data, baseMVA):
    """Calculates the physical objective cost ($) given a generation dispatch."""
    cost = 0.0
    for i, pg_val in pg_dict.items():
        # Find the cost coefficients for this generator
        row = case_data['gencost'][case_data['gencost']['gen_ID'] == i].iloc[0]
        c2, c1, c0 = row['c2'], row['c1'], row['c0']
        # Convert pu back to MW for cost calculation if coefficients require it
        # Assuming coefficients are in baseMVA terms based on your Pyomo models
        pg_mw = pg_val * baseMVA
        cost += (c2 * (pg_mw**2)) + (c1 * pg_mw) + c0
    return cost

def evaluate_benchmarks():
    parser = argparse.ArgumentParser(description="Evaluate ML-ALADIN vs Monolithic CCGA Baseline")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case14_ieee")
    parser.add_argument('--num_instances', type=int, default=5, help="Number of scenarios to benchmark")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    case_name = args.case
    baseMVA = 100.0

    print(f"=======================================================")
    print(f" BENCHMARKING: {case_name.upper()} ")
    print(f"=======================================================")

    # 1. LOAD GRID DATA
    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    case['gamma'], case['M_eta'] = 0.05, 1500

    # Clean zero-limit gens
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    # 2. SETUP ZONES & PTDF
    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zone1_buses = total_buses[:midpoint]
    zone2_buses = total_buses[midpoint:]

    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    global_kg = zonal_data['global_Kg']
    kg_and_base = ['base'] + global_kg
    boundary_buses = sorted(list(zonal_data['boundary_buses']))
    
    num_boundaries = len(boundary_buses)
    num_global_kg = len(global_kg)
    num_buses_z1 = len(zonal_data['zone1']['bus'])

    print("Pre-computing exact PTDF Matrix for baseline solver...")
    PTDF_matrix, bus_list = build_ptdf(case['bus'], case['branch'], ref_bus)

    # 3. LOAD GNN AGENTS
    print("Loading Trained Erdős-GNN Agents...")
    gnn_z1 = Zone_ADMM_GNN(num_boundaries, num_global_kg).to(device)
    gnn_z1.load_state_dict(torch.load("data/admm_models/zone1_gnn_agent.pth", map_location=device, weights_only=False))
    gnn_z1.eval()

    gnn_z2 = Zone_ADMM_GNN(num_boundaries, num_global_kg).to(device)
    gnn_z2.load_state_dict(torch.load("data/admm_models/zone2_gnn_agent.pth", map_location=device, weights_only=False))
    gnn_z2.eval()

    # 4. LOAD SCENARIOS
    csv_path = f'data/{case_name}_generated_loads.csv'
    load_profiles = pd.read_csv(csv_path)
    num_instances = min(args.num_instances, len(load_profiles))
    
    results = []
    all_residuals = []

    # 5. EXECUTION LOOP
    for s in range(num_instances):
        print(f"\n--- Scenario {s+1}/{num_instances} ---")
        scenario_row = load_profiles.iloc[s].values
        
        # Build dictionaries for exact solver
        load_vector = {bus_list[i]: scenario_row[i] for i in range(len(bus_list))}
        
        # =========================================================
        # BASELINE: EXACT CCGA
        # =========================================================
        print("Solving exact CCGA baseline...")
        start_ccga = time.time()
        opt_g_baseline, status, ccga_iters, active_S = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], load_vector, PTDF_matrix
        )
        time_ccga = time.time() - start_ccga
        
        # Convert optimal_g back to per-unit for cost calculation parity
        pg_base_pu = {k: v / baseMVA for k, v in opt_g_baseline.items()}
        cost_baseline = calculate_generation_cost(pg_base_pu, case, baseMVA)
        print(f" -> CCGA Done: {time_ccga:.2f}s | Cost: ${cost_baseline:.2f}")

        # =========================================================
        # PROPOSED: ML-ALADIN
        # =========================================================
        scenario_row_pu = scenario_row / baseMVA
        load_z1 = scenario_row_pu[:num_buses_z1].reshape(1, -1)
        load_z2 = scenario_row_pu[num_buses_z1:].reshape(1, -1)

        graph_z1 = create_pyg_dataset(zonal_data['zone1'], load_z1, baseMVA)[0].to(device)
        graph_z2 = create_pyg_dataset(zonal_data['zone2'], load_z2, baseMVA)[0].to(device)

        start_ml = time.time()
        
        # 1. GNN Inference
        with torch.no_grad():
            va_dummy = torch.zeros(1, num_global_kg + 1, num_boundaries, device=device)
            zk_dummy = torch.zeros(1, num_global_kg, device=device)
            _, _, zk_prob_z1 = gnn_z1(graph_z1, va_dummy, va_dummy, zk_dummy, zk_dummy)
            _, _, zk_prob_z2 = gnn_z2(graph_z2, va_dummy, va_dummy, zk_dummy, zk_dummy)
            zk_hard_z1 = (zk_prob_z1 > 0.5).int().squeeze().cpu().numpy()
            zk_hard_z2 = (zk_prob_z2 > 0.5).int().squeeze().cpu().numpy()
            
        gnn_time = time.time() - start_ml

        # 2. Build Models
        model_z1 = build_mi_aladin_zone(1, zonal_data['zone1'], case['branch'], zonal_data['tie_lines'], boundary_buses, ref_bus in zone1_buses, ref_bus)
        model_z2 = build_mi_aladin_zone(2, zonal_data['zone2'], case['branch'], zonal_data['tie_lines'], boundary_buses, ref_bus in zone2_buses, ref_bus)

        # Inject loads and lock binaries
        for b in model_z1.LocalBuses:
            model_z1.Pd[b].set_value(load_z1[0, list(zonal_data['zone1']['bus']['bus_i']).index(b)])
        for b in model_z2.LocalBuses:
            model_z2.Pd[b].set_value(load_z2[0, list(zonal_data['zone2']['bus']['bus_i']).index(b)])

        for k_idx, k in enumerate(global_kg):
            z1_val = zk_hard_z1.item() if zk_hard_z1.size == 1 else zk_hard_z1[k_idx]
            z2_val = zk_hard_z2.item() if zk_hard_z2.size == 1 else zk_hard_z2[k_idx]
            for i in model_z1.Gens: model_z1.xk[k, i].fix(z1_val); model_z1.zk[k].fix(z1_val)
            for i in model_z2.Gens: model_z2.xk[k, i].fix(z2_val); model_z2.zk[k].fix(z2_val)

        # 3. ALADIN Optimization Loop
        solver = pyo.SolverFactory('gurobi')
        solver.options['OutputFlag'] = 0

        lam = {('Va', k, b): 0.0 for k in kg_and_base for b in boundary_buses}
        z_target = {('Va', k, b): 0.0 for k in kg_and_base for b in boundary_buses}
        ordered_keys = [('Va', k, b) for k in kg_and_base for b in boundary_buses]
        
        dim, rho, mu, tol = len(ordered_keys), 100.0, 1e4, 1e-4
        aladin_iters = 0
        scenario_residuals = []

        for itr in range(1, 30):
            aladin_iters = itr
            for b in boundary_buses:
                for k in kg_and_base:
                    model_z1.lam_va[k, b].set_value(lam[('Va', k, b)])
                    model_z1.z_va_target[k, b].set_value(z_target[('Va', k, b)])
                    model_z2.lam_va[k, b].set_value(lam[('Va', k, b)])
                    model_z2.z_va_target[k, b].set_value(z_target[('Va', k, b)])
                    
            solver.solve(model_z1)
            solver.solve(model_z2)
            
            def get_va_val(m, k, b): return pyo.value(m.Va_base[b]) if k == 'base' else pyo.value(m.Va_k[k, b])
            x1 = {('Va', k, b): get_va_val(model_z1, k, b) for k in kg_and_base for b in boundary_buses}
            x2 = {('Va', k, b): get_va_val(model_z2, k, b) for k in kg_and_base for b in boundary_buses}

            g1 = {key: rho * (x1[key] - z_target[key]) + lam[key] for key in ordered_keys}
            g2 = {key: rho * (x2[key] - z_target[key]) - lam[key] for key in ordered_keys}

            x1_vec, x2_vec = np.array([x1[k] for k in ordered_keys]), np.array([x2[k] for k in ordered_keys])
            g1_vec, g2_vec = np.array([g1[k] for k in ordered_keys]), np.array([g2[k] for k in ordered_keys])

            if itr == 1:
                H1_mat, H2_mat = np.eye(dim) * rho, np.eye(dim) * rho
            else:
                H1_mat = damped_bfgs_update(H1_mat, x1_vec - x1_prev, g1_vec - g1_prev)
                H2_mat = damped_bfgs_update(H2_mat, x2_vec - x2_prev, g2_vec - g2_prev)

            x1_prev, g1_prev, x2_prev, g2_prev = x1_vec.copy(), g1_vec.copy(), x2_vec.copy(), g2_vec.copy()

            d_x1, d_x2, lambda_qp, s_val = solve_aladin_master_qp(x1, x2, g1, g2, H1_mat, H2_mat, ordered_keys, boundary_buses, kg_and_base, lam, mu)

            primal_residual = math.sqrt(sum(v**2 for v in s_val.values()))
            scenario_residuals.append(primal_residual)
            
            if primal_residual <= tol:
                break
                
            for key in z_target.keys():
                z_target[key] = 0.5 * ((x1[key] + d_x1[key]) + (x2[key] + d_x2[key]))
                lam[key] = lambda_qp[key]
                
        time_ml_total = time.time() - start_ml
        all_residuals.append(scenario_residuals)

        # 4. Extract ML Results & Calculate Violations
        pg_ml_pu = {}
        for i in model_z1.Gens: pg_ml_pu[i] = pyo.value(model_z1.Pg_base[i])
        for i in model_z2.Gens: pg_ml_pu[i] = pyo.value(model_z2.Pg_base[i])
        
        cost_ml = calculate_generation_cost(pg_ml_pu, case, baseMVA)
        
        # Verify algebraic violations
        bus_gen_map = case['gen'].set_index('gen_ID')['bus_i'].to_dict()
        pg_ml_mw = {k: v * baseMVA for k, v in pg_ml_pu.items()}
        max_viol, _ = check_contingency_violations(pg_ml_mw, PTDF_matrix, load_vector, case['branch'], bus_gen_map)
        
        # Calculate final metrics
        speedup = time_ccga / time_ml_total
        opt_gap = ((cost_ml - cost_baseline) / cost_baseline) * 100.0

        print(f" -> ML-ALADIN Done: {time_ml_total:.2f}s | Iterations: {aladin_iters} | Cost: ${cost_ml:.2f}")
        print(f" -> Metrics: Speedup: {speedup:.1f}x | Opt Gap: {opt_gap:.4f}% | Max Viol: {max_viol:.2f} MW")

        results.append({
            'Scenario': s + 1,
            'Time_Monolithic_s': round(time_ccga, 4),
            'Cost_Monolithic': round(cost_baseline, 2),
            'Time_ML_ALADIN_s': round(time_ml_total, 4),
            'Cost_ML_ALADIN': round(cost_ml, 2),
            'ML_ALADIN_Iters': aladin_iters,
            'Speedup_Ratio': round(speedup, 2),
            'Optimality_Gap_%': round(opt_gap, 6),
            'Max_Violation_MW': round(max_viol, 4)
        })

    # =========================================================
    # EXPORT RESULTS & GENERATE PLOT
    # =========================================================
    df_results = pd.DataFrame(results)
    out_csv = f"data/{case_name}_benchmark_summary.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n=======================================================")
    print(f" BENCHMARK COMPLETE. Results saved to {out_csv}")
    print(f" Average Speedup: {df_results['Speedup_Ratio'].mean():.2f}x")
    print(f" Average Optimality Gap: {df_results['Optimality_Gap_%'].mean():.4f}%")
    print(f" Maximum Overall Violation: {df_results['Max_Violation_MW'].max():.4f} MW")
    print(f"=======================================================\n")

    # Plot convergence of the last scenario as an example
    if len(all_residuals) > 0 and len(all_residuals[-1]) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(all_residuals[-1])+1), all_residuals[-1], marker='o', linestyle='-', color='b')
        plt.yscale('log')
        plt.axhline(y=1e-4, color='r', linestyle='--', label='Tolerance ($10^{-4}$)')
        plt.title('ML-ALADIN Primal Residual Convergence')
        plt.xlabel('ALADIN Iteration')
        plt.ylabel('Primal Residual (Log Scale)')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plot_path = f"data/{case_name}_convergence_plot.png"
        plt.savefig(plot_path)
        print(f"Convergence plot saved to {plot_path}")

if __name__ == "__main__":
    evaluate_benchmarks()