import os
import time
import argparse
import torch
import pandas as pd
import numpy as np

# Import your models and oracle
from dcopf_model import build_ptdf, run_ccga_algorithm
from gnn_erdos import Zone_ADMM_GNN, create_zonal_data
from ml_oracle import predict_active_contingencies

def run_ml_ccga_scenario(case, zonal_data, load_vector, PTDF_matrix, gnn_z1, gnn_z2, device, baseMVA=100.0, verbose=False):
    """
    Modular function to run a single scenario through the ML-Oracle and CCGA.
    Returns the per-unit dispatch, total time, iterations, and predicted set.
    """
    start_time = time.time()
    
    # 1. Instant ML Inference
    predicted_active = predict_active_contingencies(case, load_vector, gnn_z1, gnn_z2, device)
    
    if verbose:
        print(f"ML Oracle predicted active set: {predicted_active}")
        
    # 2. Exact CCGA Solver
    optimal_g, status, ccga_iters, final_active_S = run_ccga_algorithm(
        case['bus'], case['gen'], case['branch'], case['gencost'], 
        load_vector, PTDF_matrix,
        initial_active_S=predicted_active
    )
    
    solve_time = time.time() - start_time
    
    # Convert MW dispatch back to per-unit (PU) for cost calculations
    pg_ml_pu = {k: v / baseMVA for k, v in optimal_g.items()}
    
    return pg_ml_pu, solve_time, ccga_iters, predicted_active

def main():
    parser = argparse.ArgumentParser(description="Run Erdős-GNN Accelerated CCGA")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case118_ieee")
    parser.add_argument('--num_tests', type=int, default=10, help="Number of scenarios to test")
    args = parser.parse_args()

    print("=======================================================")
    print(f" INITIALIZING ML-ACCELERATED CCGA: {args.case.upper()} ")
    print("=======================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. LOAD GRID DATA
    case_path = f'../excel_outputs/{args.case}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    case['gamma'], case['M_eta'] = 1, 1500
    case['gen'].attrs['gamma'] = case['gamma'] 
    
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    # 2. BUILD PTDF MATRIX
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    PTDF_matrix, bus_list = build_ptdf(case['bus'], case['branch'], ref_bus)
    
    # 3. GET TOPOLOGY PARAMS FOR GNN
    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zonal_data = create_zonal_data(case, total_buses[:midpoint], total_buses[midpoint:])
    num_boundaries = len(zonal_data['boundary_buses'])
    num_global_kg = len(zonal_data['global_Kg'])

    # 4. LOAD THE TRAINED ORACLE MODELS
    gnn_z1 = Zone_ADMM_GNN(num_boundaries, num_global_kg).to(device)
    gnn_z2 = Zone_ADMM_GNN(num_boundaries, num_global_kg).to(device)
    
    gnn_z1.load_state_dict(torch.load(f"data/ccga_models/zone1_gnn_oracle_{args.case}.pth", map_location=device))
    gnn_z2.load_state_dict(torch.load(f"data/ccga_models/zone2_gnn_oracle_{args.case}.pth", map_location=device))
    
    print("Successfully loaded pre-trained Erdős-GNN Oracle.")

    # 5. LOAD TEST DATA
    loads_df = pd.read_csv(f'data/{args.case}_generated_loads.csv')
    num_tests = min(args.num_tests, len(loads_df))

    # 6. RUN THE EXPERIMENT
    print("\nStarting inference and CCGA evaluation...")
    
    total_ml_ccga_iters = 0
    total_time = 0.0

    for s in range(num_tests):
        print(f"\n--- Scenario {s} ---")
        load_vector = {b: loads_df.iloc[s][f"Bus_{b}_Pd"] for b in bus_list}
        
        start_time = time.time()
        
        # Step A: Instant ML Inference
        predicted_active = predict_active_contingencies(case, load_vector, gnn_z1, gnn_z2, device)
        print(f"ML Oracle predicted active set: {predicted_active}")
        
        # Step B: Feed prediction into the exact CCGA solver
        _, _, ccga_iters, final_active_S = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], 
            load_vector, PTDF_matrix,
            initial_active_S=predicted_active
        )
        
        solve_time = time.time() - start_time
        total_time += solve_time
        total_ml_ccga_iters += ccga_iters
        
        print(f"Solver Iterations Required: {ccga_iters}")
        print(f"Final True Active Set: {final_active_S}")
        print(f"Time Taken: {solve_time:.4f} seconds")

    # 7. SUMMARY
    print("\n=======================================================")
    print(" EXPERIMENT SUMMARY")
    print("=======================================================")
    print(f"Total Scenarios Tested: {num_tests}")
    print(f"Average Solver Iterations per Scenario: {total_ml_ccga_iters / num_tests:.2f}")
    print(f"Average Total Solve Time per Scenario: {total_time / num_tests:.4f} seconds")

if __name__ == "__main__":
    main()