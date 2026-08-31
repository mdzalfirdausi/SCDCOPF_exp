import os
import time
import argparse
import torch
import pandas as pd
import numpy as np

from dcopf_model import build_ptdf, run_ccga_algorithm
from gnn_erdos import Zone_ADMM_GNN, create_zonal_data
from ml_oracle import predict_active_contingencies

def run_ml_ccga_scenario(case, zonal_data, load_vector, PTDF_matrix, gnn_z1, gnn_z2, device, baseMVA=100.0, verbose=False):
    start_time = time.time()
    
    predicted_active_Kg, predicted_active_Ke = predict_active_contingencies(case, load_vector, gnn_z1, gnn_z2, device)
    
    if verbose:
        print(f"ML Oracle Predicted Active Kg: {predicted_active_Kg}")
        print(f"ML Oracle Predicted Active Ke: {predicted_active_Ke}")
        
    optimal_g, status, ccga_iters, final_active_Kg, final_active_Ke = run_ccga_algorithm(
        case['bus'], case['gen'], case['branch'], case['gencost'], 
        load_vector, PTDF_matrix,
        initial_active_Kg=predicted_active_Kg,
        initial_active_Ke=predicted_active_Ke
    )
    
    solve_time = time.time() - start_time
    pg_ml_pu = {k: v / baseMVA for k, v in optimal_g.items()}
    
    return pg_ml_pu, solve_time, ccga_iters, predicted_active_Kg, predicted_active_Ke

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

    case_path = f'../excel_outputs/{args.case}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    case['gamma'], case['M_eta'] = 1, 1500
    case['gen'].attrs['gamma'] = case['gamma'] 
    
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    PTDF_matrix, bus_list = build_ptdf(case['bus'], case['branch'], ref_bus)
    
    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zonal_data = create_zonal_data(case, total_buses[:midpoint], total_buses[midpoint:])
    
    num_boundaries = len(zonal_data['boundary_buses'])
    num_global_kg = len(zonal_data['global_Kg'])
    num_global_ke = len(zonal_data['global_Ke'])

    gnn_z1 = Zone_ADMM_GNN(num_boundaries, num_global_kg, num_global_ke).to(device)
    gnn_z2 = Zone_ADMM_GNN(num_boundaries, num_global_kg, num_global_ke).to(device)
    
    gnn_z1.load_state_dict(torch.load(f"data/ccga_models/zone1_gnn_oracle_{args.case}.pth", map_location=device))
    gnn_z2.load_state_dict(torch.load(f"data/ccga_models/zone2_gnn_oracle_{args.case}.pth", map_location=device))
    
    print("Successfully loaded pre-trained Erdős-GNN Oracle.")

    csv_path = f'data/{args.case}_generated_data.csv' 
    if not os.path.exists(csv_path):
        csv_path = f'data/{args.case}_generated_loads.csv'
        
    loads_df = pd.read_csv(csv_path)
    num_tests = min(args.num_tests, len(loads_df))

    print("\nStarting inference and CCGA evaluation...")
    total_ml_ccga_iters, total_time = 0, 0.0

    for s in range(num_tests):
        print(f"\n--- Scenario {s} ---")
        row = loads_df.iloc[s]
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}
        
        # Inject Perturbed Limits and Costs (if present)
        if f"Gen_{zonal_data['global_Kg'][0]}_Pmax" in row:
            for gen_id in zonal_data['global_Kg']:
                case['gen'].loc[case['gen']['gen_ID'] == gen_id, 'Pmax'] = row[f"Gen_{gen_id}_Pmax"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c1'] = row[f"Gen_{gen_id}_c1"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c2'] = row[f"Gen_{gen_id}_c2"]
        
        start_time = time.time()
        
        predicted_active_Kg, predicted_active_Ke = predict_active_contingencies(case, load_vector, gnn_z1, gnn_z2, device)
        print(f"ML Oracle Predicted Kg: {predicted_active_Kg}")
        print(f"ML Oracle Predicted Ke: {predicted_active_Ke}")
        
        _, _, ccga_iters, final_Kg, final_Ke = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], 
            load_vector, PTDF_matrix,
            initial_active_Kg=predicted_active_Kg,
            initial_active_Ke=predicted_active_Ke
        )
        
        solve_time = time.time() - start_time
        total_time += solve_time
        total_ml_ccga_iters += ccga_iters
        
        print(f"Solver Iterations Required: {ccga_iters}")
        print(f"Time Taken: {solve_time:.4f} seconds")

    print("\n=======================================================")
    print(" EXPERIMENT SUMMARY")
    print("=======================================================")
    print(f"Total Scenarios Tested: {num_tests}")
    print(f"Average Solver Iterations per Scenario: {total_ml_ccga_iters / num_tests:.2f}")
    print(f"Average Total Solve Time per Scenario: {total_time / num_tests:.4f} seconds")

if __name__ == "__main__":
    main()