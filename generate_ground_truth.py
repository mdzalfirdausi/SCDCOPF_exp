import os
import time
import argparse
import pandas as pd
import numpy as np

from dcopf_model import build_ptdf, run_ccga_algorithm
from gnn_erdos import create_zonal_data

def generate_labels():
    parser = argparse.ArgumentParser(description="Generate Ground-Truth Labels for CCGA")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case118_ieee")
    parser.add_argument('--start_idx', type=int, default=0, help="Start scenario index")
    parser.add_argument('--end_idx', type=int, default=None, help="End scenario index (exclusive)")
    args = parser.parse_args()
    
    case_name = args.case
    baseMVA = 100.0

    # 1. Load CSV First to fix the "None" print issue
    csv_path = f'data/{case_name}_generated_loads.csv'
    load_profiles = pd.read_csv(csv_path)
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(load_profiles)

    print(f"=======================================================")
    print(f" GENERATING LABELS (Scenarios {start} to {end}) ")
    print(f"=======================================================")

    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    case['gamma'], case['M_eta'] = 1, 1500
    case['gen'].attrs['gamma'] = case['gamma'] 
    
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zonal_data = create_zonal_data(case, total_buses[:midpoint], total_buses[midpoint:])
    global_kg = zonal_data['global_Kg']
    bus_list = sorted(case['bus']['bus_i'].tolist())
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])

    PTDF_matrix, _ = build_ptdf(case['bus'], case['branch'], ref_bus)

    # 2. Slice the Dataset based on arguments
    load_profiles = load_profiles.iloc[start:end]

    os.makedirs('data/labels', exist_ok=True)
    chunk_data = []

    print(f"Starting exact CCGA solver for {len(load_profiles)} scenarios...")
    
    # --- 3. WARM START INITIALIZATION ---
    warm_start_active_set = []
    
    for idx, (original_s, row) in enumerate(load_profiles.iterrows()):
        start_time = time.time()
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}
        
        # --- 4. PASS WARM START TO SOLVER ---
        _, _, ccga_iters, active_S = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], 
            load_vector, PTDF_matrix,
            initial_active_S=warm_start_active_set
        )
        
        # --- 5. UPDATE WARM START FOR NEXT SCENARIO ---
        warm_start_active_set = active_S.copy()
        
        row_dict = {'Scenario_ID': original_s}
        for k in global_kg:
            row_dict[f'Gen_{k}_Active'] = 1 if k in active_S else 0
            
        chunk_data.append(row_dict)
        solve_time = time.time() - start_time
        print(f" -> Solved Scenario {original_s} | True Active Set: {active_S} | Time: {solve_time:.2f}s")

    # 6. Save this specific chunk
    df_chunk = pd.DataFrame(chunk_data)
    chunk_filename = f"data/labels/{case_name}_labels_{start}_to_{end}.csv"
    df_chunk.to_csv(chunk_filename, index=False)
    print(f"\n*** Job Complete. Saved: {chunk_filename} ***\n")

if __name__ == "__main__":
    generate_labels()