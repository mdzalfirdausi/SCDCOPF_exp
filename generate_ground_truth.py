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

    # 1. Load CSV First
    csv_path = f'data/{case_name}_generated_data.csv' # Changed to _data to reflect comprehensive features
    if not os.path.exists(csv_path):
        # Fallback if you are testing with the old loads-only file
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
    
    # Cast to float to prevent Pandas LossySetitemError during injection
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
    
    # Extract both Generator and Line Contingency Sets
    zonal_data = create_zonal_data(case, total_buses[:midpoint], total_buses[midpoint:])
    global_kg = zonal_data['global_Kg']
    global_ke = zonal_data['global_Ke']
    
    bus_list = sorted(case['bus']['bus_i'].tolist())
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])

    PTDF_matrix, _ = build_ptdf(case['bus'], case['branch'], ref_bus)

    # 2. Slice the Dataset based on arguments
    load_profiles = load_profiles.iloc[start:end]

    os.makedirs('data/labels', exist_ok=True)
    chunk_data = []

    print(f"Starting exact CCGA solver for {len(load_profiles)} scenarios...")
    
    # --- 3. DUAL WARM START INITIALIZATION ---
    warm_start_active_kg = []
    warm_start_active_ke = []
    
    for idx, (original_s, row) in enumerate(load_profiles.iterrows()):
        start_time = time.time()
        
        # Extract Perturbed Loads
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}
        
        # Inject Perturbed Limits and Costs (if present in CSV)
        if f"Gen_{global_kg[0]}_Pmax" in row:
            for gen_id in global_kg:
                case['gen'].loc[case['gen']['gen_ID'] == gen_id, 'Pmax'] = row[f"Gen_{gen_id}_Pmax"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c1'] = row[f"Gen_{gen_id}_c1"]
                case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c2'] = row[f"Gen_{gen_id}_c2"]
        
        # --- 4. PASS WARM STARTS TO SOLVER ---
        # Unpack all 5 returns from the updated dcopf_model
        _, _, ccga_iters, active_Kg, active_Ke = run_ccga_algorithm(
            case['bus'], case['gen'], case['branch'], case['gencost'], 
            load_vector, PTDF_matrix,
            initial_active_Kg=warm_start_active_kg,
            initial_active_Ke=warm_start_active_ke
        )
        
        # --- 5. UPDATE WARM STARTS FOR NEXT SCENARIO ---
        warm_start_active_kg = active_Kg.copy()
        warm_start_active_ke = active_Ke.copy()
        
        # --- 6. RECORD TARGET LABELS ---
        row_dict = {'Scenario_ID': original_s}
        
        for k in global_kg:
            row_dict[f'Gen_{k}_Active'] = 1 if k in active_Kg else 0
            
        for e in global_ke:
            row_dict[f'Line_{e}_Active'] = 1 if e in active_Ke else 0
            
        chunk_data.append(row_dict)
        solve_time = time.time() - start_time
        print(f" -> Solved Scenario {original_s} | Active Kg: {len(active_Kg)} | Active Ke: {len(active_Ke)} | Time: {solve_time:.2f}s")

    # 7. Save this specific chunk
    df_chunk = pd.DataFrame(chunk_data)
    chunk_filename = f"data/labels/{case_name}_labels_{start}_to_{end}.csv"
    df_chunk.to_csv(chunk_filename, index=False)
    print(f"\n*** Job Complete. Saved: {chunk_filename} ***\n")

if __name__ == "__main__":
    generate_labels()