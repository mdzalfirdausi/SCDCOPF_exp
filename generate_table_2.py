import argparse
import os
import re
import pandas as pd

# Import the necessary mathematical builders from your main model file
from dcopf_model import build_ptdf, build_and_solve_ccga_master

def extract_presolve_stats(case_name):
    case_path = f'../excel_outputs/{case_name}.xlsx'
    
    if not os.path.exists(case_path):
        print(f"Error: Could not find {case_path}")
        return
        
    print(f"Loading {case_name} to extract Table II network statistics...")
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','branch', 'gencost'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    bus_df = case['bus']
    gen_df = case['gen']
    branch_df = case['branch']

    # 1. Filter active generators for Kg
    zero_gen_idx = [num for num, i in enumerate(gen_df.Pmax.values / baseMVA) 
                    if (i == 0 and (gen_df.Pmin.values / baseMVA)[num] == 0) or 
                    (gen_df.Pmin.values / baseMVA)[num] < 0]
    active_gen_df = gen_df.drop(index=zero_gen_idx)
    
    # 2. Build PTDF to find active Line Contingencies (Ke)
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    PTDF_matrix, bus_list = build_ptdf(bus_df, branch_df, ref_bus_id)
    
    active_Ke_list = []
    bus_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    for k in range(len(branch_df)):
        bus_i = bus_idx[branch_df.iloc[k]['bus_i']]
        bus_j = bus_idx[branch_df.iloc[k]['bus_j']]
        denom = 1.0 - (PTDF_matrix[k, bus_i] - PTDF_matrix[k, bus_j])
        if abs(denom) >= 1e-5:  # Not radial
            active_Ke_list.append(int(branch_df.iloc[k]['line_ID']))
            
    active_Kg_list = active_gen_df['gen_ID'].tolist()
    
    # 3. Create a generic base load vector
    load_vector = {b: bus_df.loc[bus_df['bus_i'] == b, 'Pd'].values[0] for b in bus_list}
    
    print(f"Building Extensive Pyomo Model (|Kg|={len(active_Kg_list)}, |Ke|={len(active_Ke_list)})...")
    print("Handing to Gurobi for Presolve analysis (NodeLimit: 0)...")
    
    log_filename = f"gurobi_presolve_{case_name}.log"
    
    try:
        build_and_solve_ccga_master(
            bus_df, active_gen_df, branch_df, case['gencost'], load_vector, 
            active_Kg_list, active_Ke_list, baseMVA, 
            time_limit=60, log_file=log_filename
        )
    except Exception as e:
        pass # Expected to interrupt on NodeLimit or memory limit

    # 4. Parse the Gurobi Log File
    if not os.path.exists(log_filename):
        print("Error: Gurobi log file was not created.")
        return

    with open(log_filename, 'r') as f:
        log_text = f.read()

    # Regex to find Before Presolve Stats
    rows_match = re.search(r'Optimize a model with (\d+) rows', log_text)
    vars_match = re.search(r'Variable types: (\d+) continuous, \d+ integer \((\d+) binary\)', log_text)
    
    # Regex to find After Presolve Stats
    presolved_match = re.search(r'Presolved: (\d+) rows, (\d+) columns', log_text)
    
    if rows_match and vars_match:
        before_cnst = int(rows_match.group(1))
        before_cv = int(vars_match.group(1))
        bv = int(vars_match.group(2)) 
        
        # --- PRINT TABLE II-A ---
        print("\n" + "="*70)
        print(" TABLE II-A: BEFORE PRESOLVE")
        print("="*70)
        print(f"{'Test Case':<28} | {'#CV':>10} | {'#BV':>10} | {'#Cnst':>10}")
        print("-" * 70)
        print(f"{case_name:<28} | {before_cv/1000:>9.1f}k | {bv/1000:>9.1f}k | {before_cnst/1000:>9.1f}k")
        print("="*70)

        # --- PRINT TABLE II-B ---
        print("\n" + "="*70)
        print(" TABLE II-B: AFTER PRESOLVE")
        print("="*70)
        print(f"{'Test Case':<28} | {'#CV':>10} | {'#BV':>10} | {'#Cnst':>10}")
        print("-" * 70)
        
        if presolved_match:
            after_cnst = int(presolved_match.group(1))
            after_total_vars = int(presolved_match.group(2))
            after_cv = after_total_vars - bv
            
            print(f"{case_name:<28} | {after_cv/1000:>9.1f}k | {bv/1000:>9.1f}k | {after_cnst/1000:>9.1f}k")
            print("="*70 + "\n")
        else:
            print(f"{case_name:<28} | {'--':>10} | {'--':>10} | {'--':>10}")
            print("="*70)
            print("\n[Note] Gurobi did not complete presolve, so post-presolve")
            print("statistics are unavailable.\n")
            
    else:
        print("Error: Could not parse before-presolve stats from Gurobi log file.")
        
    # Clean up log file
    if os.path.exists(log_filename):
        os.remove(log_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MILP Stats for Table II")
    parser.add_argument('--case', type=str, default="pglib_opf_case300_ieee")
    args = parser.parse_args()
    
    extract_presolve_stats(args.case)