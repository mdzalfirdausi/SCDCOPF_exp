import pandas as pd
import os
import sys
from dcopf_model import build_ptdf, run_ccga_algorithm

def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    chunk_size = 100  
    
    start_idx = task_id * chunk_size
    end_idx = start_idx + chunk_size

    base_data_dir = "../excel_outputs/"
    gen_data_dir = "./data/"
    
    excel_path = os.path.join(base_data_dir, "pglib_opf_case118_ieee.xlsx")
    loads_path = os.path.join(gen_data_dir, "118_ieee_generated_loads.csv")
    output_dir = os.path.join(gen_data_dir, "output_labels")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Node task {task_id}: Loading network topology from {excel_path}...")
    bus_df = pd.read_excel(excel_path, sheet_name='bus')
    gen_df = pd.read_excel(excel_path, sheet_name='gen')
    branch_df = pd.read_excel(excel_path, sheet_name='branch')
    cost_df = pd.read_excel(excel_path, sheet_name='gencost')
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    print(f"Node task {task_id}: Pre-computing PTDF Matrix...")
    PTDF_matrix, _ = build_ptdf(bus_df, branch_df, ref_bus_id)

    print(f"Node task {task_id}: Loading instances {start_idx} to {end_idx-1} from {loads_path}...")
    loads_df = pd.read_csv(loads_path, skiprows=range(1, start_idx + 1), nrows=chunk_size)

    results_list = []
    
    for idx, row in loads_df.iterrows():
        global_idx = start_idx + idx
        
        load_vector = {bus_id: row[f"Bus_{bus_id}_Pd"] for bus_id in bus_df['bus_i']}
        
        try:
            # Execute the CCGA loop
            optimal_g, status, iters = run_ccga_algorithm(
                bus_df, gen_df, branch_df, cost_df, load_vector, PTDF_matrix
            )
            
            optimal_g['Instance_ID'] = global_idx
            optimal_g['Termination_Status'] = str(status)
            optimal_g['CCGA_Iterations'] = iters
            results_list.append(optimal_g)
            
            print(f"Solved instance {global_idx} in {iters} CCGA iterations.")
            
        except Exception as e:
            print(f"Instance {global_idx} failed: {e}")

    output_filename = os.path.join(output_dir, f"labels_chunk_{task_id}.csv")
    results_df = pd.DataFrame(results_list)
    
    if not results_df.empty:
        cols = ['Instance_ID', 'Termination_Status', 'CCGA_Iterations'] + [c for c in results_df.columns if c not in ['Instance_ID', 'Termination_Status', 'CCGA_Iterations']]
        results_df = results_df[cols]
    
    results_df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    main()