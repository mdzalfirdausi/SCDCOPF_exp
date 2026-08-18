import pandas as pd
import os
import sys
import argparse
from dcopf_model import build_ptdf, run_ccga_algorithm

def main():
    # 1. Set up argparse to accept the --case flag
    parser = argparse.ArgumentParser(description="Run CCGA SCOPF for a specific power system case.")
    parser.add_argument('--case', type=str, required=True, 
                        help="The name of the case to run (e.g., 'pglib_opf_case118_ieee')")
    args = parser.parse_args()
    
    case_name = args.case

    # 2. Handle Slurm Array Indexing (Default to chunk 0 if running locally)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    chunk_size = 100
    
    start_idx = task_id * chunk_size
    end_idx = start_idx + chunk_size

    # 3. Define accurate relative paths dynamically based on the case
    base_data_dir = "../excel_outputs/"
    gen_data_dir = "./data/"
    
    excel_path = os.path.join(base_data_dir, f"{case_name}.xlsx")
    loads_path = os.path.join(gen_data_dir, f"{case_name}_generated_loads.csv")
    output_dir = os.path.join(gen_data_dir, "output_labels")
    
    os.makedirs(output_dir, exist_ok=True)

    # 4. Load Base Network Data
    print(f"Node task {task_id}: Loading network topology for case '{case_name}' from {excel_path}...")
    bus_df = pd.read_excel(excel_path, sheet_name='bus')
    gen_df = pd.read_excel(excel_path, sheet_name='gen')
    branch_df = pd.read_excel(excel_path, sheet_name='branch')
    cost_df = pd.read_excel(excel_path, sheet_name='gencost')
    
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    
    print(f"Node task {task_id}: Pre-computing PTDF Matrix (Reference Bus: {ref_bus_id})...")
    PTDF_matrix, _ = build_ptdf(bus_df, branch_df, ref_bus_id)

    # 5. Load only the assigned chunk of the generated loads
    print(f"Node task {task_id}: Loading instances {start_idx} to {end_idx-1} from {loads_path}...")
    loads_df = pd.read_csv(loads_path, skiprows=range(1, start_idx + 1), nrows=chunk_size)

    # 6. Execution Loop
    results_list = []
    
    for local_idx, row in loads_df.reset_index(drop=True).iterrows():
        global_idx = start_idx + local_idx
        
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

    # 7. Export the chunk (Added case_name prefix to avoid chunk overwrite conflicts across cases)
    output_filename = os.path.join(output_dir, f"{case_name}_labels_chunk_{task_id}.csv")
    results_df = pd.DataFrame(results_list)
    
    if not results_df.empty:
        cols = ['Instance_ID', 'Termination_Status', 'CCGA_Iterations'] + [c for c in results_df.columns if c not in ['Instance_ID', 'Termination_Status', 'CCGA_Iterations']]
        results_df = results_df[cols]
    
    results_df.to_csv(output_filename, index=False)
    print(f"Node task {task_id}: Successfully exported {len(results_df)} rows to {output_filename}")

if __name__ == "__main__":
    main()