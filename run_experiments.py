import pandas as pd
import os
import sys
from dcopf_model import solve_dc_opf

def main():
    # 1. Handle Slurm Array Indexing (Default to chunk 0 if running locally)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    chunk_size = 100  # 100 instances per node
    
    start_idx = task_id * chunk_size
    end_idx = start_idx + chunk_size

    # 2. Define Linux cluster paths (Update this to your exact cluster path)
    data_dir = "/home/username/SCDCOPF_exp/data/"
    excel_path = os.path.join(data_dir, "pglib_opf_case118_ieee.xlsx")
    loads_path = os.path.join(data_dir, "118_ieee_generated_loads.csv")
    output_dir = os.path.join(data_dir, "output_labels")
    
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load Base Network Data
    print(f"Node task {task_id}: Loading network topology...")
    bus_df = pd.read_excel(excel_path, sheet_name='bus')
    gen_df = pd.read_excel(excel_path, sheet_name='gen')
    branch_df = pd.read_excel(excel_path, sheet_name='branch')
    cost_df = pd.read_excel(excel_path, sheet_name='gencost')

    # 4. Load only the assigned chunk of the 14,000 load profiles
    print(f"Node task {task_id}: Loading instances {start_idx} to {end_idx-1}...")
    loads_df = pd.read_csv(loads_path, skiprows=range(1, start_idx + 1), nrows=chunk_size)

    # 5. Execution Loop
    results_list = []
    
    for idx, row in loads_df.iterrows():
        global_idx = start_idx + idx
        
        # Build the load vector dictionary mapping bus_i to its new load
        load_vector = {bus_id: row[f"Bus_{bus_id}_Pd"] for bus_id in bus_df['bus_i']}
        
        try:
            # Execute the MILP model
            optimal_g, status = solve_dc_opf(
                bus_df, gen_df, branch_df, cost_df, load_vector
            )
            
            # Add metadata for aggregation tracking
            optimal_g['Instance_ID'] = global_idx
            optimal_g['Termination_Status'] = str(status)
            results_list.append(optimal_g)
            
        except Exception as e:
            print(f"Instance {global_idx} failed: {e}")

    # 6. Export the chunk to a highly aggregated CSV (Exactly 100 rows)
    output_filename = os.path.join(output_dir, f"labels_chunk_{task_id}.csv")
    results_df = pd.DataFrame(results_list)
    
    # Reorder columns to put metadata first
    cols = ['Instance_ID', 'Termination_Status'] + [c for c in results_df.columns if c not in ['Instance_ID', 'Termination_Status']]
    results_df = results_df[cols]
    
    results_df.to_csv(output_filename, index=False)
    print(f"Node task {task_id}: Successfully exported {len(results_df)} rows to {output_filename}")

if __name__ == "__main__":
    main()