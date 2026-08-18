import pandas as pd
import numpy as np
import os
import argparse

def generate_load_profiles(case_name, num_instances):
    # 1. Set the seed for exact reproducibility
    np.random.seed(42)
    
    # Paths dynamically formatted based on the case name
    filepath = f"../excel_outputs/{case_name}.xlsx"
    output_dir = r"./data"
    
    print(f"Loading base network data from {filepath}...")
    bus_df = pd.read_excel(filepath, sheet_name='bus')
    
    base_Pd = bus_df['Pd'].values
    num_buses = len(base_Pd)
    
    load_matrix = np.zeros((num_instances, num_buses))
    
    print(f"Generating {num_instances} load instances...")
    
    for i in range(num_instances):
        deterministic_scale = 0.82 + (i * 0.00002)
        deterministic_Pd = base_Pd * deterministic_scale
        
        # Because the seed is set, this uniform distribution will 
        # yield the exact same arrays across every script execution
        random_noise_scale = np.random.uniform(-0.005, 0.005, size=num_buses)
        random_Pd = base_Pd * random_noise_scale
        
        load_matrix[i, :] = deterministic_Pd + random_Pd

    column_names = [f"Bus_{bus_id}_Pd" for bus_id in bus_df['bus_i']]
    dataset_df = pd.DataFrame(load_matrix, columns=column_names)
    
    # 2. Route the output to your mapped Explorer drive
    os.makedirs(output_dir, exist_ok=True)
    
    # Saves as e.g., "118_ieee_generated_loads.csv" to match run_experiments.py
    output_filename = os.path.join(output_dir, f"{case_name}_generated_loads.csv")
    
    dataset_df.to_csv(output_filename, index=False)
    print(f"Success: Exported exactly {dataset_df.shape[0]} summarized rows to {output_filename}")
    
    return dataset_df

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Generate load profiles for a specific power system case.")
    parser.add_argument('--case', type=str, required=True, 
                        help="The name of the case to run (e.g., '118_ieee', '1354_pegase')")
    parser.add_argument('--num_instances', type=int, default=1000, 
                        help="Number of load profiles to generate (default: 1000)")
    
    args = parser.parse_args()
    
    # Check if the base Excel file exists before generating
    filepath = f"../excel_outputs/{args.case}.xlsx"
    if os.path.exists(filepath):
        generated_data = generate_load_profiles(args.case, args.num_instances)
    else:
        print(f"Error: {filepath} not found. Please ensure the Excel file exists.")