import pandas as pd
import numpy as np
import os

def generate_load_profiles(file_path, output_dir, num_instances=14000):
    # 1. Set the seed for exact reproducibility
    np.random.seed(42)
    
    print("Loading base network data...")
    bus_df = pd.read_excel(file_path, sheet_name='bus')
    
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
    output_filename = os.path.join(output_dir, "118_ieee_generated_loads.csv")
    
    dataset_df.to_csv(output_filename, index=False)
    print(f"Success: Exported exactly {dataset_df.shape[0]} summarized rows to {output_filename}")
    
    return dataset_df

if __name__ == "__main__":
    filepath = "../excel_outputs/pglib_opf_case118_ieee.xlsx"
    target_directory = r"M:\projects\SCDCOPF_exp\data"
    
    if os.path.exists(filepath):
        generated_data = generate_load_profiles(filepath, target_directory)
    else:
        print(f"Error: {filepath} not found in the current directory.")