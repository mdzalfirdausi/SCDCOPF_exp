import argparse
import pandas as pd
import glob
import os

def merge_chunks():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge Ground-Truth Label Chunks")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case118_ieee")
    args = parser.parse_args()
    
    case_name = args.case
    
    # 1. Grab all chunk files from the labels directory for the specific case
    file_pattern = f"data/labels/{case_name}_labels_chunk_*.csv"
    chunk_files = glob.glob(file_pattern)
    
    if not chunk_files:
        print(f"ERROR: No chunk files found matching pattern '{file_pattern}'")
        print("Please check if the SLURM jobs finished successfully.")
        return
        
    print(f"Found {len(chunk_files)} chunk files for {case_name}. Merging...")

    # 2. Read and combine them into one DataFrame
    df_list = [pd.read_csv(file) for file in chunk_files]
    master_df = pd.concat(df_list, ignore_index=True)

    # 3. Sort by Scenario_ID so they are in perfect order (e.g., 0 to 999)
    master_df = master_df.sort_values(by='Scenario_ID').reset_index(drop=True)

    # 4. Save to the final master CSV
    output_file = f"data/labels/{case_name}_master_labels.csv"
    master_df.to_csv(output_file, index=False)

    print(f"============================================")
    print(f" MERGE COMPLETE")
    print(f" Total Scenarios: {len(master_df)}")
    print(f" Saved to: {output_file}")
    print(f"============================================")

if __name__ == "__main__":
    merge_chunks()