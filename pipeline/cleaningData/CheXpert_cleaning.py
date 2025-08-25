import pandas as pd
import os

# --- Configuration ---
# Adjust this path to where your CheXpert CSVs are located
CHEXPERT_DATA_DIR = "data/ChX chest xray dataset" 

# --- Main Cleaning Logic ---
def clean_and_save(csv_path, output_path):
    print(f"Reading from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Identify label columns (all columns except the first 5 metadata columns)
    label_cols = df.columns[5:]

    # Rule 1: Fill missing values (NaN) with 0 (Negative)
    df[label_cols] = df[label_cols].fillna(0)
    
    # Rule 2: Replace -1 (Uncertain) with 1 (Positive) - The "U-Ones" strategy
    df[label_cols] = df[label_cols].replace(-1, 1)

    # Convert to integer type for clarity
    df[label_cols] = df[label_cols].astype(int)

    df.to_csv(output_path, index=False)
    print(f"Cleaned file saved to: {output_path}\n")

# --- Run Cleaning for Both Train and Validation Sets ---
if __name__ == "__main__":
    train_csv = os.path.join(CHEXPERT_DATA_DIR, "train.csv")
    valid_csv = os.path.join(CHEXPERT_DATA_DIR, "valid.csv")

    # Define output paths for the cleaned files
    train_clean_csv = os.path.join(CHEXPERT_DATA_DIR, "train_clean.csv")
    valid_clean_csv = os.path.join(CHEXPERT_DATA_DIR, "valid_clean.csv")

    if os.path.exists(train_csv):
        clean_and_save(train_csv, train_clean_csv)
    else:
        print(f"Error: {train_csv} not found. Please check your CHEXPERT_DATA_DIR path.")
        
    if os.path.exists(valid_csv):
        clean_and_save(valid_csv, valid_clean_csv)
    else:
        print(f"Error: {valid_csv} not found. Please check your CHEXPERT_DATA_DIR path.")