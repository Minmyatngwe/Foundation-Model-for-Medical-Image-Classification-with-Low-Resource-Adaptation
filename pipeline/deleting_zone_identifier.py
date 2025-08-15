import os
import glob

# Path to your dataset
base_path = r"data/ChX chest xray dataset"

# Recursively find all files ending with .Zone.Identifier (case-insensitive)
files_to_delete = glob.glob(os.path.join(base_path, "**", "*:Zone.Identifier"), recursive=True)

# Delete each file found
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print(f"Done. Deleted {len(files_to_delete)} files.")
