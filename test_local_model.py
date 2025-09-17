import torch
from pipeline.predict_and_explain import create_foundation_model

# The path to your local model file
MODEL_PATH = "checkpoints/best_model_epoch18_auc0.7952.pth"

try:
    print(f"Attempting to load local model file: {MODEL_PATH}")
    
    # We only need to load the checkpoint to see if it's corrupt
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Check if the necessary key is inside
    if 'model_state' in checkpoint:
        print("\n--- SUCCESS! ---")
        print("The local model file is valid and can be read successfully.")
    else:
        print("\n--- ERROR ---")
        print("The file was loaded, but it's missing the 'model_state' key.")

except Exception as e:
    print(f"\n--- CRITICAL FAILURE ---")
    print("The local model file is corrupt and cannot be loaded.")
    print(f"Error details: {e}")