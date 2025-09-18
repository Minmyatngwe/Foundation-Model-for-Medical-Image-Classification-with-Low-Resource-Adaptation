import torch
import torch.nn as nn
import timm

# --- We copied the necessary function directly into this file ---
def create_foundation_model(num_classes_chexpert=14, model_name="vit_base_patch16_224"):
    """Helper to create the base ViT model structure."""
    model = timm.create_model(model_name, pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes_chexpert)
    return model

# --- The path to your local model file, relative to the 'pipeline' folder ---
# The '..' means "go up one directory"
MODEL_PATH = "/home/akshatgg/projects/CXR_MURA_analysis/project/checkpoints/best_model_epoch18_auc0.7952.pth"

try:
    print(f"Attempting to load local model file: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    if 'model_state' in checkpoint:
        print("\n--- SUCCESS! ---")
        print("The local model file is valid and can be read successfully.")
    else:
        print("\n--- ERROR ---")
        print("The file was loaded, but it's missing the 'model_state' key.")

except Exception as e:
    print(f"\n--- CRITICAL FAILURE ---")
    print("The local model file is corrupt or cannot be loaded.")
    print(f"Error details: {e}")