import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import cv2 # Uses opencv which is already installed
import timm
from peft import PeftModel

# Import tools from your previous scripts
from dataset_dataloader import get_transforms
from adapt_with_lora import create_foundation_model

# -----------------------------
# --- CONFIGURATION ---
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# --- PATHS (ACTION REQUIRED) ---
CHEXPERT_MODEL_PATH = "checkpoints/best_model_epoch9_auc0.7952.pth"
LORA_ADAPTER_PATH = "checkpoints_mura_lora/best_model_adapters/" # Updated to cleaner path

# !! YOU MUST PROVIDE AN IMAGE TO TEST !!
# Find a MURA image path, e.g., from the valid set.
IMAGE_TO_TEST = "data/MURA musculoskeletal radiographs datasets/MURA-v1.1/valid/XR_WRIST/patient11186/study1_positive/image1.png"

# -----------------------------
# Main Inference Logic
# -----------------------------
def load_full_model_for_inference(chexpert_path, lora_path):
    """Loads the base model, adds the LoRA adapters, and prepares it for inference."""
    model = create_foundation_model(num_classes_chexpert=14)
    # The fix for the PyTorch 2.6 loading error
    checkpoint = torch.load(chexpert_path, map_location='cpu', weights_only=False) 
    model.load_state_dict(checkpoint['model_state'])
    
    model.head = torch.nn.Linear(model.head.in_features, 2)
    
    print(f"Loading LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    print("Merging LoRA adapters into the base model...")
    model = model.merge_and_unload()
    
    model.to(DEVICE)
    model.eval()
    print("Model ready for inference.")
    return model

def predict(model, image_path):
    """Runs a prediction on a single image and returns probs and predicted class."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at: {image_path}")

    _, val_transform = get_transforms(img_size=IMG_SIZE)
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()

    return probabilities.cpu().numpy()[0], predicted_class, img_tensor, img_pil

# -----------------------------
# --- NEW EXPLAINABILITY METHOD: ATTENTION ROLLOUT ---
# -----------------------------
# (Keep all the other code the same, just replace this one function)
# (Keep all the other code in the file the same, just replace this one function)

def generate_attention_map(model, input_tensor, original_image_pil):
    """
    Generates and returns a PIL Image of the attention map visualization.
    """
    attention_maps = []
    hooks = []
    
    for block in model.blocks:
        def hook_fn(module, input, output):
            attention_maps.append(input[0])
        hooks.append(block.attn.attn_drop.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    h, w = input_tensor.shape[2:]
    patch_size = model.patch_embed.patch_size[0]
    num_patches = (h // patch_size) * (w // patch_size)
    
    residual_attn = torch.eye(num_patches + 1, device=DEVICE)
    attn_mats = [attn.mean(dim=1) for attn in attention_maps]
    
    for attn_mat in attn_mats:
        aug_attn_mat = attn_mat + torch.eye(attn_mat.size(1), device=DEVICE)
        aug_attn_mat = aug_attn_mat / aug_attn_mat.sum(dim=-1).unsqueeze(-1)
        residual_attn = torch.matmul(aug_attn_mat, residual_attn)

    grid_attn = residual_attn[0, 1:]
    
    grid_size = int(np.sqrt(num_patches))
    attention_grid = grid_attn.reshape(grid_size, grid_size).cpu().numpy()

    mask = cv2.resize(attention_grid, (IMG_SIZE, IMG_SIZE))
    epsilon = 1e-8
    mask = (mask - mask.min()) / (mask.max() - mask.min() + epsilon)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    original_img_np = np.array(original_image_pil.resize((IMG_SIZE, IMG_SIZE)))
    superimposed_img = np.uint8(0.6 * original_img_np + 0.4 * heatmap)

    vis_image = Image.fromarray(superimposed_img)
    
    # --- THIS IS THE FIX ---
    # Return the final image object so the UI can display it.
    return vis_image

if __name__ == "__main__":
    # Load the model with merged adapters
    inference_model = load_full_model_for_inference(CHEXPERT_MODEL_PATH, LORA_ADAPTER_PATH)
    
    # Run prediction
    class_names = ["Normal (Negative)", "Abnormal (Positive)"]
    probs, pred_idx, img_tensor, img_pil = predict(inference_model, IMAGE_TO_TEST)
    
    print(f"\n--- Prediction Results ---")
    print(f"Image: {IMAGE_TO_TEST}")
    print(f"Predicted Class: {class_names[pred_idx]} (Class {pred_idx})")
    print(f"Probabilities:")
    for i, name in enumerate(class_names):
        print(f"  - {name}: {probs[i]:.4f}")
        
    # Generate the Attention Map for the prediction
    print("\n--- Generating Attention Map ---")
    generate_attention_map(inference_model, img_tensor, img_pil)