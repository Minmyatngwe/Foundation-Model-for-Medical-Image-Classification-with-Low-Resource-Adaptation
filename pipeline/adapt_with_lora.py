import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import timm
from peft import LoraConfig, get_peft_model

# Import your new MURA dataset and the transforms from the previous script
from mura_dataset import MuraDataset, get_mura_paths_and_labels
from dataset_dataloader import get_transforms

# -----------------------------
# --- CONFIGURATION ---
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MURA_NUM_CLASSES = 2  # Abnormal (1) vs. Normal (0)
IMG_SIZE = 224
BATCH_SIZE = 32       # Can often use a larger batch size for LoRA
NUM_EPOCHS = 8        # Adaptation usually requires fewer epochs
LR = 1e-4
MODEL_NAME = "vit_base_patch16_224"
SAVE_DIR = "checkpoints_mura_lora"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- PATHS (ACTION REQUIRED) ---
# This path is correct based on your log.
CHEXPERT_MODEL_PATH = "checkpoints/best_model_epoch9_auc0.7613.pth" 

# !! YOU MUST UPDATE THIS PATH !!
MURA_ROOT_DIR = "data/MURA musculoskeletal radiographs datasets/MURA-v1.1" # e.g., "data/mura/MURA-v1.1"

# -----------------------------
# Model Loading and LoRA Setup
# -----------------------------
def create_foundation_model(num_classes_chexpert=14, model_name=MODEL_NAME):
    """Helper to create the base ViT model structure."""
    model = timm.create_model(model_name, pretrained=False) # No need to download again
    model.head = nn.Linear(model.head.in_features, num_classes_chexpert)
    return model

def apply_lora_to_model(model):
    """Applies LoRA adapters to the model and prints trainable params."""
    config = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.1, bias="none")
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return lora_model

# -----------------------------
# Main Training Logic
# -----------------------------
def main():
    # --- 1. Load Foundation Model ---
    print(f"Loading foundation model from: {CHEXPERT_MODEL_PATH}")
    foundation_model = create_foundation_model()
    checkpoint = torch.load(CHEXPERT_MODEL_PATH, map_location='cpu')
    foundation_model.load_state_dict(checkpoint['model_state'])
    
    # --- 2. Adapt for MURA Task ---
    # **CRITICAL STEP**: Replace the 14-class CheXpert head with a 2-class MURA head
    in_features = foundation_model.head.in_features
    foundation_model.head = nn.Linear(in_features, MURA_NUM_CLASSES)
    print("Model head reset for MURA's 2-class task.")

    # --- 3. Apply LoRA ---
    model = apply_lora_to_model(foundation_model).to(DEVICE)

    # --- 4. Load MURA Data ---
    if MURA_ROOT_DIR == "/path/to/your/MURA-v1.1":
        raise ValueError("Please update the `MURA_ROOT_DIR` variable in the script.")
        
    train_paths, train_labels = get_mura_paths_and_labels(os.path.join(MURA_ROOT_DIR, "train"))
    val_paths, val_labels = get_mura_paths_and_labels(os.path.join(MURA_ROOT_DIR, "valid"))
    
    train_transform, val_transform = get_transforms(img_size=IMG_SIZE)
    train_dataset = MuraDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = MuraDataset(val_paths, val_labels, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 5. Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(device=DEVICE)
    best_val_auc = -1.0

    # --- 6. Training & Validation Loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Val"):
                imgs = imgs.to(DEVICE)
                with autocast(device_type=DEVICE):
                    logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, np.round(all_preds))
        print(f"Epoch {epoch}: Val AUC = {val_auc:.4f}, Val Accuracy = {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_directory = os.path.join(SAVE_DIR, "best_model_adapters")
            print(f"  -> New best model! Saving adapter files to: {save_directory}")
            model.save_pretrained(save_directory)

    print(f"\nAdaptation complete. Best validation AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()