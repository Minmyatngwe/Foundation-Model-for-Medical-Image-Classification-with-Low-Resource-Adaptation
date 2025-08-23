# adapt_with_lora.py
# This script loads a pre-trained CheXpert model, applies LoRA, and fine-tunes on MURA.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

import timm
from peft import LoraConfig, get_peft_model

# Import MURA dataset and transforms from your other files
from mura_dataset import MuraDataset, get_mura_paths_and_labels
from dataset_dataloader import get_transforms # Re-using the same transforms

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MURA_NUM_CLASSES = 2 # Abnormal vs. Normal
IMG_SIZE = 224
BATCH_SIZE = 32 # Can often use a larger batch size for LoRA
NUM_EPOCHS = 10 # Adaptation requires fewer epochs
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "vit_base_patch16_224"
SAVE_DIR = "checkpoints_mura_lora"
os.makedirs(SAVE_DIR, exist_ok=True)

# Path to your best foundation model trained on CheXpert
# ADJUST THIS PATH
CHEXPERT_MODEL_PATH = "checkpoints/best_model_epoch_....pth" 

# -----------------------------
# LoRA and Model Setup
# -----------------------------
def create_foundation_model(num_classes_chexpert=14, model_name=MODEL_NAME):
    """Creates the same model structure as the CheXpert one."""
    model = timm.create_model(model_name, pretrained=True) # Start with ImageNet weights
    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, num_classes_chexpert)
    return model

def apply_lora(model):
    """Applies LoRA to the ViT model."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv"],  # Apply to query, key, value matrices in attention
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model

# -----------------------------
# Main Training Logic
# -----------------------------
def train_lora_adaptation(
    mura_root_dir,
    chexpert_model_path,
    device=DEVICE
):
    # --- 1. Load Data ---
    print("Loading MURA dataset...")
    train_paths, train_labels = get_mura_paths_and_labels(os.path.join(mura_root_dir, "train"))
    val_paths, val_labels = get_mura_paths_and_labels(os.path.join(mura_root_dir, "valid"))
    
    train_transform, val_transform = get_transforms(img_size=IMG_SIZE)
    
    train_dataset = MuraDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = MuraDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    # --- 2. Load Foundation Model and Adapt for MURA ---
    print("Loading CheXpert foundation model...")
    foundation_model = create_foundation_model(num_classes_chexpert=14)
    
    # Load the state dict from your saved checkpoint
    checkpoint = torch.load(chexpert_model_path, map_location='cpu')
    foundation_model.load_state_dict(checkpoint['model_state'])
    print("Foundation model loaded successfully.")

    # **Crucial Step**: Reset the classifier head for MURA's binary task
    in_features = foundation_model.head.in_features
    foundation_model.head = nn.Linear(in_features, MURA_NUM_CLASSES)
    print(f"Model head reset for {MURA_NUM_CLASSES} classes.")

    # --- 3. Apply LoRA ---
    model = apply_lora(foundation_model).to(device)

    # --- 4. Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    
    best_val_auc = -1.0

    # --- 5. Training Loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Train")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # --- 6. Validation Loop ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Val"):
                imgs = imgs.to(device)
                with autocast():
                    logits = model(imgs)
                
                # Get probabilities for the positive class (class 1)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        # --- 7. Metrics and Checkpointing ---
        val_auc = roc_auc_score(all_labels, all_preds)
        val_accuracy = accuracy_score(all_labels, np.round(all_preds)) # Accuracy at 0.5 threshold
        print(f"Epoch {epoch}: Val AUC = {val_auc:.4f}, Val Accuracy = {val_accuracy:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"  -> New best model found with AUC: {val_auc:.4f}. Saving...")
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_mura_lora_model.pth"))

    print(f"\nAdaptation complete. Best validation AUC: {best_val_auc:.4f}")

# -----------------------------
# Run the Script
# -----------------------------
if __name__ == "__main__":
    # ADJUST this path to the root of your MURA dataset
    MURA_ROOT_DIR = "/path/to/MURA-v1.1" 
    
    if not os.path.exists(CHEXPERT_MODEL_PATH):
        raise FileNotFoundError(f"CheXpert model checkpoint not found at: {CHEXPERT_MODEL_PATH}\n"
                                "Please update the path to your best-performing .pth file.")
    
    train_lora_adaptation(
        mura_root_dir=MURA_ROOT_DIR,
        chexpert_model_path=CHEXPERT_MODEL_PATH
    )