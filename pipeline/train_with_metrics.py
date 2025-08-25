import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler, autocast

import timm
from sklearn.metrics import roc_auc_score, f1_score

from pipeline.dataset_dataloader import MultiLabelXrayDataset, get_transforms, compute_pos_weight

# -----------------------------
# --- CONFIGURATION ---
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 14
IMG_SIZE = 224
BATCH_SIZE = 16       
NUM_EPOCHS = 100    
LR = 1e-4            
WEIGHT_DECAY = 1e-4
MODEL_NAME = "vit_base_patch16_224"  # Vision Transformer
SAVE_DIR = "checkpoints_full_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)
CHEXPERT_DATA_DIR = "data/ChX chest xray dataset/train/"

# -----------------------------
# Helper: build model
# -----------------------------
def create_model(num_classes=NUM_CLASSES, model_name=MODEL_NAME, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, num_classes)
    return model

# -----------------------------
# Metrics utilities
# -----------------------------
def compute_per_class_auc_and_thresholds(probs, labels, thresholds=None):
    C = probs.shape[1]
    per_class_auc = np.full(C, np.nan, dtype=np.float32)
    best_thresholds = np.full(C, 0.5, dtype=np.float32)
    best_f1 = np.zeros(C, dtype=np.float32)
    if thresholds is None: thresholds = np.linspace(0.0, 1.0, 101)

    for c in range(C):
        y_true, y_pred = labels[:, c], probs[:, c]
        if np.all(y_true == 0) or np.all(y_true == 1): continue
        
        per_class_auc[c] = roc_auc_score(y_true, y_pred)
        best_t, best_f = 0.5, -1.0
        for t in thresholds:
            f = f1_score(y_true, (y_pred >= t).astype(int), zero_division=0)
            if f > best_f: best_f, best_t = f, t
        best_thresholds[c], best_f1[c] = best_t, best_f
    return per_class_auc, best_thresholds, best_f1

# -----------------------------
# Training & validation
# -----------------------------
def train(model, train_loader, val_loader, pos_weight=None, num_epochs=NUM_EPOCHS,
          lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE, save_dir=SAVE_DIR):
    model.to(device)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_mean_auc = -1.0
    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Train")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # VALIDATION
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Val"):
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    logits = model(imgs)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))

        per_class_auc, thresholds, per_class_f1 = compute_per_class_auc_and_thresholds(all_probs, all_labels)
        valid_mean_auc = np.nanmean(per_class_auc)
        
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Mean AUC={valid_mean_auc:.4f}")
        
        if valid_mean_auc > best_mean_auc:
            best_mean_auc = valid_mean_auc
            save_path = os.path.join(save_dir, f"best_model_epoch{epoch}_auc{best_mean_auc:.4f}.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "mean_auc": best_mean_auc, "thresholds": thresholds}, save_path)
            print(f"  -> Saved new best model to {save_path}\n")

    print(f"Training complete. Best mean AUC: {best_mean_auc:.4f}")

# -----------------------------
# Main execution block
# -----------------------------
if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(CHEXPERT_DATA_DIR, 'train_clean.csv'))
    valid_df = pd.read_csv(os.path.join(CHEXPERT_DATA_DIR, 'valid_clean.csv'))

    # Apply the same path correction fix to both dataframes
    train_df['Path'] = train_df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)
    valid_df['Path'] = valid_df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)

    label_cols = [c for c in train_df.columns if c not in {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"}]

    train_transform, val_transform = get_transforms(img_size=IMG_SIZE)
    
    train_dataset = MultiLabelXrayDataset(train_df["Path"].values, train_df[label_cols].values,
                                          root_dir=CHEXPERT_DATA_DIR, transform=train_transform)
    val_dataset = MultiLabelXrayDataset(valid_df["Path"].values, valid_df[label_cols].values,
                                        root_dir=CHEXPERT_DATA_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # --Calculate pos_weight from TRAINING data only ---
    pos_weight = compute_pos_weight(train_df[label_cols].values)
    print(f"Using device: {DEVICE}")
    print(f"pos_weight calculated from training set: {pos_weight.numpy()}")

    # --Build Model and Train ---
    model = create_model(num_classes=len(label_cols), model_name=MODEL_NAME)
    train(model, train_loader, val_loader, pos_weight=pos_weight)