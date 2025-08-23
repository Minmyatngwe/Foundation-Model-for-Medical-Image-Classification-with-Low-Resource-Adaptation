# ----------------------Full training + validation + AUC + threshold-tuning script-------

import os
import time
import numpy as np 
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import timm
from sklearn.metrics import roc_auc_score, f1_score
from PyTorch_Dataset_DataLoader.dataset_dataloader import make_dataloaders_from_df, compute_pos_weight, label_cols

# ------------------Config files----------------------------

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 14
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 12
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "vit_base_patch16_224"
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------Helper: build model---------------

def create_model(num_classes = NUM_CLASSES, model_name =MODEL_NAME, pretrained = True):
    model = timm.create_model(model_name, pretrained=pretrained)
    
    if hasattr(model, 'head'):
        in_feats = model.head.in_features
        model.head = nn.Linear(in_feats, num_classes)
    elif hasattr(model, "fc"):
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    else:
        raise RuntimeError("Unknown Model head - adapt code for this architecture")
    return model

# ---------------------Metrics Utilities ---------------
def compute_per_class_auc_and_threshold(probs, labels, thresholds=None):
    """
    probs = numpy array of probablities from 0 to 1
    labels = numpy array of 0 or 1 floats
    AT the end it returns:
                        per_class_auc: numpy array of AUC scores for each class
                        best_thresholds: numpy array of optimal thresholds for each class
                        per_class_f1_at_best: numpy array of F1 scores at the best thresholds for each class

    """
    N, C = probs.shape
    per_class_auc = np.full(C, np.nan, dtype=np.float32)
    best_thresholds = np.full(C, 0.5, dtype=np.float32)
    best_f1 = np.zeroes(C, dtype=np.float32)
    
    if thresholds is None:
        thresholds =np.linspace(0.0, 1.0, 101)
        
    for c in range(C):
        y_true = labels[:, C]
        y_pred = probs[:, C]
        
        
        if np.all(y_true == 0) or np.all(y_true == 1):
            best_thresholds[c] = 0.5
            best_f1[0.5] = 0.0
        
        else:
            best_t = 0.5
            best_f = -1.0
            for t in thresholds:
                y_bin = (y_pred >= t).astpye(int)
                f = f1_score(y_true, y_bin, zero_division= 0)
                if f > best_f:
                    best_f = f
                    best_t = t
                best_thresholds[c] = best_t
                best_f1[c] = best_f
                
            return per_class_auc, best_thresholds, best_f1

# -------------------------Training & Validation------------------------

def train(model, train_loader, val_loader, pos_weight = None, num_epochs=NUM_EPOCHS, lr = LR, weight_decay=WEIGHT_DECAY, device = DEVICE, save_dir = SAVE_DIR):
    model= model.to(device)
    scaler = GradScaler() if device == "cuda" else None
    
    
    # loss with pos weight(torch expects the pos weight as tensor of the shape[C])
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_mean_auc = -1.0
    best_state = None
    best_epoch = -1
    best_threshold = None
    
    for epoch in range(1, num_epochs + 1):
        # ----------TRAIN--------------
        model.train()
        train.loss = 0.0
        n_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Train", leave= False)
        
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking = True)
            lablels = labels.to(device, non_blocking = True)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
            batch_size = imgs.size()
            train_loss += loss.item() * batch_size
            n_train += batch_size
            pbar.set_postfix(loss=loss.item())

        train_loss /= max(1, n_train)
        scheduler.step()

        # ---------- VALIDATION ----------
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Val", leave=False)
            for imgs, labels in pbar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if scaler:
                    with autocast():
                        logits = model(imgs)
                else:
                    logits = model(imgs)

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)  # shape (Nval, C)
        all_labels = np.concatenate(all_labels, axis=0)  # shape (Nval, C)
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))    # sigmoid

        # compute per-class AUC and thresholds
        per_class_auc, thresholds, per_class_f1 = compute_per_class_auc_and_threshold(all_probs, all_labels)

        # compute mean AUC (ignore nan)
        valid_mean_auc = np.nanmean(per_class_auc)
        nans = np.isnan(per_class_auc).sum()

        # Logging
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mean_auc={valid_mean_auc:.4f}, auc_nans={nans}")
        for i, (a, t, f) in enumerate(zip(per_class_auc, thresholds, per_class_f1)):
            a_str = "nan" if np.isnan(a) else f"{a:.4f}"
            print(f"  class {i:02d}: AUC={a_str}, best_thr={t:.2f}, F1@thr={f:.3f}")

        # Save best by mean AUC
        if valid_mean_auc > best_mean_auc:
            best_mean_auc = valid_mean_auc
            best_state = model.state_dict()
            best_epoch = epoch
            best_thresholds = thresholds.copy()
            torch.save({
                "epoch": epoch,
                "model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "mean_auc": best_mean_auc,
                "thresholds": best_thresholds
            }, os.path.join(save_dir, f"best_model_epoch{epoch}_auc{best_mean_auc:.4f}.pth"))
            print(f"  -> saved new best model (epoch {epoch}, mean AUC {best_mean_auc:.4f})")

    print(f"Training complete. Best epoch: {best_epoch} with mean AUC = {best_mean_auc:.4f}")
    return best_state, best_thresholds

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # --- LOAD DATAFRAME & MAKE DATALOADERS ---
    # You should have a cleaned CSV (U-Ones applied) and a validation CSV.
    # Two ways:
    # 1) If you have a single df with a 'split' column: df[df['split']=='train'], df[df['split']=='valid']
    # 2) Use separate files train_clean.csv and valid_clean.csv
    #
    # Here, example assumes you have train_clean.csv and valid_clean.csv and the dataset factory
    try:
        from PyTorch_Dataset_DataLoader.dataset_dataloader import make_dataloaders_from_df
    except Exception:
        raise RuntimeError("Please place dataset_dataloader.py in your PYTHONPATH or adapt loader creation here.")

    train_df = pd.read_csv("train_clean.csv")
    valid_df = pd.read_csv("valid_clean.csv")

    root_dir = "/path/to/chexpert/root"  # ADJUST
    label_cols = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"
    ]

    # Build train and val dataloaders (make_dataloaders_from_df returns one loader and pos_weight for that df)
    train_loader, pos_weight_train, _ = make_dataloaders_from_df(
        df=train_df, root_dir=root_dir, label_cols=label_cols,
        batch_size=BATCH_SIZE, num_workers=4, img_size=IMG_SIZE, skip_missing=True
    )

    # For validation, we need a DataLoader with val transforms and shuffle=False. If your factory doesn't support val,
    # create a small helper like below or modify make_dataloaders_from_df to accept transform type.
    from PyTorch_Dataset_DataLoader.dataset_dataloader import MultiLabelXrayDataset, get_transforms
    _, val_transform = get_transforms(img_size=IMG_SIZE)
    val_dataset = MultiLabelXrayDataset(valid_df["Path"].values, valid_df[label_cols].values,
                                       root_dir=root_dir, transform=val_transform, skip_missing=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=4, pin_memory=True, drop_last=False)

    # Use pos_weight computed on training set
    pos_weight = pos_weight_train

    # Build model
    model = create_model(num_classes=len(label_cols), model_name=MODEL_NAME, pretrained=True)

    # Train
    best_state, best_thresholds = train(model, train_loader, val_loader, pos_weight=pos_weight,
                                        num_epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE)

    # Save thresholds for inference
    np.save(os.path.join(SAVE_DIR, "best_thresholds.npy"), best_thresholds)
    print("Saved best thresholds:", best_thresholds)