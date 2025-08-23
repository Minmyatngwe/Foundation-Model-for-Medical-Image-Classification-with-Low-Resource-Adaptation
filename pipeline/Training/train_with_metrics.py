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
    
    