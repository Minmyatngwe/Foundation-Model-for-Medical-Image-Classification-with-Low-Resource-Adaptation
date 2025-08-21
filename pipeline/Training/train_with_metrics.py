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
        raise RuntimeError("UUnknown Model head - adapt code for this architecture")
    return model

