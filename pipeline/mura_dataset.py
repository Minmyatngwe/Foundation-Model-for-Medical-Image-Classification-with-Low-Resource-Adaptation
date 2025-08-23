# mura_dataset.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------
# Utility Functions
# ---------------------------
def open_image_rgb(path: str):
    """Opens an image, converts to RGB, and returns a PIL.Image."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")

def get_mura_paths_and_labels(root_dir: str):
    """Scans MURA directory and returns lists of image paths and labels."""
    root_dir = Path(root_dir)
    img_paths = []
    labels = []
    
    # MURA paths look like: MURA-v1.1/train/XR_SHOULDER/patient11189/study1_positive/image1.png
    print(f"Scanning MURA directory: {root_dir}")
    image_files = sorted(list(root_dir.glob("**/*.png")))

    for img_path in tqdm(image_files, desc="Parsing MURA file paths"):
        path_str = str(img_path)
        
        # Label is in the parent directory name, e.g., 'study1_positive'
        if "_positive" in path_str:
            labels.append(1)  # 1 for abnormal
            img_paths.append(path_str)
        elif "_negative" in path_str:
            labels.append(0)  # 0 for normal
            img_paths.append(path_str)
            
    print(f"Found {len(img_paths)} images.")
    return img_paths, labels

# ---------------------------
# MURA Dataset Class
# ---------------------------
class MuraDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = open_image_rgb(img_path)

        if self.transform is not None:
            img = self.transform(img)
            
        # For CrossEntropyLoss, labels should be Long tensors
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label