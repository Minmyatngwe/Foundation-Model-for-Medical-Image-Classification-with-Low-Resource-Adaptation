# dataset_dataloader.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------
# Config / Hyperparams
# ---------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4 # Adjust based on your machine's CPUs
PIN_MEMORY = True
LABEL_DTYPE = torch.float32  # for BCEWithLogitsLoss

# ---------------------------
# Utility: safe image open
# ---------------------------
def open_image_rgb(path: str):
    """Open an image path and return an RGB PIL.Image."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path)
    # convert to RGB (repeats grayscale to 3 channels for pretrained models)
    return img.convert("RGB")

# ---------------------------
# Dataset
# ---------------------------
class MultiLabelXrayDataset(Dataset):
    def __init__(self, img_paths, labels, root_dir=".", transform=None):
        self.root_dir = Path(root_dir)
        self.img_paths = np.array(img_paths)
        self.labels = np.array(labels).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Resolve the full image path
        rel_path = str(self.img_paths[idx])
        img_path = str(self.root_dir / rel_path)

        img = open_image_rgb(img_path)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=LABEL_DTYPE)
        return img, label

# ---------------------------
# Transforms
# ---------------------------
def get_transforms(img_size=IMG_SIZE):
    """Return (train_transforms, val_transforms) using torchvision."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_transform

# ---------------------------
# Helpers: pos_weight for imbalanced classes
# ---------------------------
def compute_pos_weight(labels_np):
    """Computes pos_weight for BCEWithLogitsLoss."""
    N = labels_np.shape[0]
    pos = labels_np.sum(axis=0)
    neg = N - pos
    pos_weight = np.where(pos > 0, neg / np.clip(pos, 1, N), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)

# ---------------------------
# DataLoader factory
# ---------------------------
def make_dataloaders_from_df(df, root_dir=".", label_cols=None,
                             batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                             img_size=IMG_SIZE):
    if label_cols is None:
        meta = {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"}
        label_cols = [c for c in df.columns if c not in meta]

    paths = df["Path"].values
    labels = df[label_cols].values

    train_transform, _ = get_transforms(img_size=img_size)

    dataset = MultiLabelXrayDataset(paths, labels, root_dir=root_dir, transform=train_transform)
    
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=PIN_MEMORY,
                        drop_last=True)

    pos_weight = compute_pos_weight(labels)
    return loader, pos_weight, label_cols

# ---------------------------
# Sanity check / Usage example
# ---------------------------
if __name__ == "__main__":
    # --- 1. CONFIGURATION ---
    CHEXPERT_DATA_DIR = "data/ChX chest xray dataset/"
    
    # --- 2. LOAD CLEANED CSV ---
    train_csv_path = os.path.join(CHEXPERT_DATA_DIR, "train_clean.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Cleaned CSV not found at {train_csv_path}. Did you run the cleaning script?")
    
    df = pd.read_csv(train_csv_path)

    # --- 3. THE DEFINITIVE FIX ---
    # The paths in the CSV start with 'CheXpert-v1.0-small/', but that folder doesn't exist.
    # We remove that incorrect prefix from the 'Path' column in the dataframe.
    print("Correcting paths in the dataframe...")
    df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)
    print("Paths corrected.")

    # The root directory should now point to the folder containing 'train/', 'valid/', etc.
    CORRECT_ROOT_DIR = CHEXPERT_DATA_DIR

    # --- 4. BUILD DATALOADER ---
    loader, pos_weight, label_cols = make_dataloaders_from_df(
        df=df,
        root_dir=CORRECT_ROOT_DIR,
        batch_size=4 
    )

    print("Successfully created DataLoader.")
    print("Labels (columns):", label_cols)
    print("pos_weight (for BCEWithLogitsLoss):", pos_weight.numpy())
    print("\n--- Sanity Check: Fetching one batch ---")

    # --- 5. TEST THE PIPELINE ---
    try:
        batch = next(iter(loader))
        images, labels = batch
        print("Images batch shape:", images.shape)
        print("Labels batch shape:", labels.shape)
        print("Labels dtype:", labels.dtype)
        print("Labels min/max:", labels.min().item(), labels.max().item())
        print("--- Sanity Check Passed! ---")
    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"File not found: {e}")
        print(f"The script is currently using root_dir: '{CORRECT_ROOT_DIR}'")
        print("This error means the path correction logic might be wrong. Please check the paths in your CSV.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")