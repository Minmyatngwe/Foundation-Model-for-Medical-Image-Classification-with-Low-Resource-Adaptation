# --------------------Imports---------------------

import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mapping.chxpert_map import label_cols

# -------------------Hyperparameters-------------------

IMG_SIZE = 224 # size to which images will be resized
BATCH_SIZE = 16 # number of images per batch
NUM_WORKERS = 6 # number of subprocesses to use for data loading
PIN_MEMORY = True # whether to pin memory for faster data transfer to GPU
LABEL_DTYPE = torch.float32 # data type for labels
IMAGE_EXTS = (".jpg", ".jpeg", ".png") # supported image file extensions

# --------------------Opening Images------------------

def open_image_rgb(path: str):
    """
    opens a rgb image
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path)
    return img.convert("RGB")

# ---------------------Dataset--------------------------

class MultiLabelXrayDataset(Dataset):
    """
    img_paths = list of locations for all the images. list/np.array of relative paths from CSV
    labels = numpy array shape (N, C) with 0/1 values
    root_dir = base location for all img_paths to start from
    transform = torchvision transform to apply to PIL Image
    """
    
    def __init__(self, img_paths, labels, root_dir=".", transform = None, skip_missing = False):
        assert len(img_paths) == len(labels), "img_paths & labels must have the same length"
        self.root_dir = Path(root_dir)
        self.img_paths = np.array(img_paths)
        self.labels = np.array(labels).astype(np.float32)
        self.transform = transform
        self.skip_missing = skip_missing
        
        if self.skip_missing:
            # this basically ensures that the images are not corrupt and have the corrupt file type which is supported
            ok_idx = []
            for i, p in enumerate(self.img_paths):
                full = self.root_dir / p
                if full.exists() and full.suffix.lower() in IMAGE_EXTS:
                    ok_idx.append(i)
            if len(ok_idx) < len(self.img_paths):
                print(f"[dataset] skipping {len(self.img_paths)-len(ok_idx)} missing/corrupt images.")
                self.img_paths = self.img_paths[ok_idx]
                self.labels = self.labels[ok_idx]
                
    def __len__(self):
        """
        checks the number of images
        """
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        gets the relative path of the images, then opens them. It also checks if the images cannot open and sees if the 
        """
        
        rel_path = str(self.img.paths[idx])
        img_path = str(self.root_dir / rel_path)
        
        try : 
            img = open_image_rgb(img_path)
        except Exception as e:
            # rechecks if the images are fine or not. If a bad image is present, it raises an error
            if self.skip_missing:
                # if a bad image is present then it chooses a random image instead. This should be very rare as the first phase of cleaning the bad images was done before
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx)
            else:
                raise
            
        if self.transform is not None:
            img = self.transform(img)
            
        label = torch.tensor(self.labels[idx], dtype=LABEL_DTYPE)
        return img, label
    
# -----------------------Transforms----------------------

def get_transforms(img_size=IMG_SIZE):
    """
    return the train and validation transforms using torchvision
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_tranform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return train_transform, val_tranform

# ------------------------Helpers: pos_weight-----------------------

def compute_pos_weight(labels_np):
    """
    labels_np = numpy array shape (N, C) of 0/1 values
    returns = torch.tensor of shape (C, ) pos_weight for BCEWithLogitsLoss
    pos_weight[c] = (N_neg / N_pos), if N_pos > 0 else 1.0
    """
    
    N = labels_np.shape[0]
    pos = labels_np.sum(axis = 0)
    #number of negative labels
    neg = N - pos
    # avoid division by zero
    pos_weight = np.where(pos > 0, neg / np.clip(pos, 1, N), 1.0) 
    # The goal of this weight is to balance the loss function by giving more importance to the less frequent classes.
    
    return torch.tensor(pos_weight, dtype=torch.float32)

# ---------------------------Dataloader----------------------------------

def make_dataloaders_from_df(df, root_dir=".", label_cols = None, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, img_size = IMG_SIZE, skip_missing = False):
    """
    df = cleaned dataframe with path column and label_cols present
    label_cols = list of label columns in dataframe. if None infer as all except Path and Metadata heuistics
    """
    # assumes Path is first column if label_cols not provided
    if label_cols is None:
        meta = {"Path",
                "Sex",
                "Age",
                "Frontal/Lateral",
                "AP/PA"}
        label_cols = [c for c in df.columns if c not in meta]
        
    paths = df["Path"].values
    labels = df[label_cols].values.astype(np.float32)
    
    # Here we assume full df is training set; user can call this twice for val with a different df subset.
    train_transform, val_transform = get_transforms(img_size=img_size)
    
    dataset = MultiLabelXrayDataset(paths, labels, root_dir=root_dir, transform=train_transform, skip_missing=skip_missing)
    
    
    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=PIN_MEMORY,
                        drop_last=True)
    
    pos_weight = compute_pos_weight(labels)
    
    return loader, pos_weight, label_cols

# ---------------Sanity chesck / Usage Example---------

if __name__ == "__main__":
    df = pd.read_csv("train_cleaned.csv")
    root_dir = "/data/ChX chest xray dataset"
    
    loader, pos_weight, label_cols = make_dataloaders_from_df(
        df = df,
        root_dir= root_dir,
        label_cols=label_cols,
        batch_size=16,
        num_workers=4,
        img_size=IMG_SIZE,
        skip_missing=True
    )
    
    print("Labels (columns):", label_cols)
    print("pos_weight (for BCEWithLogitsLoss):", pos_weight.numpy())
    
    # iterate one batch
    
    batch = next(iter(loader))
    images, labels = batch
    print("images.shape: ", images.shape)
    print("labels.shape: ", labels.shape)
    print("labels dtypes: ", labels.dtype, "min/max:", labels.min().item(), labels.max().item())
