import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def open_image_rgb(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")

def get_mura_paths_and_labels(root_dir: str):
    """Scans the MURA directory (train or valid) and extracts image paths and labels."""
    img_paths, labels = [], []
    print(f"Scanning MURA directory: {root_dir}")
    # Find all image paths, which are nested deep inside
    image_files = sorted(list(Path(root_dir).glob("**/*.png")))

    for img_path in tqdm(image_files, desc=f"Parsing {Path(root_dir).name} paths"):
        path_str = str(img_path)
        # The label is in the parent directory name, e.g., 'study1_positive'
        if "_positive" in path_str:
            labels.append(1)  # 1 for abnormal
            img_paths.append(path_str)
        elif "_negative" in path_str:
            labels.append(0)  # 0 for normal
            img_paths.append(path_str)
            
    print(f"Found {len(img_paths)} images in {Path(root_dir).name} set.")
    return img_paths, labels

class MuraDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = open_image_rgb(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        # Use LongTensor for CrossEntropyLoss
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label