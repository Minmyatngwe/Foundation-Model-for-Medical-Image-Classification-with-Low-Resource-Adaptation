import os 
from PIL import Image

def check_images(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            try:
                Image.open(os.path.join(subdir, file)).verify()
            except Exception:
                os.remove(os.path.join(subdir, file))
                print(f"Corrupt file removed: {os.path.join(subdir, file)}")

check_images("data/MURA musculoskeletal radiographs datasets/MURA-v1.1/train")