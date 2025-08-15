import pandas as pd
import numpy as np

df = pd.read_csv("data/ChX chest xray dataset/train.csv") #load train.csv

# skipping the path and gender columns because i want to fill all the unfilled values
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
# if there is any blank then i will fill it with 0
df[label_cols] = df[label_cols].fillna(0)

# if there is any value -1 which means that the data filler was uncertain so i will change it to 1 which is positive
df[label_cols] = df[label_cols].replace(-1, 1)

#converting to int form
df[label_cols] = df[label_cols].astype(int)

# now we have to split the data into x and y
X_paths = df["Path"].values
y_labels = df[label_cols].values