import pandas as pd

df = pd.read_csv("data/ChX chest xray dataset/train.csv")
df = df.fillna(0) # Fill missing values NaN with 0 which would be negative
df = df.replace(-1, 1) # Replace -1 with 1 which would be positive
df.to_csv("data/ChX chest xray dataset/train_cleaned.csv", index=False)