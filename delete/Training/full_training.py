import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

# ======================
# 1. Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2. Model
# ======================
num_classes = len(label_cols)  # from your dataset
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

# ======================
# 3. Loss and Optimizer
# ======================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

# ======================
# 4. Training Function
# ======================
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# ======================
# 5. Validation Function (compute loss + AUC)
# ======================
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    val_loss = running_loss / len(val_loader.dataset)
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    # Per-class AUC
    aucs = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        except:
            auc = np.nan  # if only one class present
        aucs.append(auc)

    mean_auc = np.nanmean(aucs)
    return val_loss, aucs, mean_auc

# ======================
# 6. Training Loop
# ======================
best_auc = 0.0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_aucs, mean_auc = validate(model, val_loader, criterion)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Mean AUC: {mean_auc:.4f}")
    print(f"Val Per-Class AUC: {val_aucs}")

    # Scheduler step
    scheduler.step(mean_auc)

    # Save best model
    if mean_auc > best_auc:
        best_auc = mean_auc
        torch.save(model.state_dict(), "best_chexpert_model.pth")
        print("✅ Saved best model")

print("Training complete. Best Val AUC:", best_auc)
