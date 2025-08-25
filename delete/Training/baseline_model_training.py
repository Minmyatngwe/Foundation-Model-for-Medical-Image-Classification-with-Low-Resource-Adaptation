import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# --- 1. Define Model ---
class CheXpertNet(nn.Module):
    def __init__(self, num_classes=19):
        super(CheXpertNet, self).__init__()
        # Use pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        # Replace FC layer with multi-label head
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# --- 2. Setup training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 19
model = CheXpertNet(num_classes=num_classes).to(device)

criterion = nn.BCEWithLogitsLoss()   # best for multi-label
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- 3. Training + Validation Loop ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)              # [batch, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, num_classes):
    from sklearn.metrics import roc_auc_score
    model.eval()
    
    total_loss = 0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_labels.append(labels.cpu())
            all_outputs.append(torch.sigmoid(outputs).cpu())
    
    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs).numpy()
    
    # Compute per-class AUC
    aucs = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        except ValueError:
            auc = float('nan')  # class may not appear in validation
        aucs.append(auc)
    
    mean_auc = np.nanmean(aucs)
    return total_loss / len(dataloader), aucs, mean_auc


# --- 4. Run training ---
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, aucs, mean_auc = validate(model, val_loader, criterion, device, num_classes)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean AUC: {mean_auc:.4f}")
    print("Per-class AUC:", aucs)
