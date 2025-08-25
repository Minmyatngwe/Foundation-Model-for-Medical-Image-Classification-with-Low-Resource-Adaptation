import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Compute per-class metrics
y_true = np.array(all_labels)  # shape: (N, C)
y_pred = np.array(all_probs)   # shape: (N, C)

aucs = []
for i, col in enumerate(label_cols):
    try:
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
    except ValueError:
        auc = np.nan  # if only one class present
    aucs.append(auc)
    print(f"{col}: AUC = {auc:.3f}")

# Precision/Recall/F1 at default 0.5 threshold
y_pred_bin = (y_pred > 0.5).astype(int)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_bin, average=None)

for i, col in enumerate(label_cols):
    print(f"{col}: Precision={prec[i]:.2f}, Recall={rec[i]:.2f}, F1={f1[i]:.2f}")

# Visualize some errors
def show_case(idx):
    path = X_paths[idx]
    img = Image.open(path).convert("L")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"True: {y_true[idx]} \nPred: {np.round(y_pred[idx], 2)}")
    plt.show()

# Show 3 false negatives for a chosen class (e.g., Pleural Effusion)
class_idx = label_cols.index("Pleural Effusion")
fn_indices = np.where((y_true[:, class_idx] == 1) & (y_pred_bin[:, class_idx] == 0))[0]

for i in fn_indices[:3]:
    show_case(i)
