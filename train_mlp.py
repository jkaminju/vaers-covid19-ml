"""
train_mlp.py — MLP Neural Network (PyTorch) for VAERS COVID-19 mortality
UW MSIS 522 HW1 — Section 2.6

Loads the existing preprocessor and sample data, trains an MLP, and
patches the existing artifact pkl files so the MLP appears in all
app tables and charts alongside the other 6 models.

Run AFTER train.py:
    python train_mlp.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibility ────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Feature lists (must match train.py exactly) ────────────────
BIN_COLS = [
    "HOSPITAL", "L_THREAT", "ER_ED_VISIT", "DISABLE",
    "RECOVD", "X_STAY", "BIRTH_DEFECT", "OFC_VISIT",
]
CAT_FEATURES = [
    "SEX", "V_ADMINBY", "VAX_MANU", "VAX_DOSE_SERIES",
    "VAX_ROUTE", "VAX_SITE",
]
NUM_FEATURES = ["AGE_YRS", "NUMDAYS", "HOSPDAYS"]

print("=" * 60)
print("1. Loading artifacts …")
print("=" * 60)

sample      = pd.read_parquet("artifacts/merged_sample.parquet")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

SYM_COLS = [c for c in sample.columns if c.startswith("SYM_")]
BIN_FEATURES = BIN_COLS + SYM_COLS
ALL_FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES
ALL_FEATURES = [c for c in ALL_FEATURES if c in sample.columns]

X = sample[ALL_FEATURES].copy()
y = sample["DIED"].values.astype(int)

for col in CAT_FEATURES:
    if col in X.columns:
        X[col] = X[col].fillna("Unknown").astype(str)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42,
)
print(f"   Train: {len(X_train_raw):,}   Test: {len(X_test_raw):,}")

# Transform using the already-fitted preprocessor
X_train_arr = preprocessor.transform(X_train_raw).astype(np.float32)
X_test_arr  = preprocessor.transform(X_test_raw).astype(np.float32)
n_features  = X_train_arr.shape[1]
print(f"   Feature dimensions: {n_features}")

# ── Class weights ──────────────────────────────────────────────
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32)
print(f"   Class weight ratio (pos/neg): {pos_weight.item():.2f}")

# ── DataLoaders ────────────────────────────────────────────────
BATCH = 512

# Hold out 20% of training set for validation curve
n_val = int(0.20 * len(X_train_arr))
idx   = np.random.RandomState(SEED).permutation(len(X_train_arr))
val_idx, tr_idx = idx[:n_val], idx[n_val:]

X_tr  = torch.from_numpy(X_train_arr[tr_idx])
y_tr  = torch.from_numpy(y_train[tr_idx].astype(np.float32))
X_val = torch.from_numpy(X_train_arr[val_idx])
y_val = torch.from_numpy(y_train[val_idx].astype(np.float32))
X_te  = torch.from_numpy(X_test_arr)
y_te  = torch.from_numpy(y_test.astype(np.float32))

train_loader = DataLoader(TensorDataset(X_tr, y_tr),  batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH)
test_loader  = DataLoader(TensorDataset(X_te,  y_te),  batch_size=BATCH)

# ── MLP Architecture ───────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

model     = MLP(n_features)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5,
)

# ── Training loop ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Training MLP (50 epochs, early stopping patience=10)")
print("=" * 60)

EPOCHS  = 50
PATIENCE = 10
history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "train_auc":  [], "val_auc":  [],
}

best_val_loss = float("inf")
patience_ctr  = 0
best_state    = None

def eval_loader(loader, crit, pos_w):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            # Apply pos_weight manually for weighted BCE
            prob = model(xb)
            weight = torch.where(yb == 1, pos_w.squeeze(), torch.ones_like(yb))
            loss   = (crit(prob, yb) * weight).mean()
            total_loss += loss.item() * len(yb)
            all_probs.extend(prob.numpy())
            all_labels.extend(yb.numpy())
    n       = len(all_labels)
    avg_loss = total_loss / n
    preds   = (np.array(all_probs) >= 0.5).astype(int)
    acc     = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    return avg_loss, acc, auc

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        prob   = model(xb)
        weight = torch.where(yb == 1, pos_weight.squeeze(), torch.ones_like(yb))
        loss   = (criterion(prob, yb) * weight).mean()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(yb)
    tr_loss /= len(y_tr)

    # ── Validate ───────────────────────────────────────────────
    val_loss, val_acc, val_auc = eval_loader(val_loader, criterion, pos_weight)
    _, tr_acc, tr_auc = eval_loader(train_loader, criterion, pos_weight)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)
    history["train_auc"].append(tr_auc)
    history["val_auc"].append(val_auc)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr  = 0
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"   Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  val_acc={val_acc:.4f}")

    if patience_ctr >= PATIENCE:
        print(f"   Early stop at epoch {epoch} (best val_loss={best_val_loss:.4f})")
        break

# Restore best weights
model.load_state_dict(best_state)
print(f"\n   Best val loss: {best_val_loss:.4f}")

# ── Test-set evaluation ────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Test-set evaluation")
print("=" * 60)

model.eval()
all_probs, all_preds = [], []
with torch.no_grad():
    for xb, _ in test_loader:
        prob = model(xb).numpy()
        all_probs.extend(prob)
        all_preds.extend((prob >= 0.5).astype(int))

y_prob_test = np.array(all_probs)
y_pred_test = np.array(all_preds)

mlp_test = {
    "accuracy":      accuracy_score(y_test, y_pred_test),
    "precision":     precision_score(y_test, y_pred_test, zero_division=0),
    "recall":        recall_score(y_test, y_pred_test,    zero_division=0),
    "f1":            f1_score(y_test, y_pred_test,        zero_division=0),
    "roc_auc":       roc_auc_score(y_test, y_prob_test),
    "avg_precision": average_precision_score(y_test, y_prob_test),
    "log_loss_val":  log_loss(y_test, y_prob_test),
}
print(f"   AUC={mlp_test['roc_auc']:.4f}  F1={mlp_test['f1']:.4f}  "
      f"Precision={mlp_test['precision']:.4f}  Recall={mlp_test['recall']:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_prob_test)
mlp_roc     = {"fpr": fpr, "tpr": tpr, "auc": mlp_test["roc_auc"]}

prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob_test)
mlp_pr = {
    "precision": prec_arr, "recall": rec_arr,
    "auc": mlp_test["avg_precision"],
}

mlp_cm = confusion_matrix(y_test, y_pred_test)

# ── Save model + history ───────────────────────────────────────
print("\n" + "=" * 60)
print("4. Saving artifacts …")
print("=" * 60)

torch.save(model.state_dict(), "models/mlp_weights.pt")
joblib.dump({"n_features": n_features}, "models/mlp_config.pkl")
joblib.dump(history, "artifacts/mlp_history.pkl")
print("   Saved models/mlp_weights.pt + mlp_config.pkl")
print("   Saved artifacts/mlp_history.pkl")

# ── Patch existing artifact pkl files ─────────────────────────
test_results    = joblib.load("artifacts/test_results.pkl")
roc_data        = joblib.load("artifacts/roc_data.pkl")
pr_data         = joblib.load("artifacts/pr_data.pkl")
confusion_mats  = joblib.load("artifacts/confusion_matrices.pkl")

test_results["mlp"]   = mlp_test
roc_data["mlp"]       = mlp_roc
pr_data["mlp"]        = mlp_pr
confusion_mats["mlp"] = mlp_cm

joblib.dump(test_results,   "artifacts/test_results.pkl")
joblib.dump(roc_data,       "artifacts/roc_data.pkl")
joblib.dump(pr_data,        "artifacts/pr_data.pkl")
joblib.dump(confusion_mats, "artifacts/confusion_matrices.pkl")
print("   Patched test_results, roc_data, pr_data, confusion_matrices")

print("\n" + "=" * 60)
print("MLP training complete!")
print("   Next: streamlit run app.py")
print("=" * 60)
