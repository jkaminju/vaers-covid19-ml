"""
train_gridsearch.py — GridSearchCV for CART, Random Forest, LightGBM
UW MSIS 522 HW1 — Sections 2.3, 2.4, 2.5

Loads existing preprocessor + data, runs GridSearchCV for the three
tree models, refits the best pipelines, patches all artifact pkl files,
and saves updated model joblibs.

Run AFTER train.py (and optionally after train_mlp.py):
    python train_gridsearch.py
"""

import os, gc, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split,
)
from sklearn.pipeline  import Pipeline
from sklearn.tree      import DecisionTreeClassifier, export_text
from sklearn.ensemble  import RandomForestClassifier
import lightgbm as lgb
import shap

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

# ── Load ───────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading artifacts …")
print("=" * 60)

sample       = pd.read_parquet("artifacts/merged_sample.parquet")
preprocessor = joblib.load("artifacts/preprocessor.pkl")
feature_names = joblib.load("artifacts/feature_names.pkl")

SYM_COLS     = [c for c in sample.columns if c.startswith("SYM_")]
BIN_FEATURES = BIN_COLS + SYM_COLS
ALL_FEATURES = [c for c in NUM_FEATURES + BIN_FEATURES + CAT_FEATURES
                if c in sample.columns]

X = sample[ALL_FEATURES].copy()
y = sample["DIED"].values.astype(int)
for col in CAT_FEATURES:
    if col in X.columns:
        X[col] = X[col].fillna("Unknown").astype(str)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42,
)
print(f"   Train: {len(X_train_raw):,}   Test: {len(X_test_raw):,}")

# Transform once — used for GridSearchCV
X_tr = preprocessor.transform(X_train_raw)
X_te = preprocessor.transform(X_test_raw)
print(f"   Transformed feature dims: {X_tr.shape[1]}")

# ── Helper: evaluate a fitted clf on test set ──────────────────
def evaluate(name, clf, X_t, y_t):
    y_pred = clf.predict(X_t)
    y_prob = clf.predict_proba(X_t)[:, 1]
    res = {
        "accuracy":      accuracy_score(y_t, y_pred),
        "precision":     precision_score(y_t, y_pred, zero_division=0),
        "recall":        recall_score(y_t, y_pred,    zero_division=0),
        "f1":            f1_score(y_t, y_pred,        zero_division=0),
        "roc_auc":       roc_auc_score(y_t, y_prob),
        "avg_precision": average_precision_score(y_t, y_prob),
        "log_loss_val":  log_loss(y_t, y_prob),
    }
    fpr, tpr, _ = roc_curve(y_t, y_prob)
    roc = {"fpr": fpr, "tpr": tpr, "auc": res["roc_auc"]}
    prec_arr, rec_arr, _ = precision_recall_curve(y_t, y_prob)
    pr  = {"precision": prec_arr, "recall": rec_arr, "auc": res["avg_precision"]}
    cm  = confusion_matrix(y_t, y_pred)
    print(f"   [{name}] AUC={res['roc_auc']:.4f}  F1={res['f1']:.4f}  "
          f"Prec={res['precision']:.4f}  Recall={res['recall']:.4f}")
    return res, roc, pr, cm

gs_results = {}   # store best params + cv_results_df

# ═══════════════════════════════════════════════════════════════
# 2. CART — GridSearchCV (5-fold)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. CART — GridSearchCV (5-fold)")
print("=" * 60)

cart_grid = {
    "max_depth":        [3, 5, 7, 10],
    "min_samples_leaf": [5, 10, 20, 50],
}
cart_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cart_gs = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=42),
    cart_grid,
    cv=cart_cv,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
    refit=True,
)
cart_gs.fit(X_tr, y_train)
print(f"   Best params : {cart_gs.best_params_}")
print(f"   Best CV F1  : {cart_gs.best_score_:.4f}")

cart_cv_df = pd.DataFrame(cart_gs.cv_results_)[
    ["param_max_depth", "param_min_samples_leaf", "mean_test_score", "std_test_score"]
].rename(columns={
    "param_max_depth": "max_depth",
    "param_min_samples_leaf": "min_samples_leaf",
    "mean_test_score": "mean_f1",
    "std_test_score": "std_f1",
})
gs_results["cart"] = {
    "best_params":   cart_gs.best_params_,
    "best_cv_score": cart_gs.best_score_,
    "cv_results_df": cart_cv_df,
    "n_splits":      5,
    "scoring":       "f1",
}

# Refit best CART as full pipeline on raw data
best_cart_clf = DecisionTreeClassifier(
    class_weight="balanced", random_state=42,
    **cart_gs.best_params_,
)
best_cart_pipe = Pipeline([("prep", preprocessor), ("clf", best_cart_clf)])
best_cart_pipe.fit(X_train_raw, y_train)

cart_res, cart_roc, cart_pr, cart_cm = evaluate(
    "cart", best_cart_pipe, X_test_raw, y_test,
)
joblib.dump(best_cart_pipe, "models/cart.joblib")
print("   Saved models/cart.joblib")

# Save text representation of best tree (top 4 levels)
cart_text = export_text(
    best_cart_pipe.named_steps["clf"],
    feature_names=feature_names,
    max_depth=4,
)
with open("artifacts/cart_tree_text.txt", "w") as f:
    f.write(f"Best CART — params: {cart_gs.best_params_}\n\n")
    f.write(cart_text)
print("   Saved artifacts/cart_tree_text.txt")

# ═══════════════════════════════════════════════════════════════
# 3. RANDOM FOREST — GridSearchCV (3-fold, large dataset)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Random Forest — GridSearchCV (3-fold)")
print("=" * 60)

rf_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [5, 8, 12],
}
rf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rf_gs = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    rf_grid,
    cv=rf_cv,
    scoring="f1",
    n_jobs=1,   # RF already uses n_jobs=-1 internally
    verbose=1,
    refit=True,
)
rf_gs.fit(X_tr, y_train)
print(f"   Best params : {rf_gs.best_params_}")
print(f"   Best CV F1  : {rf_gs.best_score_:.4f}")

rf_cv_df = pd.DataFrame(rf_gs.cv_results_)[
    ["param_n_estimators", "param_max_depth", "mean_test_score", "std_test_score"]
].rename(columns={
    "param_n_estimators": "n_estimators",
    "param_max_depth":    "max_depth",
    "mean_test_score":    "mean_f1",
    "std_test_score":     "std_f1",
})
gs_results["random_forest"] = {
    "best_params":   rf_gs.best_params_,
    "best_cv_score": rf_gs.best_score_,
    "cv_results_df": rf_cv_df,
    "n_splits":      3,
    "scoring":       "f1",
}

best_rf_clf = RandomForestClassifier(
    class_weight="balanced", random_state=42, n_jobs=-1,
    **rf_gs.best_params_,
)
best_rf_pipe = Pipeline([("prep", preprocessor), ("clf", best_rf_clf)])
best_rf_pipe.fit(X_train_raw, y_train)

rf_res, rf_roc, rf_pr, rf_cm = evaluate(
    "random_forest", best_rf_pipe, X_test_raw, y_test,
)
joblib.dump(best_rf_pipe, "models/random_forest.joblib")
print("   Saved models/random_forest.joblib")

# ═══════════════════════════════════════════════════════════════
# 4. LIGHTGBM — GridSearchCV (5-fold)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. LightGBM — GridSearchCV (5-fold)")
print("=" * 60)

lgb_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1],
}
lgb_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_gs = GridSearchCV(
    lgb.LGBMClassifier(
        is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1,
    ),
    lgb_grid,
    cv=lgb_cv,
    scoring="f1",
    n_jobs=1,
    verbose=1,
    refit=True,
)
lgb_gs.fit(X_tr, y_train)
print(f"   Best params : {lgb_gs.best_params_}")
print(f"   Best CV F1  : {lgb_gs.best_score_:.4f}")

lgb_cv_df = pd.DataFrame(lgb_gs.cv_results_)[
    ["param_n_estimators", "param_max_depth", "param_learning_rate",
     "mean_test_score", "std_test_score"]
].rename(columns={
    "param_n_estimators":  "n_estimators",
    "param_max_depth":     "max_depth",
    "param_learning_rate": "learning_rate",
    "mean_test_score":     "mean_f1",
    "std_test_score":      "std_f1",
})
gs_results["lightgbm"] = {
    "best_params":   lgb_gs.best_params_,
    "best_cv_score": lgb_gs.best_score_,
    "cv_results_df": lgb_cv_df,
    "n_splits":      5,
    "scoring":       "f1",
}

best_lgb_clf = lgb.LGBMClassifier(
    is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1,
    **lgb_gs.best_params_,
)
best_lgb_pipe = Pipeline([("prep", preprocessor), ("clf", best_lgb_clf)])
best_lgb_pipe.fit(X_train_raw, y_train)

lgb_res, lgb_roc, lgb_pr, lgb_cm = evaluate(
    "lightgbm", best_lgb_pipe, X_test_raw, y_test,
)
joblib.dump(best_lgb_pipe, "models/lightgbm.joblib")
print("   Saved models/lightgbm.joblib")

# ═══════════════════════════════════════════════════════════════
# 5. SAVE gs_results + PATCH ARTIFACT PKLs
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Saving / patching artifacts …")
print("=" * 60)

joblib.dump(gs_results, "artifacts/gridsearch_results.pkl")
print("   Saved artifacts/gridsearch_results.pkl")

test_results   = joblib.load("artifacts/test_results.pkl")
roc_data       = joblib.load("artifacts/roc_data.pkl")
pr_data        = joblib.load("artifacts/pr_data.pkl")
confusion_mats = joblib.load("artifacts/confusion_matrices.pkl")
fi_data        = joblib.load("artifacts/feature_importances.pkl")

for name, res, roc, pr, cm, pipe in [
    ("cart",          cart_res, cart_roc, cart_pr, cart_cm, best_cart_pipe),
    ("random_forest", rf_res,   rf_roc,   rf_pr,   rf_cm,   best_rf_pipe),
    ("lightgbm",      lgb_res,  lgb_roc,  lgb_pr,  lgb_cm,  best_lgb_pipe),
]:
    test_results[name]   = res
    roc_data[name]       = roc
    pr_data[name]        = pr
    confusion_mats[name] = cm
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_") and len(clf.feature_importances_) == len(feature_names):
        fi_data[name] = pd.Series(
            clf.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

joblib.dump(test_results,   "artifacts/test_results.pkl")
joblib.dump(roc_data,       "artifacts/roc_data.pkl")
joblib.dump(pr_data,        "artifacts/pr_data.pkl")
joblib.dump(confusion_mats, "artifacts/confusion_matrices.pkl")
joblib.dump(fi_data,        "artifacts/feature_importances.pkl")
print("   Patched test_results, roc_data, pr_data, confusion_matrices, feature_importances")

# ═══════════════════════════════════════════════════════════════
# 6. RECOMPUTE SHAP for updated LightGBM
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Recomputing SHAP for updated LightGBM …")
print("=" * 60)

lgb_prep  = best_lgb_pipe.named_steps["prep"]
lgb_clf_  = best_lgb_pipe.named_steps["clf"]
n_shap    = min(2000, len(X_test_raw))
X_te_arr  = lgb_prep.transform(X_test_raw.iloc[:n_shap])

explainer = shap.TreeExplainer(lgb_clf_)
print("   Computing SHAP values …", end=" ", flush=True)
shap_raw  = explainer.shap_values(X_te_arr)

if isinstance(shap_raw, list):
    shap_vals = shap_raw[1] if len(shap_raw) == 2 else shap_raw[0]
    ev_raw    = explainer.expected_value
    exp_val   = float(ev_raw[1]) if hasattr(ev_raw, "__len__") else float(ev_raw)
elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
    shap_vals = shap_raw[:, :, 1]
    ev_raw    = explainer.expected_value
    exp_val   = float(ev_raw[1]) if hasattr(ev_raw, "__len__") else float(ev_raw)
else:
    shap_vals = shap_raw
    ev_raw    = explainer.expected_value
    exp_val   = float(ev_raw[1]) if hasattr(ev_raw, "__len__") else float(ev_raw)

print(f"shape={shap_vals.shape}")

joblib.dump(shap_vals, "artifacts/shap_values.pkl")
joblib.dump(exp_val,   "artifacts/shap_expected_value.pkl")
pd.DataFrame(X_te_arr, columns=feature_names).to_parquet(
    "artifacts/X_test.parquet", index=False
)
pd.DataFrame(shap_vals, columns=feature_names).to_parquet(
    "artifacts/shap_df.parquet", index=False
)
print("   Saved updated shap_values, shap_expected_value, X_test, shap_df")

# ── Final summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("GridSearchCV complete! Best hyperparameters:")
for m, r in gs_results.items():
    print(f"  [{m}] {r['best_params']}  CV-F1={r['best_cv_score']:.4f}")
print("\nAll artifacts updated. Run: streamlit run app.py")
print("=" * 60)
