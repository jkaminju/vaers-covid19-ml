"""
train.py — VAERS COVID-19 Mortality ML Pipeline
UW MSIS 522 HW1

Run once to produce models/ and artifacts/.
Time: ~10-25 min depending on hardware.
"""

import os, gc, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

os.makedirs("models",    exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

from sklearn.compose       import ColumnTransformer
from sklearn.ensemble      import RandomForestClassifier
from sklearn.impute        import SimpleImputer
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics       import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree          import DecisionTreeClassifier
from sklearn.utils         import resample

import lightgbm as lgb
import shap

# ═══════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN VAERSDATA
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Loading VAERSDATA.csv …")

DATA_COLS = [
    "VAERS_ID", "AGE_YRS", "SEX", "STATE", "NUMDAYS", "HOSPDAYS",
    "HOSPITAL", "L_THREAT", "ER_ED_VISIT", "DISABLE", "RECOVD",
    "X_STAY", "BIRTH_DEFECT", "OFC_VISIT", "V_ADMINBY",
    "RECVDATE", "DIED",
]
df = pd.read_csv("VAERSDATA.csv", usecols=DATA_COLS, low_memory=False)
print(f"   Loaded {len(df):,} rows")

BIN_COLS = [
    "HOSPITAL", "L_THREAT", "ER_ED_VISIT", "DISABLE",
    "RECOVD", "X_STAY", "BIRTH_DEFECT", "OFC_VISIT",
]
for col in BIN_COLS:
    df[col] = (df[col].astype(str).str.strip() == "Y").astype(np.int8)
df["DIED"] = (df["DIED"].astype(str).str.strip() == "Y").astype(np.int8)

df["NUMDAYS"]  = pd.to_numeric(df["NUMDAYS"],  errors="coerce").clip(0, 365)
df["HOSPDAYS"] = pd.to_numeric(df["HOSPDAYS"], errors="coerce").clip(0, 365)
df["AGE_YRS"]  = pd.to_numeric(df["AGE_YRS"],  errors="coerce")

# Parse receive date for time-series viz
df["RECVDATE"]         = pd.to_datetime(df["RECVDATE"], errors="coerce")
df["RECV_YEAR_MONTH"]  = df["RECVDATE"].dt.to_period("M").astype(str)
df["RECV_YEAR"]        = df["RECVDATE"].dt.year
df = df.drop(columns=["RECVDATE"])

print(f"   Death rate: {df['DIED'].mean():.3%}  ({df['DIED'].sum():,} deaths)")

# ═══════════════════════════════════════════════════════════════
# 2. LOAD SYMPTOMS  (top-30 MedDRA terms as binary flags)
# ═══════════════════════════════════════════════════════════════
print("\n2. Loading VAERSSYMPTOMS.csv …")

symp = pd.read_csv(
    "VAERSSYMPTOMS.csv",
    usecols=["VAERS_ID","SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"],
    low_memory=False,
)
symp_long = (
    symp.melt(id_vars="VAERS_ID", value_name="SYMPTOM")
    .dropna(subset=["SYMPTOM"])
    .assign(SYMPTOM=lambda x: x["SYMPTOM"].str.strip())
    .query("SYMPTOM != ''")
)
top30 = symp_long["SYMPTOM"].value_counts().head(30).index.tolist()
print(f"   Top-30 symptoms from {symp_long['SYMPTOM'].nunique():,} unique terms")

symp_pivot = (
    symp_long[symp_long["SYMPTOM"].isin(top30)]
    .assign(flag=1)
    .pivot_table(
        index="VAERS_ID", columns="SYMPTOM",
        values="flag", aggfunc="max", fill_value=0,
    )
    .reset_index()
)
symp_pivot.columns.name = None

def _safe_col(s: str) -> str:
    for ch in [" ", "/", ",", "(", ")", "-", "'"]:
        s = s.replace(ch, "_")
    return f"SYM_{s}"

symp_pivot.columns = [
    _safe_col(c) if c != "VAERS_ID" else c for c in symp_pivot.columns
]
SYM_COLS = [c for c in symp_pivot.columns if c.startswith("SYM_")]
del symp, symp_long
gc.collect()

# ═══════════════════════════════════════════════════════════════
# 3. LOAD VACCINES
# ═══════════════════════════════════════════════════════════════
print("\n3. Loading VAERSVAX.csv …")

vax = (
    pd.read_csv(
        "VAERSVAX.csv",
        usecols=["VAERS_ID","VAX_MANU","VAX_DOSE_SERIES","VAX_ROUTE","VAX_SITE"],
        low_memory=False,
    )
    .drop_duplicates("VAERS_ID")
)
vax["VAX_MANU"] = (
    vax["VAX_MANU"].fillna("UNKNOWN").str.upper().str.strip()
    .replace({
        "PFIZER\\BIONTECH":   "PFIZER",
        "PFIZER-BIONTECH":    "PFIZER",
        "PFIZER\\BIONTECH ":  "PFIZER",
    })
)
print(f"   Top manufacturers: {vax['VAX_MANU'].value_counts().head(5).to_dict()}")

# ═══════════════════════════════════════════════════════════════
# 4. MERGE
# ═══════════════════════════════════════════════════════════════
print("\n4. Merging datasets …")

merged = (
    df
    .merge(symp_pivot, on="VAERS_ID", how="left")
    .merge(vax,        on="VAERS_ID", how="left")
)
merged[SYM_COLS] = merged[SYM_COLS].fillna(0).astype(np.int8)
print(f"   Merged shape: {merged.shape}")
del df, symp_pivot, vax
gc.collect()

# ═══════════════════════════════════════════════════════════════
# 5. STRATIFIED SAMPLE  (~150 k rows — all deaths + random non-deaths)
# ═══════════════════════════════════════════════════════════════
print("\n5. Stratified sampling (target 150 k) …")

died1 = merged[merged["DIED"] == 1]
died0 = merged[merged["DIED"] == 0]
n0    = min(150_000 - len(died1), len(died0))
sample = (
    pd.concat([died1, resample(died0, n_samples=n0, random_state=42, replace=False)])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
print(f"   Sample: {len(sample):,} rows | deaths: {sample['DIED'].sum():,} ({sample['DIED'].mean():.3%})")

sample.to_parquet("artifacts/merged_sample.parquet", index=False)
print("   Saved artifacts/merged_sample.parquet")
del died1, died0, merged
gc.collect()

# ═══════════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING + PREPROCESSOR
# ═══════════════════════════════════════════════════════════════
print("\n6. Feature engineering …")

NUM_FEATURES = ["AGE_YRS", "NUMDAYS", "HOSPDAYS"]
BIN_FEATURES = BIN_COLS + SYM_COLS
CAT_FEATURES = ["SEX", "V_ADMINBY", "VAX_MANU", "VAX_DOSE_SERIES", "VAX_ROUTE", "VAX_SITE"]
ALL_FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES

X = sample[ALL_FEATURES].copy()
y = sample["DIED"].values.astype(int)

for col in CAT_FEATURES:
    X[col] = X[col].fillna("Unknown").astype(str)

print(f"   X shape: {X.shape}  |  positives: {y.sum():,}")
print(f"   Num={len(NUM_FEATURES)}  Bin={len(BIN_FEATURES)}  Cat={len(CAT_FEATURES)}")

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ]), NUM_FEATURES),
    ("bin", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0)),
    ]), BIN_FEATURES),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("enc", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ]), CAT_FEATURES),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42,
)
print(f"   Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")

pd.Series(y_test, name="DIED").to_frame().to_parquet("artifacts/y_test.parquet",         index=False)
X_test.reset_index(drop=True).to_parquet("artifacts/X_test_original.parquet", index=False)

# ═══════════════════════════════════════════════════════════════
# 7. MODEL DEFINITIONS  (6 classifiers)
# ═══════════════════════════════════════════════════════════════
MODELS = {
    "logistic":      LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=42,
    ),
    "ridge":         LogisticRegression(
        penalty="l2", C=0.1, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=42,
    ),
    "lasso":         LogisticRegression(
        penalty="l1", C=0.1, solver="saga",  max_iter=2000,
        class_weight="balanced", random_state=42,
    ),
    "cart":          DecisionTreeClassifier(
        max_depth=8, class_weight="balanced", random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1,
        class_weight="balanced", random_state=42,
    ),
    "lightgbm":      lgb.LGBMClassifier(
        n_estimators=300, is_unbalance=True,
        n_jobs=-1, random_state=42, verbose=-1,
    ),
}

pipelines = {
    name: Pipeline([("prep", preprocessor), ("clf", clf)])
    for name, clf in MODELS.items()
}

# ═══════════════════════════════════════════════════════════════
# 8. 5-FOLD STRATIFIED CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. 5-Fold Cross-Validation")
print("=" * 60)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_results = {}

for name, pipe in pipelines.items():
    print(f"   CV [{name}] …", end=" ", flush=True)
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
    cv_results[name] = res
    print(f"AUC={res['test_roc_auc'].mean():.4f}  F1={res['test_f1'].mean():.4f}")

joblib.dump(cv_results, "artifacts/cv_results.pkl")
print("   Saved artifacts/cv_results.pkl")

# ═══════════════════════════════════════════════════════════════
# 9. FINAL FIT + FULL TEST-SET EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. Final fit + test evaluation")
print("=" * 60)

test_results   = {}
roc_data       = {}
pr_data        = {}
confusion_mats = {}
feature_imps   = {}   # for tree models
coeff_dict     = {}   # for linear models

for name, pipe in pipelines.items():
    print(f"   Fitting [{name}] …", end=" ", flush=True)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    cm     = confusion_matrix(y_test, y_pred)

    test_results[name] = {
        "accuracy":      accuracy_score(y_test, y_pred),
        "precision":     precision_score(y_test, y_pred,  zero_division=0),
        "recall":        recall_score(y_test, y_pred,     zero_division=0),
        "f1":            f1_score(y_test, y_pred,         zero_division=0),
        "roc_auc":       roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
        "log_loss_val":  log_loss(y_test, y_prob),
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": test_results[name]["roc_auc"]}

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_data[name] = {
        "precision": prec_arr, "recall": rec_arr,
        "auc": test_results[name]["avg_precision"],
    }

    confusion_mats[name] = cm

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        feature_imps[name] = clf.feature_importances_
    if hasattr(clf, "coef_") and clf.coef_ is not None:
        coeff_dict[name] = clf.coef_[0]

    joblib.dump(pipe, f"models/{name}.joblib")
    print(f"AUC={test_results[name]['roc_auc']:.4f}  F1={test_results[name]['f1']:.4f}")

joblib.dump(test_results,   "artifacts/test_results.pkl")
joblib.dump(roc_data,       "artifacts/roc_data.pkl")
joblib.dump(pr_data,        "artifacts/pr_data.pkl")
joblib.dump(confusion_mats, "artifacts/confusion_matrices.pkl")
print("   Saved test_results, roc_data, pr_data, confusion_matrices")

# ── Preprocessor + feature names ──────────────────────────────
lgb_prep  = pipelines["lightgbm"].named_steps["prep"]
cat_enc   = lgb_prep.named_transformers_["cat"].named_steps["enc"]
feature_names = (
    NUM_FEATURES + BIN_FEATURES
    + list(cat_enc.get_feature_names_out(CAT_FEATURES))
)

joblib.dump(lgb_prep,      "artifacts/preprocessor.pkl")
joblib.dump(feature_names, "artifacts/feature_names.pkl")
print(f"   Saved preprocessor + {len(feature_names)} feature names")

# ── Feature importances with names ───────────────────────────
fi_named = {}
for name, imps in feature_imps.items():
    if len(imps) == len(feature_names):
        fi_named[name] = pd.Series(
            imps, index=feature_names
        ).sort_values(ascending=False)
joblib.dump(fi_named, "artifacts/feature_importances.pkl")

# ── Linear model coefficients with names ─────────────────────
coeff_named = {}
for name, coefs in coeff_dict.items():
    if len(coefs) == len(feature_names):
        coeff_named[name] = pd.Series(
            coefs, index=feature_names
        ).sort_values(key=abs, ascending=False)
joblib.dump(coeff_named, "artifacts/model_coefficients.pkl")
print(f"   Saved feature importances for: {list(fi_named.keys())}")
print(f"   Saved coefficients for: {list(coeff_named.keys())}")

# ── Transformed test set (for SHAP + app) ────────────────────
n_shap       = min(2000, len(X_test))
X_test_arr   = lgb_prep.transform(X_test.iloc[:n_shap])
pd.DataFrame(X_test_arr, columns=feature_names).to_parquet(
    "artifacts/X_test.parquet", index=False
)
print(f"   Saved artifacts/X_test.parquet  {X_test_arr.shape}")

# ═══════════════════════════════════════════════════════════════
# 10. SHAP  (LightGBM, 2 000 rows)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. SHAP analysis (LightGBM)")
print("=" * 60)

lgb_clf   = pipelines["lightgbm"].named_steps["clf"]
explainer = shap.TreeExplainer(lgb_clf)

print("   Computing SHAP values …", end=" ", flush=True)
shap_raw = explainer.shap_values(X_test_arr)

# Normalise to (n_samples, n_features) for positive class
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

# Save as both pkl (legacy) and parquet (DataFrame with feature names)
joblib.dump(shap_vals, "artifacts/shap_values.pkl")
joblib.dump(exp_val,   "artifacts/shap_expected_value.pkl")
pd.DataFrame(shap_vals, columns=feature_names).to_parquet(
    "artifacts/shap_df.parquet", index=False
)
print("   Saved shap_values.pkl, shap_expected_value.pkl, shap_df.parquet")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Training complete!")
print("   Artifacts -> artifacts/")
print("   Models    -> models/")
print("   Next step -> streamlit run app.py")
print("=" * 60)
