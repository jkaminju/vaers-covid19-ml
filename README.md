# VAERS COVID-19 Adverse Event Mortality Analysis

**UW MSIS 522 — Data Science Workflow, HW1**

A machine-learning pipeline that predicts in-hospital mortality from
VAERS COVID-19 adverse event reports (Dec 2020 – Jun 2024), with a
five-tab Streamlit dashboard for exploration, model comparison,
SHAP explainability, and interactive prediction.

**Live app:** https://jkaminju-vaers-covid19.streamlit.app/

---

## Project Structure

```
├── app.py                  # Streamlit dashboard (5 tabs)
├── train.py                # Main training pipeline (6 models + SHAP)
├── train_mlp.py            # MLP Neural Network (PyTorch) — Section 2.6
├── train_gridsearch.py     # GridSearchCV for CART, RF, LightGBM — Sections 2.3-2.5
├── requirements.txt
├── artifacts/              # Pre-computed outputs (parquet + pkl)
│   ├── merged_sample.parquet
│   ├── cv_results.pkl
│   ├── test_results.pkl
│   ├── roc_data.pkl / pr_data.pkl
│   ├── confusion_matrices.pkl
│   ├── feature_importances.pkl / model_coefficients.pkl
│   ├── feature_names.pkl / preprocessor.pkl
│   ├── X_test.parquet / X_test_original.parquet / y_test.parquet
│   ├── shap_values.pkl / shap_expected_value.pkl / shap_df.parquet
│   ├── mlp_history.pkl
│   ├── gridsearch_results.pkl
│   └── cart_tree_text.txt
└── models/                 # Fitted model pipelines
    ├── logistic.joblib
    ├── ridge.joblib
    ├── lasso.joblib
    ├── cart.joblib
    ├── random_forest.joblib
    ├── lightgbm.joblib
    ├── mlp_weights.pt
    └── mlp_config.pkl
```

---

## Data

Download the three VAERS CSV files and place them in the project root
(they are not included in this repo due to file size):

| File | Size | Rows |
|---|---|---|
| `VAERSDATA.csv` | ~877 MB | ~1M |
| `VAERSSYMPTOMS.csv` | ~100 MB | ~1.36M |
| `VAERSVAX.csv` | ~76 MB | ~1.07M |

Source: https://vaers.hhs.gov/data/datasets.html

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/jkaminju/vaers-covid19-ml.git
cd vaers-covid19-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place VAERS CSV files in the project root (see Data section above)
```

---

## How to Run

### Option A — Use pre-trained artifacts (fastest)

The `artifacts/` and `models/` folders are committed to the repo.
Skip training and go straight to the app:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Option B — Retrain from scratch

```bash
# Step 1 — Main pipeline: 6 classifiers + SHAP (~10-25 min)
python train.py

# Step 2 — MLP Neural Network (~3-5 min)
python train_mlp.py

# Step 3 — GridSearchCV for CART, RF, LightGBM (~15-30 min)
python train_gridsearch.py

# Step 4 — Launch the app
streamlit run app.py
```

> All scripts use `random_state=42` for reproducibility.

---

## Models

| Model | Section | Best Hyperparameters | Test AUC |
|---|---|---|---|
| Logistic Regression | 2.2 | penalty=None | 0.946 |
| Ridge (L2) | 2.2 | C=0.1 | 0.946 |
| LASSO (L1) | 2.2 | C=0.1 | 0.946 |
| CART | 2.3 | max_depth=10, min_samples_leaf=10 | 0.928 |
| Random Forest | 2.4 | n_estimators=100, max_depth=12 | 0.954 |
| LightGBM | 2.5 | n_estimators=300, max_depth=7, lr=0.1 | 0.967 |
| MLP (PyTorch) | 2.6 | 128-128, dropout=0.3, Adam | 0.966 |

---

## Streamlit App Tabs

| Tab | Contents |
|---|---|
| Executive Summary | KPI cards, model leaderboard, key findings |
| Descriptive Analytics | Age, sex, manufacturer, symptoms, correlation heatmap, choropleth, time series — each with written interpretation |
| Model Performance | GridSearch results, CV table, test metrics, ROC/PR curves, confusion matrices, MLP training curves, feature importances, coefficients |
| SHAP Analysis & Interactive Prediction | Beeswarm, mean\|SHAP\| bar, dependence plot, waterfall, force plot, SHAP interpretation, live prediction widget with model selector + waterfall |
| COVID Feature Explorer | Per-feature distribution, fatality rate buckets, SHAP scatter, partial dependence plot |
