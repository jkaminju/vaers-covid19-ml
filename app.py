"""
app.py — VAERS COVID-19 Mortality Dashboard
UW MSIS 522 HW1

Run:  streamlit run app.py
Requires:  python train.py  (run first to produce artifacts/ and models/)
"""

import os, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VAERS COVID-19 Dashboard",
    page_icon="💉",
    layout="wide",
)

# ── Model display names ───────────────────────────────────────
MODEL_LABELS = {
    "logistic":      "Logistic Regression (Linear)",
    "ridge":         "Ridge (L2 Logistic)",
    "lasso":         "LASSO (L1 Logistic)",
    "cart":          "CART (Decision Tree)",
    "random_forest": "Random Forest",
    "lightgbm":      "LightGBM",
}

def pretty(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").title())

# ═══════════════════════════════════════════════════════════════
# ARTIFACT LOADERS  (all cached)
# ═══════════════════════════════════════════════════════════════

def _artifacts_ready() -> bool:
    required = [
        "artifacts/merged_sample.parquet",
        "artifacts/cv_results.pkl",
        "artifacts/test_results.pkl",
        "artifacts/shap_values.pkl",
        "artifacts/feature_names.pkl",
        "artifacts/X_test.parquet",
        "artifacts/shap_expected_value.pkl",
        "artifacts/y_test.parquet",
        "artifacts/roc_data.pkl",
        "artifacts/confusion_matrices.pkl",
    ]
    return all(os.path.exists(p) for p in required)


@st.cache_data(show_spinner="Loading sample data …")
def load_sample() -> pd.DataFrame:
    return pd.read_parquet("artifacts/merged_sample.parquet")

@st.cache_data(show_spinner="Loading CV results …")
def load_cv():
    return joblib.load("artifacts/cv_results.pkl")

@st.cache_data(show_spinner="Loading test results …")
def load_test():
    return joblib.load("artifacts/test_results.pkl")

@st.cache_data(show_spinner="Loading SHAP artifacts …")
def load_shap_data():
    shap_vals  = joblib.load("artifacts/shap_values.pkl")
    exp_val    = float(joblib.load("artifacts/shap_expected_value.pkl"))
    feat_names = joblib.load("artifacts/feature_names.pkl")
    X_test_df  = pd.read_parquet("artifacts/X_test.parquet")
    return shap_vals, exp_val, feat_names, X_test_df.values

@st.cache_data(show_spinner="Loading SHAP DataFrame …")
def load_shap_df() -> pd.DataFrame:
    return pd.read_parquet("artifacts/shap_df.parquet")

@st.cache_data(show_spinner="Loading ROC data …")
def load_roc():
    return joblib.load("artifacts/roc_data.pkl")

@st.cache_data(show_spinner="Loading PR data …")
def load_pr():
    path = "artifacts/pr_data.pkl"
    return joblib.load(path) if os.path.exists(path) else {}

@st.cache_data(show_spinner="Loading confusion matrices …")
def load_cm():
    return joblib.load("artifacts/confusion_matrices.pkl")

@st.cache_data(show_spinner="Loading feature importances …")
def load_fi():
    path = "artifacts/feature_importances.pkl"
    return joblib.load(path) if os.path.exists(path) else {}

@st.cache_data(show_spinner="Loading model coefficients …")
def load_coeff():
    path = "artifacts/model_coefficients.pkl"
    return joblib.load(path) if os.path.exists(path) else {}

@st.cache_data(show_spinner="Loading original test features …")
def load_X_test_original() -> pd.DataFrame:
    path = "artifacts/X_test_original.parquet"
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_resource(show_spinner="Loading LightGBM pipeline …")
def load_lgb_pipeline():
    return joblib.load("models/lightgbm.joblib")

# ── Guard ─────────────────────────────────────────────────────
if not _artifacts_ready():
    st.error(
        "**Artifacts not found.** Run `python train.py` first, then refresh."
    )
    st.stop()

# ── Tab layout ────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Executive Summary",
    "📊 Data Visualization",
    "🤖 Model Reports",
    "🔍 SHAP Analysis",
    "🦠 COVID Feature Explorer",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.title("VAERS COVID-19 Adverse Event Reports: Mortality Analysis")
    st.markdown("*UW MSIS 522 — Data Science Workflow, HW1*")

    df        = load_sample()
    test_res  = load_test()
    cv_res    = load_cv()

    best_model = max(test_res, key=lambda k: test_res[k]["roc_auc"])
    best_auc   = test_res[best_model]["roc_auc"]
    total_recs = len(df)
    death_rate = df["DIED"].mean() * 100
    age_min    = int(df["AGE_YRS"].dropna().min())
    age_max    = int(df["AGE_YRS"].dropna().max())
    n_states   = df["STATE"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sample Records",          f"{total_recs:,}")
    c2.metric("Fatality Rate (sample)",  f"{death_rate:.2f}%")
    c3.metric("Age Range",               f"{age_min}–{age_max} yrs")
    c4.metric("States Represented",      str(n_states))
    c5.metric(f"Best AUC ({pretty(best_model)})", f"{best_auc:.4f}")

    st.divider()

    # ── Leaderboard ──────────────────────────────────────────
    st.subheader("Model Leaderboard (Test Set)")
    rows = []
    for m, res in test_res.items():
        rows.append({
            "Model":         pretty(m),
            "AUC-ROC":       round(res["roc_auc"],       4),
            "Avg Precision": round(res.get("avg_precision", float("nan")), 4),
            "F1":            round(res["f1"],             4),
            "Precision":     round(res["precision"],      4),
            "Recall":        round(res["recall"],         4),
            "Accuracy":      round(res["accuracy"],       4),
            "Log Loss":      round(res.get("log_loss_val", float("nan")), 4),
        })
    lb = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    st.dataframe(lb, use_container_width=True, hide_index=True)

    # ── Key findings ──────────────────────────────────────────
    st.divider()
    st.subheader("Key Findings")
    n_sym = sum(1 for c in df.columns if c.startswith("SYM_"))
    lasso_zero = ""
    coeff = load_coeff()
    if "lasso" in coeff:
        n_zero = (coeff["lasso"].abs() < 1e-6).sum()
        lasso_zero = f"LASSO zeroed {n_zero} of {len(coeff['lasso'])} features, " \
                     f"demonstrating automatic feature selection. "

    st.markdown(f"""
**Dataset scope:** VAERS COVID-19 adverse event reports (Dec 2020 – Jun 2024).
This analysis uses a representative stratified sample of **{total_recs:,} records**,
retaining all {df['DIED'].sum():,} reported deaths plus a random non-death sample.
The in-sample fatality rate is **{death_rate:.2f}%**, reflecting real-world class imbalance.

**Features engineered:** {len(df.columns)} total columns — demographics, clinical outcomes,
vaccine metadata, and binary flags for the top-{n_sym} MedDRA symptom terms.

**Models trained:** Six classifiers with 5-fold stratified cross-validation and
class-balanced weighting:
- **Linear models:** Logistic Regression (unregularized), Ridge (L2), LASSO (L1).
  {lasso_zero}
- **Tree models:** CART Decision Tree, Random Forest (200 trees), LightGBM (300 boosting rounds).

**Best model:** **{pretty(best_model)}** achieved AUC-ROC **{best_auc:.4f}** on the
hold-out test set, indicating strong discriminative ability.
Ensemble tree methods outperform linear classifiers, pointing to non-linear
interactions between age, co-morbidities, vaccine manufacturer, and symptom severity.
""")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — DATA VISUALIZATION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.title("Exploratory Data Visualization")
    df = load_sample()

    # ── Age distribution ──────────────────────────────────────
    st.subheader("Age Distribution by Outcome")
    fig_age = px.histogram(
        df.dropna(subset=["AGE_YRS"]),
        x="AGE_YRS", nbins=60,
        color="DIED",
        color_discrete_map={0: "#4C78A8", 1: "#E45756"},
        labels={"AGE_YRS": "Age (years)", "DIED": "Died"},
        barmode="overlay", opacity=0.75,
        title="Age Distribution: Died (red) vs Survived (blue)",
    )
    fig_age.update_layout(legend_title_text="Outcome (1=Died)")
    st.plotly_chart(fig_age, use_container_width=True)
    st.markdown(
        "*Fatalities (red) are heavily concentrated in the 70–85 age range, while "
        "non-fatal reports (blue) are spread more evenly across middle age — confirming "
        "age as the dominant demographic risk factor. Reporters under 40 make up the "
        "largest share of total reports but contribute very few deaths. This skew "
        "motivates age as the top feature in every model trained.*"
    )

    col_a, col_b = st.columns(2)

    # ── Death rate by sex ─────────────────────────────────────
    with col_a:
        st.subheader("Fatality Rate by Sex")
        sex_df = (
            df.groupby("SEX")["DIED"]
            .agg(rate="mean", n="size").reset_index()
            .query("n >= 50")
            .assign(rate_pct=lambda d: (d["rate"] * 100).round(2))
            .sort_values("rate_pct", ascending=False)
        )
        fig_sex = px.bar(
            sex_df, x="SEX", y="rate_pct", color="SEX",
            text="rate_pct",
            labels={"SEX": "Sex", "rate_pct": "Fatality Rate (%)"},
        )
        fig_sex.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_sex, use_container_width=True)
        st.markdown(
            "*Male reporters show a notably higher fatality rate than female reporters, "
            "consistent with established COVID-19 epidemiology where male sex is an "
            "independent risk factor for severe outcomes. This may reflect hormonal "
            "differences in immune response and higher rates of relevant comorbidities "
            "in men. The 'Unknown' sex category carries an inflated rate due to "
            "under-reported records that skew toward unresolved severe cases.*"
        )

    # ── Death rate by manufacturer ────────────────────────────
    with col_b:
        st.subheader("Fatality Rate by Vaccine Manufacturer")
        manu_df = (
            df.groupby("VAX_MANU")["DIED"]
            .agg(rate="mean", n="size").reset_index()
            .query("n >= 100")
            .assign(rate_pct=lambda d: (d["rate"] * 100).round(2))
            .sort_values("rate_pct", ascending=False)
        )
        fig_manu = px.bar(
            manu_df, x="VAX_MANU", y="rate_pct", color="VAX_MANU",
            text="rate_pct",
            labels={"VAX_MANU": "Manufacturer", "rate_pct": "Fatality Rate (%)"},
        )
        fig_manu.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_manu, use_container_width=True)
        st.markdown(
            "*Janssen (J&J) shows the highest reported fatality rate, likely reflecting "
            "its early rollout to older and harder-to-reach populations rather than an "
            "intrinsic product effect. Pfizer and Moderna, administered in far larger "
            "volumes across a broader demographic mix, show lower raw fatality rates. "
            "These differences should be interpreted cautiously — reporting demographics "
            "and rollout timing confound direct manufacturer comparisons.*"
        )

    # ── Top-15 symptoms ───────────────────────────────────────
    st.subheader("Top-15 Reported Symptoms")
    sym_cols_all = [c for c in df.columns if c.startswith("SYM_")]
    if sym_cols_all:
        sym_total = df[sym_cols_all].sum().sort_values(ascending=False).head(15)
        sym_death = (
            df[df["DIED"] == 1][sym_cols_all].sum()
            .reindex(sym_total.index)
        )
        sym_plot = pd.DataFrame({
            "Symptom":    [c.replace("SYM_","").replace("_"," ") for c in sym_total.index],
            "All Reports": sym_total.values,
            "Deaths":      sym_death.values,
        })
        fig_sym = px.bar(
            sym_plot.melt(id_vars="Symptom", var_name="Group", value_name="Count"),
            x="Count", y="Symptom", color="Group", orientation="h",
            barmode="group",
            color_discrete_map={"All Reports": "#4C78A8", "Deaths": "#E45756"},
            title="Symptom Counts — All Reports vs Deaths",
        )
        fig_sym.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_sym, use_container_width=True)
        st.markdown(
            "*Dyspnoea (shortness of breath), pyrexia (fever), and fatigue dominate "
            "both the overall report population and the death subset, identifying them "
            "as key signals of vaccine-related adverse events. Death-specific counts "
            "skew toward cardiovascular and respiratory symptoms — such as chest pain "
            "and dyspnoea — relative to the general reporting population. The ratio of "
            "death counts to all-report counts for each symptom can serve as a rough "
            "proxy for symptom-level lethality risk.*"
        )

    # ── Correlation heatmap ───────────────────────────────────
    st.subheader("Correlation Heatmap (Numeric & Binary Features)")
    heat_cols = [
        "AGE_YRS", "NUMDAYS", "HOSPDAYS",
        "HOSPITAL", "L_THREAT", "ER_ED_VISIT", "DISABLE",
        "RECOVD", "X_STAY", "BIRTH_DEFECT", "OFC_VISIT", "DIED",
    ]
    heat_cols = [c for c in heat_cols if c in df.columns]
    corr = df[heat_cols].astype(float).corr()
    fig_heat, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.4, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Pearson Correlation Matrix", pad=12)
    st.pyplot(fig_heat, use_container_width=True)
    plt.close(fig_heat)
    st.markdown(
        "*Age (`AGE_YRS`) shows the strongest positive correlation with `DIED`, "
        "confirming it as the most predictive demographic variable in the dataset. "
        "`HOSPITAL` and `L_THREAT` correlate positively with mortality, as expected — "
        "severe cases requiring hospitalisation or posing life-threatening risk more "
        "often result in death. `RECOVD` shows a strong negative correlation with "
        "`DIED`, which is mechanistically sound: patients who fully recovered did not "
        "die, making these two outcomes mutually exclusive in most records.*"
    )

    col_c, col_d = st.columns(2)

    # ── Dose series ───────────────────────────────────────────
    with col_c:
        st.subheader("Fatality Rate by Vaccine Dose Series")
        if "VAX_DOSE_SERIES" in df.columns:
            dose_df = (
                df.groupby("VAX_DOSE_SERIES")["DIED"]
                .agg(rate="mean", n="size").reset_index()
                .query("n >= 100")
                .assign(rate_pct=lambda d: (d["rate"] * 100).round(2))
                .sort_values("VAX_DOSE_SERIES")
            )
            fig_dose = px.bar(
                dose_df, x="VAX_DOSE_SERIES", y="rate_pct",
                text="rate_pct",
                labels={"VAX_DOSE_SERIES": "Dose Series", "rate_pct": "Fatality Rate (%)"},
                color="rate_pct", color_continuous_scale="Reds",
            )
            fig_dose.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig_dose, use_container_width=True)
            st.markdown(
                "*First-dose fatality rates are elevated because early rollout "
                "prioritised nursing home residents and severely immunocompromised "
                "individuals — the highest-risk populations. Later doses (3rd, 4th) "
                "show lower fatality rates, consistent with a healthier booster-seeking "
                "population and broader immunity at the time of administration. This "
                "dose-series pattern reflects vaccination campaign prioritisation "
                "strategies rather than a causal protective effect of additional doses.*"
            )

    # ── Onset lag boxplot ─────────────────────────────────────
    with col_d:
        st.subheader("Onset-to-Report Lag (NUMDAYS) by Outcome")
        box_df = df.dropna(subset=["NUMDAYS"]).copy()
        box_df["Outcome"] = box_df["DIED"].map({0: "Survived", 1: "Died"})
        fig_box = px.box(
            box_df, x="Outcome", y="NUMDAYS", color="Outcome",
            color_discrete_map={"Survived": "#4C78A8", "Died": "#E45756"},
            labels={"NUMDAYS": "Days from Onset to Report"},
            points=False,
        )
        fig_box.update_layout(yaxis_range=[0, 120])
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown(
            "*Deceased patients tend to have shorter onset-to-report lag times than "
            "survivors, suggesting fatal adverse events are identified and filed rapidly "
            "by healthcare providers. Longer lags among survivors likely reflect milder "
            "events noticed and self-reported retrospectively. This pattern underscores "
            "the importance of early post-vaccination monitoring windows in capturing "
            "the highest-severity events.*"
        )

    # ── Reports over time ─────────────────────────────────────
    if "RECV_YEAR_MONTH" in df.columns:
        st.subheader("Reports Over Time by Outcome")
        ts = (
            df.groupby(["RECV_YEAR_MONTH", "DIED"])
            .size().reset_index(name="Count")
            .sort_values("RECV_YEAR_MONTH")
        )
        ts["Outcome"] = ts["DIED"].map({0: "Survived", 1: "Died"})
        # Keep last 42 months for readability
        months = sorted(ts["RECV_YEAR_MONTH"].unique())
        ts = ts[ts["RECV_YEAR_MONTH"].isin(months[-42:])]
        fig_ts = px.line(
            ts, x="RECV_YEAR_MONTH", y="Count", color="Outcome",
            color_discrete_map={"Survived": "#4C78A8", "Died": "#E45756"},
            markers=True, title="Monthly Report Volume",
            labels={"RECV_YEAR_MONTH": "Month", "Count": "# Reports"},
        )
        fig_ts.update_xaxes(tickangle=45)
        st.plotly_chart(fig_ts, use_container_width=True)
        st.markdown(
            "*Report volume surged sharply in early 2021 during the initial rollout, "
            "peaked mid-2021 as vaccination rates reached maximum uptake, then declined "
            "steadily through 2022–2024 as campaign momentum slowed. Death reports "
            "follow the overall volume trend, remaining a consistently small fraction "
            "throughout — indicating the absolute fatality count was highest during the "
            "peak reporting period. The gradual decline in both series after 2021 "
            "reflects reduced booster uptake and improved safety signal processing "
            "reducing duplicate or late filings.*"
        )

    # ── Choropleth map ────────────────────────────────────────
    st.subheader("Fatality Rate by US State")
    state_df = (
        df.dropna(subset=["STATE"])
        .groupby("STATE")["DIED"]
        .agg(rate="mean", n="size").reset_index()
        .query("n >= 20")
        .assign(rate_pct=lambda d: (d["rate"] * 100).round(3))
    )
    fig_map = px.choropleth(
        state_df, locations="STATE", locationmode="USA-states",
        color="rate_pct", scope="usa",
        color_continuous_scale="Reds",
        labels={"rate_pct": "Fatality Rate (%)"},
        title="VAERS COVID-19 Fatality Rate by State (≥20 reports)",
        height=550,
    )
    fig_map.update_layout(geo=dict(showlakes=True, lakecolor="rgb(255,255,255)"))
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown(
        "*South Dakota and Kentucky show the highest state-level fatality rates, "
        "likely driven by older rural populations, limited healthcare access, and "
        "early rollout demographics rather than state-specific vaccine risks. "
        "Plains and southeastern states cluster toward higher rates, consistent with "
        "regions that have larger elderly rural populations and fewer healthcare "
        "facilities per capita. Western and northeastern states generally show lower "
        "rates, reflecting earlier broader uptake across younger and more urban "
        "populations with better access to follow-up care.*"
    )

    # ── Age boxplot by hospitalization ───────────────────────
    st.subheader("Age Distribution by Clinical Outcome")
    outcome_cols = {
        "HOSPITAL": "Hospitalised", "L_THREAT": "Life-Threatening",
        "ER_ED_VISIT": "ER Visit", "DISABLE": "Disability", "DIED": "Died",
    }
    avail = {k: v for k, v in outcome_cols.items() if k in df.columns}
    age_outcome_rows = []
    for col, label in avail.items():
        tmp = df.dropna(subset=["AGE_YRS"])
        age_outcome_rows.append({"Outcome": f"{label}=1", "AGE_YRS": tmp[tmp[col]==1]["AGE_YRS"].values})
        age_outcome_rows.append({"Outcome": f"{label}=0", "AGE_YRS": tmp[tmp[col]==0]["AGE_YRS"].values})
    age_long = pd.concat([
        pd.DataFrame({"Outcome": r["Outcome"], "Age": r["AGE_YRS"]})
        for r in age_outcome_rows
    ], ignore_index=True)
    fig_age_box = px.box(
        age_long, x="Outcome", y="Age", color="Outcome",
        title="Age Distribution by Clinical Outcome",
        labels={"Age": "Age (years)"},
    )
    fig_age_box.update_layout(showlegend=False, xaxis_tickangle=30)
    st.plotly_chart(fig_age_box, use_container_width=True)
    st.markdown(
        "*Across every clinical outcome category, patients who experienced the adverse "
        "outcome (=1) are consistently older than those who did not, reinforcing age as "
        "a universal risk amplifier for vaccine-related adverse events of all severities. "
        "The median age of deceased patients is markedly higher than for the "
        "hospitalised or ER-visit groups, indicating that age compounds risk at every "
        "step of the severity ladder. The tight interquartile range for the Died=1 "
        "group suggests most fatalities are concentrated in a narrow older age band "
        "rather than distributed across all ages.*"
    )

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL REPORTS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.title("Model Performance Reports")
    cv_res   = load_cv()
    test_res = load_test()
    roc_d    = load_roc()
    pr_d     = load_pr()
    cms      = load_cm()
    fi       = load_fi()
    coeff    = load_coeff()

    # ── CV summary ────────────────────────────────────────────
    st.subheader("5-Fold Cross-Validation (Training Set)")
    cv_rows = []
    for name, res in cv_res.items():
        row = {"Model": pretty(name)}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            vals = res[f"test_{metric}"]
            row[metric.upper().replace("_","-")] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        cv_rows.append(row)
    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    # ── CV box plots ──────────────────────────────────────────
    st.subheader("CV Fold Score Distributions")
    metric_pick = st.selectbox(
        "Select CV metric", ["roc_auc", "f1", "precision", "recall", "accuracy"],
        key="cv_metric_pick",
    )
    box_data = []
    for name, res in cv_res.items():
        for v in res[f"test_{metric_pick}"]:
            box_data.append({"Model": pretty(name), "Score": v})
    fig_cv_box = px.box(
        pd.DataFrame(box_data), x="Model", y="Score", color="Model",
        title=f"CV {metric_pick.upper().replace('_','-')} across 5 folds",
    )
    fig_cv_box.update_layout(showlegend=False, xaxis_tickangle=20)
    st.plotly_chart(fig_cv_box, use_container_width=True)

    st.divider()

    # ── Test set metrics ──────────────────────────────────────
    st.subheader("Test-Set Metrics (Hold-out 30%)")
    trows = []
    for name, res in test_res.items():
        trows.append({
            "Model":         pretty(name),
            "AUC-ROC":       round(res["roc_auc"],                    4),
            "Avg Precision": round(res.get("avg_precision", np.nan),   4),
            "F1":            round(res["f1"],                          4),
            "Precision":     round(res["precision"],                   4),
            "Recall":        round(res["recall"],                      4),
            "Accuracy":      round(res["accuracy"],                    4),
            "Log Loss":      round(res.get("log_loss_val", np.nan),    4),
        })
    t_df = pd.DataFrame(trows).sort_values("AUC-ROC", ascending=False)
    st.dataframe(t_df, use_container_width=True, hide_index=True)

    col_e, col_f = st.columns(2)

    # ── AUC-ROC bar ───────────────────────────────────────────
    with col_e:
        auc_bar = pd.DataFrame([
            {"Model": pretty(k), "AUC-ROC": v["roc_auc"]}
            for k, v in test_res.items()
        ]).sort_values("AUC-ROC")
        fig_auc = px.bar(
            auc_bar, x="AUC-ROC", y="Model", orientation="h",
            color="AUC-ROC", color_continuous_scale="Greens", text="AUC-ROC",
            title="AUC-ROC by Model",
        )
        fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_auc.update_layout(xaxis_range=[0, 1.05])
        st.plotly_chart(fig_auc, use_container_width=True)

    # ── F1 bar ─────────────────────────────────────────────────
    with col_f:
        f1_bar = pd.DataFrame([
            {"Model": pretty(k), "F1": v["f1"]}
            for k, v in test_res.items()
        ]).sort_values("F1")
        fig_f1 = px.bar(
            f1_bar, x="F1", y="Model", orientation="h",
            color="F1", color_continuous_scale="Blues", text="F1",
            title="F1 Score by Model",
        )
        fig_f1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_f1.update_layout(xaxis_range=[0, 1.05])
        st.plotly_chart(fig_f1, use_container_width=True)

    # ── ROC curves ────────────────────────────────────────────
    st.subheader("ROC Curves (All Models)")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="gray"))
    colors = px.colors.qualitative.Plotly
    for i, (name, rd) in enumerate(roc_d.items()):
        fig_roc.add_trace(go.Scatter(
            x=rd["fpr"], y=rd["tpr"],
            mode="lines", name=f"{pretty(name)} (AUC={rd['auc']:.4f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        title="ROC Curves — Test Set", legend=dict(x=0.6, y=0.1),
        width=800, height=500,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── PR curves ─────────────────────────────────────────────
    if pr_d:
        st.subheader("Precision-Recall Curves (All Models)")
        fig_pr = go.Figure()
        for i, (name, pd_) in enumerate(pr_d.items()):
            fig_pr.add_trace(go.Scatter(
                x=pd_["recall"], y=pd_["precision"],
                mode="lines", name=f"{pretty(name)} (AP={pd_['auc']:.4f})",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_pr.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            title="Precision-Recall Curves — Test Set",
            legend=dict(x=0.5, y=0.9), width=800, height=500,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Confusion matrices ────────────────────────────────────
    st.subheader("Confusion Matrices (Test Set)")
    n_models = len(cms)
    cm_cols  = st.columns(min(n_models, 3))
    for i, (name, cm) in enumerate(cms.items()):
        with cm_cols[i % 3]:
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
            ax_cm.set_title(pretty(name), fontsize=9)
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
            tn, fp, fn, tp = cm.ravel()
            st.caption(
                f"TP={tp} | FP={fp} | FN={fn} | TN={tn} | "
                f"Specificity={(tn/(tn+fp)):.3f}"
            )

    st.divider()

    # ── Feature importance (tree models) ─────────────────────
    if fi:
        st.subheader("Feature Importance — Tree Models (Top 20)")
        for name, imp_series in fi.items():
            top20 = imp_series.head(20)[::-1]
            fig_fi = px.bar(
                x=top20.values, y=top20.index, orientation="h",
                color=top20.values, color_continuous_scale="Oranges",
                title=f"{pretty(name)} — Feature Importance",
                labels={"x": "Importance", "y": "Feature"},
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Coefficients (linear models) ─────────────────────────
    if coeff:
        st.subheader("Feature Coefficients — Linear Models (Top 20 by |coef|)")
        for name, coef_series in coeff.items():
            top20 = coef_series.head(20)[::-1]
            fig_coef = px.bar(
                x=top20.values, y=top20.index, orientation="h",
                color=top20.values, color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title=f"{pretty(name)} — Coefficients (sorted by |value|)",
                labels={"x": "Coefficient", "y": "Feature"},
            )
            lasso_zero_n = (coef_series.abs() < 1e-6).sum()
            st.plotly_chart(fig_coef, use_container_width=True)
            if name == "lasso":
                st.info(f"LASSO zeroed {lasso_zero_n}/{len(coef_series)} features "
                        f"({100*lasso_zero_n/len(coef_series):.1f}%).")

# ═══════════════════════════════════════════════════════════════
# TAB 4 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.title("SHAP Analysis (LightGBM)")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) values measure each feature's "
        "contribution to individual predictions. Positive = pushes toward **death**."
    )

    shap_vals, exp_val, feat_names, X_test_arr = load_shap_data()
    shap_df_  = load_shap_df()

    # ── A1. Beeswarm ─────────────────────────────────────────
    st.subheader("A1 — SHAP Summary (Beeswarm)")
    with st.spinner("Rendering beeswarm …"):
        fig_bee = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_vals, X_test_arr, feature_names=feat_names,
            show=False, max_display=20,
        )
        st.pyplot(fig_bee, use_container_width=True)
        plt.close(fig_bee)

    # ── A2. Mean |SHAP| bar ───────────────────────────────────
    st.subheader("A2 — Mean |SHAP| Feature Importance (Top 20)")
    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_imp = (
        pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_abs})
        .sort_values("Mean |SHAP|", ascending=False)
        .head(20)
    )
    fig_shap_bar = px.bar(
        shap_imp[::-1], x="Mean |SHAP|", y="Feature", orientation="h",
        color="Mean |SHAP|", color_continuous_scale="Oranges",
        title="Top 20 Features by Mean |SHAP| Value",
    )
    st.plotly_chart(fig_shap_bar, use_container_width=True)

    # ── A3. Dependence plot ───────────────────────────────────
    st.subheader("A3 — SHAP Dependence Plot")
    col_dp1, col_dp2 = st.columns(2)
    with col_dp1:
        # Offer top-20 features by importance as options
        top20_feats = shap_imp["Feature"].tolist()
        dep_feat = st.selectbox("Primary feature", top20_feats, key="dep_feat")
    with col_dp2:
        interact_feat = st.selectbox(
            "Interaction color feature",
            ["(Auto)"] + top20_feats,
            key="interact_feat",
        )

    dep_idx = feat_names.index(dep_feat)
    feat_vals = X_test_arr[:, dep_idx]
    dep_shap  = shap_df_[dep_feat].values if dep_feat in shap_df_.columns else shap_vals[:, dep_idx]

    if interact_feat == "(Auto)":
        color_vals = feat_vals
        color_label = dep_feat
    else:
        int_idx = feat_names.index(interact_feat)
        color_vals = X_test_arr[:, int_idx]
        color_label = interact_feat

    dep_plot_df = pd.DataFrame({
        dep_feat:     feat_vals,
        "SHAP Value": dep_shap,
        color_label:  color_vals,
    })
    fig_dep = px.scatter(
        dep_plot_df, x=dep_feat, y="SHAP Value",
        color=color_label, color_continuous_scale="RdBu",
        opacity=0.4, title=f"SHAP Dependence: {dep_feat}",
        labels={"SHAP Value": f"SHAP value for {dep_feat}"},
    )
    fig_dep.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_dep, use_container_width=True)

    # ── A4. Waterfall ─────────────────────────────────────────
    st.subheader("A4 — SHAP Waterfall (Single Prediction)")
    row_idx = st.slider(
        "Select test-set row index", 0, len(shap_vals) - 1, 0, key="wf_idx"
    )
    with st.spinner("Rendering waterfall …"):
        explanation = shap.Explanation(
            values=shap_vals[row_idx],
            base_values=exp_val,
            data=X_test_arr[row_idx],
            feature_names=feat_names,
        )
        fig_wf = plt.figure()
        shap.plots.waterfall(explanation, show=False, max_display=15)
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

    # ── A5. SHAP Force Plot ───────────────────────────────────
    st.subheader("A5 — SHAP Force Plot (Single Prediction)")
    force_row = st.slider(
        "Select row for force plot", 0, len(shap_vals) - 1, 0, key="fp_idx"
    )
    with st.spinner("Computing force plot …"):
        try:
            fp = shap.force_plot(
                exp_val, shap_vals[force_row], X_test_arr[force_row],
                feature_names=feat_names, matplotlib=False,
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{fp.html()}</body>"
            st.components.v1.html(shap_html, height=220, scrolling=True)
        except Exception:
            exp2 = shap.Explanation(
                values=shap_vals[force_row], base_values=exp_val,
                data=X_test_arr[force_row], feature_names=feat_names,
            )
            fig_fb = plt.figure()
            shap.plots.waterfall(exp2, show=False, max_display=15)
            st.pyplot(fig_fb)
            plt.close(fig_fb)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — COVID FEATURE EXPLORER
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.title("COVID Feature Explorer")
    st.markdown(
        "Explore how individual features relate to COVID-19 vaccine adverse event "
        "outcomes. Select any feature below to see its distribution, fatality-rate "
        "breakdown, SHAP impact, and model partial-dependence."
    )

    df       = load_sample()
    shap_df_ = load_shap_df()
    feat_names_list = joblib.load("artifacts/feature_names.pkl")

    # Feature categories for the selector
    NUM_EXP = ["AGE_YRS", "NUMDAYS", "HOSPDAYS"]
    BIN_EXP = [
        "HOSPITAL", "L_THREAT", "ER_ED_VISIT", "DISABLE",
        "RECOVD", "X_STAY", "BIRTH_DEFECT", "OFC_VISIT",
    ]
    CAT_EXP = ["SEX", "VAX_MANU", "VAX_DOSE_SERIES", "VAX_ROUTE", "VAX_SITE", "STATE"]
    SYM_EXP = [c for c in df.columns if c.startswith("SYM_")]

    all_explor = NUM_EXP + CAT_EXP + BIN_EXP + SYM_EXP
    all_explor = [c for c in all_explor if c in df.columns]

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        feat_sel = st.selectbox("Select a feature to explore", all_explor, key="explorer_feat")
        st.markdown("**Feature type:**")
        if feat_sel in NUM_EXP:
            st.info("Numerical (continuous)")
            n_bins = st.slider("Histogram bins", 10, 80, 40, key="exp_bins")
        elif feat_sel in CAT_EXP:
            st.info("Categorical")
        elif feat_sel in BIN_EXP:
            st.info("Binary (Yes/No outcome)")
        else:
            st.info("Binary symptom flag")

    # ── Distribution ──────────────────────────────────────────
    st.subheader(f"Distribution: {feat_sel}")

    if feat_sel in NUM_EXP:
        tmp = df.dropna(subset=[feat_sel]).copy()
        tmp["Outcome"] = tmp["DIED"].map({0: "Survived", 1: "Died"})
        fig_dist = px.histogram(
            tmp, x=feat_sel, color="Outcome", nbins=n_bins,
            color_discrete_map={"Survived": "#4C78A8", "Died": "#E45756"},
            barmode="overlay", opacity=0.7,
            title=f"Distribution of {feat_sel} by Outcome",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Binned death rate
        st.subheader(f"Fatality Rate by {feat_sel} Bucket")
        tmp["Bucket"] = pd.qcut(tmp[feat_sel], q=10, duplicates="drop")
        bucket_stats = (
            tmp.groupby("Bucket", observed=True)["DIED"]
            .agg(rate="mean", n="size").reset_index()
            .assign(
                rate_pct=lambda d: (d["rate"] * 100).round(2),
                bucket_str=lambda d: d["Bucket"].astype(str),
            )
        )
        fig_rate = px.bar(
            bucket_stats, x="bucket_str", y="rate_pct",
            text="rate_pct", color="rate_pct", color_continuous_scale="Reds",
            title=f"Fatality Rate by {feat_sel} Decile",
            labels={"bucket_str": f"{feat_sel} Range", "rate_pct": "Fatality Rate (%)"},
        )
        fig_rate.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_rate, use_container_width=True)

        # Summary stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Mean (Survived)", f"{tmp[tmp['DIED']==0][feat_sel].mean():.1f}")
        col_s2.metric("Mean (Died)",     f"{tmp[tmp['DIED']==1][feat_sel].mean():.1f}")
        col_s3.metric("Median (Survived)", f"{tmp[tmp['DIED']==0][feat_sel].median():.1f}")
        col_s4.metric("Median (Died)",   f"{tmp[tmp['DIED']==1][feat_sel].median():.1f}")

    elif feat_sel in CAT_EXP:
        cat_stats = (
            df.groupby(feat_sel)["DIED"]
            .agg(rate="mean", n="size").reset_index()
            .query("n >= 20")
            .assign(
                rate_pct=lambda d: (d["rate"] * 100).round(2),
                pct_of_total=lambda d: (d["n"] / d["n"].sum() * 100).round(1),
            )
            .sort_values("rate_pct", ascending=False)
        )
        fig_cat = px.bar(
            cat_stats, x=feat_sel, y="rate_pct",
            text="rate_pct", color="rate_pct", color_continuous_scale="Reds",
            title=f"Fatality Rate by {feat_sel}",
            labels={"rate_pct": "Fatality Rate (%)", feat_sel: feat_sel},
            hover_data={"n": True, "pct_of_total": True},
        )
        fig_cat.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_cat, use_container_width=True)

        # Count distribution
        fig_cnt = px.bar(
            cat_stats.sort_values("n", ascending=False),
            x=feat_sel, y="n", color="n",
            color_continuous_scale="Blues",
            title=f"Report Count by {feat_sel}",
            labels={"n": "# Reports"},
        )
        st.plotly_chart(fig_cnt, use_container_width=True)

    else:  # binary
        val_stats = (
            df.groupby(feat_sel)["DIED"]
            .agg(rate="mean", n="size").reset_index()
            .assign(
                rate_pct=lambda d: (d["rate"] * 100).round(2),
                label=lambda d: d[feat_sel].map({0: "No / Not Reported", 1: "Yes"}),
            )
        )
        fig_bin = px.bar(
            val_stats, x="label", y="rate_pct",
            text="rate_pct", color="label",
            color_discrete_map={"No / Not Reported": "#4C78A8", "Yes": "#E45756"},
            title=f"Fatality Rate: {feat_sel}",
            labels={"label": feat_sel, "rate_pct": "Fatality Rate (%)"},
        )
        fig_bin.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_bin, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        yes_rate = float(val_stats[val_stats[feat_sel]==1]["rate_pct"])
        no_rate  = float(val_stats[val_stats[feat_sel]==0]["rate_pct"])
        col_b1.metric(f"Fatality when {feat_sel}=1", f"{yes_rate:.2f}%")
        col_b2.metric(f"Fatality when {feat_sel}=0", f"{no_rate:.2f}%")
        ratio = yes_rate / max(no_rate, 0.001)
        st.info(f"Patients with **{feat_sel}=1** have **{ratio:.1f}x** the fatality rate of those without.")

    # ── SHAP impact for selected feature ─────────────────────
    st.divider()
    st.subheader(f"SHAP Impact: {feat_sel}")

    # Find all SHAP columns that correspond to this raw feature
    shap_feats = [f for f in shap_df_.columns if f == feat_sel or f.startswith(f"cat__{feat_sel}_") or f.startswith(f"{feat_sel}_")]
    if not shap_feats and feat_sel in shap_df_.columns:
        shap_feats = [feat_sel]

    # Fallback: fuzzy search
    if not shap_feats:
        shap_feats = [f for f in shap_df_.columns if feat_sel.lower() in f.lower()]

    if shap_feats:
        # Aggregate SHAP for this feature (sum if multiple OHE columns)
        agg_shap = shap_df_[shap_feats].sum(axis=1)
        mean_shap = agg_shap.mean()
        st.markdown(
            f"Average SHAP contribution of **{feat_sel}**: "
            f"`{mean_shap:+.4f}` "
            f"({'pushes toward death' if mean_shap > 0 else 'pushes toward survival'})"
        )

        if feat_sel in feat_names_list:
            dep_idx = feat_names_list.index(feat_sel)
            X_test_arr_  = pd.read_parquet("artifacts/X_test.parquet").values
            dep_df_ = pd.DataFrame({
                feat_sel:    X_test_arr_[:, dep_idx],
                "SHAP":      agg_shap.values,
            })
            fig_shap_dep = px.scatter(
                dep_df_, x=feat_sel, y="SHAP", opacity=0.4,
                color="SHAP", color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title=f"SHAP Value vs {feat_sel}",
                labels={"SHAP": f"SHAP value"},
            )
            fig_shap_dep.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_shap_dep, use_container_width=True)
    else:
        st.info(f"SHAP values not directly available for {feat_sel} "
                f"(it may be one-hot-encoded under a different column name).")

    # ── Partial Dependence (model-based) ─────────────────────
    st.divider()
    st.subheader(f"Partial Dependence Plot (LightGBM): {feat_sel}")
    st.caption("Holds all other features at their median/mode value and varies the selected feature across its range.")

    X_orig = load_X_test_original()

    if not X_orig.empty and feat_sel in X_orig.columns and feat_sel in NUM_EXP:
        try:
            lgb_pipe_exp = load_lgb_pipeline()
            feat_min = float(X_orig[feat_sel].quantile(0.01))
            feat_max = float(X_orig[feat_sel].quantile(0.99))
            grid     = np.linspace(feat_min, feat_max, 50)

            # Background: median/mode for all features
            bg = X_orig.copy()
            for c in bg.columns:
                if c in NUM_EXP:
                    bg[c] = bg[c].median()
                else:
                    mode_val = bg[c].mode()
                    bg[c] = mode_val.iloc[0] if len(mode_val) > 0 else bg[c].iloc[0]
            bg = bg.iloc[[0]].copy()

            pdp_rows = []
            for gv in grid:
                row = bg.copy()
                row[feat_sel] = gv
                prob = lgb_pipe_exp.predict_proba(row)[0, 1]
                pdp_rows.append({"Feature Value": gv, "Predicted Mortality Prob": prob})

            pdp_df = pd.DataFrame(pdp_rows)
            fig_pdp = px.line(
                pdp_df, x="Feature Value", y="Predicted Mortality Prob",
                markers=True,
                title=f"Partial Dependence: {feat_sel} → Mortality Probability",
                labels={"Feature Value": feat_sel, "Predicted Mortality Prob": "Pred. Probability (Death)"},
            )
            fig_pdp.add_hline(
                y=load_sample()["DIED"].mean(),
                line_dash="dot", line_color="orange",
                annotation_text="Avg death rate",
            )
            st.plotly_chart(fig_pdp, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not compute partial dependence: {exc}")
    elif feat_sel in CAT_EXP and not X_orig.empty and feat_sel in X_orig.columns:
        try:
            lgb_pipe_exp = load_lgb_pipeline()
            unique_vals  = X_orig[feat_sel].dropna().unique().tolist()
            bg = X_orig.copy()
            for c in bg.columns:
                if c in NUM_EXP:
                    bg[c] = bg[c].median()
                else:
                    mode_val = bg[c].mode()
                    bg[c] = mode_val.iloc[0] if len(mode_val) > 0 else bg[c].iloc[0]
            bg = bg.iloc[[0]].copy()

            pdp_rows = []
            for val in unique_vals:
                row = bg.copy()
                row[feat_sel] = str(val)
                prob = lgb_pipe_exp.predict_proba(row)[0, 1]
                pdp_rows.append({"Feature Value": str(val), "Predicted Mortality Prob": prob})

            pdp_df = pd.DataFrame(pdp_rows).sort_values("Predicted Mortality Prob", ascending=False)
            fig_pdp = px.bar(
                pdp_df, x="Feature Value", y="Predicted Mortality Prob",
                color="Predicted Mortality Prob", color_continuous_scale="Reds",
                title=f"Partial Dependence: {feat_sel} → Mortality Probability",
                text="Predicted Mortality Prob",
            )
            fig_pdp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig_pdp, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not compute partial dependence: {exc}")
    elif feat_sel in BIN_EXP + SYM_EXP and not X_orig.empty and feat_sel in X_orig.columns:
        try:
            lgb_pipe_exp = load_lgb_pipeline()
            bg = X_orig.copy()
            for c in bg.columns:
                if c in NUM_EXP:
                    bg[c] = bg[c].median()
                else:
                    mode_val = bg[c].mode()
                    bg[c] = mode_val.iloc[0] if len(mode_val) > 0 else bg[c].iloc[0]
            bg = bg.iloc[[0]].copy()

            pdp_rows = []
            for val in [0, 1]:
                row = bg.copy()
                row[feat_sel] = val
                prob = lgb_pipe_exp.predict_proba(row)[0, 1]
                pdp_rows.append({
                    "Feature Value": "Yes (1)" if val else "No (0)",
                    "Predicted Mortality Prob": prob,
                })

            pdp_df = pd.DataFrame(pdp_rows)
            fig_pdp = px.bar(
                pdp_df, x="Feature Value", y="Predicted Mortality Prob",
                color="Feature Value",
                color_discrete_map={"No (0)": "#4C78A8", "Yes (1)": "#E45756"},
                title=f"Partial Dependence: {feat_sel} → Mortality Probability",
                text="Predicted Mortality Prob",
            )
            fig_pdp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig_pdp, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not compute partial dependence: {exc}")
    else:
        st.info("Partial dependence plot requires `X_test_original.parquet` — re-run train.py.")
