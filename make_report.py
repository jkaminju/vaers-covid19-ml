"""
Generate a polished PDF report from all Streamlit app screenshots.
Organises shots into sections with title pages and captions.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Image, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import KeepTogether
from PIL import Image as PILImage
import os

OUT  = r"C:\uw\522\data"
DATA = r"C:\uw\522\data\final_caps"

# ── Colours ──────────────────────────────────────────────────────────────────
C_NAVY   = HexColor("#1B2A4A")
C_RED    = HexColor("#C0392B")
C_LGRAY  = HexColor("#F4F6F9")
C_MGRAY  = HexColor("#BDC3C7")
C_DKGRAY = HexColor("#555555")

PAGE_W, PAGE_H = letter          # 8.5 × 11 in
MARGIN = 0.65 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def _s(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=styles[parent], **kw)
    styles.add(s)
    return s

S_COVER_TITLE  = _s("CoverTitle",  fontSize=28, leading=34, textColor=white,
                     alignment=TA_CENTER, spaceAfter=12, fontName="Helvetica-Bold")
S_COVER_SUB    = _s("CoverSub",    fontSize=14, leading=18, textColor=C_LGRAY,
                     alignment=TA_CENTER, spaceAfter=6,  fontName="Helvetica")
S_COVER_META   = _s("CoverMeta",   fontSize=10, leading=14, textColor=C_MGRAY,
                     alignment=TA_CENTER, fontName="Helvetica")
S_SECTION_HDR  = _s("SectionHdr",  fontSize=18, leading=22, textColor=white,
                     alignment=TA_CENTER, fontName="Helvetica-Bold")
S_SECTION_SUB  = _s("SectionSub",  fontSize=11, leading=15, textColor=C_LGRAY,
                     alignment=TA_CENTER, fontName="Helvetica")
S_CAPTION      = _s("Caption",     fontSize=8.5, leading=11, textColor=C_DKGRAY,
                     alignment=TA_CENTER, fontName="Helvetica-Oblique", spaceBefore=3)
S_BODY         = _s("Body",        fontSize=9.5, leading=13, textColor=C_DKGRAY,
                     fontName="Helvetica", spaceAfter=6)

# ── Helpers ───────────────────────────────────────────────────────────────────
def img_path(name):
    return os.path.join(DATA, name)

def fit_image(path, max_w, max_h, caption=""):
    """Return [Image, caption_para] fitting within max dimensions."""
    if not os.path.exists(path):
        return []
    with PILImage.open(path) as im:
        pw, ph = im.size
    ratio = min(max_w / pw, max_h / ph)
    w, h = pw * ratio, ph * ratio
    elems = [Image(path, width=w, height=h)]
    if caption:
        elems.append(Paragraph(caption, S_CAPTION))
    return elems

def section_divider(title, subtitle="", color=C_NAVY):
    """Full-width coloured section banner."""
    data = [[Paragraph(title, S_SECTION_HDR)]]
    if subtitle:
        data.append([Paragraph(subtitle, S_SECTION_SUB)])
    t = Table(data, colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), color),
        ("TOPPADDING",   (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 18),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    return t

def rule():
    return HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MGRAY,
                      spaceAfter=4, spaceBefore=4)

# ── Cover page ────────────────────────────────────────────────────────────────
def cover_page():
    # Navy banner via a Table
    banner_data = [
        [Paragraph("VAERS COVID-19<br/>Adverse Event Analysis", S_COVER_TITLE)],
        [Paragraph("Machine-Learning Pipeline Report", S_COVER_SUB)],
        [Spacer(1, 0.15*inch)],
        [Paragraph("UW MSIS 522 – Data Science Workflow", S_COVER_META)],
        [Paragraph("February 2026", S_COVER_META)],
    ]
    banner = Table(banner_data, colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
    ]))

    summary_rows = [
        ["Dataset",    "VAERS COVID-19 Reports, Dec 2020 – Jun 2024"],
        ["Records",    "~1 million adverse event reports"],
        ["Target",     "Mortality (DIED == Y)"],
        ["Best model", "LightGBM  AUC = 0.9665  CV-F1 = 0.682"],
        ["MLP",        "PyTorch MLP  AUC = 0.9657  F1 = 0.742"],
        ["Runner-up",  "Random Forest  AUC = 0.9535  CV-F1 = 0.653"],
        ["Baseline",   "Logistic Regression  AUC = 0.946"],
        ["Live app",   "https://jkaminju-vaers-covid19.streamlit.app/"],
    ]
    tbl = Table(summary_rows, colWidths=[1.8*inch, CONTENT_W - 1.8*inch])
    tbl.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), C_NAVY),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 0), (1, -1), C_DKGRAY),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [C_LGRAY, white]),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.5, C_MGRAY),
    ]))

    return [
        Spacer(1, 0.3*inch),
        banner,
        Spacer(1, 0.4*inch),
        Paragraph("Project Overview", _s("OvHdr", fontSize=12, fontName="Helvetica-Bold",
                                         textColor=C_NAVY, spaceAfter=8)),
        Paragraph(
            "This report documents a five-tab Streamlit application that ingests VAERS "
            "COVID-19 adverse event data and delivers: descriptive analytics, five "
            "supervised-learning classifiers with cross-validation, SHAP explainability, "
            "and an interactive per-feature explorer. Screenshots of every section of the "
            "live application are included below.",
            S_BODY),
        Spacer(1, 0.2*inch),
        Paragraph("Key Results", _s("KRHdr", fontSize=12, fontName="Helvetica-Bold",
                                    textColor=C_NAVY, spaceAfter=8)),
        tbl,
        PageBreak(),
    ]

# ── Image blocks ──────────────────────────────────────────────────────────────
IH = 6.8 * inch   # max image height (most of the page)
IH_HALF = 3.2 * inch  # half-page height for pairs

def image_page(path, caption="", max_h=IH):
    elems = fit_image(path, CONTENT_W, max_h, caption)
    if not elems:
        return []
    return elems + [PageBreak()]

def image_pair(path1, cap1, path2, cap2, max_h=IH_HALF):
    e1 = fit_image(path1, CONTENT_W, max_h, cap1)
    e2 = fit_image(path2, CONTENT_W, max_h, cap2)
    return (e1 or []) + [Spacer(1, 0.15*inch)] + (e2 or []) + [PageBreak()]

# ── Build story ───────────────────────────────────────────────────────────────
def build_story():
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story += cover_page()

    # ── Tab 1 – Executive Summary ─────────────────────────────────────────────
    story += [
        section_divider("Tab 1 – Executive Summary",
                        "High-level KPIs, model leaderboard, and key findings"),
        Spacer(1, 0.2*inch),
    ]
    story += image_page(img_path("t1_01_top.png"),
                        "Executive Summary – KPI cards (total reports, deaths, fatality rate, best AUC)")
    story += image_page(img_path("t1_02_leaderboard.png"),
                        "Executive Summary – Model leaderboard ranking all 7 classifiers by AUC")
    story += image_page(img_path("t1_03_findings.png"),
                        "Executive Summary – Key findings and clinical insights")

    # ── Tab 2 – Data Visualization ────────────────────────────────────────────
    story += [
        section_divider("Tab 2 – Data Visualization",
                        "Demographic distributions, symptom frequency, correlation heatmap, "
                        "temporal trends, choropleth map, and written interpretations"),
        Spacer(1, 0.2*inch),
    ]
    story += image_page(img_path("t2_01_age_hist.png"),
                        "Age distribution: Died (red) vs Survived (blue) — fatalities "
                        "concentrate in the 70–85 age band")
    story += image_page(img_path("t2_02_age_interp.png"),
                        "Age histogram with written interpretation")
    story += image_page(img_path("t2_03_sex_manu.png"),
                        "Fatality rate by sex (left) and by vaccine manufacturer (right)")
    story += image_page(img_path("t2_04_sex_manu_interp.png"),
                        "Sex and manufacturer charts with written interpretations")
    story += image_page(img_path("t2_05_symptoms.png"),
                        "Top-15 reported symptoms: all reports vs deaths — dyspnoea and fever dominate")
    story += image_page(img_path("t2_06_symptoms_interp.png"),
                        "Symptom frequency chart with written interpretation")
    story += image_page(img_path("t2_07_corr_heatmap.png"),
                        "Pearson correlation heatmap — AGE_YRS strongest positive predictor; "
                        "RECOVD strongly negative")
    story += image_page(img_path("t2_08_corr_interp.png"),
                        "Correlation heatmap with written interpretation")
    story += image_page(img_path("t2_09_dose_onset.png"),
                        "Fatality rate by vaccine dose series (left) and "
                        "onset-to-report lag by outcome (right)")
    story += image_page(img_path("t2_10_dose_interp.png"),
                        "Dose series and onset lag charts with written interpretations")
    story += image_page(img_path("t2_11_timeseries.png"),
                        "Monthly report volume over time — 2021 surge then steady decline")
    story += image_page(img_path("t2_12_timeseries_interp.png"),
                        "Time series with written interpretation")
    story += image_page(img_path("t2_13_choropleth.png"),
                        "US Choropleth — VAERS COVID-19 fatality rate by state")
    story += image_page(img_path("t2_14_choropleth_interp.png"),
                        "Choropleth map with written interpretation")
    story += image_page(img_path("t2_15_age_boxplot.png"),
                        "Age distribution by clinical outcome — older patients across "
                        "every severity category")
    story += image_page(img_path("t2_16_age_boxplot_interp.png"),
                        "Age by clinical outcome with written interpretation")

    # ── Tab 3 – Model Reports ─────────────────────────────────────────────────
    story += [
        section_divider("Tab 3 – Model Reports",
                        "GridSearchCV results, CV metrics, ROC/PR curves, confusion matrices, "
                        "MLP training curves, feature importances, coefficients"),
        Spacer(1, 0.2*inch),
    ]
    story += image_page(img_path("t3_01_gridsearch_top.png"),
                        "GridSearchCV best hyperparameters for CART, Random Forest, and LightGBM")
    story += image_page(img_path("t3_02_cart_heatmap.png"),
                        "CART grid search heatmap — CV-F1 by max_depth × min_samples_leaf")
    story += image_page(img_path("t3_03_lgb_top5.png"),
                        "LightGBM top-5 GridSearchCV configurations ranked by CV-F1")
    story += image_page(img_path("t3_04_cv_table.png"),
                        "5-fold cross-validation results table (AUC, F1, Precision, Recall) — 7 models")
    story += image_page(img_path("t3_05_cv_boxplots.png"),
                        "Cross-validation metric distributions – boxplots by model")
    story += image_page(img_path("t3_06_test_metrics.png"),
                        "Hold-out test set metrics table")
    story += image_page(img_path("t3_07_auc_f1_bars.png"),
                        "Test AUC and F1 bar chart comparison across all 7 models")
    story += image_page(img_path("t3_08_roc_curves.png"),
                        "ROC curves – all models (LightGBM AUC = 0.9665, MLP AUC = 0.9657)")
    story += image_page(img_path("t3_09_pr_curves.png"),
                        "Precision–Recall curves – all models")
    story += image_page(img_path("t3_10_conf_matrices_1.png"),
                        "Confusion matrices – first set of models at threshold 0.5")
    story += image_page(img_path("t3_11_conf_matrices_2.png"),
                        "Confusion matrices – remaining models at threshold 0.5")
    story += image_page(img_path("t3_12_mlp_curves.png"),
                        "MLP training curves – loss and AUC over epochs (early stop at epoch 36)")
    story += image_page(img_path("t3_13_fi_cart.png"),
                        "CART decision-tree feature importances (Gini impurity)")
    story += image_page(img_path("t3_14_fi_rf.png"),
                        "Random Forest feature importances (mean decrease in impurity)")
    story += image_page(img_path("t3_15_fi_lgbm.png"),
                        "LightGBM feature importances (gain)")
    story += image_page(img_path("t3_16_coeff_logistic.png"),
                        "Logistic Regression coefficients (top positive / negative predictors)")
    story += image_page(img_path("t3_17_coeff_ridge_lasso.png"),
                        "Ridge (L2) and LASSO (L1) coefficients side by side")

    # ── Tab 4 – SHAP Analysis & Interactive Prediction ────────────────────────
    story += [
        section_divider("Tab 4 – SHAP Analysis & Interactive Prediction",
                        "LightGBM explainability: beeswarm, mean |SHAP|, dependence, waterfall, "
                        "force plot, written interpretation, and live prediction widget"),
        Spacer(1, 0.2*inch),
    ]
    story += image_page(img_path("t4_01_title_intro.png"),
                        "SHAP tab title and introduction")
    story += image_page(img_path("t4_02_beeswarm.png"),
                        "SHAP beeswarm plot – top features (AGE_YRS dominates; RECOVD strongly negative)")
    story += image_page(img_path("t4_03_beeswarm_bottom.png"),
                        "SHAP beeswarm – lower-ranked features")
    story += image_page(img_path("t4_04_shap_interp.png"),
                        "Written SHAP interpretation – What drives predictions?")
    story += image_page(img_path("t4_05_shap_interp2.png"),
                        "Written SHAP interpretation continued – protective vs risk factors")
    story += image_page(img_path("t4_06_mean_shap_bar.png"),
                        "Mean |SHAP| bar chart – global feature importance ranking")
    story += image_page(img_path("t4_07_dependence_plot.png"),
                        "SHAP dependence plot – selected feature vs. SHAP value colored by interaction")
    story += image_page(img_path("t4_08_waterfall.png"),
                        "Waterfall plot – individual prediction explanation (feature contributions)")
    story += image_page(img_path("t4_09_force_plot.png"),
                        "Force plot – individual prediction breakdown (push toward/away from mortality)")
    story += image_page(img_path("t4_10_interactive_top.png"),
                        "Interactive Prediction widget – title and model selector")
    story += image_page(img_path("t4_11_interactive_inputs.png"),
                        "Interactive Prediction – feature input fields (age, outcomes, symptoms)")
    story += image_page(img_path("t4_12_interactive_result.png"),
                        "Interactive Prediction – mortality probability badge and gauge")
    story += image_page(img_path("t4_13_interactive_waterfall.png"),
                        "Interactive Prediction – SHAP waterfall for custom input")

    # ── Tab 5 – COVID Feature Explorer ────────────────────────────────────────
    story += [
        section_divider("Tab 5 – COVID Feature Explorer",
                        "Per-feature distribution, fatality rate buckets, SHAP scatter, partial dependence"),
        Spacer(1, 0.2*inch),
    ]
    story += image_page(img_path("t5_01_top.png"),
                        "COVID Feature Explorer – feature selector and description")
    story += image_page(img_path("t5_02_hist_fatality.png"),
                        "AGE_YRS distribution histogram and fatality rate by decile")
    story += image_page(img_path("t5_03_stats_shap.png"),
                        "Summary statistics (mean age survived vs died) and SHAP scatter header")
    story += image_page(img_path("t5_04_shap_scatter.png"),
                        "SHAP scatter plot – feature value vs. SHAP contribution to mortality risk")
    story += image_page(img_path("t5_05_pdp.png"),
                        "Partial Dependence Plot – feature vs. predicted mortality probability")

    # Strip trailing PageBreak if present
    while story and isinstance(story[-1], PageBreak):
        story.pop()

    return story


# ── Page template with footer ─────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_DKGRAY)
    canvas.drawString(MARGIN, 0.4 * inch,
                      "VAERS COVID-19 ML Pipeline Report  |  UW MSIS 522  |  2026")
    canvas.drawRightString(PAGE_W - MARGIN, 0.4 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(C_MGRAY)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 0.52 * inch, PAGE_W - MARGIN, 0.52 * inch)
    canvas.restoreState()


def main():
    out_path = os.path.join(OUT, "VAERS_ML_Report.pdf")
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=0.75 * inch,
        title="VAERS COVID-19 ML Pipeline Report",
        author="UW MSIS 522",
    )

    story = build_story()
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"PDF saved: {out_path}")
    print(f"Pages: check the file.")


if __name__ == "__main__":
    main()
