"""
app.py
AskMyData — Pixel-Perfect, Fully-Responsive Analytics Dashboard
Features:
- Dual exports: Instant CSV data slice & Print-Ready HTML Executive Brief
- Conversational follow-up memory across sequential user questions
- Smart business dimension detection (excludes Order_ID, UUIDs, codes from AI insights)
- On-demand user execution (no auto-analysis on page load)
- Dynamic schema-aware filters, inline dataset cancellation, fluid typography
- Universal bottom copyright footer
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import io

from modules.data_loader import load_dataset
from modules.profiler import profile_dataset
from modules.data_quality import assess_data_quality
from modules.query_planner import generate_plan
from modules.analysis_engine import execute_plan
from modules.visualizer import create_visualization
from modules.recommender import generate_recommendations

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AskMyData | AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# FLUID RESPONSIVE CSS DESIGN SYSTEM (PC & MOBILE OPTIMIZED)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    *, *::before, *::after {
        box-sizing: border-box;
    }

    html, body, [class*="css"], .stMarkdown, p, div, span, label {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 10 !important;
    }

    .block-container {
        padding-top: clamp(1.2rem, 2.5vw, 2.4rem) !important;
        padding-bottom: clamp(1.5rem, 3vw, 3rem) !important;
        padding-left: clamp(0.85rem, 2.2vw, 2.5rem) !important;
        padding-right: clamp(0.85rem, 2.2vw, 2.5rem) !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }

    /* --- SIDEBAR DESIGN --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070D18 0%, #0D1627 60%, #15102A 100%) !important;
        width: clamp(240px, 20vw, 280px) !important;
        border-right: 1px solid #1E293B !important;
    }

    section[data-testid="stSidebar"] div.stButton {
        margin-bottom: 3px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: clamp(7px, 1vh, 10px) 14px !important;
        border-radius: 9px !important;
        font-size: clamp(0.82rem, 0.9vw, 0.9rem) !important;
        border: none !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        transition: all 0.15s ease !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #141F32 !important;
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35) !important;
    }

    /* Sidebar In-Card Dataset Box with Inline Cross */
    .sb-dataset-card-inline {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 11px;
        padding: 9px 11px;
        margin-top: 6px;
        margin-bottom: 8px;
    }

    div[data-testid="stSidebar"] div.sb-inline-cancel-btn > button {
        background: transparent !important;
        border: none !important;
        color: #64748B !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0 !important;
        margin-top: 2px !important;
        height: auto !important;
        width: 100% !important;
        min-height: unset !important;
        box-shadow: none !important;
        justify-content: center !important;
    }
    div[data-testid="stSidebar"] div.sb-inline-cancel-btn > button:hover {
        color: #EF4444 !important;
        background: rgba(239, 68, 68, 0.15) !important;
        border-radius: 5px !important;
    }

    .sb-excel-icon {
        background: #059669;
        color: white;
        font-weight: 800;
        font-size: 0.95rem;
        width: 30px;
        height: 30px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #141F32 !important;
        border: 1px dashed #2E3F5B !important;
        border-radius: 11px !important;
        padding: 10px 8px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 700 !important;
        width: 100% !important;
        padding: 7px 10px !important;
        box-shadow: 0 2px 5px rgba(124, 58, 237, 0.3) !important;
    }

    /* Hide redundant native upload file chip completely */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"],
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] ul {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Filters Expander & Selectboxes */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid #23334E !important;
        background-color: #141F32 !important;
        border-radius: 11px !important;
        margin-top: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details summary span p {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #1A2639 !important;
        border: 1px solid #2B3A55 !important;
        border-radius: 7px !important;
        min-height: 34px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #FFFFFF !important;
        font-size: 0.82rem !important;
    }

    /* --- HERO HEADER SECTION --- */
    .hero-card {
        margin-bottom: clamp(14px, 2vh, 22px);
    }
    .hello-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: clamp(0.78rem, 0.85vw, 0.85rem);
        font-weight: 600;
        color: #475569;
        margin-bottom: 2px;
    }
    .hero-headline {
        font-size: clamp(1.4rem, 2.3vw, 2.1rem);
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        line-height: 1.15;
        margin-bottom: 4px;
    }
    .hero-headline span {
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-tagline {
        font-size: clamp(0.85rem, 1vw, 0.95rem);
        color: #64748B;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .hero-desc {
        font-size: clamp(0.82rem, 0.9vw, 0.92rem);
        color: #475569;
        max-width: 820px;
        line-height: 1.65;
        text-align: justify !important;
        text-justify: inter-word !important;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* --- RESPONSIVE HORIZONTAL OVERVIEW STRIP --- */
    .overview-strip {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: clamp(10px, 1.4vw, 15px);
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: clamp(8px, 1.2vw, 14px);
        margin-bottom: clamp(14px, 2vh, 22px);
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .current-ds-block {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-right: 14px;
        border-right: 1px solid #F1F5F9;
        flex: 1 1 240px;
    }
    .metric-pill-item {
        background: #F8FAFC;
        border: 1px solid #EDF2F7;
        border-radius: 10px;
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1 1 130px;
    }
    .metric-pill-icon {
        width: 30px;
        height: 30px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        flex-shrink: 0;
    }
    .metric-pill-val {
        font-size: clamp(1rem, 1.25vw, 1.2rem);
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-pill-lbl {
        font-size: clamp(0.68rem, 0.72vw, 0.74rem);
        color: #64748B;
        font-weight: 600;
    }

    /* --- ASK YOUR DATA CARD --- */
    .ask-deck-card {
        background: linear-gradient(135deg, #EEF4FF 0%, #F5F7FF 60%, #FAF5FF 100%);
        border: 1px solid #E0EAFF;
        border-radius: 15px;
        padding: clamp(14px, 1.8vw, 22px);
        margin-bottom: clamp(16px, 2.2vh, 24px);
    }
    .ask-deck-header {
        display: flex;
        align-items: center;
        gap: 7px;
        font-size: clamp(1.05rem, 1.3vw, 1.2rem);
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .ask-deck-sub {
        font-size: clamp(0.78rem, 0.85vw, 0.86rem);
        color: #64748B;
        margin-bottom: 12px;
    }

    /* --- LATEST RESULT HEADER & BANNER --- */
    .section-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .section-title {
        font-size: clamp(0.95rem, 1.15vw, 1.05rem);
        font-weight: 800;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .section-link {
        font-size: clamp(0.76rem, 0.82vw, 0.82rem);
        font-weight: 600;
        color: #2563EB;
        cursor: pointer;
    }

    .answer-callout {
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        border-radius: 10px;
        padding: clamp(9px, 1.2vw, 13px) 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: #166534;
        font-size: clamp(0.84rem, 0.95vw, 0.92rem);
        font-weight: 600;
        margin-bottom: 14px;
        line-height: 1.45;
    }

    .idle-callout {
        background: #F8FAFC;
        border: 1px dashed #CBD5E1;
        border-radius: 12px;
        padding: clamp(24px, 3.5vw, 36px);
        text-align: center;
        color: #64748B;
        margin-bottom: 18px;
    }

    /* Content Cards */
    .white-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 13px;
        padding: clamp(12px, 1.4vw, 18px);
        height: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    /* Compact HTML Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: clamp(0.76rem, 0.82vw, 0.84rem);
    }
    .styled-table th {
        font-weight: 700;
        color: #475569;
        text-align: left;
        padding: 7px 9px;
        border-bottom: 1px solid #E2E8F0;
        background: #FAFAFC;
    }
    .styled-table td {
        padding: 7px 9px;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
    }
    .styled-table tr:hover {
        background: #F8FAFC;
    }

    .insight-sub-card {
        border-radius: 9px;
        padding: 10px 12px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 8px;
    }

    .rec-item {
        display: flex;
        align-items: flex-start;
        gap: 9px;
        margin-bottom: 10px;
    }
    .rec-number {
        background: #2563EB;
        color: #FFFFFF;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
        flex-shrink: 0;
        margin-top: 2px;
    }

    .stTextInput>div>div>input {
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        padding: 8px 12px !important;
        font-size: clamp(0.82rem, 0.9vw, 0.88rem) !important;
    }
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: clamp(0.78rem, 0.85vw, 0.86rem) !important;
        padding: 7px 12px !important;
    }

    /* Small download buttons */
    div[data-testid="stDownloadButton"] > button {
        background: #F8FAFC !important;
        color: #2563EB !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 7px !important;
        font-size: 0.72rem !important;
        padding: 3px 8px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #EFF6FF !important;
        border-color: #93C5FD !important;
    }

    .copyright-footer {
        margin-top: clamp(28px, 4vh, 44px);
        padding: 18px 0 10px 0;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: clamp(0.76rem, 0.82vw, 0.84rem);
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    @media (max-width: 768px) {
        .overview-strip {
            flex-direction: column;
            align-items: stretch;
        }
        .current-ds-block {
            border-right: none;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 10px;
            padding-right: 0;
        }
        .hero-desc {
            max-width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TABLE RENDERING HELPER (Pure HTML)
# -----------------------------------------------------------------------------
def render_table(data_obj):
    """Renders clean, styled tabular HTML directly into the DOM."""
    if isinstance(data_obj, pd.DataFrame):
        html_tbl = data_obj.to_html(classes="styled-table", index=False, justify="left")
        st.markdown(html_tbl, unsafe_allow_html=True)
    elif isinstance(data_obj, dict):
        df_dict = pd.DataFrame(data_obj)
        html_tbl = df_dict.to_html(classes="styled-table", index=True, justify="left")
        st.markdown(html_tbl, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SMART BUSINESS DIMENSION DETECTOR
# -----------------------------------------------------------------------------
def get_meaningful_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Filters out technical ID columns, barcodes, or high-cardinality keys.
    Returns prioritized business dimensions (Region, Category, Segment, etc.).
    """
    if df is None or df.empty:
        return []

    excluded_tokens = ["id", "uuid", "guid", "code", "hash", "key", "number", "num", "phone", "email"]
    meaningful = []

    for col in df.columns:
        c_lower = col.lower().strip()
        if any(c_lower.endswith(tok) or c_lower.startswith(tok) or f"_{tok}" in c_lower for tok in excluded_tokens):
            continue

        if df[col].dtype == "object" or df[col].dtype.name == "category":
            n_unique = df[col].nunique(dropna=True)
            if 1 < n_unique <= (len(df) * 0.8) and n_unique <= 60:
                meaningful.append(col)

    if not meaningful:
        meaningful = [c for c in df.columns if df[c].dtype == "object" and "id" not in c.lower()]

    return meaningful


# -----------------------------------------------------------------------------
# DYNAMIC AI INSIGHTS & RECOMMENDATIONS GENERATORS
# -----------------------------------------------------------------------------
def generate_dynamic_insights(df, query, result, quality):
    """
    Synthesizes contextual AI insights computed directly from active dataset,
    the user's asked question, and current data quality scores.
    """
    insights = []

    if df is None or df.empty:
        return [
            {"title": "Awaiting Dataset", "desc": "Upload a CSV/Excel file or restore sample data to see automated insights.", "type": "neutral"}
        ]

    # Insight 1: Analysis Question Specific Finding
    if isinstance(result, pd.DataFrame) and not result.empty and len(result.columns) >= 2:
        grp_col = result.columns[0]
        val_col = result.columns[1]
        top_name = str(result.iloc[0][grp_col])
        top_val = result.iloc[0][val_col]
        total_val = result[val_col].sum() if pd.api.types.is_numeric_dtype(result[val_col]) else 1
        pct = (top_val / total_val * 100) if total_val > 0 else 0
        
        insights.append({
            "title": f"Top Performer in {grp_col}",
            "desc": f"For '{query}': <b>{top_name}</b> leads with {top_val:,.0f} ({pct:.1f}% of total {val_col}).",
            "type": "positive"
        })
    elif isinstance(result, (int, float)):
        insights.append({
            "title": "Calculated Aggregate",
            "desc": f"Executed query yielded an aggregate value of <b>{result:,.2f}</b> across active rows.",
            "type": "positive"
        })
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            lead_metric = num_cols[0]
            avg_val = df[lead_metric].mean()
            insights.append({
                "title": f"Baseline {lead_metric} Average",
                "desc": f"The average {lead_metric} across {len(df):,} records stands at <b>{avg_val:,.1f}</b>.",
                "type": "positive"
            })

    # Insight 2: Segment Variance (Using Business Dimensions only)
    meaningful_dims = get_meaningful_categorical_columns(df)
    if meaningful_dims:
        sample_cat = meaningful_dims[0]
        top_cat_counts = df[sample_cat].value_counts()
        if not top_cat_counts.empty:
            lead_seg = top_cat_counts.index[0]
            seg_pct = (top_cat_counts.iloc[0] / len(df)) * 100
            insights.append({
                "title": f"{sample_cat} Concentration",
                "desc": f"Dominant group <b>{lead_seg}</b> accounts for {seg_pct:.1f}% of entire active transaction volume.",
                "type": "purple"
            })

    # Insight 3: Data Quality & Hygiene
    missing_cells = quality.get("metrics", {}).get("total_missing_cells", 0)
    outlier_count = sum(quality.get("metrics", {}).get("columns_with_outliers", {}).values())
    
    if missing_cells > 0:
        insights.append({
            "title": "Data Hygiene Notice",
            "desc": f"Identified <b>{missing_cells} missing values</b> across the active dataset needing imputation.",
            "type": "warning"
        })
    elif outlier_count > 0:
        insights.append({
            "title": "Outlier Detection",
            "desc": f"Detected <b>{outlier_count} statistical outlier points</b> that may skew average metrics.",
            "type": "warning"
        })
    else:
        insights.append({
            "title": "Data Integrity Score",
            "desc": f"Data quality is rated at <b>{quality.get('score', 95)}/100</b> with zero missing fields in active slice.",
            "type": "positive"
        })

    return insights[:3]


def generate_dynamic_recommendations(df, query, result, quality):
    """
    Synthesizes targeted business action items based on meaningful dimensions,
    active filter slice, and the specific query answered.
    """
    recs = []

    if df is None or df.empty:
        return [
            "Upload your dataset or restore sample data to receive strategic recommendations.",
            "Ensure uploaded CSV/Excel files contain headers in the first row.",
            "Include both metric columns (Sales, Profit) and segment columns (Region, Category)."
        ]

    # Recommendation 1: Response to Query or Top Entities
    if isinstance(result, pd.DataFrame) and not result.empty and len(result.columns) >= 2:
        grp_col = result.columns[0]
        top_name = str(result.iloc[0][grp_col])
        recs.append(f"Focus strategic marketing and resource allocation on <b>{top_name}</b>, which dominates the {grp_col} distribution.")
        if len(result) > 1:
            bottom_name = str(result.iloc[-1][grp_col])
            recs.append(f"Investigate underperformance in <b>{bottom_name}</b> to identify whether low {result.columns[1].lower()} stems from supply gaps or reduced demand.")
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            recs.append(f"Analyze the margin spread between <b>{num_cols[0]}</b> and <b>{num_cols[1]}</b> to locate revenue leakage.")
        else:
            recs.append("Segment metrics across regional or category dimensions to identify primary revenue drivers.")

    # Recommendation 2: Inventory / Campaign Optimization on Business Dimensions
    meaningful_dims = get_meaningful_categorical_columns(df)
    if meaningful_dims:
        recs.append(f"Create prioritized promotional cohorts segmented by <b>{meaningful_dims[0]}</b> to lift repeat conversion rates.")

    # Recommendation 3: Data Quality Action Item
    missing_cells = quality.get("metrics", {}).get("total_missing_cells", 0)
    if missing_cells > 0:
        recs.append(f"Standardize ingestion pipelines to clean <b>{missing_cells} records</b> missing key attributes.")
    else:
        recs.append("Establish automated periodic data quality audits to sustain the high 90+ integrity score.")

    return recs[:3]


# -----------------------------------------------------------------------------
# EXECUTIVE BRIEF HTML REPORT BUILDER
# -----------------------------------------------------------------------------
def build_executive_html_report(file_name, query, answer_headline, result_df, recommendations, quality_score):
    """Generates a standalone, print-ready HTML executive report."""
    table_rows = ""
    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
        headers = "".join([f"<th style='padding:8px 12px; border-bottom:2px solid #CBD5E1; text-align:left; font-size:12px; color:#475569;'>{c}</th>" for c in result_df.columns])
        table_rows += f"<thead><tr style='background:#F8FAFC;'>{headers}</tr></thead><tbody>"
        for _, row in result_df.head(8).iterrows():
            cells = "".join([f"<td style='padding:8px 12px; border-bottom:1px solid #F1F5F9; font-size:12px; color:#1E293B;'>{v}</td>" for v in row.values])
            table_rows += f"<tr>{cells}</tr>"
        table_rows += "</tbody>"

    recs_list = "".join([f"<li style='margin-bottom:8px; font-size:13px; color:#334155; line-height:1.5;'>{r}</li>" for r in recommendations])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Executive Brief - AskMyData</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
            body {{
                font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
                margin: 40px auto;
                max-width: 800px;
                color: #0F172A;
                line-height: 1.5;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #2563EB;
                padding-bottom: 16px;
                margin-bottom: 24px;
            }}
            .brand {{ font-size: 24px; font-weight: 800; color: #2563EB; }}
            .pill {{ background: #EFF6FF; color: #2563EB; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 12px; }}
            .card {{
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 18px;
                margin-bottom: 20px;
            }}
            .headline {{
                background: #F0FDF4;
                border: 1px solid #DCFCE7;
                color: #166534;
                padding: 12px 16px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 20px;
            }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            .footer {{ margin-top: 40px; text-align: center; font-size: 11px; color: #94A3B8; border-top: 1px solid #E2E8F0; padding-top: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <div class="brand">AskMyData</div>
                <div style="font-size:12px; color:#64748B;">Executive Analytics Brief</div>
            </div>
            <div style="text-align:right;">
                <span class="pill">Dataset: {file_name}</span>
                <div style="font-size:11px; color:#94A3B8; margin-top:4px;">Quality Score: {quality_score}/100</div>
            </div>
        </div>

        <div style="margin-bottom:14px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Analysis Query</div>
            <div style="font-size:18px; font-weight:800; color:#0F172A;">"{query}"</div>
        </div>

        <div class="headline">
            {answer_headline}
        </div>

        <div class="card">
            <div style="font-size:14px; font-weight:700; margin-bottom:8px;">Tabular Summary</div>
            <table>
                {table_rows}
            </table>
        </div>

        <div class="card" style="background:#FFFFFF; border:1px solid #E2E8F0;">
            <div style="font-size:14px; font-weight:700; margin-bottom:8px; color:#0F172A;">Strategic AI Recommendations</div>
            <ol style="padding-left:18px; margin:0;">
                {recs_list}
            </ol>
        </div>

        <div class="footer">
            Generated automatically by AskMyData | Confidential Analytics Report
        </div>
    </body>
    </html>
    """
    return html_content.encode("utf-8")


# -----------------------------------------------------------------------------
# PERSISTENT SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "dataset_cleared" not in st.session_state:
    st.session_state.dataset_cleared = False

if "df" not in st.session_state:
    sample_path = os.path.join("data", "sample_sales.csv")
    if os.path.exists(sample_path) and not st.session_state.dataset_cleared:
        sample_df, _ = load_dataset(sample_path)
        st.session_state.df = sample_df
        st.session_state.file_name = "sales_data.xlsx"
    else:
        st.session_state.df = None
        st.session_state.file_name = "No dataset loaded"

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_plan" not in st.session_state:
    st.session_state.last_plan = None

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DATASET CONTROLLER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 18px; padding-left: 2px;">
            <div style="background:#2563EB; width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#FFF; font-weight:800; font-size:1.05rem;">📊</div>
            <div>
                <div style="font-size:1.15rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.5px; line-height:1.1;">AskMyData</div>
                <div style="font-size:0.72rem; color:#94A3B8;">AI-Powered Data Analytics</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    def navigate_to(page_name):
        st.session_state.nav_page = page_name

    nav_items = [
        ("Home", "🏠 Home"),
        ("Analysis", "📊 Analysis"),
        ("AI Insights", "💡 AI Insights"),
        ("Data Quality", "🛡️ Data Quality"),
        ("Data Preview", "📅 Data Preview"),
        ("History", "🕘 History")
    ]

    for page_key, label in nav_items:
        is_active = (st.session_state.nav_page == page_key)
        st.button(
            label,
            key=f"nav_btn_{page_key}",
            type="primary" if is_active else "secondary",
            on_click=navigate_to,
            args=(page_key,),
            use_container_width=True
        )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#64748B; letter-spacing:0.5px; margin-bottom:3px;'>YOUR DATA</div>", unsafe_allow_html=True)

    df_raw = st.session_state.df
    file_name = st.session_state.file_name
    row_count_display = len(df_raw) if df_raw is not None else 0
    col_count_display = len(df_raw.columns) if df_raw is not None else 0

    if df_raw is not None and not df_raw.empty:
        st.markdown('<div class="sb-dataset-card-inline">', unsafe_allow_html=True)
        c_info, c_close = st.columns([8.2, 1.8])
        with c_info:
            st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px;">
                    <div class="sb-excel-icon">X</div>
                    <div style="overflow:hidden; max-width:130px;">
                        <div style="font-weight:700; font-size:0.82rem; color:#FFFFFF; white-space:nowrap; text-overflow:ellipsis;" title="{file_name}">{file_name}</div>
                        <div style="font-size:0.7rem; color:#94A3B8;">{row_count_display:,} rows • {col_count_display} cols</div>
                        <div style="color:#10B981; font-weight:700; font-size:0.7rem; margin-top:1px;">✔ Loaded</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with c_close:
            st.markdown('<div class="sb-inline-cancel-btn">', unsafe_allow_html=True)
            if st.button("✕", key="btn_card_cross_cancel", help="Remove current dataset"):
                st.session_state.df = None
                st.session_state.file_name = "No dataset loaded"
                st.session_state.dataset_cleared = True
                st.session_state.last_result = None
                st.session_state.last_plan = None
                st.session_state.current_query = ""
                if "last_uploaded_name" in st.session_state:
                    del st.session_state["last_uploaded_name"]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No dataset active.")
        sample_path = os.path.join("data", "sample_sales.csv")
        if os.path.exists(sample_path):
            if st.button("↺ Restore Sample Data", key="btn_restore_sample_data", use_container_width=True):
                sample_df, _ = load_dataset(sample_path)
                st.session_state.df = sample_df
                st.session_state.file_name = "sales_data.xlsx"
                st.session_state.dataset_cleared = False
                st.session_state.last_result = None
                st.session_state.current_query = ""
                st.rerun()

    uploaded_file = st.file_uploader(
        "Upload New File",
        type=["csv", "xlsx"],
        key="dataset_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
            with st.spinner("Loading new dataset..."):
                new_df, err = load_dataset(uploaded_file)
                if not err and new_df is not None:
                    st.session_state.df = new_df
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.session_state.dataset_cleared = False
                    st.session_state.last_result = None
                    st.session_state.current_query = ""
                    st.rerun()

    df = df_raw.copy() if df_raw is not None else None
    active_filters_selected = {}

    with st.expander("🔍 Filters", expanded=True):
        if df_raw is not None and not df_raw.empty:
            cat_candidates = get_meaningful_categorical_columns(df_raw)[:3]

            date_candidates = [
                c for c in df_raw.columns 
                if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(df_raw[c])
            ]

            if not cat_candidates and not date_candidates:
                st.caption("No filterable attributes found.")

            for col in cat_candidates:
                raw_options = sorted([str(x) for x in df_raw[col].dropna().unique().tolist()])
                options = ["All"] + raw_options
                sel_val = st.selectbox(
                    f"{col}",
                    options,
                    index=0,
                    key=f"dyn_filter_{col}"
                )
                if sel_val != "All":
                    active_filters_selected[col] = sel_val

            if date_candidates:
                d_col = date_candidates[0]
                date_filter_options = ["All", "Recent Period"]
                sel_date = st.selectbox(
                    f"{d_col}",
                    date_filter_options,
                    index=0,
                    key=f"dyn_filter_date_{d_col}"
                )
                if sel_date != "All":
                    active_filters_selected[d_col] = sel_date
        else:
            st.caption("Upload data to view filters.")

    if df is not None and active_filters_selected:
        for col, chosen_val in active_filters_selected.items():
            if col in df.columns:
                if chosen_val != "Recent Period":
                    df = df[df[col].astype(str) == str(chosen_val)]
                else:
                    df = df.tail(int(len(df) * 0.5))

    st.markdown("""
        <div style="margin-top:24px; padding: 10px; display:flex; align-items:center; gap:8px; font-size:0.76rem; color:#94A3B8;">
            <span style="color:#38BDF8; font-size:1rem;">✨</span>
            <div>Turn your data into<br><b style="color:#FFF;">answers with AI</b></div>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# METADATA & PROFILE COMPUTATION
# -----------------------------------------------------------------------------
profile = profile_dataset(df) if df is not None and not df.empty else {"numeric_columns": [], "categorical_columns": [], "date_columns": []}
quality = assess_data_quality(df) if df is not None and not df.empty else {"score": 92, "status": "Good", "metrics": {"total_missing_cells": 0, "columns_with_outliers": {}}}
schema_info = {
    "columns": list(df.columns) if df is not None else [],
    "numeric_columns": profile.get("numeric_columns", []),
    "categorical_columns": profile.get("categorical_columns", []),
    "date_columns": profile.get("date_columns", [])
}


# -----------------------------------------------------------------------------
# PAGE ROUTING: HOME
# -----------------------------------------------------------------------------
if st.session_state.nav_page == "Home":

    # 1. HERO HEADER WITH JUSTIFIED INTRODUCTORY PARAGRAPH
    st.markdown("""
        <div class="hero-card">
            <div class="hello-pill">👋 Hello!</div>
            <div class="hero-headline">Welcome to <span>AskMyData</span></div>
            <div class="hero-tagline">Ask questions. Understand your data. Get insights.</div>
            <p class="hero-desc">
                AskMyData bridges the gap between raw data and actionable business decisions. Whether you are exploring revenue metrics, isolating regional trends, or evaluating data health, our conversational engine transforms your natural language queries into automated execution plans, interactive visualizations, and data-backed strategic recommendations without requiring a single line of SQL or code.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 2. HORIZONTAL OVERVIEW STRIP
    active_rows = len(df) if df is not None else 0
    active_cols = len(df.columns) if df is not None else 0
    missing_count = quality.get("metrics", {}).get("total_missing_cells", 0)
    score_val = quality.get("score", 92) if df is not None else 0

    st.markdown(f"""
        <div class="overview-strip">
            <div class="current-ds-block">
                <div style="background:#E6F4EA; color:#059669; width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; font-weight:800;">X</div>
                <div style="overflow:hidden;">
                    <div style="font-size:0.7rem; color:#64748B; font-weight:600;">Current Dataset</div>
                    <div style="font-size:0.92rem; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:5px; white-space:nowrap; text-overflow:ellipsis;">
                        {file_name} <span style="background:#E6F4EA; color:#059669; font-size:0.68rem; padding:1px 6px; border-radius:10px; font-weight:700;">{"✔ Loaded" if df is not None else "✕ Empty"}</span>
                    </div>
                    <div style="font-size:0.7rem; color:#94A3B8;">{active_rows:,} rows • {active_cols} cols</div>
                </div>
            </div>
            <div class="metric-pill-item">
                <div class="metric-pill-icon" style="background:#EFF6FF; color:#2563EB;">📅</div>
                <div>
                    <div class="metric-pill-val">{active_rows:,}</div>
                    <div class="metric-pill-lbl">Total Rows</div>
                </div>
            </div>
            <div class="metric-pill-item">
                <div class="metric-pill-icon" style="background:#F3E8FF; color:#7C3AED;">📑</div>
                <div>
                    <div class="metric-pill-val">{active_cols}</div>
                    <div class="metric-pill-lbl">Total Columns</div>
                </div>
            </div>
            <div class="metric-pill-item">
                <div class="metric-pill-icon" style="background:#FEF3C7; color:#D97706;">⚠️</div>
                <div>
                    <div class="metric-pill-val">{missing_count}</div>
                    <div class="metric-pill-lbl">Missing Values</div>
                </div>
            </div>
            <div class="metric-pill-item">
                <div class="metric-pill-icon" style="background:#DCFCE7; color:#16A34A;">🛡️</div>
                <div>
                    <div class="metric-pill-val">{score_val}/100</div>
                    <div class="metric-pill-lbl">Data Quality Score</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. ASK YOUR DATA CARD
    st.markdown("""
        <div class="ask-deck-card">
            <div class="ask-deck-header">
                <span style="color:#2563EB;">✨</span> Ask Your Data
            </div>
            <div class="ask-deck-sub">
                Ask questions in plain English and get instant insights, charts and recommendations.
            </div>
    """, unsafe_allow_html=True)

    in_col, btn_col = st.columns([8.2, 1.8], gap="small")
    with in_col:
        input_query = st.text_input(
            "Query Box",
            value=st.session_state.current_query,
            placeholder="Type a question (e.g. Which region has highest sales?)...",
            label_visibility="collapsed"
        )
    with btn_col:
        btn_ask = st.button("Ask My Data ➔", type="primary", use_container_width=True)

    st.markdown("<div style='font-size:0.78rem; font-weight:700; color:#475569; margin-top:10px; margin-bottom:6px;'>Try asking:</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    chip_clicked = False
    chosen_chip_query = ""

    if c1.button("📈 Which category has highest sales?", key="t1", use_container_width=True):
        chosen_chip_query = "Which category has highest sales?"
        chip_clicked = True
    if c2.button("⭐ Top 5 products?", key="t2", use_container_width=True):
        chosen_chip_query = "What are the top 5 products by revenue?"
        chip_clicked = True
    if c3.button("🏢 Average metric value?", key="t3", use_container_width=True):
        chosen_chip_query = "What is the average profit or quantity?"
        chip_clicked = True
    if c4.button("📈 Show trend analysis", key="t4", use_container_width=True):
        chosen_chip_query = "Show me the general trend breakdown"
        chip_clicked = True

    if chip_clicked:
        st.session_state.current_query = chosen_chip_query
        input_query = chosen_chip_query

    st.markdown("</div>", unsafe_allow_html=True)

    # 4. LATEST ANALYSIS RESULT SECTION
    st.markdown("""
        <div class="section-title-row">
            <div class="section-title">
                <span style="color:#2563EB;">📈</span> Latest Analysis Result
            </div>
            <div class="section-link">View full analysis ➔</div>
        </div>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No dataset active or no records match the active filter criteria. Please upload a file or adjust filters.")
        st.stop()

    user_triggered = (btn_ask and input_query.strip() != "") or chip_clicked

    if user_triggered:
        st.session_state.current_query = input_query

        # Extract conversational follow-up context from the preceding run
        prev_context = None
        if st.session_state.last_plan and st.session_state.last_result is not None:
            last_top = None
            if isinstance(st.session_state.last_result, pd.DataFrame) and not st.session_state.last_result.empty:
                last_top = str(st.session_state.last_result.iloc[0, 0])
            prev_context = {
                "last_query": st.session_state.history[0]["query"] if st.session_state.history else None,
                "last_group_column": st.session_state.last_plan.get("group_column"),
                "last_value_column": st.session_state.last_plan.get("value_column"),
                "last_top_entity": last_top
            }

        with st.spinner("Analyzing dataset..."):
            plan, plan_err = generate_plan(input_query, schema_info, previous_context=prev_context)
            if not plan_err and df is not None:
                res, exec_err = execute_plan(df, plan)
                if not exec_err:
                    st.session_state.last_result = res
                    st.session_state.last_plan = plan
                    st.session_state.history.insert(0, {"query": input_query, "time": "Just now"})
            else:
                num_cols = profile.get("numeric_columns", [])
                cat_cols = get_meaningful_categorical_columns(df)
                sample_metric = num_cols[0] if num_cols else df.columns[-1]
                sample_group = cat_cols[0] if cat_cols else df.columns[0]
                active_plan = {"operation": "groupby", "group_column": sample_group, "value_column": sample_metric}
                active_res, _ = execute_plan(df, active_plan)
                st.session_state.last_result = active_res
                st.session_state.last_plan = active_plan

    if st.session_state.last_result is not None:
        active_res = st.session_state.last_result
        active_plan = st.session_state.last_plan

        if isinstance(active_res, pd.DataFrame) and not active_res.empty:
            first_col = active_res.columns[0]
            val_col = active_res.columns[1] if len(active_res.columns) > 1 else first_col
            top_entity = active_res.iloc[0][first_col]
            top_val = active_res.iloc[0][val_col]
            tot = active_res[val_col].sum() if pd.api.types.is_numeric_dtype(active_res[val_col]) else 1
            pct = (top_val / tot * 100) if tot > 0 else 0
            answer_headline = f"{top_entity} leads with total {val_col} at <b>₹{top_val:,.0f}</b> ({pct:.1f}% of filtered total)."
        elif isinstance(active_res, (int, float)):
            answer_headline = f"Calculated value across active records: <b>₹{active_res:,.2f}</b>."
        else:
            answer_headline = f"Displaying {len(df)} active records matching selected filters."

        st.markdown(f"""
            <div class="answer-callout">
                <span style="font-size:1.05rem;">✔</span>
                <div>{answer_headline}</div>
            </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1], gap="medium")

        with res_col1:
            st.markdown("<div class='white-card'>", unsafe_allow_html=True)
            chart_title = f"{active_plan.get('value_column', 'Metric')} by {active_plan.get('group_column', 'Category')}" if active_plan else "Visual Breakdown"
            st.markdown(f"<div style='font-size:0.88rem; font-weight:700; color:#0F172A; margin-bottom:10px;'>{chart_title}</div>", unsafe_allow_html=True)
            fig = create_visualization(active_res, active_plan)
            if fig:
                fig.update_layout(height=280, margin=dict(l=8, r=8, t=28, b=8))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='white-card'>", unsafe_allow_html=True)
            hdr_left, hdr_csv, hdr_rep = st.columns([5, 2.5, 2.5])
            with hdr_left:
                st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#0F172A; margin-top:2px;'>Result Summary</div>", unsafe_allow_html=True)
            with hdr_csv:
                if isinstance(active_res, pd.DataFrame) and not active_res.empty:
                    csv_data = active_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 CSV",
                        data=csv_data,
                        file_name="query_result.csv",
                        mime="text/csv",
                        key="btn_download_result_csv"
                    )
            with hdr_rep:
                if isinstance(active_res, pd.DataFrame) and not active_res.empty:
                    # Pre-calculate recommendations for report injection
                    report_recs = generate_dynamic_recommendations(df, st.session_state.current_query, active_res, quality)
                    html_report = build_executive_html_report(
                        file_name=file_name,
                        query=st.session_state.current_query,
                        answer_headline=answer_headline,
                        result_df=active_res,
                        recommendations=report_recs,
                        quality_score=quality.get("score", 92)
                    )
                    st.download_button(
                        label="📄 Report",
                        data=html_report,
                        file_name="Executive_Summary_Report.html",
                        mime="text/html",
                        key="btn_download_exec_report"
                    )

            if isinstance(active_res, pd.DataFrame) and len(active_res.columns) >= 2:
                tbl_display = active_res.copy().reset_index(drop=True)
                tbl_display.index = tbl_display.index + 1
                tbl_display = tbl_display.reset_index().rename(columns={"index": "#"})
                val_col = tbl_display.columns[2]
                tot_sum = tbl_display[val_col].sum() if pd.api.types.is_numeric_dtype(tbl_display[val_col]) else 1
                tbl_display["% of Total"] = tbl_display[val_col].apply(lambda v: f"{(v/tot_sum*100):.1f}%" if isinstance(v, (int, float)) else "-")
                tbl_display[val_col] = tbl_display[val_col].apply(lambda v: f"₹{v:,.0f}" if isinstance(v, (int, float)) else str(v))
                render_table(tbl_display.head(5))
            else:
                render_table(df.head(5))
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="idle-callout">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">💡</div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #1E293B; margin-bottom: 4px;">Ready to Explore</div>
                <div style="font-size: 0.84rem;">Type a question above or click one of the suggested prompts to generate instant charts, summaries, and answers.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:clamp(14px, 2vh, 22px);'></div>", unsafe_allow_html=True)

    # 5. DYNAMIC LOWER DECK: AI INSIGHTS & AI RECOMMENDATIONS
    dynamic_insights = generate_dynamic_insights(df, st.session_state.current_query, st.session_state.last_result, quality)
    dynamic_recommendations = generate_dynamic_recommendations(df, st.session_state.current_query, st.session_state.last_result, quality)

    low_col1, low_col2 = st.columns([1, 1], gap="medium")

    with low_col1:
        st.markdown("""
            <div class="white-card">
                <div class="section-title-row">
                    <div style="font-size:0.88rem; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:6px;">
                        <span style="color:#2563EB;">💡</span> AI Insights
                    </div>
                    <div class="section-link">View all insights ➔</div>
                </div>
        """, unsafe_allow_html=True)

        for ins in dynamic_insights:
            if ins["type"] == "positive":
                card_bg = "#F0FDF4"
                card_border = "#DCFCE7"
                badge_bg = "#DCFCE7"
                badge_color = "#16A34A"
                icon = "↗"
            elif ins["type"] == "purple":
                card_bg = "#F5F3FF"
                card_border = "#EDE9FE"
                badge_bg = "#EDE9FE"
                badge_color = "#7C3AED"
                icon = "🏆"
            elif ins["type"] == "warning":
                card_bg = "#FEF3C7"
                card_border = "#FDE68A"
                badge_bg = "#FDE68A"
                badge_color = "#D97706"
                icon = "⚠️"
            else:
                card_bg = "#F8FAFC"
                card_border = "#E2E8F0"
                badge_bg = "#EDF2F7"
                badge_color = "#475569"
                icon = "ℹ️"

            st.markdown(f"""
                <div class="insight-sub-card" style="background:{card_bg}; border:1px solid {card_border};">
                    <div style="background:{badge_bg}; color:{badge_color}; padding:5px; border-radius:6px; font-size:0.85rem;">{icon}</div>
                    <div>
                        <div style="font-weight:700; font-size:0.82rem; color:#0F172A;">{ins['title']}</div>
                        <div style="font-size:0.76rem; color:#475569;">{ins['desc']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with low_col2:
        st.markdown("""
            <div class="white-card">
                <div class="section-title-row">
                    <div style="font-size:0.88rem; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:6px;">
                        <span style="color:#2563EB;">🎯</span> AI Recommendations
                    </div>
                    <div class="section-link">View all recommendations ➔</div>
                </div>
        """, unsafe_allow_html=True)

        for idx, rec in enumerate(dynamic_recommendations, 1):
            st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-number">{idx}</div>
                    <div style="font-size:0.8rem; color:#334155; line-height:1.45;">
                        {rec}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE ROUTING: ANALYSIS
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Analysis":
    st.markdown("""
        <div class="hero-card">
            <div class="hello-pill">📊 Deep Dive</div>
            <div class="hero-headline">Statistical <span>Distributions</span></div>
            <div class="hero-tagline">Detailed variance, cardinality, and column-level properties.</div>
        </div>
    """, unsafe_allow_html=True)

    if df is not None and not df.empty:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("<div class='white-card'><b style='font-size:0.95rem; color:#0F172A;'>Numeric Variables Summary</b>", unsafe_allow_html=True)
            num_desc = df.describe().T
            if not num_desc.empty:
                num_desc = num_desc[["count", "mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
                for col in ["mean", "std", "min", "median", "max"]:
                    num_desc[col] = num_desc[col].apply(lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) else str(v))
                render_table(num_desc.reset_index().rename(columns={"index": "Metric Column"}))
            else:
                st.info("No numeric columns present in the active dataset.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='white-card'><b style='font-size:0.95rem; color:#0F172A;'>Categorical Overview</b>", unsafe_allow_html=True)
            cat_cols = get_meaningful_categorical_columns(df)
            if cat_cols:
                cat_summary = []
                for c in cat_cols:
                    cat_summary.append({
                        "Column": c,
                        "Unique Values": df[c].nunique(),
                        "Top Value": str(df[c].mode().iloc[0]) if not df[c].empty else "-",
                        "Top Freq": int(df[c].value_counts().iloc[0]) if not df[c].empty else 0
                    })
                render_table(pd.DataFrame(cat_summary))
            else:
                st.info("No categorical attributes present.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No dataset active. Upload a CSV/XLSX or restore sample data.")


# -----------------------------------------------------------------------------
# PAGE ROUTING: AI INSIGHTS
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "AI Insights":
    st.markdown("""
        <div class="hero-card">
            <div class="hello-pill">💡 Intelligence</div>
            <div class="hero-headline">AI Strategic <span>Insights</span></div>
            <div class="hero-tagline">Automated deductions derived from schema distributions and data hygiene.</div>
        </div>
    """, unsafe_allow_html=True)

    dynamic_insights = generate_dynamic_insights(df, st.session_state.current_query, st.session_state.last_result, quality)
    dynamic_recommendations = generate_dynamic_recommendations(df, st.session_state.current_query, st.session_state.last_result, quality)

    i1, i2 = st.columns(2, gap="medium")
    with i1:
        st.markdown("""
            <div class='white-card'>
                <div style='font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:12px;'>
                    🧹 Data Hygiene Action Items
                </div>
        """, unsafe_allow_html=True)
        for ins in dynamic_insights:
            st.markdown(f"""
                <div style='margin-bottom: 10px; padding: 10px 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 9px;'>
                    <div style='font-weight: 700; font-size: 0.85rem; color: #0F172A;'>{ins['title']}</div>
                    <div style='font-size: 0.8rem; color: #475569;'>{ins['desc']}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with i2:
        st.markdown("""
            <div class='white-card'>
                <div style='font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:12px;'>
                    📈 Strategic Business Directions
                </div>
        """, unsafe_allow_html=True)
        for idx, rec in enumerate(dynamic_recommendations, 1):
            st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-number">{idx}</div>
                    <div style="font-size:0.84rem; color:#334155; line-height:1.5;">{rec}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE ROUTING: DATA QUALITY
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Data Quality":
    st.markdown("""
        <div class="hero-card">
            <div class="hello-pill">🛡️ Audit</div>
            <div class="hero-headline">Data Quality & <span>Integrity</span></div>
            <div class="hero-tagline">Automated audit checking for missing cells, duplicates, and variance outliers.</div>
        </div>
    """, unsafe_allow_html=True)

    score_val = quality.get("score", 92) if df is not None else 0
    q1, q2, q3 = st.columns(3, gap="medium")
    q1.metric("Missing Values", f"{quality['metrics']['total_missing_cells']} cells")
    q2.metric("Duplicate Rows", f"{quality.get('metrics', {}).get('duplicate_rows', 0)} rows")
    q3.metric("Outliers Detected", f"{sum(quality['metrics']['columns_with_outliers'].values())} points")

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:8px;'>Audit Status: {quality.get('status', 'Good')} ({score_val}/100)</div>", unsafe_allow_html=True)

    if quality.get("issues"):
        for issue in quality["issues"]:
            st.warning(f"[{issue.get('severity', 'Notice')}] {issue.get('message', '')}")
    else:
        st.success("No critical quality defects detected in the active dataset slice.")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE ROUTING: DATA PREVIEW
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "Data Preview":
    st.markdown("""
        <div class="hero-card">
            <div class="hello-pill">📅 Tabular View</div>
            <div class="hero-headline">Raw Dataset <span>Records</span></div>
            <div class="hero-tagline">Inspect schema records, native values, and column definitions.</div>
        </div>
    """, unsafe_allow_html=True)

    if df is not None and not df.empty:
        st.markdown("<div class='white-card' style='overflow-x: auto;'>", unsafe_allow_html=True)
        render_table(df.head(25))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No active dataset loaded.")


# -----------------------------------------------------------------------------
# PAGE ROUTING: HISTORY (GLOBALLY UNIQUE WIDGET KEYS)
# -----------------------------------------------------------------------------
elif st.session_state.nav_page == "History":
    st.subheader("🕘 Session Query History")
    st.caption("Review queries executed during this analytics session.")

    if not st.session_state.history:
        st.info("No queries have been executed yet in this session.")
    else:
        for idx, item in enumerate(st.session_state.history):
            h1, h2 = st.columns([8.2, 1.8], gap="small")
            with h1:
                st.markdown(
                    f"<div class='white-card' style='padding:12px 14px; margin-bottom:8px;'>"
                    f"<b>Question:</b> {item['query']} "
                    f"<span style='float:right; color:#94A3B8; font-size:0.78rem;'>{item['time']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with h2:
                if st.button("Re-run", key=f"hist_btn_{idx}_{hash(item['query'])}", use_container_width=True):
                    st.session_state.current_query = item['query']
                    st.session_state.nav_page = "Home"
                    st.session_state.last_result = None
                    st.rerun()

# -----------------------------------------------------------------------------
# COPYRIGHT FOOTER (UNIVERSAL BOTTOM BAR)
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="copyright-footer">
        © 2026 AskMyData. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)