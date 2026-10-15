"""
modules/recommender.py
Generates data-backed strategic business recommendations
using Gemini AI (gemini-3.8-flash) with contextual fallback generation.
"""
import os
import re
import streamlit as st
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _get_api_key() -> str | None:
    """Safely retrieves API key across local (.env/os) and Streamlit Cloud (secrets)."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return None


def generate_recommendations(df: pd.DataFrame, query: str, result, quality_info: dict) -> list[str]:
    """
    Synthesizes 3 targeted, actionable business recommendations based on
    the current query, executed output, and schema health.
    """
    if df is None or df.empty:
        return [
            "Upload an active dataset to receive automated business recommendations.",
            "Ensure the file contains numeric metrics (e.g. Sales, Revenue, Units) and categorical groups (e.g. Region, Category).",
            "Keep headers on row 1 for instant automated schema profiling."
        ]

    api_key = _get_api_key()

    if not api_key or not HAS_GENAI:
        return _fallback_recommendations(df, query, result, quality_info)

    try:
        genai.configure(api_key=api_key)

        model_name = "gemini-3.8-flash"
        try:
            available = [m.name for m in genai.list_models() if "generateContent" in getattr(m, "supported_generation_methods", [])]
            for cand in ["gemini-3.8-flash", "models/gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
                if cand in available or cand.replace("models/", "") in [m.replace("models/", "") for m in available]:
                    model_name = cand
                    break
        except Exception:
            pass

        model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.3}
        )

        res_preview = "No query executed yet."
        if isinstance(result, pd.DataFrame) and not result.empty:
            res_preview = result.head(5).to_string()
        elif isinstance(result, (int, float)):
            res_preview = f"Calculated Aggregate: {result:,.2f}"

        prompt = f"""
You are an executive business consultant and senior analytics advisor.
Generate EXACTLY 3 high-impact, prioritized business recommendations based on this analysis:

USER QUESTION: "{query}"
ANALYSIS OUTPUT RESULT:
{res_preview}

DATA HEALTH:
- Data Quality Score: {quality_info.get('score', 90)}/100
- Missing Records: {quality_info.get('metrics', {}).get('total_missing_cells', 0)}

RULES:
1. Provide exactly 3 concise, numbered bullet points.
2. Bold the primary entity or metric in each action item (e.g. "Focus marketing on **Electronics**...").
3. Make recommendations practical (e.g. resource reallocation, pricing review, bundle promotions, or data cleaning).
4. Do NOT output preamble, titles, or closing text. Just the 3 numbered bullet points.
"""

        response = model.generate_content(prompt)
        text = response.text.strip()
        lines = [line.strip() for line in text.split("\n") if line.strip() and re.match(r"^\d+[\.\)]", line.strip())]

        cleaned_recs = []
        for l in lines[:3]:
            clean_l = re.sub(r"^\d+[\.\)]\s*", "", l)
            cleaned_recs.append(clean_l)

        if cleaned_recs:
            return cleaned_recs
        return _fallback_recommendations(df, query, result, quality_info)

    except Exception:
        return _fallback_recommendations(df, query, result, quality_info)


def _fallback_recommendations(df: pd.DataFrame, query: str, result, quality_info: dict) -> list[str]:
    """Generates rule-based recommendations when the API is unreachable."""
    recs = []

    if isinstance(result, pd.DataFrame) and not result.empty and len(result.columns) >= 2:
        grp_col = result.columns[0]
        val_col = result.columns[1]
        lead_name = str(result.iloc[0][grp_col])
        recs.append(f"Prioritize promotional budget and stock availability for **{lead_name}**, which commands the largest volume of {val_col.lower()}.")
        if len(result) > 1:
            tail_name = str(result.iloc[-1][grp_col])
            recs.append(f"Audit underperformance in **{tail_name}** to verify whether low {val_col.lower()} is driven by marketing gaps or inventory constraints.")
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            recs.append(f"Establish benchmark monitoring on **{num_cols[0]}** to flag performance drop-offs week over week.")

    cat_cols = [c for c in df.columns if df[c].dtype == "object" and "id" not in c.lower()]
    if cat_cols:
        top_seg = df[cat_cols[0]].mode().iloc[0] if not df[cat_cols[0]].empty else cat_cols[0]
        recs.append(f"Design personalized customer cohort campaigns centered around **{top_seg}** to lift retention and repeat transactions.")

    missing = quality_info.get("metrics", {}).get("total_missing_cells", 0)
    if missing > 0:
        recs.append(f"Sanitize the ingestion pipeline to resolve **{missing} null fields**, preventing downstream reporting variance.")
    else:
        recs.append("Conduct automated quarterly schema audits to maintain current high data integrity standards.")

    return recs[:3]