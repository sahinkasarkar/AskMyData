"""
modules/recommender.py
Generates contextual, actionable business recommendations from analysis results.
Integrates with Google Gemini API when available, with a rule-based statistical fallback.
"""
import os
import json
import pandas as pd

try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _heuristic_recommendations(df: pd.DataFrame, query: str, result) -> list[str]:
    """Fallback rule engine generating contextual recommendations without API access."""
    recs = []

    if isinstance(result, pd.DataFrame) and not result.empty and len(result.columns) >= 2:
        grp = result.columns[0]
        val = result.columns[1]
        top_name = str(result.iloc[0][grp])
        
        recs.append(f"Double down on <b>{top_name}</b>, which generated the highest {val.lower()} in the current view.")
        
        if len(result) > 1:
            bottom_name = str(result.iloc[-1][grp])
            recs.append(f"Audit operational performance for <b>{bottom_name}</b> to identify whether low {val.lower()} stems from supply gaps or reduced demand.")
        
        recs.append(f"Track quarter-over-quarter retention across <b>{grp}</b> cohorts to maintain margin stability.")
    elif df is not None and not df.empty:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            recs.append(f"Analyze distribution variance across <b>{num_cols[0]}</b> to detect margin leakage.")
        recs.append("Segment customers by order frequency to build targeted retention campaigns.")
        recs.append("Establish a weekly data validation audit to maintain database health.")
    else:
        recs = [
            "Upload your dataset to start generating tailored strategic recommendations.",
            "Confirm that column headers clearly distinguish categories from numeric metrics.",
            "Use filters to segment your data before evaluating high-level KPIs."
        ]

    return recs[:3]


def generate_recommendations(df: pd.DataFrame, query: str, result, quality_score: int = 90) -> list[str]:
    """
    Produces 3 prioritized recommendations based on active data and the latest query.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not HAS_GENAI or df is None or df.empty:
        return _heuristic_recommendations(df, query, result)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        result_summary = ""
        if isinstance(result, pd.DataFrame):
            result_summary = result.head(5).to_string(index=False)
        elif result is not None:
            result_summary = str(result)

        prompt = f"""
You are a senior business data analyst. Given the context below, provide exactly 3 concise, highly actionable, strategic business recommendations.

Query: "{query}"
Dataset Columns: {list(df.columns)}
Latest Result Summary:
{result_summary}
Data Quality Score: {quality_score}/100

Format: Return a JSON array containing exactly 3 strings. Each string should be 1-2 sentences with <b>tags</b> around key entities or numbers.
Example format:
["Focus on <b>North Region</b>...", "Review stock for <b>Laptops</b>...", "Address <b>14 missing values</b>..."]
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        recs = json.loads(text)
        if isinstance(recs, list) and len(recs) >= 3:
            return [str(r) for r in recs[:3]]
        return _heuristic_recommendations(df, query, result)
    except Exception:
        return _heuristic_recommendations(df, query, result)