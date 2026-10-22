"""
modules/query_planner.py
Translates natural language questions into executable analytical plans
using Gemini AI (gemini-3.8-flash) with conversational follow-up memory and robust rule-based fallbacks.
Supports ascending/descending polarity (lowest vs highest).
"""
import os
import json
import re
import streamlit as st

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


def generate_plan(query: str, schema_info: dict, previous_context: dict = None) -> tuple[dict | None, str | None]:
    """
    Generates a structured execution plan (dict) from a user's natural language query.
    Returns: (plan_dict, error_message_or_None)
    """
    if not query or not query.strip():
        return None, "Empty query provided."

    api_key = _get_api_key()

    if not api_key or not HAS_GENAI:
        return _heuristic_planner(query, schema_info, previous_context)

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
            generation_config={"temperature": 0.0}
        )

        context_prompt = ""
        if previous_context:
            context_prompt = f"""
PREVIOUS CONVERSATION CONTEXT:
- Last Query: {previous_context.get('last_query')}
- Last Group/Entity Column: {previous_context.get('last_group_column')}
- Last Metric/Value Column: {previous_context.get('last_value_column')}
- Last Leading Entity: {previous_context.get('last_top_entity')}
If the current user query is a follow-up, inherit context appropriately.
"""

        system_instruction = f"""
You are an expert data analytics query planner.
Translate the user's natural language question into a structured JSON execution plan based on this dataset schema:

DATASET SCHEMA:
- All Columns: {schema_info.get('columns', [])}
- Numeric Metrics: {schema_info.get('numeric_columns', [])}
- Categorical Dimensions: {schema_info.get('categorical_columns', [])}
- Date Columns: {schema_info.get('date_columns', [])}

{context_prompt}

CRITICAL SORTING RULES:
- If query asks for 'lowest', 'least', 'worst', 'minimum', 'bottom': set "sort_ascending": true and operation "bottom_n" or "groupby".
- If query asks for 'highest', 'top', 'best', 'most': set "sort_ascending": false and operation "top_n" or "groupby".

VALID OPERATIONS:
1. "groupby": Aggregate a numeric metric across a categorical dimension.
2. "top_n": Rank top entities by a metric (highest).
3. "bottom_n": Rank lowest entities by a metric (lowest).
4. "filter": Filter records where a dimension matches a value.
5. "aggregate": Compute a single scalar (sum, mean, min, max, count).
6. "trend": Time-series aggregation over a date column.
7. "correlation": Relate two numeric metrics.

STRICT JSON SCHEMA OUTPUT ONLY:
{{
    "operation": "groupby" | "top_n" | "bottom_n" | "filter" | "aggregate" | "trend" | "correlation",
    "group_column": "<exact column name or null>",
    "value_column": "<exact column name or null>",
    "agg_func": "sum" | "mean" | "count" | "min" | "max",
    "limit": <integer or null>,
    "filter_condition": {{"column": "<col>", "value": "<val>"}} or null,
    "sort_ascending": false
}}

Respond with RAW JSON ONLY. No markdown, no backticks, no preamble.
"""

        prompt = f"{system_instruction}\nUser Question: {query}"
        response = model.generate_content(prompt)
        text_resp = response.text.strip()

        if text_resp.startswith("```"):
            text_resp = re.sub(r"^```(?:json)?\s*", "", text_resp)
            text_resp = re.sub(r"\s*```$", "", text_resp)

        plan = json.loads(text_resp.strip())
        return plan, None

    except Exception:
        return _heuristic_planner(query, schema_info, previous_context)


def _heuristic_planner(query: str, schema_info: dict, previous_context: dict = None) -> tuple[dict, None]:
    """Fallback planner mapping user keywords to columns with strict polarity awareness."""
    q_lower = query.lower()
    columns = schema_info.get("columns", [])
    num_cols = schema_info.get("numeric_columns", [])
    cat_cols = schema_info.get("categorical_columns", [])
    date_cols = schema_info.get("date_columns", [])

    matched_num = [c for c in num_cols if c.lower() in q_lower]
    matched_cat = [c for c in cat_cols if c.lower() in q_lower]

    val_col = matched_num[0] if matched_num else (num_cols[0] if num_cols else (columns[-1] if columns else None))
    grp_col = matched_cat[0] if matched_cat else (cat_cols[0] if cat_cols else (columns[0] if columns else None))

    if previous_context:
        if not matched_num and previous_context.get("last_value_column"):
            val_col = previous_context["last_value_column"]
        if not matched_cat and previous_context.get("last_group_column"):
            grp_col = previous_context["last_group_column"]

    is_bottom = any(w in q_lower for w in ["lowest", "least", "bottom", "worst", "minimum", "min"])
    limit_match = re.search(r"\b(top|bottom)\s+(\d+)\b", q_lower)

    if is_bottom:
        lim = int(limit_match.group(2)) if limit_match else None
        return {
            "operation": "bottom_n" if limit_match else "groupby",
            "group_column": grp_col,
            "value_column": val_col,
            "agg_func": "sum",
            "limit": lim,
            "sort_ascending": True
        }, None

    if limit_match or any(w in q_lower for w in ["top", "best", "highest", "most"]):
        lim = int(limit_match.group(2)) if limit_match else 5
        return {
            "operation": "top_n",
            "group_column": grp_col,
            "value_column": val_col,
            "agg_func": "sum",
            "limit": lim,
            "sort_ascending": False
        }, None

    if date_cols and any(w in q_lower for w in ["trend", "over time", "monthly", "daily", "timeline", "by date"]):
        return {
            "operation": "trend",
            "group_column": date_cols[0],
            "value_column": val_col,
            "agg_func": "sum",
            "limit": None,
            "sort_ascending": True
        }, None

    if any(w in q_lower for w in ["average", "avg", "mean"]):
        return {
            "operation": "aggregate",
            "group_column": None,
            "value_column": val_col,
            "agg_func": "mean",
            "limit": None,
            "sort_ascending": False
        }, None

    return {
        "operation": "groupby",
        "group_column": grp_col,
        "value_column": val_col,
        "agg_func": "sum",
        "limit": 10,
        "sort_ascending": False
    }, None