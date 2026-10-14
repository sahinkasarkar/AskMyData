"""
modules/query_planner.py
Translates natural language questions into deterministic JSON analysis execution plans.
Supports conversational follow-ups by incorporating conversation history and previous query context.
"""
import os
import json
import re

try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _build_heuristic_plan(query: str, schema_info: dict, previous_context: dict | None = None) -> dict:
    """Fallback planner supporting basic conversational entity inheritance."""
    query_lower = query.lower()
    columns = schema_info.get("columns", [])
    num_cols = schema_info.get("numeric_columns", [])
    cat_cols = schema_info.get("categorical_columns", [])
    date_cols = schema_info.get("date_columns", [])

    filters = []

    # Check for pronoun or continuation cues ("that", "it", "its", "there")
    is_followup = any(w in query_lower for w in ["that", "it", "its", "there", "in that", "for that", "further"])
    if is_followup and previous_context and previous_context.get("last_top_entity"):
        filters.append({
            "column": previous_context.get("last_group_column"),
            "operator": "==",
            "value": previous_context.get("last_top_entity")
        })

    # Target metric detection
    target_metric = None
    for col in num_cols:
        if col.lower() in query_lower:
            target_metric = col
            break
    if not target_metric:
        if previous_context and previous_context.get("last_value_column"):
            target_metric = previous_context["last_value_column"]
        else:
            for prio in ["sales", "revenue", "profit", "amount", "price", "quantity"]:
                matched = [c for c in num_cols if prio in c.lower()]
                if matched:
                    target_metric = matched[0]
                    break
    if not target_metric and num_cols:
        target_metric = num_cols[0]

    # Target grouping column detection
    target_group = None
    for col in cat_cols + date_cols:
        if col.lower() in query_lower:
            target_group = col
            break

    if not target_group:
        if any(w in query_lower for w in ["month", "monthly", "date", "trend", "year", "quarter"]):
            if date_cols:
                target_group = date_cols[0]
        if not target_group:
            for prio in ["category", "product", "region", "segment", "city", "country"]:
                # If we filtered on previous group column, pick a different dimension
                if previous_context and prio == str(previous_context.get("last_group_column", "")).lower():
                    continue
                matched = [c for c in cat_cols if prio in c.lower()]
                if matched:
                    target_group = matched[0]
                    break
    if not target_group and cat_cols:
        target_group = cat_cols[0]

    # Operation & limits
    operation = "groupby"
    agg_func = "sum"
    if any(w in query_lower for w in ["average", "avg", "mean"]):
        agg_func = "mean"
    elif any(w in query_lower for w in ["count", "number of", "how many"]):
        agg_func = "count"

    limit = None
    top_match = re.search(r"top\s*(\d+)", query_lower)
    if top_match:
        limit = int(top_match.group(1))
    elif any(w in query_lower for w in ["highest", "best", "leading", "most"]):
        limit = 5

    return {
        "operation": operation,
        "group_column": target_group,
        "value_column": target_metric,
        "agg_func": agg_func,
        "filters": filters,
        "sort_ascending": False,
        "limit": limit
    }


def generate_plan(query: str, schema_info: dict, previous_context: dict | None = None) -> tuple[dict, str | None]:
    """
    Generates a structured execution plan for a user query, factoring in recent conversation context.
    Returns: (plan_dict, error_message_or_None)
    """
    if not query or not query.strip():
        return {}, "Empty query provided."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not HAS_GENAI:
        return _build_heuristic_plan(query, schema_info, previous_context), None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

        context_str = json.dumps(previous_context, indent=2) if previous_context else "None"

        prompt = f"""
You are an expert analytics query planner. Given the user's dataset schema, conversation context, and latest query, generate a strictly valid JSON execution plan.

DATASET SCHEMA:
- All Columns: {schema_info.get('columns', [])}
- Numeric Columns: {schema_info.get('numeric_columns', [])}
- Categorical Columns: {schema_info.get('categorical_columns', [])}
- Date Columns: {schema_info.get('date_columns', [])}

PREVIOUS QUERY CONTEXT:
{context_str}

USER QUERY: "{query}"

JSON FORMAT SPECIFICATION:
{{
  "operation": "groupby" | "aggregate" | "top_n" | "timeseries",
  "group_column": "<exact column name from schema to group by, or null>",
  "value_column": "<exact numeric column name to calculate, or null>",
  "agg_func": "sum" | "mean" | "count" | "min" | "max" | "median",
  "filters": [
    {{
      "column": "<column name>",
      "operator": "==" | "!=" | ">" | "<",
      "value": "<value to filter>"
    }}
  ],
  "sort_ascending": false,
  "limit": <integer limit such as 5, 10, or null>
}}

Instructions:
1. If the user asks a follow-up (e.g., "What about its top category?", "Break that down"), use the PREVIOUS QUERY CONTEXT to filter the dataset to the leading entity of the previous question.
2. Only use exact column names present in the dataset schema.
3. Return ONLY the raw JSON object.
"""
        response = model.generate_content(prompt)
        plan = json.loads(response.text)
        return plan, None
    except Exception:
        fallback = _build_heuristic_plan(query, schema_info, previous_context)
        return fallback, None