"""
modules/chatbot_engine.py
Interactive AI chatbot engine grounded in the uploaded dataset schema, profile, and sample rows.
Uses safe API key retrieval, native string formatting, and gemini-3.8-flash model resolution.
"""
import os
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


def _get_supported_model_name(api_key: str) -> str:
    """
    Selects the active model version supported by your Google AI project.
    """
    candidate_priorities = [
        "gemini-3.8-flash",
        "models/gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]

    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        for cand in candidate_priorities:
            if cand in available_models or cand.replace("models/", "") in [m.replace("models/", "") for m in available_models]:
                return cand
        if available_models:
            return available_models[0]
    except Exception:
        pass

    return "gemini-3.8-flash"


def generate_chat_response(messages: list, df: pd.DataFrame, schema_info: dict, quality_info: dict) -> str:
    """
    Generates a conversational response from Gemini grounded in the loaded dataset context.
    """
    if df is None or df.empty:
        return "No dataset is currently active. Please upload a dataset or restore sample data to start asking questions."

    api_key = _get_api_key()

    # Offline / Heuristic Fallback Engine
    if not api_key or not HAS_GENAI:
        last_user_msg = messages[-1]["content"].lower() if messages else ""
        
        num_cols = schema_info.get("numeric_columns", [])
        matched_num = [c for c in num_cols if c.lower() in last_user_msg]
        
        if "rows" in last_user_msg or "size" in last_user_msg or "records" in last_user_msg:
            return f"The active dataset has **{len(df):,} rows** and **{len(df.columns)} columns**."
        if "columns" in last_user_msg or "features" in last_user_msg:
            return f"The dataset contains the following columns: `{', '.join(df.columns)}`."
        if matched_num and any(w in last_user_msg for w in ["total", "sum"]):
            col = matched_num[0]
            val = pd.to_numeric(df[col], errors="coerce").sum()
            return f"The total sum of **{col}** is **{val:,.2f}**."
        if matched_num and any(w in last_user_msg for w in ["average", "mean", "avg"]):
            col = matched_num[0]
            val = pd.to_numeric(df[col], errors="coerce").mean()
            return f"The average of **{col}** is **{val:,.2f}**."

        return (
            f"I am analyzing the dataset with columns: `{', '.join(df.columns[:8])}`. "
            "To unlock full conversational AI reasoning and complex interpretations, ensure your `GEMINI_API_KEY` is configured."
        )

    try:
        genai.configure(api_key=api_key)

        # Context generation using native pandas methods (no tabulate library required)
        dataset_sample = df.head(8).to_string(index=False)
        describe_sample = (
            df.describe().T[["count", "mean", "min", "max"]].to_string()
            if not df.describe().empty
            else "None"
        )

        cat_breakdown = ""
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]
        for c in cat_cols[:3]:
            top_vals = df[c].value_counts().head(5).to_dict()
            cat_breakdown += f"\n- Top {c} distribution: {top_vals}"

        system_context = f"""
You are an expert AI Data Analyst Assistant inside the 'AskMyData' platform.
The user is asking you direct business, analytical, or strategic questions about their active dataset.

DATASET CONTEXT:
- Columns: {list(df.columns)}
- Total Rows: {len(df):,}
- Total Columns: {len(df.columns)}
- Numeric Columns: {schema_info.get('numeric_columns', [])}
- Categorical Dimensions: {schema_info.get('categorical_columns', [])}
- Date Columns: {schema_info.get('date_columns', [])}
- Data Quality Score: {quality_info.get('score', 90)}/100
{cat_breakdown}

STATISTICAL SUMMARY:
{describe_sample}

SAMPLE DATA ROWS:
{dataset_sample}

INSTRUCTIONS:
1. Answer the user's questions clearly, accurately, and strategically based directly on the provided dataset context.
2. If asked "How to improve sales", "Why are profits low", or similar business strategy questions:
   - Cite specific categories, products, or regions present in the data (e.g., Electronics, Furniture, etc.).
   - Provide 3-4 concrete, numbered tactical steps (such as bundling, targeted promotions, margin adjustments, inventory rebalancing).
3. Use clean markdown formatting (**bolding**, bullet points) for readability.
"""

        conversation_history = f"{system_context}\n\nCONVERSATION HISTORY:\n"
        for msg in messages[:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['content']}\n"
        
        conversation_history += f"User: {messages[-1]['content']}\nAssistant:"

        active_model_name = _get_supported_model_name(api_key)
        model = genai.GenerativeModel(active_model_name)
        response = model.generate_content(conversation_history)

        if response and response.text:
            return response.text.strip()
        return "I could not generate a response. Please rephrase your question."

    except Exception as e:
        return f"Unable to generate response: {str(e)}"