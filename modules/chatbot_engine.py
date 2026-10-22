"""
modules/chatbot_engine.py
Interactive AI chatbot engine grounded in the active dataset schema, profile, and sample records.
Features:
- Dynamic model discovery via genai.list_models() to prevent 404 version mismatches
- Instant key diagnostic probe (`check key`) to verify active secrets on Streamlit Cloud
- Intelligent fallback analysis when all models are quota-limited
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
        return key.strip().strip('"').strip("'")

    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("'")
    except Exception:
        pass

    return None


@st.cache_data(ttl=1800, show_spinner=False)
def _get_usable_models(api_key: str) -> list[str]:
    """
    Queries Google API to discover exact model names supported for generateContent.
    Caches the result for 30 minutes to eliminate latency.
    """
    usable = []
    try:
        genai.configure(api_key=api_key)
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                # Clean 'models/' prefix to avoid 404 endpoint mapping errors
                clean_name = m.name.replace("models/", "")
                usable.append(clean_name)
    except Exception:
        pass

    # Preferred priority order if available
    preferred_order = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash"
    ]
    sorted_models = [m for m in preferred_order if m in usable]
    for m in usable:
        if m not in sorted_models:
            sorted_models.append(m)

    return sorted_models if sorted_models else ["gemini-2.0-flash", "gemini-2.5-flash"]


def _build_offline_analysis(df: pd.DataFrame, schema_info: dict, quality_info: dict, last_user_msg: str) -> str:
    """
    Constructs an immediate, data-backed analytical response using meaningful
    business dimensions (filtering out Order_ID, UUIDs) and data quality metrics.
    """
    excluded = ["id", "code", "key", "number", "num", "phone", "email"]
    meaningful_cats = [
        c for c in df.columns 
        if (df[c].dtype == "object" or df[c].dtype.name == "category")
        and not any(tok in c.lower() for tok in excluded)
    ]
    
    num_cols = schema_info.get("numeric_columns", [])
    q_lower = last_user_msg.lower()

    if any(k in q_lower for k in ["anomal", "outlier", "missing", "defect", "quality"]):
        missing = quality_info.get("metrics", {}).get("total_missing_cells", 0)
        outliers = sum(quality_info.get("metrics", {}).get("columns_with_outliers", {}).values())
        score = quality_info.get("score", 95)
        
        return (
            f"🛡️ **Automated Data Quality Audit**\n\n"
            f"- **Data Integrity Score**: **{score}/100**\n"
            f"- **Missing Cells**: **{missing}** across {len(df):,} records.\n"
            f"- **Statistical Outliers**: **{outliers}** points outside standard variance intervals.\n"
            f"- **Sanitization Advice**: Audit flagged numeric fields for negative values or unnormalized scale spikes."
        )

    lead_cat = meaningful_cats[0] if meaningful_cats else df.columns[0]
    lead_metric = num_cols[0] if num_cols else df.columns[-1]

    for c in meaningful_cats:
        if c.lower() in q_lower:
            lead_cat = c
            break

    grouped = df.groupby(lead_cat)[lead_metric].sum().sort_values(ascending=False)
    top_name = grouped.index[0]
    top_val = grouped.iloc[0]
    bottom_name = grouped.index[-1]
    bottom_val = grouped.iloc[-1]
    total_val = grouped.sum()
    top_share = (top_val / total_val * 100) if total_val > 0 else 0

    return (
        f"📊 **Data-Driven Strategy Breakdown (Local Recovery Mode)**\n\n"
        f"Analysis based on **{lead_cat}** against **{lead_metric}**:\n\n"
        f"1. **Core Growth Engine ({top_name})**: Generates **₹{top_val:,.0f}** ({top_share:.1f}% of total `{lead_metric}`). Maintain high stock availability and priority marketing.\n"
        f"2. **Lagging Segment ({bottom_name})**: Produces **₹{bottom_val:,.0f}**. Review pricing competitiveness, demand softness, or catalog visibility.\n"
        f"3. **Cross-Category Bundling**: Combine high-volume `{top_name}` SKUs with underperforming segments to lift basket size.\n"
        f"4. **Re-engagement Campaign**: Run targeted promotions on secondary segments to diversify revenue."
    )


def generate_chat_response(messages: list, df: pd.DataFrame, schema_info: dict, quality_info: dict) -> str:
    """
    Generates a conversational response from Gemini grounded in the loaded dataset.
    """
    if df is None or df.empty:
        return "No dataset is currently active. Please upload a dataset or restore sample data to start asking questions."

    api_key = _get_api_key()
    last_user_msg = messages[-1]["content"] if messages else ""

    # Instant diagnostic tool
    if last_user_msg.strip().lower() == "check key":
        if not api_key:
            return "❌ **Diagnostic Result**: No `GEMINI_API_KEY` was found in `os.environ` or `st.secrets`."
        models_available = _get_usable_models(api_key)
        return (
            f"🔑 **Diagnostic Result**: Active key detected with prefix `{api_key[:8]}` and suffix `{api_key[-4:]}`.\n\n"
            f"**Supported Models Found**: `{', '.join(models_available[:5])}`"
        )

    if not api_key or not HAS_GENAI:
        return _build_offline_analysis(df, schema_info, quality_info, last_user_msg)

    try:
        genai.configure(api_key=api_key)

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
1. Answer clearly, accurately, and strategically based directly on the provided dataset context.
2. If asked about anomalies or defects, reference the Data Quality Score and variance distributions.
3. If asked how to improve sales, cite actual categories or products from the data and give 3-4 actionable steps.
4. Use clean markdown formatting (**bolding**, bullet points).
"""

        conversation_history = f"{system_context}\n\nCONVERSATION HISTORY:\n"
        for msg in messages[:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['content']}\n"

        conversation_history += f"User: {last_user_msg}\nAssistant:"

        # Get the actual model strings enabled on this account
        candidate_models = _get_usable_models(api_key)

        last_err = None
        for m_name in candidate_models:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(conversation_history)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_err = e
                continue

        # If all discovered models are rate limited or fail
        if last_err is not None:
            err_str = str(last_err)
            if "429" in err_str or "quota" in err_str.lower():
                return _build_offline_analysis(df, schema_info, quality_info, last_user_msg)
            return f"⚠️ **Gemini API Error Detail**: `{err_str}`\n\n*Models queried:* `{candidate_models[:3]}`"

        return _build_offline_analysis(df, schema_info, quality_info, last_user_msg)

    except Exception as e:
        return f"⚠️ **Chatbot Execution Error**: `{str(e)}`"