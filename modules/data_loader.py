"""
modules/data_loader.py
Dataset loader supporting CSV and Excel formats with caching and error checking.
"""
import streamlit as st
import pandas as pd
import io

@st.cache_data(show_spinner=False)
def _parse_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Cached internal parser that converts raw file bytes into a DataFrame."""
    buffer = io.BytesIO(file_bytes)
    if filename.endswith(".csv"):
        return pd.read_csv(buffer)
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported format. Please upload a .csv or .xlsx file.")


def load_dataset(file_input) -> tuple[pd.DataFrame | None, str | None]:
    """
    Loads dataset from either a local file path string or an uploaded file buffer.
    Returns: (DataFrame, error_message_or_None)
    """
    if file_input is None:
        return None, "No file provided."

    try:
        if isinstance(file_input, str):
            # Local file path
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
            df = _parse_file(raw_bytes, file_input)
        else:
            # Streamlit UploadedFile object
            raw_bytes = file_input.getvalue()
            df = _parse_file(raw_bytes, file_input.name)

        if df.empty:
            return None, "Uploaded file contains no records."

        # Strip whitespace from column names
        df.columns = df.columns.astype(str).str.strip()
        return df, None

    except Exception as e:
        return None, f"Failed to load dataset: {str(e)}"