"""
profiler.py
Extracts structural overview, data types, null statistics, 
and mathematical distributions from any Pandas DataFrame.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profiles a DataFrame and returns structured metadata.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    # Classify column types
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    date_cols: List[str] = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Missing value statistics
    null_counts = df.isnull().sum()
    missing_stats = {}
    for col in df.columns:
        missing_count = int(null_counts[col])
        missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        missing_stats[col] = {
            "count": missing_count,
            "percentage": missing_pct
        }

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Numeric summary statistics
    numeric_summary = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if not series.empty:
            numeric_summary[col] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "std": round(float(series.std()), 2) if len(series) > 1 else 0.0
            }

    # Categorical summary statistics
    categorical_summary = {}
    for col in categorical_cols:
        series = df[col].dropna()
        if not series.empty:
            top_val = series.mode().iloc[0] if not series.empty else "N/A"
            top_freq = int((series == top_val).sum()) if not series.empty else 0
            categorical_summary[col] = {
                "unique_values": int(series.nunique()),
                "top_category": str(top_val),
                "top_category_frequency": top_freq
            }

    # Date summary statistics
    date_summary = {}
    for col in date_cols:
        series = df[col].dropna()
        if not series.empty:
            min_date = series.min()
            max_date = series.max()
            date_summary[col] = {
                "min_date": str(min_date.date()) if hasattr(min_date, "date") else str(min_date),
                "max_date": str(max_date.date()) if hasattr(max_date, "date") else str(max_date),
                "days_span": int((max_date - min_date).days) if hasattr(max_date - min_date, "days") else None
            }

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "missing_statistics": missing_stats,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "date_summary": date_summary
    }