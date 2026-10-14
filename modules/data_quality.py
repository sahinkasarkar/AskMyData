"""
data_quality.py
Calculates an explainable Data Quality Score (0-100) and detects:
- Missing value concentrations
- Duplicate rows
- Numeric outliers via Interquartile Range (IQR)
- Constant / single-value columns
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a transparent 0-100 Data Quality Score with detailed issue logging.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    if total_rows == 0 or total_cols == 0:
        return {
            "score": 0,
            "status": "Poor",
            "issues": ["The dataset contains no data."],
            "metrics": {}
        }

    issues: List[Dict[str, str]] = []
    penalties = 0.0

    # 1. Missing Values Evaluation
    total_missing = int(df.isnull().sum().sum())
    missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0.0
    
    if total_missing > 0:
        # Scale: 1% missing = 1.5 pts deduction, capped at 30 pts
        missing_penalty = min(30.0, missing_pct * 1.5)
        penalties += missing_penalty
        
        # Identify columns with highest missing rate
        high_null_cols = [
            f"{col} ({round((df[col].isnull().sum() / total_rows) * 100, 1)}%)"
            for col in df.columns if df[col].isnull().sum() > 0
        ]
        issues.append({
            "severity": "Medium" if missing_pct < 10 else "High",
            "message": f"Dataset has {total_missing} missing cell(s) ({round(missing_pct, 2)}% of total data). Affected columns: {', '.join(high_null_cols)}."
        })

    # 2. Duplicate Records Evaluation
    duplicate_count = int(df.duplicated().sum())
    dup_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0.0
    
    if duplicate_count > 0:
        # Scale: 1% duplicate = 2 pts deduction, capped at 20 pts
        dup_penalty = min(20.0, dup_pct * 2.0)
        penalties += dup_penalty
        issues.append({
            "severity": "Medium" if dup_pct < 5 else "High",
            "message": f"Found {duplicate_count} duplicate row(s) ({round(dup_pct, 1)}% of rows)."
        })

    # 3. Numeric Outlier Detection via IQR
    outlier_summary = {}
    total_outliers = 0
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) >= 8:  # Only evaluate distributions with sufficient sample size
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                count = int(len(outliers))
                if count > 0:
                    outlier_summary[col] = count
                    total_outliers += count

    if total_outliers > 0:
        # Scale: Outlier presence penalty capped at 15 pts
        outlier_penalty = min(15.0, total_outliers * 3.0)
        penalties += outlier_penalty
        outlier_details = [f"{col}: {cnt} outlier(s)" for col, cnt in outlier_summary.items()]
        issues.append({
            "severity": "Low",
            "message": f"Potential numeric outliers detected using IQR rule: {', '.join(outlier_details)}."
        })

    # 4. Single-Value / Zero-Variance Columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if constant_cols:
        const_penalty = min(15.0, len(constant_cols) * 5.0)
        penalties += const_penalty
        issues.append({
            "severity": "Low",
            "message": f"Column(s) with single constant value detected: {', '.join(constant_cols)}."
        })

    # Final Score Calculation
    final_score = max(0, int(round(100.0 - penalties)))

    if final_score >= 90:
        status = "Excellent"
    elif final_score >= 75:
        status = "Good"
    elif final_score >= 50:
        status = "Fair"
    else:
        status = "Poor"

    return {
        "score": final_score,
        "status": status,
        "penalties_applied": round(penalties, 1),
        "issues": issues,
        "metrics": {
            "total_missing_cells": total_missing,
            "missing_percentage": round(missing_pct, 2),
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(dup_pct, 2),
            "columns_with_outliers": outlier_summary
        }
    }