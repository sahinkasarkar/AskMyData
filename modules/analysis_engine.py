"""
modules/analysis_engine.py
Executes structured JSON query plans deterministically on Pandas DataFrames.
Features automatic type coercion, filter application, temporal binning, and error handling.
"""
import pandas as pd
import numpy as np


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Coerces strings with currency symbols or commas into clean numeric floats."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    
    cleaned = (
        series.astype(str)
        .str.replace(r"[₹$,\s]", "", regex=True)
        .str.replace(r"[^\d.-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def execute_plan(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame | float | None, str | None]:
    """
    Executes a structured query plan on a DataFrame.
    Returns: (result_data, error_message_or_None)
    """
    if df is None or df.empty:
        return None, "Dataset is empty or not loaded."
    if not plan:
        return None, "Invalid or empty execution plan."

    try:
        working_df = df.copy()
        operation = plan.get("operation", "groupby")
        group_col = plan.get("group_column")
        val_col = plan.get("value_column")
        agg_func = plan.get("agg_func", "sum")
        sort_asc = plan.get("sort_ascending", False)
        limit = plan.get("limit")
        plan_filters = plan.get("filters", [])

        # 0. Apply query plan level filters (e.g. from conversational context)
        if plan_filters and isinstance(plan_filters, list):
            for f in plan_filters:
                f_col = f.get("column")
                f_val = f.get("value")
                f_op = f.get("operator", "==")
                if f_col and f_col in working_df.columns and f_val is not None:
                    if f_op == "==":
                        working_df = working_df[working_df[f_col].astype(str) == str(f_val)]
                    elif f_op == "!=":
                        working_df = working_df[working_df[f_col].astype(str) != str(f_val)]

        # Fallback column selection if planner missed matching
        if not val_col or val_col not in working_df.columns:
            num_cols = [c for c in working_df.columns if pd.api.types.is_numeric_dtype(working_df[c])]
            val_col = num_cols[0] if num_cols else working_df.columns[-1]

        # Clean the value column to guarantee valid numeric arithmetic
        working_df[val_col] = _clean_numeric_series(working_df[val_col])

        # 1. SCALAR AGGREGATE
        if operation == "aggregate" or not group_col:
            val_series = working_df[val_col].dropna()
            if agg_func == "mean":
                result = float(val_series.mean())
            elif agg_func == "count":
                result = float(val_series.count())
            elif agg_func == "min":
                result = float(val_series.min())
            elif agg_func == "max":
                result = float(val_series.max())
            elif agg_func == "median":
                result = float(val_series.median())
            else:
                result = float(val_series.sum())
            return result, None

        # 2. GROUP BY / TOP-N / TIME-SERIES
        if group_col not in working_df.columns:
            available = [c for c in working_df.columns if c != val_col]
            group_col = available[0] if available else working_df.columns[0]

        # If temporal column grouping, format clean date representations
        if "date" in group_col.lower() or pd.api.types.is_datetime64_any_dtype(working_df[group_col]):
            try:
                working_df[group_col] = pd.to_datetime(working_df[group_col], errors="coerce")
                working_df = working_df.dropna(subset=[group_col])
                working_df["_temporal_key"] = working_df[group_col].dt.strftime("%Y-%m")
                group_col = "_temporal_key"
            except Exception:
                pass

        # Perform aggregation
        grouped = working_df.groupby(group_col, as_index=False)[val_col]

        if agg_func == "mean":
            res_df = grouped.mean()
        elif agg_func == "count":
            res_df = grouped.count()
        elif agg_func == "min":
            res_df = grouped.min()
        elif agg_func == "max":
            res_df = grouped.max()
        elif agg_func == "median":
            res_df = grouped.median()
        else:
            res_df = grouped.sum()

        # Sort results
        res_df = res_df.sort_values(by=val_col, ascending=sort_asc)

        # Apply Top-N / Limit truncation
        if limit and isinstance(limit, int) and limit > 0:
            res_df = res_df.head(limit)

        if group_col == "_temporal_key":
            res_df = res_df.rename(columns={"_temporal_key": "Period"})

        res_df = res_df.reset_index(drop=True)
        return res_df, None

    except Exception as e:
        return None, f"Execution failed: {str(e)}"