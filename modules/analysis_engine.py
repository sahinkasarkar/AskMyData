"""
modules/analysis_engine.py
Deterministically executes generated analytical plans against active Pandas DataFrames.
Strictly respects sort_ascending and limit parameters.
"""
import pandas as pd
import numpy as np


def execute_plan(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame | float | int | None, str | None]:
    """
    Executes a structured analytical plan on the active DataFrame.
    Returns: (result_data, error_string_or_None)
    """
    if df is None or df.empty:
        return None, "Dataset is empty."

    if not plan or not isinstance(plan, dict):
        return None, "Invalid execution plan."

    op = plan.get("operation", "groupby")
    grp = plan.get("group_column")
    val = plan.get("value_column")
    agg_func = plan.get("agg_func", "sum").lower()
    limit = plan.get("limit")
    sort_asc = plan.get("sort_ascending", False)
    cond = plan.get("filter_condition")

    try:
        data = df.copy()

        # 1. Apply preliminary filter if specified
        if cond and isinstance(cond, dict):
            c_col = cond.get("column")
            c_val = cond.get("value")
            if c_col in data.columns and c_val is not None:
                data = data[data[c_col].astype(str).str.lower() == str(c_val).lower()]
                if data.empty:
                    return None, f"No records found where {c_col} matches '{c_val}'."

        # 2. Filter Operation
        if op == "filter":
            return data.head(limit or 25), None

        # 3. Scalar Aggregate Operation
        if op == "aggregate":
            target_col = val or data.select_dtypes(include=[np.number]).columns[0]
            series = pd.to_numeric(data[target_col], errors="coerce").dropna()
            if agg_func == "mean":
                return float(series.mean()), None
            elif agg_func == "count":
                return int(series.count()), None
            elif agg_func == "min":
                return float(series.min()), None
            elif agg_func == "max":
                return float(series.max()), None
            return float(series.sum()), None

        # Resolve columns for grouping operations
        if not grp or grp not in data.columns:
            obj_cols = [c for c in data.columns if data[c].dtype == "object" and "id" not in c.lower()]
            grp = obj_cols[0] if obj_cols else data.columns[0]

        if not val or val not in data.columns:
            num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            val = num_cols[0] if num_cols else data.columns[-1]

        data[val] = pd.to_numeric(data[val], errors="coerce").fillna(0)

        # 4. GroupBy, Top_N, and Bottom_N Operations
        if op in ["groupby", "top_n", "bottom_n"]:
            res = data.groupby(grp)[val].agg(agg_func).reset_index()
            # Strict sort direction enforcement
            res = res.sort_values(by=val, ascending=sort_asc)
            if limit and limit > 0:
                res = res.head(limit)
            return res, None

        # 5. Trend (Time-Series) Operation
        if op == "trend":
            try:
                data["_parsed_date"] = pd.to_datetime(data[grp], errors="coerce")
                data = data.dropna(subset=["_parsed_date"])
                res = data.groupby(pd.Grouper(key="_parsed_date", freq="M"))[val].agg(agg_func).reset_index()
                res["_parsed_date"] = res["_parsed_date"].dt.strftime("%Y-%m")
                res = res.rename(columns={"_parsed_date": grp})
                res = res.sort_values(by=grp, ascending=True)
                return res, None
            except Exception:
                res = data.groupby(grp)[val].agg(agg_func).reset_index().sort_values(by=grp)
                return res, None

        # 6. Correlation Operation
        if op == "correlation":
            num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                c1 = val if val in num_cols else num_cols[0]
                c2 = [c for c in num_cols if c != c1][0]
                corr_val = data[c1].corr(data[c2])
                corr_df = pd.DataFrame({"Metric 1": [c1], "Metric 2": [c2], "Correlation": [round(corr_val, 3)]})
                return corr_df, None

        # Default fallback
        res = data.groupby(grp)[val].sum().reset_index().sort_values(by=val, ascending=sort_asc)
        return res.head(limit or 10), None

    except Exception as e:
        return None, f"Execution failed: {str(e)}"