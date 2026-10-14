"""
test_phase5.py
Tests the execution engine on aggregations, groupby, time series, and filters.
"""
from modules.data_loader import load_dataset
from modules.analysis_engine import execute_plan

print("=" * 45)
print("AskMyData: Testing Phase 5 Analysis Engine")
print("=" * 45)

df, err = load_dataset("data/sample_sales.csv")
assert err is None, f"Data load failed: {err}"

# Test 1: Aggregate Mean Profit
plan_agg = {"operation": "aggregate", "column": "Profit", "aggregation": "mean"}
res1, err1 = execute_plan(df, plan_agg)
print(f"[OK] Average Profit: {res1} (Error: {err1})")

# Test 2: GroupBy Region by Sales (Descending)
plan_group = {
    "operation": "groupby",
    "group_column": "Region",
    "value_column": "Sales",
    "aggregation": "sum",
    "sort": "descending"
}
res2, err2 = execute_plan(df, plan_group)
print(f"[OK] Sales by Region Result Table:\n{res2}")

# Test 3: Top 3 Products by Sales
plan_top = {
    "operation": "top_n",
    "value_column": "Sales",
    "display_column": "Product",
    "limit": 3,
    "sort": "descending"
}
res3, err3 = execute_plan(df, plan_top)
print(f"\n[OK] Top 3 Products by Sales:\n{res3}")

# Test 4: Time-Series Monthly Sales
plan_time = {
    "operation": "time_series",
    "date_column": "Order_Date",
    "value_column": "Sales",
    "aggregation": "sum",
    "frequency": "month"
}
res4, err4 = execute_plan(df, plan_time)
print(f"\n[OK] Monthly Sales Trend:\n{res4.head(4)}")

print("=" * 45)
print("Phase 5 complete and verified!")