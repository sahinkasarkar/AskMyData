"""
test_phase7.py
Tests the visualizer module on aggregated and time-series outputs.
"""
from modules.data_loader import load_dataset
from modules.analysis_engine import execute_plan
from modules.visualizer import create_visualization

print("=" * 45)
print("AskMyData: Testing Phase 7 Visualizer")
print("=" * 45)

df, _ = load_dataset("data/sample_sales.csv")

# Test 1: Bar chart on Regional GroupBy
plan_bar = {
    "operation": "groupby",
    "group_column": "Region",
    "value_column": "Sales",
    "aggregation": "sum",
    "sort": "descending"
}
res_bar, _ = execute_plan(df, plan_bar)
fig_bar = create_visualization(res_bar, plan_bar)
assert fig_bar is not None, "Failed to create Bar chart"
print(f"[OK] Bar Chart created successfully: type={fig_bar.data[0].type}")

# Test 2: Line chart on Time Series
plan_line = {
    "operation": "time_series",
    "date_column": "Order_Date",
    "value_column": "Sales",
    "aggregation": "sum",
    "frequency": "month"
}
res_line, _ = execute_plan(df, plan_line)
fig_line = create_visualization(res_line, plan_line)
assert fig_line is not None, "Failed to create Line chart"
print(f"[OK] Line Chart created successfully: type={fig_line.data[0].type}")

# Test 3: Scalar value returns None (as expected)
scalar_res = 12500.50
fig_scalar = create_visualization(scalar_res, {"operation": "aggregate"})
assert fig_scalar is None, "Scalar should not create a chart"
print("[OK] Scalar correctly handled without chart crash")

print("=" * 45)
print("Phase 7 complete and verified!")