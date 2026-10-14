"""
test_phase6.py
Verifies natural language query translation and end-to-end execution.
"""
from modules.data_loader import load_dataset
from modules.profiler import profile_dataset
from modules.query_planner import generate_plan
from modules.analysis_engine import execute_plan

print("=" * 45)
print("AskMyData: Testing Phase 6 Query Planner")
print("=" * 45)

df, err = load_dataset("data/sample_sales.csv")
assert err is None, f"Data load failed: {err}"

profile = profile_dataset(df)
schema_info = {
    "columns": list(df.columns),
    "numeric_columns": profile["numeric_columns"],
    "categorical_columns": profile["categorical_columns"],
    "date_columns": profile["date_columns"]
}

# Test Question 1: Simple GroupBy
q1 = "What is the total sales for each region?"
print(f"\nUser Question: '{q1}'")
plan1, err_plan1 = generate_plan(q1, schema_info)
if err_plan1:
    print(f"[FAIL] Plan 1 Error: {err_plan1}")
else:
    print(f"[OK] Generated Plan: {plan1}")
    res1, err_res1 = execute_plan(df, plan1)
    print(f"Execution Output:\n{res1}")

# Test Question 2: Top N
q2 = "Who are the top 3 customers by sales?"
print(f"\nUser Question: '{q2}'")
plan2, err_plan2 = generate_plan(q2, schema_info)
if err_plan2:
    print(f"[FAIL] Plan 2 Error: {err_plan2}")
else:
    print(f"[OK] Generated Plan: {plan2}")
    res2, err_res2 = execute_plan(df, plan2)
    print(f"Execution Output:\n{res2}")

print("=" * 45)
print("Phase 6 complete and verified!")