"""
test_phase3.py
Verifies dataset profiling against sample_sales.csv.
"""
from modules.data_loader import load_dataset
from modules.profiler import profile_dataset

print("=" * 45)
print("AskMyData: Testing Phase 3 Data Profiler")
print("=" * 45)

df, err = load_dataset("data/sample_sales.csv")
assert err is None, f"Failed to load dataset: {err}"

profile = profile_dataset(df)

print(f"[OK] Total Rows: {profile['total_rows']}")
print(f"[OK] Total Columns: {profile['total_cols']}")
print(f"[OK] Duplicates Detected: {profile['duplicate_rows']}")
print(f"[OK] Numeric Columns: {profile['numeric_columns']}")
print(f"[OK] Categorical Columns: {profile['categorical_columns']}")
print(f"[OK] Date Columns: {profile['date_columns']}")

print("\n--- Numeric Column Summary (Sales) ---")
print(profile["numeric_summary"].get("Sales"))

print("\n--- Categorical Summary (Region) ---")
print(profile["categorical_summary"].get("Region"))

print("\n--- Date Summary (Order_Date) ---")
print(profile["date_summary"].get("Order_Date"))

print("=" * 45)
print("Phase 3 complete and verified!")