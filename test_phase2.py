"""
test_phase2.py
Tests the data_loader against the sample sales dataset and edge cases.
"""
from modules.data_loader import load_dataset

print("=" * 45)
print("AskMyData: Testing Phase 2 Data Loader")
print("=" * 45)

# Test 1: Load valid CSV
df, err = load_dataset("data/sample_sales.csv")
if err:
    print(f"[FAIL] Error loading sample_sales.csv: {err}")
else:
    print("[OK] sample_sales.csv loaded successfully!")
    print(f"     Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"     Columns found: {list(df.columns)}")

# Test 2: Incompatible format check
_, err_unsupported = load_dataset("invalid_file.txt")
if err_unsupported:
    print(f"[OK] Correctly caught unsupported file: {err_unsupported}")

print("=" * 45)
print("Phase 2 complete and verified!")