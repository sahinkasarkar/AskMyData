"""
test_phase4.py
Verifies Data Quality Scoring against sample_sales.csv.
"""
from modules.data_loader import load_dataset
from modules.data_quality import assess_data_quality

print("=" * 45)
print("AskMyData: Testing Phase 4 Data Quality")
print("=" * 45)

df, err = load_dataset("data/sample_sales.csv")
assert err is None, f"Failed to load dataset: {err}"

quality = assess_data_quality(df)

print(f"Data Quality Score: {quality['score']}/100 ({quality['status']})")
print(f"Total Penalties: -{quality['penalties_applied']} points")
print("\nDetected Quality Issues:")
for idx, issue in enumerate(quality["issues"], 1):
    print(f"  {idx}. [{issue['severity']}] {issue['message']}")

print("\nQuality Metrics:")
print(f"  - Missing cells: {quality['metrics']['total_missing_cells']} ({quality['metrics']['missing_percentage']}%)")
print(f"  - Duplicate rows: {quality['metrics']['duplicate_rows']} ({quality['metrics']['duplicate_percentage']}%)")
print(f"  - Outliers: {quality['metrics']['columns_with_outliers']}")

print("=" * 45)
print("Phase 4 complete and verified!")