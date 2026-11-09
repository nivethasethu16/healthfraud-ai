import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_FILE = Path("data/synthetic_claims.csv")

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\n====================================")
print(" HEALTHCARE CLAIMS DATA VALIDATION")
print("====================================")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print(f"\nDataset loaded successfully.")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# --------------------------------------------------
# 3. REQUIRED COLUMN CHECK
# --------------------------------------------------

required_columns = [
    "claim_id",
    "provider_id",
    "patient_age",
    "procedure_code",
    "diagnosis_code",
    "claim_amount",
    "procedure_complexity",
    "length_of_stay",
    "provider_claim_count",
    "provider_avg_amount",
    "repeat_billing_count",
    "clinical_note",
    "billed_description",
    "scenario_type",
    "review_label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

print("\n--- COLUMN VALIDATION ---")

if missing_columns:
    print("FAILED: Missing columns:", missing_columns)
else:
    print("PASSED: All required columns exist.")

# Stop if essential columns are missing
if missing_columns:
    raise ValueError("Dataset structure is incomplete.")

# --------------------------------------------------
# 4. MISSING VALUE CHECK
# --------------------------------------------------

print("\n--- MISSING VALUE CHECK ---")

missing_values = df.isnull().sum()

if missing_values.sum() == 0:
    print("PASSED: No missing values found.")
else:
    print("WARNING: Missing values detected.")
    print(missing_values[missing_values > 0])

# Check empty text fields too
text_columns = [
    "claim_id",
    "provider_id",
    "procedure_code",
    "diagnosis_code",
    "clinical_note",
    "billed_description",
    "scenario_type"
]

empty_text = (
    df[text_columns]
    .fillna("")
    .astype(str)
    .apply(lambda col: col.str.strip().eq(""))
)

print("\nEmpty text values:", empty_text.sum().sum())

# --------------------------------------------------
# 5. DUPLICATE CHECK
# --------------------------------------------------

print("\n--- DUPLICATE CHECK ---")

duplicate_ids = df["claim_id"].duplicated().sum()

if duplicate_ids == 0:
    print("PASSED: No duplicate claim IDs.")
else:
    print(f"FAILED: {duplicate_ids} duplicate claim IDs.")

# --------------------------------------------------
# 6. NUMERICAL VALIDATION
# --------------------------------------------------

print("\n--- NUMERICAL VALIDATION ---")

checks = {
    "Patient age (1-99)": (
        df["patient_age"].between(1, 99).all()
    ),

    "Claim amount positive": (
        (df["claim_amount"] > 0).all()
    ),

    "Procedure complexity (1-5)": (
        df["procedure_complexity"].between(1, 5).all()
    ),

    "Length of stay non-negative": (
        (df["length_of_stay"] >= 0).all()
    ),

    "Provider claim count positive": (
        (df["provider_claim_count"] > 0).all()
    ),

    "Provider average amount positive": (
        (df["provider_avg_amount"] > 0).all()
    ),

    "Repeat billing count non-negative": (
        (df["repeat_billing_count"] >= 0).all()
    )
}

for check, passed in checks.items():
    print(
        f"{'PASSED' if passed else 'FAILED'}: {check}"
    )

# --------------------------------------------------
# 7. CATEGORICAL VALIDATION
# --------------------------------------------------

print("\n--- CATEGORICAL VALIDATION ---")

valid_procedures = [
    "consultation",
    "diagnostics",
    "emergency",
    "surgery",
    "intensive_care"
]

valid_scenarios = [
    "normal",
    "legitimate_high_cost",
    "unusual_billing",
    "repeat_billing",
    "documentation_mismatch"
]

valid_diagnoses = [
    "general",
    "infection",
    "respiratory",
    "cardiac",
    "injury"
]

print(
    "Procedure codes valid:",
    df["procedure_code"].isin(valid_procedures).all()
)

print(
    "Diagnosis codes valid:",
    df["diagnosis_code"].isin(valid_diagnoses).all()
)

print(
    "Scenario types valid:",
    df["scenario_type"].isin(valid_scenarios).all()
)

print(
    "Review labels valid:",
    df["review_label"].isin([0, 1]).all()
)

# --------------------------------------------------
# 8. LABEL CONSISTENCY CHECK
# --------------------------------------------------

print("\n--- REVIEW LABEL VALIDATION ---")

flagged_scenarios = [
    "unusual_billing",
    "repeat_billing",
    "documentation_mismatch"
]

expected_labels = (
    df["scenario_type"]
    .isin(flagged_scenarios)
    .astype(int)
)

label_mismatches = (
    df["review_label"] != expected_labels
).sum()

if label_mismatches == 0:
    print("PASSED: Review labels match scenario types.")
else:
    print(
        f"FAILED: {label_mismatches} label mismatches."
    )

# --------------------------------------------------
# 9. PROVIDER VALIDATION
# --------------------------------------------------

print("\n--- PROVIDER VALIDATION ---")

provider_count = df["provider_id"].nunique()

print(f"Unique providers: {provider_count}")

if provider_count == 50:
    print("PASSED: Expected 50 providers found.")
else:
    print("WARNING: Provider count differs from 50.")

# --------------------------------------------------
# 10. SCENARIO DISTRIBUTION
# --------------------------------------------------

print("\n--- SCENARIO DISTRIBUTION ---")

print(df["scenario_type"].value_counts())

print("\n--- REVIEW LABEL DISTRIBUTION ---")

print(df["review_label"].value_counts())

# --------------------------------------------------
# 11. DATA SUMMARY
# --------------------------------------------------

print("\n--- NUMERICAL SUMMARY ---")

print(
    df[
        [
            "patient_age",
            "claim_amount",
            "procedure_complexity",
            "length_of_stay",
            "repeat_billing_count"
        ]
    ].describe().round(2)
)

# --------------------------------------------------
# 12. FINAL RESULT
# --------------------------------------------------

print("\n====================================")

if (
    len(df) == 5000
    and duplicate_ids == 0
    and missing_values.sum() == 0
    and label_mismatches == 0
    and all(checks.values())
    and not missing_columns
):
    print(" DATASET VALIDATION COMPLETED")
    print(" Core validation checks passed.")
else:
    print(" DATASET VALIDATION NEEDS ATTENTION")
    print(" Review failed checks and warnings.")

print("====================================")