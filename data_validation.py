import numpy as np
import pandas as pd
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_FILE = Path("data/synthetic_claims.csv")

EXPECTED_RECORDS = 5000
EXPECTED_PROVIDERS = 50

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


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("\n========================================")
print(" HEALTHCARE CLAIMS DATA VALIDATION")
print("========================================")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# ==================================================
# 3. REQUIRED COLUMN CHECK
# ==================================================

print("\n--- COLUMN VALIDATION ---")

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

extra_columns = [
    col for col in df.columns
    if col not in required_columns
]

column_check = not missing_columns and not extra_columns

if column_check:
    print("PASSED: All required columns exist.")
else:
    print("FAILED: Column structure does not match.")

    if missing_columns:
        print("Missing columns:", missing_columns)

    if extra_columns:
        print("Unexpected columns:", extra_columns)

if missing_columns:
    raise ValueError(
        "Essential columns are missing. Validation stopped."
    )


# ==================================================
# 4. MISSING VALUE CHECK
# ==================================================

print("\n--- MISSING VALUE CHECK ---")

missing_values = df[required_columns].isnull().sum()
missing_count = int(missing_values.sum())

if missing_count == 0:
    print("PASSED: No missing values found.")
else:
    print("FAILED: Missing values detected.")
    print(missing_values[missing_values > 0])

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

empty_text_count = int(empty_text.sum().sum())

if empty_text_count == 0:
    print("PASSED: No empty text values.")
else:
    print(f"FAILED: {empty_text_count} empty text values found.")


# ==================================================
# 5. DUPLICATE CHECK
# ==================================================

print("\n--- DUPLICATE CHECK ---")

duplicate_ids = int(df["claim_id"].duplicated().sum())

if duplicate_ids == 0:
    print("PASSED: No duplicate claim IDs.")
else:
    print(f"FAILED: {duplicate_ids} duplicate claim IDs.")


# ==================================================
# 6. NUMERICAL VALIDATION
# ==================================================

print("\n--- NUMERICAL VALIDATION ---")

numeric_columns = [
    "patient_age",
    "claim_amount",
    "procedure_complexity",
    "length_of_stay",
    "provider_claim_count",
    "provider_avg_amount",
    "repeat_billing_count",
    "review_label"
]

numeric_data = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

numeric_conversion_check = numeric_data.notna().all().all()

finite_check = np.isfinite(
    numeric_data.to_numpy(dtype=float)
).all()

numeric_checks = {
    "Numeric values are valid": numeric_conversion_check,
    "Numeric values are finite": finite_check,
    "Patient age (1-99)": (
        numeric_data["patient_age"].between(1, 99).all()
    ),
    "Claim amount positive": (
        (numeric_data["claim_amount"] > 0).all()
    ),
    "Procedure complexity (1-5)": (
        numeric_data["procedure_complexity"].between(1, 5).all()
    ),
    "Length of stay non-negative": (
        (numeric_data["length_of_stay"] >= 0).all()
    ),
    "Provider claim count positive": (
        (numeric_data["provider_claim_count"] > 0).all()
    ),
    "Provider average amount positive": (
        (numeric_data["provider_avg_amount"] > 0).all()
    ),
    "Repeat billing count non-negative": (
        (numeric_data["repeat_billing_count"] >= 0).all()
    )
}

for check, passed in numeric_checks.items():
    print(
        f"{'PASSED' if passed else 'FAILED'}: {check}"
    )


# ==================================================
# 7. CATEGORICAL VALIDATION
# ==================================================

print("\n--- CATEGORICAL VALIDATION ---")

categorical_checks = {
    "Procedure codes valid": (
        df["procedure_code"].isin(valid_procedures).all()
    ),
    "Diagnosis codes valid": (
        df["diagnosis_code"].isin(valid_diagnoses).all()
    ),
    "Scenario types valid": (
        df["scenario_type"].isin(valid_scenarios).all()
    ),
    "Review labels are binary": (
        df["review_label"].isin([0, 1]).all()
    )
}

for check, passed in categorical_checks.items():
    print(
        f"{'PASSED' if passed else 'FAILED'}: {check}"
    )


# ==================================================
# 8. REVIEW LABEL VALIDATION
# ==================================================

print("\n--- REVIEW LABEL VALIDATION ---")

label_counts = df["review_label"].value_counts()

both_classes_present = set(
    df["review_label"].dropna().unique()
) == {0, 1}

if both_classes_present:
    print("PASSED: Both review-label classes are present.")
else:
    print("FAILED: Expected both labels 0 and 1.")

print("\nReview label distribution:")
print(label_counts)

print("\nReview label percentages:")
print(
    (
        df["review_label"]
        .value_counts(normalize=True) * 100
    ).round(2)
)

# Labels are generated probabilistically.
# Therefore, they are NOT required to match scenario_type.


# ==================================================
# 9. SCENARIO AND LABEL ANALYSIS
# ==================================================

print("\n--- SCENARIO VS REVIEW LABEL ---")

scenario_label_table = pd.crosstab(
    df["scenario_type"],
    df["review_label"],
    margins=True
)

print(scenario_label_table)

print(
    "\nNote: Different labels within a scenario are expected."
)
print(
    "Scenario type does not directly determine the review label."
)


# ==================================================
# 10. PROVIDER VALIDATION
# ==================================================

print("\n--- PROVIDER VALIDATION ---")

provider_count = df["provider_id"].nunique()

print(f"Unique providers: {provider_count}")

provider_count_check = (
    provider_count == EXPECTED_PROVIDERS
)

print(
    f"{'PASSED' if provider_count_check else 'FAILED'}: "
    f"Expected {EXPECTED_PROVIDERS} providers."
)

# Provider history values should remain consistent
# for each provider in this generated dataset.

provider_history_check = (
    df.groupby("provider_id")[
        ["provider_claim_count", "provider_avg_amount"]
    ].nunique().le(1).all().all()
)

print(
    f"{'PASSED' if provider_history_check else 'FAILED'}: "
    "Provider history values are consistent."
)


# ==================================================
# 11. SCENARIO DISTRIBUTION
# ==================================================

print("\n--- SCENARIO DISTRIBUTION ---")

print(df["scenario_type"].value_counts())


# ==================================================
# 12. NUMERICAL SUMMARY
# ==================================================

print("\n--- NUMERICAL SUMMARY ---")

print(
    numeric_data[
        [
            "patient_age",
            "claim_amount",
            "procedure_complexity",
            "length_of_stay",
            "repeat_billing_count"
        ]
    ].describe().round(2)
)


# ==================================================
# 13. FINAL VALIDATION RESULT
# ==================================================

print("\n========================================")

record_count_check = len(df) == EXPECTED_RECORDS

print(
    f"{'PASSED' if record_count_check else 'FAILED'}: "
    f"Expected {EXPECTED_RECORDS} records."
)

all_checks = (
    column_check
    and missing_count == 0
    and empty_text_count == 0
    and duplicate_ids == 0
    and record_count_check
    and all(numeric_checks.values())
    and all(categorical_checks.values())
    and both_classes_present
    and provider_count_check
    and provider_history_check
)

if all_checks:
    print(" DATASET VALIDATION PASSED")
    print(" All core validation checks passed.")
else:
    print(" DATASET VALIDATION FAILED")
    print(" Please review the failed checks.")

print("========================================")