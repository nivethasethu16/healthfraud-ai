import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

np.random.seed(42)

NUM_CLAIMS = 5000
NUM_PROVIDERS = 50

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. HEALTHCARE CATEGORIES
# --------------------------------------------------

procedures = {
    "consultation": 2000,
    "diagnostics": 6000,
    "emergency": 18000,
    "surgery": 50000,
    "intensive_care": 75000
}

diagnoses = {
    "consultation": ["general", "infection"],
    "diagnostics": ["respiratory", "cardiac", "general"],
    "emergency": ["injury", "cardiac", "infection"],
    "surgery": ["injury", "cardiac", "general"],
    "intensive_care": ["cardiac", "respiratory", "infection"]
}

# --------------------------------------------------
# 3. CLINICAL DOCUMENTATION
# --------------------------------------------------

clinical_notes = {
    "consultation": [
        "Patient attended routine medical consultation.",
        "Patient reported mild symptoms and received assessment.",
        "Routine follow-up examination completed."
    ],
    "diagnostics": [
        "Diagnostic tests performed for further evaluation.",
        "Patient underwent laboratory and imaging assessment.",
        "Clinical investigation completed."
    ],
    "emergency": [
        "Patient presented with acute symptoms requiring emergency assessment.",
        "Urgent clinical evaluation and treatment provided.",
        "Patient required emergency medical attention."
    ],
    "surgery": [
        "Patient underwent a documented surgical procedure.",
        "Surgical intervention performed following clinical evaluation.",
        "Post-operative monitoring documented."
    ],
    "intensive_care": [
        "Patient required intensive monitoring and critical care.",
        "Patient received intensive clinical management.",
        "Critical care monitoring and treatment documented."
    ]
}

# --------------------------------------------------
# 4. PROVIDER PROFILES
# --------------------------------------------------

provider_ids = [
    f"HOSP_{i:03d}"
    for i in range(1, NUM_PROVIDERS + 1)
]

provider_profiles = {}

for provider in provider_ids:
    provider_profiles[provider] = {
        "claim_count": np.random.randint(100, 500),
        "billing_factor": np.random.uniform(0.85, 1.15)
    }

# --------------------------------------------------
# 5. GENERATE CLAIM RECORDS
# --------------------------------------------------

records = []

scenarios = [
    "normal",
    "legitimate_high_cost",
    "unusual_billing",
    "repeat_billing",
    "documentation_mismatch"
]

scenario_probabilities = [0.70, 0.10, 0.07, 0.06, 0.07]

for i in range(NUM_CLAIMS):

    provider = np.random.choice(provider_ids)

    provider_profile = provider_profiles[provider]

    scenario = np.random.choice(
        scenarios,
        p=scenario_probabilities
    )

    # Select billed procedure
    procedure = np.random.choice(list(procedures.keys()))

    base_amount = procedures[procedure]

    complexity = np.random.randint(1, 6)

    # --------------------------------------------------
    # CLAIM AMOUNT
    # --------------------------------------------------

    claim_amount = (
        base_amount
        * complexity
        * provider_profile["billing_factor"]
        * np.random.uniform(0.75, 1.25)
    )

    # --------------------------------------------------
    # LENGTH OF STAY
    # --------------------------------------------------

    stay_parameters = {
        "consultation": 0,
        "diagnostics": 0,
        "emergency": 1,
        "surgery": 3,
        "intensive_care": 5
    }

    if procedure in ["consultation", "diagnostics"]:
        length_of_stay = 0
    else:
        length_of_stay = np.random.poisson(
            stay_parameters[procedure] * complexity / 3
        )

    # --------------------------------------------------
    # REPEAT BILLING
    # --------------------------------------------------

    repeat_billing_count = np.random.poisson(0.3)

    # --------------------------------------------------
    # SCENARIO-SPECIFIC VARIATIONS
    # --------------------------------------------------

    if scenario == "legitimate_high_cost":

        claim_amount *= np.random.uniform(2.0, 3.0)

        length_of_stay += np.random.randint(2, 6)

    elif scenario == "unusual_billing":

        claim_amount *= np.random.uniform(2.0, 3.5)

    elif scenario == "repeat_billing":

        repeat_billing_count += np.random.randint(2, 5)

    elif scenario == "documentation_mismatch":

        procedure = "surgery"

        base_amount = procedures[procedure]

        claim_amount = (
            base_amount
            * complexity
            * provider_profile["billing_factor"]
            * np.random.uniform(0.75, 1.25)
        )

    # --------------------------------------------------
    # PROVIDER HISTORY
    # --------------------------------------------------

    provider_claim_count = provider_profile["claim_count"]

    provider_avg_amount = (
    procedures[procedure]
    * 3
    * provider_profile["billing_factor"]
    )
    # --------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------

    diagnosis = np.random.choice(diagnoses[procedure])

    # --------------------------------------------------
    # CLINICAL DOCUMENTATION
    # --------------------------------------------------

    note = np.random.choice(clinical_notes[procedure])

    billed_description = (
        procedure.replace("_", " ").title()
        + " procedure"
    )

    # Introduce documentation mismatch
    if scenario == "documentation_mismatch":

        note = (
            "Patient attended routine consultation "
            "for mild symptoms. No major intervention "
            "was documented."
        )

    # --------------------------------------------------
    # GROUND-TRUTH REVIEW LABEL
    # --------------------------------------------------

    review_label = int(
        scenario in [
            "unusual_billing",
            "repeat_billing",
            "documentation_mismatch"
        ]
    )

    # --------------------------------------------------
    # STORE RECORD
    # --------------------------------------------------

    records.append({

        "claim_id": f"CLM_{i+1:05d}",

        "provider_id": provider,

        "patient_age": np.random.randint(1, 100),

        "procedure_code": procedure,

        "diagnosis_code": diagnosis,

        "claim_amount": round(claim_amount, 2),

        "procedure_complexity": complexity,

        "length_of_stay": length_of_stay,

        "provider_claim_count": provider_claim_count,

        "provider_avg_amount": round(provider_avg_amount, 2),

        "repeat_billing_count": repeat_billing_count,

        "clinical_note": note,

        "billed_description": billed_description,

        "scenario_type": scenario,

        "review_label": review_label
    })

# --------------------------------------------------
# 6. CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)

# --------------------------------------------------
# 7. SAVE DATASET
# --------------------------------------------------

output_file = OUTPUT_DIR / "synthetic_claims.csv"

df.to_csv(output_file, index=False)

# --------------------------------------------------
# 8. DISPLAY SUMMARY
# --------------------------------------------------

print("\n====================================")
print(" HEALTHCARE CLAIMS DATA GENERATOR")
print("====================================")

print(f"\nTotal claims generated: {len(df)}")

print("\nScenario distribution:")
print(df["scenario_type"].value_counts())

print("\nReview label distribution:")
print(df["review_label"].value_counts())

print("\nDataset preview:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print(f"\nDataset saved to: {output_file}")

print("\nData generation completed!")