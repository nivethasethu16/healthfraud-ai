
import numpy as np
import pandas as pd
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

RANDOM_SEED = 42
NUM_CLAIMS = 5000
NUM_PROVIDERS = 50

rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "synthetic_claims.csv"


# ==================================================
# 2. HEALTHCARE CATEGORIES
# ==================================================

procedures = {
    "consultation": {
        "base_amount": 2000,
        "diagnoses": ["general", "infection"],
        "stay_mean": 0,
        "notes": [
            "Patient attended routine medical consultation.",
            "Patient reported mild symptoms and received assessment.",
            "Routine follow-up examination completed."
        ]
    },
    "diagnostics": {
        "base_amount": 6000,
        "diagnoses": ["respiratory", "cardiac", "general"],
        "stay_mean": 0,
        "notes": [
            "Diagnostic tests performed for further evaluation.",
            "Patient underwent laboratory and imaging assessment.",
            "Clinical investigation completed."
        ]
    },
    "emergency": {
        "base_amount": 18000,
        "diagnoses": ["injury", "cardiac", "infection"],
        "stay_mean": 1,
        "notes": [
            "Patient presented with acute symptoms requiring emergency assessment.",
            "Urgent clinical evaluation and treatment provided.",
            "Patient required emergency medical attention."
        ]
    },
    "surgery": {
        "base_amount": 50000,
        "diagnoses": ["injury", "cardiac", "general"],
        "stay_mean": 3,
        "notes": [
            "Patient underwent a documented surgical procedure.",
            "Surgical intervention performed following clinical evaluation.",
            "Post-operative monitoring documented."
        ]
    },
    "intensive_care": {
        "base_amount": 75000,
        "diagnoses": ["cardiac", "respiratory", "infection"],
        "stay_mean": 5,
        "notes": [
            "Patient required intensive monitoring and critical care.",
            "Patient received intensive clinical management.",
            "Critical care monitoring and treatment documented."
        ]
    }
}

procedure_names = list(procedures.keys())

procedure_probabilities = [
    0.30,  # Consultation
    0.25,  # Diagnostics
    0.18,  # Emergency
    0.17,  # Surgery
    0.10   # Intensive care
]

scenarios = [
    "normal",
    "legitimate_high_cost",
    "unusual_billing",
    "repeat_billing",
    "documentation_mismatch"
]

scenario_probabilities = [0.70, 0.10, 0.07, 0.06, 0.07]


# ==================================================
# 3. PROVIDER PROFILES
# ==================================================

provider_ids = [
    f"HOSP_{i:03d}"
    for i in range(1, NUM_PROVIDERS + 1)
]

provider_profiles = {}

# Approximate expected average claim amount across
# the procedure mix used in this synthetic dataset.

average_procedure_amount = sum(
    procedures[name]["base_amount"] * probability
    for name, probability in zip(
        procedure_names,
        procedure_probabilities
    )
) * 3

for provider in provider_ids:

    historical_claim_count = int(
        rng.integers(200, 1500)
    )

    billing_factor = rng.uniform(0.85, 1.15)

    historical_average_amount = (
        average_procedure_amount
        * billing_factor
        * rng.uniform(0.90, 1.10)
    )

    provider_profiles[provider] = {
        "claim_count": historical_claim_count,
        "billing_factor": billing_factor,
        "average_amount": historical_average_amount
    }

# Providers with larger historical claim volumes
# are more likely to appear in the generated dataset.

provider_weights = np.array([
    provider_profiles[p]["claim_count"]
    for p in provider_ids
], dtype=float)

provider_weights /= provider_weights.sum()


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def generate_procedure():

    return rng.choice(
        procedure_names,
        p=procedure_probabilities
    )


def generate_review_probability(
    amount_ratio,
    repeat_count,
    documentation_mismatch,
    length_of_stay,
    expected_stay
):
    """
    Estimate the probability that a claim should
    be flagged for manual review.

    This is a synthetic rule-based probability,
    not a real-world fraud probability.
    """

    billing_signal = np.clip(
        (amount_ratio - 1.25) / 1.40,
        0,
        1
    )

    repeat_signal = np.clip(
        repeat_count / 3,
        0,
        1
    )

    documentation_signal = int(
        documentation_mismatch
    )

    # Additional stay can provide context for
    # a higher-cost legitimate claim.
    stay_support = np.clip(
        (length_of_stay - expected_stay) / 4,
        0,
        1
    )

    risk_score = (
        0.55 * billing_signal
        + 0.35 * repeat_signal
        + 0.70 * documentation_signal
        - 0.30 * stay_support
    )

    logit = -2.30 + (3.20 * risk_score)

    probability = 1 / (1 + np.exp(-logit))

    return float(np.clip(probability, 0.01, 0.95))


# ==================================================
# 5. GENERATE CLAIM RECORDS
# ==================================================

records = []

for i in range(NUM_CLAIMS):

    # --------------------------------------------------
    # PROVIDER SELECTION
    # --------------------------------------------------

    provider = rng.choice(
        provider_ids,
        p=provider_weights
    )

    provider_profile = provider_profiles[provider]

    # --------------------------------------------------
    # SCENARIO SELECTION
    # --------------------------------------------------

    scenario = rng.choice(
        scenarios,
        p=scenario_probabilities
    )

    # --------------------------------------------------
    # PROCEDURE SELECTION
    # --------------------------------------------------

    if scenario == "legitimate_high_cost":

        # These claims are expensive for legitimate
        # clinical reasons, not automatically suspicious.
        procedure = rng.choice(
            ["emergency", "surgery", "intensive_care"],
            p=[0.20, 0.45, 0.35]
        )

        complexity = int(rng.integers(4, 6))

    else:

        procedure = generate_procedure()

        complexity = int(rng.integers(1, 6))

    procedure_info = procedures[procedure]

    # --------------------------------------------------
    # EXPECTED CLAIM AMOUNT
    # --------------------------------------------------

    expected_amount = (
        procedure_info["base_amount"]
        * complexity
        * provider_profile["billing_factor"]
    )

    # --------------------------------------------------
    # SCENARIO-BASED BILLING VARIATION
    # --------------------------------------------------

    if scenario == "unusual_billing":

        # Some unusual claims have moderate deviations;
        # others have larger deviations.
        if rng.random() < 0.30:
            billing_multiplier = rng.uniform(1.25, 1.60)
        else:
            billing_multiplier = rng.uniform(1.60, 2.40)

    elif scenario == "legitimate_high_cost":

        # A moderate increase accompanied by greater
        # clinical complexity and longer hospital stay.
        billing_multiplier = rng.uniform(1.05, 1.40)

    else:

        billing_multiplier = rng.uniform(0.85, 1.25)

    claim_amount = (
        expected_amount
        * billing_multiplier
        * rng.uniform(0.95, 1.05)
    )

    # --------------------------------------------------
    # LENGTH OF STAY
    # --------------------------------------------------

    expected_stay = procedure_info["stay_mean"]

    if expected_stay == 0:
        length_of_stay = 0
    else:
        length_of_stay = int(
            rng.poisson(expected_stay * complexity / 3)
        )

    if scenario == "legitimate_high_cost":
        length_of_stay += int(rng.integers(2, 6))

    # --------------------------------------------------
    # REPEAT BILLING
    # --------------------------------------------------

    repeat_billing_count = int(
        rng.poisson(0.25)
    )

    if scenario == "repeat_billing":
        repeat_billing_count += int(
            rng.integers(1, 4)
        )

    # --------------------------------------------------
    # DOCUMENTATION
    # --------------------------------------------------

    # Most documentation-mismatch scenarios create
    # an actual mismatch. A small number of other
    # scenarios can also contain accidental mismatches.
    if scenario == "documentation_mismatch":
        documentation_mismatch = rng.random() < 0.85
    else:
        documentation_mismatch = rng.random() < 0.015

    if documentation_mismatch:

        other_procedures = [
            name for name in procedure_names
            if name != procedure
        ]

        note_category = rng.choice(other_procedures)

    else:
        note_category = procedure

    note = rng.choice(
        procedures[note_category]["notes"]
    )

    billed_description = (
        procedure.replace("_", " ").title()
        + " procedure"
    )

    # --------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------

    diagnosis = rng.choice(
        procedure_info["diagnoses"]
    )

    # --------------------------------------------------
    # PROVIDER HISTORY
    # --------------------------------------------------

    provider_claim_count = provider_profile["claim_count"]

    provider_avg_amount = provider_profile["average_amount"]

    # --------------------------------------------------
    # REVIEW LABEL
    # --------------------------------------------------

    amount_ratio = claim_amount / expected_amount

    review_probability = generate_review_probability(
        amount_ratio=amount_ratio,
        repeat_count=repeat_billing_count,
        documentation_mismatch=documentation_mismatch,
        length_of_stay=length_of_stay,
        expected_stay=expected_stay
    )

    # A probabilistic label avoids making scenario
    # categories directly determine the target.
    review_label = int(
        rng.random() < review_probability
    )

    # --------------------------------------------------
    # STORE RECORD
    # --------------------------------------------------

    records.append({

        "claim_id": f"CLM_{i + 1:05d}",

        "provider_id": provider,

        "patient_age": int(rng.integers(1, 100)),

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


# ==================================================
# 6. CREATE DATAFRAME
# ==================================================

df = pd.DataFrame(records)


# ==================================================
# 7. VALIDATE DATASET
# ==================================================

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

assert len(df) == NUM_CLAIMS, "Incorrect number of claims."

assert list(df.columns) == required_columns, (
    "Dataset columns do not match the expected structure."
)

assert df["claim_id"].is_unique, "Duplicate claim IDs found."

assert not df.isnull().any().any(), "Missing values found."

assert set(df["review_label"].unique()).issubset({0, 1}), (
    "Invalid review labels found."
)

assert (df["claim_amount"] > 0).all(), (
    "Invalid claim amounts found."
)


# ==================================================
# 8. SAVE DATASET
# ==================================================

df.to_csv(OUTPUT_FILE, index=False)


# ==================================================
# 9. DISPLAY SUMMARY
# ==================================================

print("\n========================================")
print(" HEALTHCARE CLAIMS DATA GENERATOR")
print("========================================")

print(f"\nTotal claims generated: {len(df)}")

print(f"\nTotal providers: {df['provider_id'].nunique()}")

print("\nScenario distribution:")
print(df["scenario_type"].value_counts())

print("\nReview label distribution:")
print(df["review_label"].value_counts())

print("\nReview label percentages:")
print(
    (df["review_label"].value_counts(normalize=True) * 100)
    .round(2)
    .astype(str)
    + "%"
)

print("\nProvider history preview:")
print(
    df[
        [
            "provider_id",
            "provider_claim_count",
            "provider_avg_amount"
        ]
    ].drop_duplicates().head()
)

print("\nDataset preview:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nValidation:")
print("All required columns present.")
print("All claim IDs unique.")
print("No missing values.")
print("Review labels are binary.")
print("Claim amounts are positive.")

print(f"\nDataset saved to: {OUTPUT_FILE}")

print("\nData generation completed!")