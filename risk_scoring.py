
import pandas as pd
import numpy as np

from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.55

# Provisional review-priority bands.
# High begins at the selected XGBoost classification threshold.
LOW_LIMIT = 0.35
HIGH_LIMIT = THRESHOLD

# --------------------------------------------------
# 2. LOAD REPORTS
# --------------------------------------------------

print("\n" + "=" * 55)
print(" HEALTHFRAUD AI - RISK SCORING")
print("=" * 55)

print("\nLoading prediction and explanation reports...")

tuned_df = pd.read_csv(
    REPORT_DIR / "tuned_predictions.csv"
)

anomaly_df = pd.read_csv(
    REPORT_DIR / "anomaly_predictions.csv"
)

explanation_df = pd.read_csv(
    REPORT_DIR / "claim_explanations.csv"
)

claim_ids_df = pd.read_csv(
    DATA_DIR / "test_claim_ids.csv"
)

# --------------------------------------------------
# 3. VALIDATE CLAIM IDENTIFIERS
# --------------------------------------------------

if "claim_id" not in claim_ids_df.columns:
    raise ValueError("claim_id column missing from test_claim_ids.csv")

claim_ids = claim_ids_df["claim_id"].astype(str)

if not (
    len(claim_ids)
    == len(tuned_df)
    == len(anomaly_df)
):
    raise ValueError(
        "Prediction records and test claim IDs have different lengths."
    )

if claim_ids.duplicated().any():
    raise ValueError("Duplicate claim IDs found in test data.")

# Attach IDs using the original test-set row order.
tuned_df.insert(0, "claim_id", claim_ids.to_numpy())
anomaly_df.insert(0, "claim_id", claim_ids.to_numpy())

# --------------------------------------------------
# 4. VALIDATE PREDICTIONS
# --------------------------------------------------

required_tuned = {
    "claim_id",
    "actual_label",
    "predicted_label",
    "fraud_probability"
}

required_anomaly = {
    "claim_id",
    "actual_label",
    "anomaly_prediction",
    "anomaly_score"
}

required_explanation = {
    "claim_id",
    "actual_label",
    "predicted_label",
    "fraud_probability",
    "top_risk_contributors",
    "top_normal_contributors"
}

if not required_tuned.issubset(tuned_df.columns):
    raise ValueError("Unexpected columns in tuned_predictions.csv")

if not required_anomaly.issubset(anomaly_df.columns):
    raise ValueError("Unexpected columns in anomaly_predictions.csv")

if not required_explanation.issubset(explanation_df.columns):
    raise ValueError("Unexpected columns in claim_explanations.csv")

if explanation_df["claim_id"].astype(str).duplicated().any():
    raise ValueError("Duplicate claim IDs in claim explanations.")

if not set(tuned_df["claim_id"]).issubset(
    set(explanation_df["claim_id"].astype(str))
):
    raise ValueError("Some tuned predictions have no SHAP explanation.")

# --------------------------------------------------
# 5. MERGE REPORTS BY CLAIM ID
# --------------------------------------------------

print("\nMatching records by claim ID...")

risk_df = tuned_df.merge(
    anomaly_df,
    on="claim_id",
    how="inner",
    suffixes=("_xgb", "_anomaly"),
    validate="one_to_one"
)

risk_df = risk_df.merge(
    explanation_df,
    on="claim_id",
    how="inner",
    suffixes=("", "_shap"),
    validate="one_to_one"
)

if len(risk_df) != len(claim_ids):
    raise ValueError(
        "Some claims were lost during merging. Check claim IDs."
    )

# --------------------------------------------------
# 6. CROSS-CHECK LABELS AND PROBABILITIES
# --------------------------------------------------

if not (
    risk_df["actual_label_xgb"].to_numpy()
    == risk_df["actual_label_anomaly"].to_numpy()
).all():
    raise ValueError("Actual labels differ between prediction reports.")

if not (
    risk_df["actual_label_xgb"].to_numpy()
    == risk_df["actual_label"].to_numpy()
).all():
    raise ValueError("Actual labels differ in SHAP explanations.")

if not np.allclose(
    risk_df["fraud_probability"],
    risk_df["fraud_probability_shap"],
    atol=1e-6
):
    raise ValueError(
        "XGBoost probabilities differ from SHAP explanation report."
    )

# Retain one consistent set of columns.
risk_df = risk_df.drop(
    columns=[
        "actual_label",
        "actual_label_anomaly",
        "actual_label_shap",
        "predicted_label_shap",
        "fraud_probability_shap"
    ],
    errors="ignore"
)

risk_df = risk_df.rename(
    columns={
        "actual_label_xgb": "actual_label"
    }
)
# --------------------------------------------------
# 7. GENERATE RISK CATEGORIES
# --------------------------------------------------

print("\nAssigning review-priority categories...")

risk_df["risk_category"] = np.select(
    [
        risk_df["fraud_probability"] >= HIGH_LIMIT,
        risk_df["fraud_probability"] >= LOW_LIMIT
    ],
    [
        "High",
        "Medium"
    ],
    default="Low"
)

risk_df["review_priority"] = risk_df["risk_category"].map(
    {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }
)

# Keep anomaly information separate from XGBoost risk.
risk_df["anomaly_status"] = np.where(
    risk_df["anomaly_prediction"] == 1,
    "Anomalous",
    "Not anomalous"
)

# --------------------------------------------------
# 8. ADD REVIEW INDICATORS
# --------------------------------------------------

risk_df["requires_review"] = (
    (risk_df["predicted_label"] == 1)
    | (risk_df["anomaly_prediction"] == 1)
)

# --------------------------------------------------
# 9. SELECT FINAL REPORT COLUMNS
# --------------------------------------------------

final_columns = [
    "claim_id",
    "actual_label",
    "fraud_probability",
    "predicted_label",
    "risk_category",
    "review_priority",
    "anomaly_score",
    "anomaly_prediction",
    "anomaly_status",
    "requires_review",
    "top_risk_contributors",
    "top_normal_contributors"
]

risk_df = risk_df[final_columns]

# Sort by primary model probability.
risk_df = risk_df.sort_values(
    by="fraud_probability",
    ascending=False
)

# --------------------------------------------------
# 10. SAVE FINAL REPORT
# --------------------------------------------------

output_path = REPORT_DIR / "risk_scoring.csv"

risk_df.to_csv(
    output_path,
    index=False
)

# --------------------------------------------------
# 11. DISPLAY SUMMARY
# --------------------------------------------------

print("\n========== RISK SCORING SUMMARY ==========")

print("Total claims:", len(risk_df))

print("\nRisk category distribution:")
print(
    risk_df["risk_category"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
)

print("\nClaims requiring review:",
      int(risk_df["requires_review"].sum()))

print("\nAnomalous claims:",
      int((risk_df["anomaly_prediction"] == 1).sum()))

print("\nTop 10 highest-risk claims:")

print(
    risk_df[
        [
            "claim_id",
            "fraud_probability",
            "risk_category",
            "anomaly_status"
        ]
    ].head(10).to_string(index=False)
)

print("\nRisk scoring completed successfully.")
print("Saved:", output_path)
