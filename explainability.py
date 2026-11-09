
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.55

# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

print("\n" + "=" * 55)
print(" HEALTHFRAUD AI - EXPLAINABILITY ANALYSIS")
print("=" * 55)

print("\nLoading trained model...")

model_path = MODEL_DIR / "xgboost_tuned.joblib"

if not model_path.exists():
    raise FileNotFoundError("Tuned model not found.")

model = joblib.load(model_path)

print("Model loaded:", model_path.name)

# --------------------------------------------------
# 3. LOAD TEST DATA
# --------------------------------------------------

print("\nLoading processed test data...")

X_test = joblib.load(DATA_DIR / "X_test.joblib")

y_test = np.asarray(
    joblib.load(DATA_DIR / "y_test.joblib")
).ravel()

feature_names = [
    str(name)
    for name in joblib.load(DATA_DIR / "feature_names.joblib")
]

if X_test.shape[0] != len(y_test):
    raise ValueError("Test data and labels do not match.")

if X_test.shape[1] != len(feature_names):
    raise ValueError("Feature names do not match.")

print("Test records:", X_test.shape[0])
print("Number of features:", X_test.shape[1])

# Dense data is only for displaying feature values in plots.
if hasattr(X_test, "toarray"):
    X_test_dense = X_test.toarray()
else:
    X_test_dense = np.asarray(X_test)

X_test_df = pd.DataFrame(
    X_test_dense,
    columns=feature_names
)

# --------------------------------------------------
# 4. LOAD CLAIM IDS
# --------------------------------------------------

print("\nLoading claim identifiers...")

claim_ids_df = pd.read_csv(
    DATA_DIR / "test_claim_ids.csv"
)

if "claim_id" not in claim_ids_df.columns:
    raise ValueError("claim_id column missing.")

claim_ids = (
    claim_ids_df["claim_id"]
    .astype(str)
    .to_numpy()
)

if len(claim_ids) != X_test.shape[0]:
    raise ValueError("Claim IDs do not match test records.")

if len(np.unique(claim_ids)) != len(claim_ids):
    raise ValueError("Duplicate claim IDs found.")

# --------------------------------------------------
# 5. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating XGBoost predictions...")

# IMPORTANT: Preserve the original sparse input.
fraud_probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    fraud_probabilities >= THRESHOLD
).astype(int)

# --------------------------------------------------
# 6. VERIFY PREDICTIONS
# --------------------------------------------------

print("\nVerifying predictions against tuning report...")

tuned_path = REPORT_DIR / "tuned_predictions.csv"

if not tuned_path.exists():
    raise FileNotFoundError("Tuned predictions report missing.")

tuned_df = pd.read_csv(tuned_path)

if len(tuned_df) != len(fraud_probabilities):
    raise ValueError("Prediction report row count mismatch.")

if not np.allclose(
    fraud_probabilities,
    tuned_df["fraud_probability"].to_numpy(),
    atol=1e-6
):
    raise ValueError(
        "Predictions do not match tuned_predictions.csv. "
        "Stopping."
    )

print("Prediction verification passed.")

# --------------------------------------------------
# 7. GENERATE SHAP VALUES
# --------------------------------------------------

print("\nCalculating SHAP contributions...")

# Use XGBoost's native contribution calculation.
# This preserves sparse-input behavior.
dmatrix = xgb.DMatrix(X_test)

contributions = model.get_booster().predict(
    dmatrix,
    pred_contribs=True
)

# Last column is the base-value contribution.
shap_values = contributions[:, :-1]

if shap_values.shape != X_test.shape:
    raise ValueError("SHAP contribution dimensions mismatch.")

print("SHAP calculation completed.")

# --------------------------------------------------
# 8. GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

print("\nCalculating global feature importance...")

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": np.abs(shap_values).mean(axis=0)
})

importance_df = importance_df.sort_values(
    "mean_absolute_shap",
    ascending=False
)

importance_df.to_csv(
    REPORT_DIR / "feature_importance_shap.csv",
    index=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

# --------------------------------------------------
# 9. SAVE INDIVIDUAL SHAP VALUES
# --------------------------------------------------

print("\nSaving individual explanations...")

shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

shap_df.insert(0, "claim_id", claim_ids)

shap_df.to_csv(
    REPORT_DIR / "shap_explanations.csv",
    index=False
)

# --------------------------------------------------
# 10. CLAIM-LEVEL EXPLANATIONS
# --------------------------------------------------

print("\nGenerating claim-level explanations...")

claim_explanations = []

for i in range(X_test.shape[0]):

    contributions_row = shap_values[i]

    positive_indices = np.argsort(
        contributions_row
    )[::-1][:5]

    negative_indices = np.argsort(
        contributions_row
    )[:5]

    positive_features = [
        f"{feature_names[j]} ({contributions_row[j]:.4f})"
        for j in positive_indices
        if contributions_row[j] > 0
    ]

    negative_features = [
        f"{feature_names[j]} ({contributions_row[j]:.4f})"
        for j in negative_indices
        if contributions_row[j] < 0
    ]

    claim_explanations.append({
        "claim_id": claim_ids[i],
        "actual_label": int(y_test[i]),
        "predicted_label": int(predictions[i]),
        "fraud_probability": float(fraud_probabilities[i]),
        "top_risk_contributors": "; ".join(positive_features),
        "top_normal_contributors": "; ".join(negative_features)
    })

claim_df = pd.DataFrame(claim_explanations)

claim_df.to_csv(
    REPORT_DIR / "claim_explanations.csv",
    index=False
)

# --------------------------------------------------
# 11. SHAP SUMMARY PLOT
# --------------------------------------------------

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_test_df,
    show=False,
    max_display=15
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 12. SHAP FEATURE IMPORTANCE PLOT
# --------------------------------------------------

print("\nGenerating SHAP feature importance plot...")

top_features = importance_df.head(15).sort_values(
    "mean_absolute_shap",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["mean_absolute_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Top 15 Features - SHAP Importance")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 13. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 55)
print(" EXPLAINABILITY ANALYSIS COMPLETED")
print("=" * 55)

print("\nGenerated Reports:")
print("1. feature_importance_shap.csv")
print("2. shap_explanations.csv")
print("3. claim_explanations.csv")
print("4. shap_summary.png")
print("5. shap_feature_importance.png")

print("\nAll reports saved inside the reports folder.")