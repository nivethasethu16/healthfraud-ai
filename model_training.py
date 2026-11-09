
import joblib
import numpy as np
import pandas as pd
import json

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
THRESHOLD = 0.5

# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("\n" + "=" * 55)
print(" HEALTHFRAUD AI - XGBOOST MODEL TRAINING")
print("=" * 55)

X_train = joblib.load(DATA_DIR / "X_train.joblib")
X_test = joblib.load(DATA_DIR / "X_test.joblib")

y_train = np.asarray(
    joblib.load(DATA_DIR / "y_train.joblib")
).ravel()

y_test = np.asarray(
    joblib.load(DATA_DIR / "y_test.joblib")
).ravel()

feature_names = joblib.load(
    DATA_DIR / "feature_names.joblib"
)

print("\nDataset loaded successfully.")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])

# --------------------------------------------------
# 3. HANDLE CLASS IMBALANCE
# --------------------------------------------------

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

if positive_count == 0:
    raise ValueError("Training data contains no positive labels.")

scale_pos_weight = negative_count / positive_count

print("\nClass distribution:")
print("Normal claims:", negative_count)
print("Flagged claims:", positive_count)

print("Class imbalance weight:", round(scale_pos_weight, 3))

# --------------------------------------------------
# 4. INITIALIZE XGBOOST
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("Model training completed successfully.")

# --------------------------------------------------
# 6. GENERATE PREDICTIONS
# --------------------------------------------------

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= THRESHOLD).astype(int)

# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print(" MODEL EVALUATION RESULTS")
print("=" * 55)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Flagged"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(cm)

# --------------------------------------------------
# 8. FEATURE IMPORTANCE
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Important Features:")

print(
    importance_df.head(15).to_string(index=False)
)

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

# --------------------------------------------------
# 9. SAVE PREDICTIONS
# --------------------------------------------------

predictions = pd.DataFrame({
    "actual_label": y_test,
    "predicted_label": y_pred,
    "fraud_probability": y_prob
})

predictions.to_csv(
    REPORT_DIR / "model_predictions.csv",
    index=False
)

# --------------------------------------------------
# 10. SAVE EVALUATION METRICS
# --------------------------------------------------

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "threshold": THRESHOLD,
    "scale_pos_weight": float(scale_pos_weight),
    "confusion_matrix": cm.tolist()
}

with open(REPORT_DIR / "model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# 11. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "xgboost_model.joblib"
)

print("\nModel saved successfully.")

print("\nGenerated files:")
print("models/xgboost_model.joblib")
print("reports/model_metrics.json")
print("reports/model_predictions.csv")
print("reports/feature_importance.csv")

print("\n" + "=" * 55)
print(" XGBOOST TRAINING PIPELINE COMPLETED")
print("=" * 55)