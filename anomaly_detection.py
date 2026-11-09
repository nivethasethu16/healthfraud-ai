
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

RANDOM_STATE = 42

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading preprocessed data...")

X_train = joblib.load(DATA_DIR / "X_train.joblib")
X_test = joblib.load(DATA_DIR / "X_test.joblib")

y_train = np.asarray(
    joblib.load(DATA_DIR / "y_train.joblib")
).ravel()

y_test = np.asarray(
    joblib.load(DATA_DIR / "y_test.joblib")
).ravel()

# --------------------------------------------------
# 3. CONFIGURE CONTAMINATION
# --------------------------------------------------

contamination = float(np.mean(y_train == 1))

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Contamination:", round(contamination, 4))

# --------------------------------------------------
# 4. TRAIN ISOLATION FOREST
# --------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=contamination,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)

print("Training completed.")

# --------------------------------------------------
# 5. GENERATE ANOMALY SCORES
# --------------------------------------------------

# Lower decision scores indicate more anomalous observations.
anomaly_scores = -model.decision_function(X_test)

# Isolation Forest: -1 = anomaly, 1 = normal.
raw_predictions = model.predict(X_test)

predictions = (raw_predictions == -1).astype(int)

# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

roc_auc = roc_auc_score(y_test, anomaly_scores)

pr_auc = average_precision_score(
    y_test, anomaly_scores
)

cm = confusion_matrix(y_test, predictions)

print("\n========== ISOLATION FOREST RESULTS ==========")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Flagged"],
    zero_division=0
))

# --------------------------------------------------
# 7. SAVE PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "actual_label": y_test,
    "anomaly_prediction": predictions,
    "anomaly_score": anomaly_scores
})

results.to_csv(
    REPORT_DIR / "anomaly_predictions.csv",
    index=False
)

# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "isolation_forest.joblib"
)

# --------------------------------------------------
# 9. SAVE METRICS
# --------------------------------------------------

metrics = {
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "contamination": contamination,
    "confusion_matrix": cm.tolist()
}

import json

with open(
    REPORT_DIR / "anomaly_metrics.json", "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nSaved successfully:")
print("models/isolation_forest.joblib")
print("reports/anomaly_predictions.csv")
print("reports/anomaly_metrics.json")

print("\nISOLATION FOREST COMPLETED.")
