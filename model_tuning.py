
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
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

RANDOM_STATE = 42

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading preprocessed datasets...")

X = joblib.load(DATA_DIR / "X_train.joblib")
X_test = joblib.load(DATA_DIR / "X_test.joblib")

y = np.asarray(
    joblib.load(DATA_DIR / "y_train.joblib")
).ravel()

y_test = np.asarray(
    joblib.load(DATA_DIR / "y_test.joblib")
).ravel()

features = joblib.load(
    DATA_DIR / "feature_names.joblib"
)

# --------------------------------------------------
# 3. CREATE VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

negative = np.sum(y_train == 0)
positive = np.sum(y_train == 1)

scale_weight = negative / positive

print("Training records:", len(y_train))
print("Validation records:", len(y_val))

# --------------------------------------------------
# 4. MODEL CONFIGURATIONS
# --------------------------------------------------

configs = [
    {
        "name": "XGB_150",
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.1
    },
    {
        "name": "XGB_200",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05
    },
    {
        "name": "XGB_150_deep",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05
    }
]

# --------------------------------------------------
# 5. TRAIN AND COMPARE
# --------------------------------------------------

results = []
best_model = None
best_config = None
best_score = -1

for config in configs:

    print("\nTraining:", config["name"])

    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(
        y_val, probabilities
    )

    roc_auc = roc_auc_score(
        y_val, probabilities
    )

    results.append({
        "model": config["name"],
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    })

    print("PR-AUC:", round(pr_auc, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    # Select using PR-AUC, not accuracy.
    if pr_auc > best_score:
        best_score = pr_auc
        best_model = model
        best_config = config

# --------------------------------------------------
# 6. SELECT THRESHOLD ON VALIDATION DATA
# --------------------------------------------------

print("\nSelecting classification threshold...")

val_probabilities = best_model.predict_proba(
    X_val
)[:, 1]

threshold_results = []

for threshold in np.arange(0.20, 0.71, 0.05):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

threshold_df = pd.DataFrame(threshold_results)

# Choose the threshold with the best validation F1.
best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nSelected threshold:", best_threshold)
print(threshold_df.to_string(index=False))

# --------------------------------------------------
# 7. FINAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on untouched test data...")

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

precision = precision_score(
    y_test, test_predictions, zero_division=0
)

recall = recall_score(
    y_test, test_predictions, zero_division=0
)

f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, test_probabilities
)

pr_auc = average_precision_score(
    y_test, test_probabilities
)

cm = confusion_matrix(
    y_test, test_predictions
)

print("\n========== FINAL RESULTS ==========")

print("Selected model:", best_config["name"])
print("Threshold:", best_threshold)
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
    test_predictions,
    target_names=["Normal", "Flagged"],
    zero_division=0
))

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

joblib.dump(
    best_model,
    MODEL_DIR / "xgboost_tuned.joblib"
)

threshold_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False
)

pd.DataFrame(results).to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

pd.DataFrame({
    "actual_label": y_test,
    "predicted_label": test_predictions,
    "fraud_probability": test_probabilities
}).to_csv(
    REPORT_DIR / "tuned_predictions.csv",
    index=False
)

metrics = {
    "selected_model": best_config["name"],
    "threshold": best_threshold,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "confusion_matrix": cm.tolist()
}

with open(
    REPORT_DIR / "tuned_metrics.json", "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nTuning completed.")
print("Saved: models/xgboost_tuned.joblib")
print("Saved: reports/tuned_metrics.json")
print("Saved: reports/model_comparison.csv")
print("Saved: reports/threshold_comparison.csv")
print("Saved: reports/tuned_predictions.csv")
