import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_FILE = Path("data/synthetic_claims.csv")

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "review_label"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\n========================================")
print(" HEALTHCARE CLAIMS DATA PREPROCESSING")
print("========================================")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# --------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
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

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[TARGET].isnull().any():
    raise ValueError("Target column contains missing values.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

# --------------------------------------------------
# 4. PREPARE CLINICAL TEXT
# --------------------------------------------------

df["claim_text"] = (
    df["clinical_note"].fillna("").astype(str)
    + " "
    + df["billed_description"].fillna("").astype(str)
)

# --------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

# Keep claim IDs separately for traceability.
claim_ids = df["claim_id"].copy()

# Exclude identifiers, scenario labels and target.
# Clinical text is combined into claim_text.

X = df.drop(
    columns=[
        "claim_id",
        "scenario_type",
        "review_label",
        "clinical_note",
        "billed_description"
    ]
)

y = df[TARGET].astype(int)

print("\nFeatures and target separated.")
print(f"Input columns: {len(X.columns)}")
print(f"Target: {TARGET}")

# --------------------------------------------------
# 6. DEFINE FEATURE GROUPS
# --------------------------------------------------

numerical_features = [
    "patient_age",
    "claim_amount",
    "procedure_complexity",
    "length_of_stay",
    "provider_claim_count",
    "provider_avg_amount",
    "repeat_billing_count"
]

categorical_features = [
    "provider_id",
    "procedure_code",
    "diagnosis_code"
]

text_feature = "claim_text"

# --------------------------------------------------
# 7. NUMERICAL PREPROCESSING
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# --------------------------------------------------
# 8. CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# --------------------------------------------------
# 9. TEXT PREPROCESSING
# --------------------------------------------------

text_pipeline = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words="english"
)

# --------------------------------------------------
# 10. COMBINE PREPROCESSING PIPELINES
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "text",
            text_pipeline,
            text_feature
        )
    ],
    remainder="drop"
)

# --------------------------------------------------
# 11. TRAIN-TEST SPLIT
# --------------------------------------------------

(
    X_train,
    X_test,
    y_train,
    y_test,
    ids_train,
    ids_test
) = train_test_split(
    X,
    y,
    claim_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain-test split completed.")

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# --------------------------------------------------
# 12. FIT PREPROCESSOR ON TRAINING DATA
# --------------------------------------------------

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

# Transform test data using the same fitted pipeline.
X_test_processed = preprocessor.transform(
    X_test
)

print("Preprocessing completed.")

print(
    "\nTraining feature shape:",
    X_train_processed.shape
)

print(
    "Testing feature shape:",
    X_test_processed.shape
)

# --------------------------------------------------
# 13. SAVE PREPROCESSOR
# --------------------------------------------------

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.joblib"
)

# --------------------------------------------------
# 14. SAVE PROCESSED DATASETS
# --------------------------------------------------

datasets = {
    "X_train": X_train_processed,
    "X_test": X_test_processed,
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": preprocessor.get_feature_names_out()
}

for filename, data in datasets.items():

    joblib.dump(
        data,
        PROCESSED_DIR / f"{filename}.joblib"
    )

# Save claim IDs separately for prediction traceability.
ids_train.to_csv(
    PROCESSED_DIR / "train_claim_ids.csv",
    index=False
)

ids_test.to_csv(
    PROCESSED_DIR / "test_claim_ids.csv",
    index=False
)

# --------------------------------------------------
# 15. DISPLAY LABEL DISTRIBUTION
# --------------------------------------------------

print("\n--- TRAINING LABEL DISTRIBUTION ---")
print(y_train.value_counts().sort_index())

print("\n--- TESTING LABEL DISTRIBUTION ---")
print(y_test.value_counts().sort_index())

# --------------------------------------------------
# 16. FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print(" PREPROCESSING COMPLETED SUCCESSFULLY")
print("========================================")

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print(
    f"Processed features: "
    f"{X_train_processed.shape[1]}"
)

print("\nGenerated files:")

print("models/preprocessor.joblib")

for filename in datasets:
    print(f"data/processed/{filename}.joblib")

print("data/processed/train_claim_ids.csv")
print("data/processed/test_claim_ids.csv")

print("\nData preprocessing completed!")