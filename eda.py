import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_FILE = Path("data/synthetic_claims.csv")
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "review_label"

NUMERICAL_FEATURES = [
    "patient_age",
    "claim_amount",
    "procedure_complexity",
    "length_of_stay",
    "provider_claim_count",
    "provider_avg_amount",
    "repeat_billing_count"
]

CATEGORICAL_FEATURES = [
    "provider_id",
    "procedure_code",
    "diagnosis_code",
    "scenario_type"
]

TEXT_FEATURES = [
    "clinical_note",
    "billed_description"
]

REQUIRED_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + TEXT_FEATURES
    + ["claim_id", TARGET]
)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\n========================================")
print(" HEALTHCARE CLAIMS EXPLORATORY ANALYSIS")
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
# 3. VALIDATE DATASET
# --------------------------------------------------

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("Dataset is empty.")

if df[TARGET].isnull().any():
    raise ValueError("Target contains missing values.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

print("\nDataset validation completed.")


# --------------------------------------------------
# 4. BASIC DATASET INFORMATION
# --------------------------------------------------

print("\n--- DATASET INFORMATION ---")

print("\nColumn data types:")
print(df.dtypes)

print("\nDataset statistics:")
print(df.describe(include="all").transpose())

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 5. DUPLICATE ANALYSIS
# --------------------------------------------------

duplicate_rows = df.duplicated().sum()
duplicate_ids = df["claim_id"].duplicated().sum()

print("\n--- DUPLICATE ANALYSIS ---")
print(f"Duplicate rows: {duplicate_rows}")
print(f"Duplicate claim IDs: {duplicate_ids}")


# --------------------------------------------------
# 6. TARGET DISTRIBUTION
# --------------------------------------------------

print("\n--- TARGET DISTRIBUTION ---")

label_counts = (
    df[TARGET]
    .value_counts()
    .sort_index()
)

label_percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print(label_counts)
print("\nPercentage distribution:")
print(label_percentages.round(2))

plt.figure(figsize=(7, 5))

plt.bar(
    ["Normal (0)", "Review (1)"],
    [
        label_counts.get(0, 0),
        label_counts.get(1, 0)
    ]
)

plt.title("Claim Review Label Distribution")
plt.xlabel("Claim Category")
plt.ylabel("Number of Claims")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "target_distribution.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# 7. MISSING VALUE ANALYSIS
# --------------------------------------------------

missing_summary = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (
        df.isnull().mean() * 100
    )
})

missing_summary = missing_summary.sort_values(
    by="missing_percentage",
    ascending=False
)

missing_summary.to_csv(
    OUTPUT_DIR / "missing_values.csv"
)

print("\n--- MISSING VALUE SUMMARY ---")
print(missing_summary)


# --------------------------------------------------
# 8. NUMERICAL FEATURE ANALYSIS
# --------------------------------------------------

available_numeric = [
    col for col in NUMERICAL_FEATURES
    if pd.api.types.is_numeric_dtype(df[col])
]

print("\n--- NUMERICAL FEATURE STATISTICS ---")

print(
    df[available_numeric]
    .describe()
    .round(2)
)

for feature in available_numeric:

    plt.figure(figsize=(8, 5))

    for label in sorted(df[TARGET].unique()):

        values = df.loc[
            df[TARGET] == label,
            feature
        ].dropna()

        plt.hist(
            values,
            bins=30,
            alpha=0.5,
            label=f"Review label {label}"
        )

    plt.title(f"{feature} Distribution by Review Label")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}_distribution.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 9. NUMERICAL CORRELATION ANALYSIS
# --------------------------------------------------

correlation = df[
    available_numeric + [TARGET]
].corr(numeric_only=True)

print("\n--- CORRELATION WITH TARGET ---")

print(
    correlation[TARGET]
    .sort_values(ascending=False)
)

correlation.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

plt.title("Numerical Feature Correlation")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_matrix.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 10. CATEGORICAL FEATURE ANALYSIS
# --------------------------------------------------

print("\n--- CATEGORICAL FEATURE ANALYSIS ---")

for feature in CATEGORICAL_FEATURES:

    print(f"\n{feature} distribution:")

    print(
        df[feature]
        .value_counts(dropna=False)
        .head(10)
    )

    top_values = (
        df[feature]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(9, 5))

    top_values.plot(kind="bar")

    plt.title(f"Top 10 Values: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Claims")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}_distribution.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 11. PROVIDER REVIEW ANALYSIS
# --------------------------------------------------

provider_analysis = (
    df.groupby("provider_id")
    .agg(
        total_claims=(TARGET, "count"),
        review_claims=(TARGET, "sum"),
        review_rate=(TARGET, "mean")
    )
)

provider_analysis["review_rate"] *= 100

provider_analysis = provider_analysis.sort_values(
    by="review_rate",
    ascending=False
)

provider_analysis.to_csv(
    OUTPUT_DIR / "provider_review_analysis.csv"
)

print("\n--- PROVIDER REVIEW ANALYSIS ---")

print(
    provider_analysis.head(10).round(2)
)


# --------------------------------------------------
# 12. TEXT FEATURE ANALYSIS
# --------------------------------------------------

print("\n--- TEXT FEATURE ANALYSIS ---")

for feature in TEXT_FEATURES:

    text_lengths = (
        df[feature]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df[f"{feature}_length"] = text_lengths

    print(f"\n{feature}:")
    print(text_lengths.describe().round(2))

    plt.figure(figsize=(8, 5))

    for label in sorted(df[TARGET].unique()):

        values = df.loc[
            df[TARGET] == label,
            f"{feature}_length"
        ]

        plt.hist(
            values,
            bins=30,
            alpha=0.5,
            label=f"Review label {label}"
        )

    plt.title(f"{feature} Length Distribution")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}_length.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 13. SCENARIO AND LABEL RELATIONSHIP
# --------------------------------------------------

scenario_analysis = pd.crosstab(
    df["scenario_type"],
    df[TARGET],
    margins=True
)

scenario_analysis.to_csv(
    OUTPUT_DIR / "scenario_label_analysis.csv"
)

print("\n--- SCENARIO VS REVIEW LABEL ---")
print(scenario_analysis)


# --------------------------------------------------
# 14. SAVE ANALYSIS SUMMARY
# --------------------------------------------------

summary = []

summary.append("HEALTHCARE CLAIMS EDA SUMMARY")
summary.append("=" * 40)

summary.append(f"Total records: {len(df)}")
summary.append(f"Total columns: {len(df.columns)}")

summary.append(
    f"Duplicate rows: {duplicate_rows}"
)

summary.append(
    f"Duplicate claim IDs: {duplicate_ids}"
)

summary.append("\nTarget distribution:")

summary.append(
    label_counts.to_string()
)

summary.append("\nMissing values:")

summary.append(
    missing_summary.to_string()
)

summary.append("\nProvider review analysis:")

summary.append(
    provider_analysis.head(10).round(2).to_string()
)

with open(
    OUTPUT_DIR / "eda_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(summary))


# --------------------------------------------------
# 15. FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print(" EDA COMPLETED SUCCESSFULLY")
print("========================================")

print(f"\nAnalysis files saved in: {OUTPUT_DIR}")

print("\nGenerated reports include:")

print("- Target distribution")
print("- Missing value analysis")
print("- Numerical feature distributions")
print("- Correlation analysis")
print("- Categorical feature distributions")
print("- Provider review analysis")
print("- Text length analysis")
print("- Scenario and label analysis")
print("- EDA summary")

print("\nExploratory data analysis completed!")