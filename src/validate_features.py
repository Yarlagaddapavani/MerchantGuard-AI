import pandas as pd
from pathlib import Path


# ============================================================
# Load feature dataset
# ============================================================

FILE = Path(
    "data/processed/merchantguard_features.csv"
)

df = pd.read_csv(
    FILE,
    parse_dates=["TX_DATETIME"]
)


print("\n========================================")
print("MERCHANTGUARD FEATURE VALIDATION")
print("========================================")


# ============================================================
# 1. Dataset size
# ============================================================

print("\n========== DATASET SIZE ==========")

print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))


# ============================================================
# 2. Feature list
# ============================================================

print("\n========== ALL COLUMNS ==========")

for column in df.columns:
    print("-", column)


# ============================================================
# 3. Missing values
# ============================================================

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(
    missing[missing > 0]
)

if missing.sum() == 0:
    print("No missing values found.")


# ============================================================
# 4. Fraud distribution
# ============================================================

print("\n========== FRAUD DISTRIBUTION ==========")

print(
    df["TX_FRAUD"].value_counts()
)

print("\nFraud percentage:")

print(
    df["TX_FRAUD"].mean() * 100
)


# ============================================================
# 5. Terminal statistics
# ============================================================

print("\n========== TERMINAL STATISTICS ==========")

print(
    "Unique terminals:",
    df["TERMINAL_ID"].nunique()
)

print(
    "Average transactions per terminal:",
    len(df) / df["TERMINAL_ID"].nunique()
)


# ============================================================
# 6. Check feature ranges
# ============================================================

print("\n========== FEATURE RANGES ==========")

features = [
    "terminal_transaction_count",
    "terminal_fraud_count",
    "terminal_fraud_rate",
    "terminal_avg_amount",
    "amount_deviation",
    "seconds_since_previous",
    "transactions_last_1h",
    "transactions_last_24h",
    "amount_last_1h"
]


for feature in features:

    print(
        f"\n{feature}"
    )

    print(
        "  Min:",
        df[feature].min()
    )

    print(
        "  Max:",
        df[feature].max()
    )

    print(
        "  Mean:",
        df[feature].mean()
    )


# ============================================================
# 7. Check suspicious values
# ============================================================

print("\n========== NEGATIVE VALUES ==========")

for feature in features:

    negative_count = (
        df[feature] < 0
    ).sum()

    print(
        f"{feature}: {negative_count}"
    )


# ============================================================
# 8. Check date range
# ============================================================

print("\n========== DATE RANGE ==========")

print(
    "Start:",
    df["TX_DATETIME"].min()
)

print(
    "End:",
    df["TX_DATETIME"].max()
)


# ============================================================
# 9. Check target values
# ============================================================

print("\n========== TARGET VALUES ==========")

print(
    df["TX_FRAUD"].unique()
)


# ============================================================
# 10. Final message
# ============================================================

print("\n========================================")
print("VALIDATION COMPLETE")
print("========================================")