import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")


# ============================================================
# Load training data
# ============================================================

print("Loading training data...")

train = pd.read_csv(
    DATA_DIR / "train.csv",
    parse_dates=["TX_DATETIME"]
)


# ============================================================
# Create hourly windows
# ============================================================

print("Creating hourly merchant windows...")

train["hour_window"] = (
    train["TX_DATETIME"]
    .dt.floor("1h")
)


# ============================================================
# Aggregate terminal behavior
# ============================================================

hourly = (
    train
    .groupby(
        ["TERMINAL_ID", "hour_window"]
    )
    .agg(
        transaction_count=(
            "TRANSACTION_ID",
            "count"
        ),

        total_amount=(
            "TX_AMOUNT",
            "sum"
        ),

        average_amount=(
            "TX_AMOUNT",
            "mean"
        ),

        fraud_count=(
            "TX_FRAUD",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# Calculate historical baseline per terminal
# ============================================================

print("Calculating terminal baselines...")


baseline = (
    hourly
    .groupby("TERMINAL_ID")
    .agg(

        baseline_avg_transactions=(
            "transaction_count",
            "mean"
        ),

        baseline_std_transactions=(
            "transaction_count",
            "std"
        ),

        baseline_avg_amount=(
            "total_amount",
            "mean"
        ),

        baseline_std_amount=(
            "total_amount",
            "std"
        ),

        baseline_avg_transaction_amount=(
            "average_amount",
            "mean"
        ),

        baseline_transaction_windows=(
            "hour_window",
            "count"
        )
    )
    .reset_index()
)


# ============================================================
# Replace missing standard deviations
# ============================================================

baseline[
    "baseline_std_transactions"
] = (
    baseline[
        "baseline_std_transactions"
    ]
    .fillna(0)
)


baseline[
    "baseline_std_amount"
] = (
    baseline[
        "baseline_std_amount"
    ]
    .fillna(0)
)


# ============================================================
# Minimum-history indicator
# ============================================================

baseline["has_strong_history"] = (
    baseline[
        "baseline_transaction_windows"
    ] >= 10
).astype(int)


# ============================================================
# Save baseline
# ============================================================

OUTPUT = (
    DATA_DIR /
    "merchant_baseline.csv"
)


baseline.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n========================================")
print("MERCHANT BASELINE COMPLETE")
print("========================================")

print(
    "Unique terminals:",
    len(baseline)
)

print(
    "Terminals with >=10 historical windows:",
    baseline[
        "has_strong_history"
    ].sum()
)

print("\nBaseline statistics:")

print(
    baseline[
        [
            "baseline_avg_transactions",
            "baseline_std_transactions",
            "baseline_avg_amount",
            "baseline_std_amount",
            "baseline_transaction_windows"
        ]
    ].describe()
)

print("\nSaved to:")

print(OUTPUT)