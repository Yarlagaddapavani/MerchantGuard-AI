import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "data/processed/merchantguard_transactions.csv"
)

OUTPUT_FILE = Path(
    "data/processed/merchantguard_features.csv"
)


# ============================================================
# Load data
# ============================================================

print("Loading transaction data...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["TX_DATETIME"]
)

print(f"Transactions loaded: {len(df):,}")


# ============================================================
# Sort by time
# ============================================================

df = df.sort_values(
    "TX_DATETIME"
).reset_index(drop=True)


# ============================================================
# Basic time features
# ============================================================

print("Creating time features...")

df["hour"] = df["TX_DATETIME"].dt.hour

df["day_of_week"] = (
    df["TX_DATETIME"].dt.dayofweek
)

df["date"] = (
    df["TX_DATETIME"].dt.date
)


# ============================================================
# Transaction amount
# ============================================================

df["amount_log"] = np.log1p(
    df["TX_AMOUNT"]
)


# ============================================================
# LEAKAGE-SAFE TERMINAL HISTORY
# ============================================================
#
# IMPORTANT:
# We use only information from transactions that happened
# BEFORE the current transaction.
#
# This prevents future information from influencing the
# current risk decision.
# ============================================================

print("Creating historical terminal features...")


# Previous transaction count for each terminal
df["terminal_transaction_count"] = (
    df.groupby("TERMINAL_ID")
      .cumcount()
)


# Previous fraud count
df["terminal_fraud_count"] = (
    df.groupby("TERMINAL_ID")["TX_FRAUD"]
      .transform(
          lambda x: x.shift(1)
                    .fillna(0)
                    .cumsum()
      )
)


# Previous fraud rate
df["terminal_fraud_rate"] = (
    df["terminal_fraud_count"]
    /
    df["terminal_transaction_count"].replace(
        0,
        np.nan
    )
)

df["terminal_fraud_rate"] = (
    df["terminal_fraud_rate"]
    .fillna(0)
)


# ============================================================
# Historical average transaction amount
# ============================================================

df["terminal_avg_amount"] = (
    df.groupby("TERMINAL_ID")["TX_AMOUNT"]
      .transform(
          lambda x:
          x.shift(1)
           .expanding()
           .mean()
           .reset_index(
               level=0,
               drop=True
           )
      )
)

df["terminal_avg_amount"] = (
    df["terminal_avg_amount"]
    .fillna(df["TX_AMOUNT"].median())
)


# ============================================================
# Amount deviation from historical behavior
# ============================================================

df["amount_deviation"] = (
    df["TX_AMOUNT"]
    /
    (df["terminal_avg_amount"] + 0.01)
)


# ============================================================
# Time since previous transaction
# ============================================================

print("Creating transaction velocity features...")


df["previous_transaction_time"] = (
    df.groupby("TERMINAL_ID")["TX_DATETIME"]
      .shift(1)
)


df["seconds_since_previous"] = (
    (
        df["TX_DATETIME"]
        -
        df["previous_transaction_time"]
    )
    .dt.total_seconds()
)


df["seconds_since_previous"] = (
    df["seconds_since_previous"]
    .fillna(86400)
)


# ============================================================
# Transactions in previous 1 hour
# ============================================================

print("Creating 1-hour transaction velocity...")


df["transactions_last_1h"] = (
    df.set_index("TX_DATETIME")
      .groupby("TERMINAL_ID")["TRANSACTION_ID"]
      .rolling("1h")
      .count()
      .reset_index(
          level=0,
          drop=True
      )
      .values
)


# The rolling window includes the current transaction.
# Remove it so the feature only represents previous activity.

df["transactions_last_1h"] = (
    df["transactions_last_1h"] - 1
).clip(lower=0)


# ============================================================
# Transactions in previous 24 hours
# ============================================================

print("Creating 24-hour transaction velocity...")


df["transactions_last_24h"] = (
    df.set_index("TX_DATETIME")
      .groupby("TERMINAL_ID")["TRANSACTION_ID"]
      .rolling("24h")
      .count()
      .reset_index(
          level=0,
          drop=True
      )
      .values
)


df["transactions_last_24h"] = (
    df["transactions_last_24h"] - 1
).clip(lower=0)


# ============================================================
# Previous 1-hour transaction amount
# ============================================================

print("Creating previous 1-hour transaction amount...")


df["amount_last_1h"] = (
    df.set_index("TX_DATETIME")
      .groupby("TERMINAL_ID")["TX_AMOUNT"]
      .rolling("1h")
      .sum()
      .reset_index(
          level=0,
          drop=True
      )
      .values
)


# Remove current transaction amount

df["amount_last_1h"] = (
    df["amount_last_1h"]
    -
    df["TX_AMOUNT"]
)

df["amount_last_1h"] = (
    df["amount_last_1h"]
    .clip(lower=0)
)


# ============================================================
# Clean values
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


numeric_columns = (
    df.select_dtypes(
        include=["number"]
    ).columns
)


df[numeric_columns] = (
    df[numeric_columns]
    .fillna(0)
)


# ============================================================
# Remove helper column
# ============================================================

df = df.drop(
    columns=[
        "previous_transaction_time"
    ]
)


# ============================================================
# Save feature dataset
# ============================================================

print("Saving feature dataset...")


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n========================================")
print("FEATURE ENGINEERING COMPLETE")
print("========================================")

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print("\nFeatures created:")

features = [
    "hour",
    "day_of_week",
    "amount_log",
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
    print(f" - {feature}")


print("\nSaved to:")

print(OUTPUT_FILE)