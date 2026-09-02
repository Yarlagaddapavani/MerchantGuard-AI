import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


# ============================================================
# Load trained models
# ============================================================

print("Loading models...")

fraud_model = joblib.load(
    MODEL_DIR / "random_forest_fraud_model.joblib"
)

anomaly_model = joblib.load(
    MODEL_DIR / "isolation_forest_model.joblib"
)

anomaly_scaler = joblib.load(
    MODEL_DIR / "anomaly_scaler.joblib"
)


# ============================================================
# Features used by Random Forest
# ============================================================

FRAUD_FEATURES = [
    "TX_AMOUNT",
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


# ============================================================
# Load TEST transactions
# ============================================================

print("Loading test transactions...")

test = pd.read_csv(
    DATA_DIR / "test.csv",
    parse_dates=["TX_DATETIME"]
)


# ============================================================
# Transaction-level fraud probability
# ============================================================

print("Calculating fraud probabilities...")

test["fraud_probability"] = (
    fraud_model
    .predict_proba(
        test[FRAUD_FEATURES]
    )[:, 1]
)


# ============================================================
# Create hourly terminal windows
# ============================================================

print("Creating terminal-hour windows...")

test["hour_window"] = (
    test["TX_DATETIME"]
    .dt.floor("1h")
)


hourly = (
    test
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

        average_fraud_probability=(
            "fraud_probability",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# Load TRAINING baseline
# ============================================================

print("Loading frozen merchant baseline...")

baseline = pd.read_csv(
    DATA_DIR / "merchant_baseline.csv"
)


# ============================================================
# Merge baseline
# ============================================================

hourly = hourly.merge(
    baseline,
    on="TERMINAL_ID",
    how="left"
)


# ============================================================
# Fill missing baseline values
# ============================================================

hourly[
    "baseline_avg_transactions"
] = hourly[
    "baseline_avg_transactions"
].fillna(1)


hourly[
    "baseline_std_transactions"
] = hourly[
    "baseline_std_transactions"
].fillna(0)


hourly[
    "baseline_avg_amount"
] = hourly[
    "baseline_avg_amount"
].fillna(
    hourly["total_amount"].median()
)


hourly[
    "baseline_std_amount"
] = hourly[
    "baseline_std_amount"
].fillna(0)


hourly[
    "baseline_transaction_windows"
] = hourly[
    "baseline_transaction_windows"
].fillna(0)


# ============================================================
# Behavioral deviation
# ============================================================

print("Calculating behavioral deviation...")

hourly["volume_deviation"] = (
    hourly["transaction_count"]
    /
    (
        hourly["baseline_avg_transactions"]
        + 0.01
    )
)


hourly["amount_deviation"] = (
    hourly["total_amount"]
    /
    (
        hourly["baseline_avg_amount"]
        + 0.01
    )
)


# ============================================================
# Z-score style deviation
# ============================================================

hourly["volume_zscore"] = (
    (
        hourly["transaction_count"]
        -
        hourly["baseline_avg_transactions"]
    )
    /
    (
        hourly["baseline_std_transactions"]
        + 0.1
    )
)


hourly["amount_zscore"] = (
    (
        hourly["total_amount"]
        -
        hourly["baseline_avg_amount"]
    )
    /
    (
        hourly["baseline_std_amount"]
        + 1
    )
)


# ============================================================
# Evidence strength
# ============================================================

hourly["evidence_strength"] = (
    1
    -
    np.exp(
        -hourly["transaction_count"]
        /
        3
    )
)


# ============================================================
# Isolation Forest anomaly score
# ============================================================

print("Calculating anomaly scores...")


ANOMALY_FEATURES = [
    "transaction_count",
    "total_amount",
    "average_amount"
]


X_anomaly = hourly[
    ANOMALY_FEATURES
]


X_scaled = anomaly_scaler.transform(
    X_anomaly
)


hourly["anomaly_prediction"] = (
    anomaly_model.predict(
        X_scaled
    ) == -1
).astype(int)


hourly["anomaly_score"] = (
    -anomaly_model.decision_function(
        X_scaled
    )
)


# ============================================================
# Normalize anomaly score
# ============================================================

anomaly_min = (
    hourly["anomaly_score"].min()
)

anomaly_max = (
    hourly["anomaly_score"].max()
)


hourly["anomaly_signal"] = (
    (
        hourly["anomaly_score"]
        -
        anomaly_min
    )
    /
    (
        anomaly_max
        -
        anomaly_min
        +
        0.000001
    )
)


# ============================================================
# Fraud probability signal
# ============================================================

hourly["fraud_signal"] = (
    hourly["average_fraud_probability"]
)


# ============================================================
# Behavioral signal
# ============================================================

volume_signal = (
    hourly["volume_zscore"]
    .clip(lower=0)
    / 5
).clip(upper=1)


amount_signal = (
    hourly["amount_zscore"]
    .clip(lower=0)
    / 5
).clip(upper=1)


hourly["behavior_signal"] = (
    0.6 * volume_signal
    +
    0.4 * amount_signal
)


## ============================================================
# MerchantGuard base risk
# ============================================================

hourly["base_risk"] = (
    0.55 * hourly["fraud_signal"]
    +
    0.20 * hourly["anomaly_signal"]
    +
    0.20 * hourly["behavior_signal"]
    +
    0.05 * hourly["evidence_strength"]
)


## ============================================================
# Risk score
# ============================================================

hourly["risk_score"] = (
    hourly["base_risk"]
    * 100
)


# ============================================================
# High-value behavior adjustment
# ============================================================

hourly.loc[
    hourly["amount_deviation"] > 5,
    "risk_score"
] += 5


# ============================================================
# Clip risk score
# ============================================================

hourly["risk_score"] = (
    hourly["risk_score"]
    .clip(0, 100)
)


# ============================================================
# Risk categories
# ============================================================

hourly["risk_level"] = pd.cut(
    hourly["risk_score"],
    bins=[
        -1,
        30,
        60,
        80,
        100
    ],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]
)


# ============================================================
# Estimated exposure
# ============================================================
#
# This is NOT used for prediction.
# It is a retrospective business metric.
#
# We use the actual fraud labels only here, AFTER scoring.
# ============================================================

test_labels = (
    test
    .groupby(
        [
            "TERMINAL_ID",
            "hour_window"
        ]
    )
    .agg(
        fraud_count=(
            "TX_FRAUD",
            "sum"
        )
    )
    .reset_index()
)


hourly = hourly.merge(
    test_labels,
    on=[
        "TERMINAL_ID",
        "hour_window"
    ],
    how="left"
)


hourly["fraud_count"] = (
    hourly["fraud_count"]
    .fillna(0)
)


hourly["actual_fraud_rate"] = (
    hourly["fraud_count"]
    /
    hourly["transaction_count"]
)


hourly["estimated_exposure"] = (
    hourly["total_amount"]
    *
    hourly["actual_fraud_rate"]
)


# ============================================================
# Recommended actions
# ============================================================

conditions = [
    hourly["risk_score"] < 30,

    hourly["risk_score"].between(
        30,
        60
    ),

    hourly["risk_score"].between(
        60,
        80
    ),

    hourly["risk_score"] >= 80
]


actions = [
    "Normal monitoring",

    "Enhanced monitoring",

    "Additional verification",

    "Merchant review + enhanced verification"
]


hourly["recommended_action"] = np.select(
    conditions,
    actions,
    default="Manual review"
)


# ============================================================
# Save
# ============================================================

OUTPUT = (
    DATA_DIR /
    "merchantguard_v3_results.csv"
)


hourly.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# Evaluation
# ============================================================

high_risk = hourly[
    hourly["risk_level"].isin(
        ["HIGH", "CRITICAL"]
    )
]


fraud_windows = (
    hourly["fraud_count"] > 0
)


fraud_alerts = (
    high_risk["fraud_count"] > 0
).sum()


print("\n========================================")
print("MERCHANTGUARD V3 COMPLETE")
print("========================================")


print(
    "\nTotal terminal-hour windows:",
    f"{len(hourly):,}"
)


print(
    "\nRisk distribution:"
)

print(
    hourly[
        "risk_level"
    ].value_counts()
)


print(
    "\nHIGH + CRITICAL alerts:",
    len(high_risk)
)


print(
    "Fraud-containing alerts:",
    fraud_alerts
)


if len(high_risk) > 0:

    precision = (
        fraud_alerts
        /
        len(high_risk)
    )

else:

    precision = 0


print(
    "Alert precision:",
    f"{precision:.4f}"
)


total_fraud_windows = (
    fraud_windows.sum()
)


if total_fraud_windows > 0:

    capture = (
        fraud_alerts
        /
        total_fraud_windows
    )

else:

    capture = 0


print(
    "Fraud-window capture:",
    f"{capture:.4f}"
)


# ============================================================
# Top alerts
# ============================================================

print(
    "\n========== TOP 15 ALERTS =========="
)


top = (
    hourly
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(15)
)


print(
    top[
        [
            "TERMINAL_ID",
            "hour_window",
            "transaction_count",
            "total_amount",
            "fraud_count",
            "fraud_signal",
            "anomaly_prediction",
            "volume_deviation",
            "amount_deviation",
            "evidence_strength",
            "risk_score",
            "risk_level",
            "recommended_action"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT
)