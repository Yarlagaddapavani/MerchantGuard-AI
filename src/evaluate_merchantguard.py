import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Load MerchantGuard results
# ============================================================

FILE = Path(
    "data/processed/merchantguard_v3_results.csv"
)

df = pd.read_csv(FILE)


print("\n========================================")
print("MERCHANTGUARD FINAL EVALUATION")
print("========================================")


# ============================================================
# Basic statistics
# ============================================================

print("\n========== TEST DATA ==========")

print(
    "Terminal-hour windows:",
    f"{len(df):,}"
)

print(
    "Fraud-containing windows:",
    int(
        (df["fraud_count"] > 0).sum()
    )
)


# ============================================================
# Risk levels
# ============================================================

print("\n========== RISK LEVELS ==========")

risk_levels = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]

for level in risk_levels:

    subset = df[
        df["risk_level"] == level
    ]

    if len(subset) == 0:

        print(
            f"{level}: 0"
        )

        continue

    fraud_windows = (
        subset["fraud_count"] > 0
    ).sum()

    precision = (
        fraud_windows
        /
        len(subset)
    )

    print(
        f"{level}: {len(subset):,}"
        f" | Fraud windows: {fraud_windows:,}"
        f" | Precision: {precision:.4f}"
    )


# ============================================================
# High + Critical
# ============================================================

high_risk = df[
    df["risk_level"].isin(
        ["HIGH", "CRITICAL"]
    )
]


print(
    "\n========== HIGH + CRITICAL =========="
)

print(
    "Alerts:",
    len(high_risk)
)


fraud_alerts = (
    high_risk["fraud_count"] > 0
).sum()


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


# ============================================================
# Fraud capture
# ============================================================

total_fraud_windows = (
    df["fraud_count"] > 0
).sum()


if total_fraud_windows > 0:

    fraud_capture = (
        fraud_alerts
        /
        total_fraud_windows
    )

else:

    fraud_capture = 0


print(
    "Fraud-window capture:",
    f"{fraud_capture:.4f}"
)


# ============================================================
# Exposure
# ============================================================

print(
    "\n========== FINANCIAL EXPOSURE =========="
)

print(
    "Total exposure in HIGH/CRITICAL:",
    high_risk[
        "estimated_exposure"
    ].sum()
)


print(
    "Average exposure per alert:",
    high_risk[
        "estimated_exposure"
    ].mean()
)


# ============================================================
# Anomaly performance
# ============================================================

print(
    "\n========== ANOMALY SIGNAL =========="
)

anomalies = df[
    df["anomaly_prediction"] == 1
]


anomaly_fraud = (
    anomalies["fraud_count"] > 0
).sum()


print(
    "Anomalous windows:",
    len(anomalies)
)


print(
    "Anomalous windows with fraud:",
    anomaly_fraud
)


if len(anomalies) > 0:

    anomaly_precision = (
        anomaly_fraud
        /
        len(anomalies)
    )

else:

    anomaly_precision = 0


print(
    "Anomaly precision:",
    f"{anomaly_precision:.4f}"
)


# ============================================================
# Risk score comparison
# ============================================================

print(
    "\n========== RISK SCORE COMPARISON =========="
)


fraud_windows = df[
    df["fraud_count"] > 0
]

normal_windows = df[
    df["fraud_count"] == 0
]


print(
    "Average risk score - fraud windows:",
    f"{fraud_windows['risk_score'].mean():.2f}"
)


print(
    "Average risk score - normal windows:",
    f"{normal_windows['risk_score'].mean():.2f}"
)


# ============================================================
# Top alerts
# ============================================================

print(
    "\n========== TOP 10 ALERTS =========="
)


top = (
    df
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(10)
)


print(
    top[
        [
            "TERMINAL_ID",
            "hour_window",
            "transaction_count",
            "total_amount",
            "fraud_count",
            "anomaly_prediction",
            "volume_deviation",
            "amount_deviation",
            "evidence_strength",
            "risk_score",
            "risk_level"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n========================================"
)

print(
    "EVALUATION COMPLETE"
)

print(
    "========================================"
)