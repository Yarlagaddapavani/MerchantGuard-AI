import pandas as pd
from pathlib import Path


# ============================================================
# Load test risk scores
# ============================================================

FILE = Path(
    "data/processed/test_risk_scores.csv"
)

df = pd.read_csv(
    FILE,
    parse_dates=["hour_window"]
)


print("\n========================================")
print("MERCHANTGUARD ALERT ANALYSIS")
print("========================================")


# ============================================================
# Basic statistics
# ============================================================

print("\n========== RISK SCORE STATISTICS ==========")

print(
    df["risk_score"].describe()
)


# ============================================================
# Risk level distribution
# ============================================================

print("\n========== RISK LEVEL DISTRIBUTION ==========")

print(
    df["risk_level"].value_counts()
)


# ============================================================
# Anomaly vs fraud
# ============================================================

print("\n========== ANOMALY ANALYSIS ==========")

anomalies = df[
    df["anomaly_prediction"] == 1
]

print(
    "Total anomalies:",
    len(anomalies)
)

print(
    "Anomalies containing fraud:",
    len(
        anomalies[
            anomalies["fraud_count"] > 0
        ]
    )
)

print(
    "Anomaly fraud rate:",
    anomalies["fraud_rate"].mean()
)


# ============================================================
# Fraud-containing windows
# ============================================================

print("\n========== FRAUD WINDOWS ==========")

fraud_windows = df[
    df["fraud_count"] > 0
]

print(
    "Total fraud-containing windows:",
    len(fraud_windows)
)

print(
    "Average fraud rate:",
    fraud_windows["fraud_rate"].mean()
)

print(
    "Average transaction count:",
    fraud_windows[
        "transaction_count"
    ].mean()
)


# ============================================================
# Top 20 risk alerts
# ============================================================

print("\n========== TOP 20 RISK ALERTS ==========")

top_alerts = (
    df
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(20)
)


columns = [
    "TERMINAL_ID",
    "hour_window",
    "transaction_count",
    "fraud_count",
    "fraud_rate",
    "total_amount",
    "anomaly_prediction",
    "anomaly_score",
    "risk_score",
    "risk_level"
]


print(
    top_alerts[
        columns
    ].to_string(
        index=False
    )
)


# ============================================================
# High and critical alerts
# ============================================================

print("\n========== HIGH / CRITICAL ALERTS ==========")

high_risk = df[
    df["risk_level"].isin(
        ["HIGH", "CRITICAL"]
    )
]


print(
    "Total high/critical:",
    len(high_risk)
)

print(
    "Containing fraud:",
    len(
        high_risk[
            high_risk["fraud_count"] > 0
        ]
    )
)


# ============================================================
# Estimated exposure
# ============================================================

print(
    "\n========== EXPOSURE =========="
)

print(
    "Total estimated exposure:",
    high_risk[
        "estimated_exposure"
    ].sum()
)

print(
    "Average high-risk exposure:",
    high_risk[
        "estimated_exposure"
    ].mean()
)


print(
    "\n========================================"
)

print(
    "ALERT ANALYSIS COMPLETE"
)

print(
    "========================================"
)