import pandas as pd
import numpy as np

from pathlib import Path


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")


# ============================================================
# Load hourly anomaly data
# ============================================================

print("Loading anomaly data...")

validation = pd.read_csv(
    DATA_DIR / "validation_hourly_anomalies.csv",
    parse_dates=["hour_window"]
)

test = pd.read_csv(
    DATA_DIR / "test_hourly_anomalies.csv",
    parse_dates=["hour_window"]
)


# ============================================================
# Create risk features
# ============================================================

def create_risk_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Fraud-rate deviation
    # --------------------------------------------------------

    # Add smoothing to avoid unstable rates for tiny samples
    df["smoothed_fraud_rate"] = (
        df["fraud_count"] + 1
    ) / (
        df["transaction_count"] + 10
    )


    # --------------------------------------------------------
    # Transaction volume score
    # --------------------------------------------------------

    volume_mean = (
        df["transaction_count"]
        .mean()
    )

    volume_std = (
        df["transaction_count"]
        .std()
        + 0.001
    )

    df["volume_zscore"] = (
        df["transaction_count"]
        - volume_mean
    ) / volume_std


    # --------------------------------------------------------
    # Amount score
    # --------------------------------------------------------

    amount_mean = (
        df["total_amount"]
        .mean()
    )

    amount_std = (
        df["total_amount"]
        .std()
        + 0.001
    )

    df["amount_zscore"] = (
        df["total_amount"]
        - amount_mean
    ) / amount_std


    # --------------------------------------------------------
    # Normalize anomaly score
    # --------------------------------------------------------

    anomaly_min = (
        df["anomaly_score"].min()
    )

    anomaly_max = (
        df["anomaly_score"].max()
    )

    df["anomaly_normalized"] = (
        (
            df["anomaly_score"]
            - anomaly_min
        )
        /
        (
            anomaly_max
            - anomaly_min
            + 0.000001
        )
    )


    # --------------------------------------------------------
    # Normalize fraud rate
    # --------------------------------------------------------

    fraud_min = (
        df["smoothed_fraud_rate"].min()
    )

    fraud_max = (
        df["smoothed_fraud_rate"].max()
    )

    df["fraud_rate_normalized"] = (
        (
            df["smoothed_fraud_rate"]
            - fraud_min
        )
        /
        (
            fraud_max
            - fraud_min
            + 0.000001
        )
    )


    # --------------------------------------------------------
    # Normalize volume
    # --------------------------------------------------------

    volume_positive = (
        df["volume_zscore"]
        .clip(lower=0)
    )

    df["volume_normalized"] = (
        volume_positive
        /
        (
            volume_positive.max()
            + 0.000001
        )
    )


    # --------------------------------------------------------
    # Normalize amount
    # --------------------------------------------------------

    amount_positive = (
        df["amount_zscore"]
        .clip(lower=0)
    )

    df["amount_normalized"] = (
        amount_positive
        /
        (
            amount_positive.max()
            + 0.000001
        )
    )


    # ========================================================
    # MerchantGuard Risk Score
    # ========================================================

    df["risk_score"] = (
        0.40
        * df["anomaly_normalized"]
        +
        0.30
        * df["fraud_rate_normalized"]
        +
        0.20
        * df["volume_normalized"]
        +
        0.10
        * df["amount_normalized"]
    )


    # Convert to 0–100
    df["risk_score"] = (
        df["risk_score"]
        * 100
    )


    # ========================================================
    # Risk category
    # ========================================================

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[
            -1,
            30,
            60,
            80,
            101
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ]
    )


    # ========================================================
    # Estimated exposure
    # ========================================================

    # This is a prototype estimate, not a real financial-loss
    # calculation.

    df["estimated_exposure"] = (
        df["total_amount"]
        *
        df["smoothed_fraud_rate"]
    )


    # ========================================================
    # Recommended action
    # ========================================================

    conditions = [
        df["risk_score"] < 30,
        df["risk_score"].between(30, 60),
        df["risk_score"].between(60, 80),
        df["risk_score"] >= 80
    ]

    actions = [
        "Normal monitoring",
        "Enhanced monitoring",
        "Additional verification",
        "Merchant review + enhanced verification"
    ]

    df["recommended_action"] = np.select(
        conditions,
        actions,
        default="Manual review"
    )


    return df


# ============================================================
# Process datasets
# ============================================================

print("Calculating validation risk scores...")

validation_risk = create_risk_features(
    validation
)


print("Calculating test risk scores...")

test_risk = create_risk_features(
    test
)


# ============================================================
# Save results
# ============================================================

validation_risk.to_csv(
    DATA_DIR /
    "validation_risk_scores.csv",
    index=False
)

test_risk.to_csv(
    DATA_DIR /
    "test_risk_scores.csv",
    index=False
)


# ============================================================
# Display summary
# ============================================================

print("\n========================================")
print("MERCHANTGUARD RISK ENGINE COMPLETE")
print("========================================")


print("\n========== VALIDATION RISK LEVELS ==========")

print(
    validation_risk[
        "risk_level"
    ].value_counts()
)


print("\n========== TEST RISK LEVELS ==========")

print(
    test_risk[
        "risk_level"
    ].value_counts()
)


print("\n========== TOP TEST ALERTS ==========")

top_alerts = (
    test_risk
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(10)
)


print(
    top_alerts[
        [
            "TERMINAL_ID",
            "hour_window",
            "transaction_count",
            "fraud_count",
            "fraud_rate",
            "risk_score",
            "risk_level",
            "estimated_exposure",
            "recommended_action"
        ]
    ].to_string(
        index=False
    )
)


print("\nFiles created:")

print(
    " - data/processed/validation_risk_scores.csv"
)

print(
    " - data/processed/test_risk_scores.csv"
)