import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# Load data
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    parse_dates=["TX_DATETIME"]
)

validation_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    parse_dates=["TX_DATETIME"]
)

test_df = pd.read_csv(
    DATA_DIR / "test.csv",
    parse_dates=["TX_DATETIME"]
)


# ============================================================
# Create hourly terminal aggregates
# ============================================================

def create_hourly_features(df):

    df = df.copy()

    df["hour_window"] = (
        df["TX_DATETIME"]
        .dt.floor("1h")
    )

    hourly = (
        df.groupby(
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

    # Fraud rate within the hour
    hourly["fraud_rate"] = (
        hourly["fraud_count"]
        /
        hourly["transaction_count"]
    )

    return hourly


print("Creating hourly terminal features...")

train_hourly = create_hourly_features(
    train_df
)

validation_hourly = create_hourly_features(
    validation_df
)

test_hourly = create_hourly_features(
    test_df
)


# # ============================================================
# Select anomaly features
# ============================================================

ANOMALY_FEATURES = [
    "transaction_count",
    "total_amount",
    "average_amount"
]



X_train = train_hourly[
    ANOMALY_FEATURES
]

X_validation = validation_hourly[
    ANOMALY_FEATURES
]

X_test = test_hourly[
    ANOMALY_FEATURES
]


# ============================================================
# Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# Train Isolation Forest
# ============================================================

print("Training Isolation Forest...")


model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train_scaled
)


print("Isolation Forest training complete.")


# ============================================================
# Generate anomaly scores
# ============================================================

def add_anomaly_scores(
    dataframe,
    scaled_features
):

    dataframe = dataframe.copy()

    predictions = model.predict(
        scaled_features
    )

    raw_scores = model.decision_function(
        scaled_features
    )

    dataframe["anomaly_prediction"] = (
        predictions == -1
    ).astype(int)

    dataframe["anomaly_score"] = (
        -raw_scores
    )

    return dataframe


validation_hourly = add_anomaly_scores(
    validation_hourly,
    X_validation_scaled
)

test_hourly = add_anomaly_scores(
    test_hourly,
    X_test_scaled
)


# ============================================================
# Save model and scaler
# ============================================================

joblib.dump(
    model,
    MODEL_DIR /
    "isolation_forest_model.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR /
    "anomaly_scaler.joblib"
)


# ============================================================
# Save anomaly datasets
# ============================================================

validation_hourly.to_csv(
    DATA_DIR /
    "validation_hourly_anomalies.csv",
    index=False
)

test_hourly.to_csv(
    DATA_DIR /
    "test_hourly_anomalies.csv",
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n========================================")
print("MERCHANT ANOMALY DETECTION COMPLETE")
print("========================================")


print(
    "\nTraining hourly windows:",
    f"{len(train_hourly):,}"
)

print(
    "Validation hourly windows:",
    f"{len(validation_hourly):,}"
)

print(
    "Test hourly windows:",
    f"{len(test_hourly):,}"
)


print(
    "\nValidation anomalies:",
    int(
        validation_hourly[
            "anomaly_prediction"
        ].sum()
    )
)

print(
    "Test anomalies:",
    int(
        test_hourly[
            "anomaly_prediction"
        ].sum()
    )
)


print(
    "\nModels saved:"
)

print(
    " - models/isolation_forest_model.joblib"
)

print(
    " - models/anomaly_scaler.joblib"
)