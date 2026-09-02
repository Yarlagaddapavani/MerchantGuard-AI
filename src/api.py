import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


# ============================================================
# Create FastAPI application
# ============================================================

app = FastAPI(
    title="MerchantGuard AI",
    version="1.0.0",
    description="AI-powered merchant fraud and risk detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load models
# ============================================================

print("Loading MerchantGuard models...")

fraud_model = joblib.load(
    MODEL_DIR / "random_forest_fraud_model.joblib"
)

anomaly_model = joblib.load(
    MODEL_DIR / "isolation_forest_model.joblib"
)

anomaly_scaler = joblib.load(
    MODEL_DIR / "anomaly_scaler.joblib"
)

baseline = pd.read_csv(
    DATA_DIR / "merchant_baseline.csv"
)

print("Models loaded successfully.")


# ============================================================
# Fraud model features
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
# API request model
# ============================================================

class TransactionRequest(BaseModel):

    transaction_amount: float
    hour: int
    day_of_week: int

    terminal_transaction_count: int = 0
    terminal_fraud_count: int = 0
    terminal_fraud_rate: float = 0.0
    terminal_avg_amount: float = 50.0

    seconds_since_previous: float = 3600.0

    transactions_last_1h: int = 0
    transactions_last_24h: int = 0

    amount_last_1h: float = 0.0


# ============================================================
# Health check
# ============================================================

@app.get("/")
def root():

    return {
        "application": "MerchantGuard AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "fraud_model": "loaded",
        "anomaly_model": "loaded",
        "baseline": "loaded"
    }

# ============================================================
# Analytics endpoint
# ============================================================

@app.get("/analytics")
def analytics():

    try:

        results_file = (
            DATA_DIR /
            "merchantguard_v3_results.csv"
        )

        df = pd.read_csv(results_file)

        total_windows = len(df)

        fraud_windows = int(
            (df["fraud_count"] > 0).sum()
        )

        high_critical = df[
            df["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            )
        ]

        high_alerts = len(high_critical)

        fraud_alerts = int(
            (
                high_critical["fraud_count"] > 0
            ).sum()
        )

        alert_precision = (
            fraud_alerts / high_alerts
            if high_alerts > 0
            else 0
        )

        fraud_capture = (
            fraud_alerts / fraud_windows
            if fraud_windows > 0
            else 0
        )

        financial_exposure = float(
            high_critical[
                "estimated_exposure"
            ].sum()
        )

        risk_distribution = (
            df["risk_level"]
            .value_counts()
            .to_dict()
        )

        return {

            "total_windows":
                total_windows,

            "fraud_windows":
                fraud_windows,

            "high_alerts":
                high_alerts,

            "fraud_alerts":
                fraud_alerts,

            "alert_precision":
                round(alert_precision, 4),

            "fraud_capture":
                round(fraud_capture, 4),

            "financial_exposure":
                round(financial_exposure, 2),

            "risk_distribution": {

                "LOW":
                    int(
                        risk_distribution.get(
                            "LOW",
                            0
                        )
                    ),

                "MEDIUM":
                    int(
                        risk_distribution.get(
                            "MEDIUM",
                            0
                        )
                    ),

                "HIGH":
                    int(
                        risk_distribution.get(
                            "HIGH",
                            0
                        )
                    ),

                "CRITICAL":
                    int(
                        risk_distribution.get(
                            "CRITICAL",
                            0
                        )
                    )
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



# ============================================================
# Top risk alerts endpoint
# ============================================================

@app.get("/alerts")
def alerts():

    try:

        results_file = (
            DATA_DIR /
            "merchantguard_v3_results.csv"
        )

        df = pd.read_csv(results_file)

        # Keep only HIGH and CRITICAL alerts
        alerts_df = df[
            df["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            )
        ].copy()

        # Sort highest risk first
        alerts_df = alerts_df.sort_values(
            by=["risk_score"],
            ascending=False
        ).head(10)

        alert_list = []

        for _, row in alerts_df.iterrows():

            # Try to find the terminal column
            terminal = "Unknown"

            for column in [
                "TERMINAL_ID",
                "terminal_id",
                "Terminal_ID"
            ]:
                if column in df.columns:
                    terminal = str(row[column])
                    break

            # Try to find a time column
            timestamp = "Test window"

            for column in [
                "TX_DATETIME",
                "timestamp",
                "window_start",
                "hour_window",
                "datetime"
            ]:
                if column in df.columns:
                    timestamp = str(row[column])
                    break

            fraud_probability = float(
                row.get(
                    "average_fraud_probability",
                    0
                )
            )

            amount = float(
                row.get(
                    "total_amount",
                    0
                )
            )

            alert_list.append({

                "timestamp":
                    timestamp,

                "terminal":
                    terminal,

                "risk_level":
                    str(row["risk_level"]),

                "risk_score":
                    round(
                        float(row["risk_score"]),
                        2
                    ),

                "fraud_probability":
                    round(
                        fraud_probability * 100,
                        2
                    ),

                "amount":
                    round(
                        amount,
                        2
                    ),

                "fraud_count":
                    int(
                        row.get(
                            "fraud_count",
                            0
                        )
                    ),

                "recommended_action":
                    str(
                        row.get(
                            "recommended_action",
                            "Review alert"
                        )
                    )
            })

        return {
            "alerts": alert_list
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ============================================================
# Fraud prediction endpoint
# ============================================================

@app.post("/predict")
def predict(transaction: TransactionRequest):

    try:

        amount = transaction.transaction_amount

        if amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="Transaction amount must be greater than 0."
            )

        if transaction.hour < 0 or transaction.hour > 23:
            raise HTTPException(
                status_code=400,
                detail="Hour must be between 0 and 23."
            )

        if transaction.day_of_week < 0 or transaction.day_of_week > 6:
            raise HTTPException(
                status_code=400,
                detail="day_of_week must be between 0 and 6."
            )


        # ----------------------------------------------------
        # Derived features
        # ----------------------------------------------------

        amount_log = np.log1p(amount)

        amount_deviation = (
            amount /
            (
                transaction.terminal_avg_amount
                + 0.01
            )
        )


        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        features = pd.DataFrame(
            [{
                "TX_AMOUNT": amount,

                "hour": transaction.hour,

                "day_of_week":
                    transaction.day_of_week,

                "amount_log":
                    amount_log,

                "terminal_transaction_count":
                    transaction.terminal_transaction_count,

                "terminal_fraud_count":
                    transaction.terminal_fraud_count,

                "terminal_fraud_rate":
                    transaction.terminal_fraud_rate,

                "terminal_avg_amount":
                    transaction.terminal_avg_amount,

                "amount_deviation":
                    amount_deviation,

                "seconds_since_previous":
                    transaction.seconds_since_previous,

                "transactions_last_1h":
                    transaction.transactions_last_1h,

                "transactions_last_24h":
                    transaction.transactions_last_24h,

                "amount_last_1h":
                    transaction.amount_last_1h
            }]
        )


        # ----------------------------------------------------
        # Fraud probability
        # ----------------------------------------------------

        fraud_probability = float(
            fraud_model.predict_proba(
                features[FRAUD_FEATURES]
            )[0, 1]
        )


        # ----------------------------------------------------
        # Anomaly features
        # ----------------------------------------------------

        anomaly_input = pd.DataFrame(
            [{
                "transaction_count":
                    transaction.terminal_transaction_count,

                "total_amount":
                    amount,

                "average_amount":
                    amount
            }]
        )


        anomaly_scaled = (
            anomaly_scaler.transform(
                anomaly_input
            )
        )


        anomaly_prediction = int(
            anomaly_model.predict(
                anomaly_scaled
            )[0] == -1
        )


        anomaly_raw = float(
            -anomaly_model.decision_function(
                anomaly_scaled
            )[0]
        )


        # ----------------------------------------------------
        # Risk signals
        # ----------------------------------------------------

        fraud_signal = fraud_probability

        anomaly_signal = float(
            np.clip(
                anomaly_raw,
                0,
                1
            )
        )


        evidence_strength = float(
            1 -
            np.exp(
                -transaction.terminal_transaction_count
                / 3
            )
        )


        # ----------------------------------------------------
        # Behavioral signal
        # ----------------------------------------------------

        volume_deviation = (
            transaction.terminal_transaction_count
            /
            (
                transaction.terminal_transaction_count
                + 1
            )
        )


        amount_ratio = (
            amount /
            (
                transaction.terminal_avg_amount
                + 0.01
            )
        )


        amount_signal = float(
            np.clip(
                (amount_ratio - 1) / 5,
                0,
                1
            )
        )


        behavior_signal = (
            0.6 * volume_deviation
            +
            0.4 * amount_signal
        )


        # ----------------------------------------------------
        # MerchantGuard risk score
        # ----------------------------------------------------

        base_risk = (
            0.40 * fraud_signal
            +
            0.30 * anomaly_signal
            +
            0.20 * behavior_signal
            +
            0.10 * evidence_strength
        )


        risk_score = (
            base_risk
            *
            (
                0.50
                +
                0.50 *
                evidence_strength
            )
            *
            100
        )


        # High-value transaction adjustment

        if amount_ratio > 5:

            risk_score += 5


        risk_score = float(
            np.clip(
                risk_score,
                0,
                100
            )
        )


        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if risk_score < 30:

            risk_level = "LOW"

            action = "Normal monitoring"

        elif risk_score < 60:

            risk_level = "MEDIUM"

            action = "Enhanced monitoring"

        elif risk_score < 80:

            risk_level = "HIGH"

            action = "Additional verification"

        else:

            risk_level = "CRITICAL"

            action = (
                "Merchant review + "
                "enhanced verification"
            )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "transaction_amount":
                round(amount, 2),

            "fraud_probability":
                round(fraud_probability, 4),

            "fraud_probability_percent":
                round(
                    fraud_probability * 100,
                    2
                ),

            "anomaly_detected":
                bool(anomaly_prediction),

            "anomaly_score":
                round(anomaly_raw, 4),

            "risk_score":
                round(risk_score, 2),

            "risk_level":
                risk_level,

            "recommended_action":
                action,

            "signals": {

                "fraud_signal":
                    round(fraud_signal, 4),

                "anomaly_signal":
                    round(anomaly_signal, 4),

                "behavior_signal":
                    round(behavior_signal, 4),

                "evidence_strength":
                    round(
                        evidence_strength,
                        4
                    )
            }

        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )