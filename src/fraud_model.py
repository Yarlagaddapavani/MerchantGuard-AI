import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Load datasets
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(
    DATA_DIR / "train.csv"
)

validation_df = pd.read_csv(
    DATA_DIR / "validation.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "test.csv"
)


print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Validation rows: {len(validation_df):,}"
)

print(
    f"Test rows: {len(test_df):,}"
)


# ============================================================
# Features
# ============================================================
#
# We intentionally do NOT use:
#
# TRANSACTION_ID
# CUSTOMER_ID
# TERMINAL_ID
# TX_FRAUD
# TX_FRAUD_SCENARIO
# TX_DATETIME
# date
#
# IDs are identifiers, not meaningful numeric predictors.
# ============================================================

FEATURES = [
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


TARGET = "TX_FRAUD"


# ============================================================
# Prepare X and y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# Train Random Forest
# ============================================================

print("\nTraining Random Forest...")


model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Training complete.")


# ============================================================
# Evaluation function
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y,
    threshold=0.5
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)


    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )


    print(
        f"\n========== {name} =========="
    )

    print(
        f"Threshold: {threshold}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)


    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc
    }


# ============================================================
# Validation evaluation
# ============================================================

validation_metrics = evaluate_model(
    "VALIDATION",
    model,
    X_validation,
    y_validation
)


# ============================================================
# Held-out test evaluation
# ============================================================

test_metrics = evaluate_model(
    "HELD-OUT TEST",
    model,
    X_test,
    y_test
)


# ============================================================
# Feature importance
# ============================================================

print(
    "\n========== FEATURE IMPORTANCE =========="
)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# Save model
# ============================================================

MODEL_FILE = (
    MODEL_DIR /
    "random_forest_fraud_model.joblib"
)


joblib.dump(
    model,
    MODEL_FILE
)


print(
    "\nModel saved to:"
)

print(MODEL_FILE)


# ============================================================
# Save feature list
# ============================================================

FEATURE_FILE = (
    MODEL_DIR /
    "fraud_features.txt"
)


with open(
    FEATURE_FILE,
    "w"
) as f:

    for feature in FEATURES:
        f.write(
            feature + "\n"
        )


print(
    "Feature list saved to:"
)

print(FEATURE_FILE)


print(
    "\n========================================"
)

print(
    "FRAUD MODEL COMPLETE"
)

print(
    "========================================"
)