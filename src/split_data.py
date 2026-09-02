import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "data/processed/merchantguard_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)


# ============================================================
# Load data
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["TX_DATETIME"]
)

df = df.sort_values(
    "TX_DATETIME"
).reset_index(drop=True)


# ============================================================
# Define chronological periods
# ============================================================

train_end = pd.Timestamp("2018-04-20 23:59:59")

validation_end = pd.Timestamp("2018-04-25 23:59:59")


train_df = df[
    df["TX_DATETIME"] <= train_end
].copy()


validation_df = df[
    (df["TX_DATETIME"] > train_end)
    &
    (df["TX_DATETIME"] <= validation_end)
].copy()


test_df = df[
    df["TX_DATETIME"] > validation_end
].copy()


# ============================================================
# Save splits
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# ============================================================
# Function to print statistics
# ============================================================

def print_stats(name, data):

    print(f"\n========== {name} ==========")

    print(
        "Rows:",
        f"{len(data):,}"
    )

    print(
        "Start:",
        data["TX_DATETIME"].min()
    )

    print(
        "End:",
        data["TX_DATETIME"].max()
    )

    print(
        "Fraud:",
        int(data["TX_FRAUD"].sum())
    )

    print(
        "Fraud rate:",
        f"{data['TX_FRAUD'].mean() * 100:.4f}%"
    )


# ============================================================
# Display statistics
# ============================================================

print_stats(
    "TRAIN",
    train_df
)

print_stats(
    "VALIDATION",
    validation_df
)

print_stats(
    "HELD-OUT TEST",
    test_df
)


# ============================================================
# Final confirmation
# ============================================================

print("\n========================================")
print("CHRONOLOGICAL SPLIT COMPLETE")
print("========================================")

print("\nFiles created:")

print(" - data/processed/train.csv")
print(" - data/processed/validation.csv")
print(" - data/processed/test.csv")