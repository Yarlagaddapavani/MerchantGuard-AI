import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_DIR = Path("data/raw/simulated-data-raw/data")
OUTPUT_FILE = Path("data/processed/merchantguard_transactions.csv")


# --------------------------------------------------
# Find daily dataset files
# --------------------------------------------------

files = sorted(RAW_DIR.glob("*.pkl"))

print(f"Total daily files available: {len(files)}")


# --------------------------------------------------
# Select first 30 days
# --------------------------------------------------

selected_files = files[:30]

print(f"Files selected: {len(selected_files)}")


# --------------------------------------------------
# Load daily files
# --------------------------------------------------

dataframes = []

for file in selected_files:

    print(f"Loading: {file.name}")

    df = pd.read_pickle(file)

    dataframes.append(df)


# --------------------------------------------------
# Combine all days
# --------------------------------------------------

combined_df = pd.concat(
    dataframes,
    ignore_index=True
)


# --------------------------------------------------
# Sort chronologically
# --------------------------------------------------

combined_df = combined_df.sort_values(
    "TX_DATETIME"
).reset_index(drop=True)


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n========== FINAL DATASET ==========")

print("Rows:", len(combined_df))

print("Columns:", len(combined_df.columns))

print(
    "Start:",
    combined_df["TX_DATETIME"].min()
)

print(
    "End:",
    combined_df["TX_DATETIME"].max()
)

print(
    "Fraud transactions:",
    combined_df["TX_FRAUD"].sum()
)

print(
    "Fraud percentage:",
    combined_df["TX_FRAUD"].mean() * 100
)

print(
    "Unique customers:",
    combined_df["CUSTOMER_ID"].nunique()
)

print(
    "Unique terminals:",
    combined_df["TERMINAL_ID"].nunique()
)

print("\nDataset saved to:")

print(OUTPUT_FILE)