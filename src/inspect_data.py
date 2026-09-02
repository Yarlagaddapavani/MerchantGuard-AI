import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Locate one daily dataset file
# --------------------------------------------------

DATA_PATH = Path(
    "data/raw/simulated-data-raw/data/2018-04-01.pkl"
)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_pickle(DATA_PATH)


# --------------------------------------------------
# 3. Basic information
# --------------------------------------------------

print("\n========== DATASET SHAPE ==========")
print(df.shape)


print("\n========== COLUMNS ==========")
print(df.columns.tolist())


print("\n========== FIRST 5 ROWS ==========")
print(df.head())


print("\n========== DATA TYPES ==========")
print(df.dtypes)


print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# --------------------------------------------------
# 4. Fraud distribution
# --------------------------------------------------

print("\n========== FRAUD DISTRIBUTION ==========")
print(df["TX_FRAUD"].value_counts())


print("\n========== FRAUD PERCENTAGE ==========")
print(df["TX_FRAUD"].value_counts(normalize=True) * 100)


# --------------------------------------------------
# 5. Number of unique customers and terminals
# --------------------------------------------------

print("\n========== UNIQUE CUSTOMERS ==========")
print(df["CUSTOMER_ID"].nunique())


print("\n========== UNIQUE TERMINALS ==========")
print(df["TERMINAL_ID"].nunique())


# --------------------------------------------------
# 6. Transaction amount statistics
# --------------------------------------------------

print("\n========== TRANSACTION AMOUNT ==========")
print(df["TX_AMOUNT"].describe())


# --------------------------------------------------
# 7. Date range
# --------------------------------------------------

print("\n========== DATE RANGE ==========")
print("Start:", df["TX_DATETIME"].min())
print("End:  ", df["TX_DATETIME"].max())