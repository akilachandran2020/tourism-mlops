import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "tourism_project/data"
RAW_PATH = os.path.join(DATA_DIR, "tourism.csv")

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"{RAW_PATH} not found in repo.")

df = pd.read_csv(RAW_PATH)

# Drop unnecessary columns
df = df.drop(columns=[c for c in df.columns if "unnamed" in c.lower() or "customer" in c.lower()], errors="ignore")

# Train-test split
if "ProdTaken" not in df.columns:
    raise ValueError("ProdTaken column not found in dataset.")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ProdTaken"])

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("✅ Data preparation done.")
print(f"Train saved to {TRAIN_PATH}")
print(f"Test saved to {TEST_PATH}")
