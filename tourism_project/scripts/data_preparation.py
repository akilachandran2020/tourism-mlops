import pandas as pd
import os

# Define file paths
TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Example preprocessing (modify as needed)
train = train.fillna(0)
test = test.fillna(0)

# Create output folder for cleaned data
OUTPUT_DIR = "tourism_project/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save cleaned files
train.to_csv(f"{OUTPUT_DIR}/train_clean.csv", index=False)
test.to_csv(f"{OUTPUT_DIR}/test_clean.csv", index=False)

print("Data preparation completed successfully!")
