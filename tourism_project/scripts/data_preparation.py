import pandas as pd

TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Example cleaning — modify as needed
train = train.fillna(0)
test = test.fillna(0)

# Write back to same CSVs
train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print("Data preparation completed — train.csv and test.csv updated.")
