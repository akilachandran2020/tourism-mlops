import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Example preprocessing
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

train.to_csv("train_clean.csv", index=False)
test.to_csv("test_clean.csv", index=False)

print("Data preparation complete!")
