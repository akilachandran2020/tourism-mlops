import pandas as pd
import os
from xgboost import XGBRegressor
import joblib
import mlflow

print("Starting model training...")

TRAIN_CLEAN_PATH = "tourism_project/data/processed/train_clean.csv"

print("🔎 Checking if cleaned train file exists:")
print("    Exists:", os.path.exists(TRAIN_CLEAN_PATH))

if not os.path.exists(TRAIN_CLEAN_PATH):
    print("ERROR: File NOT FOUND! Here are all files in data/processed:")
    os.system("ls -R tourism_project/data/processed")
    raise FileNotFoundError(TRAIN_CLEAN_PATH)

train = pd.read_csv(TRAIN_CLEAN_PATH)

TARGET_COLUMN = "target"  # update if needed
print("Columns in training data:", train.columns)

X = train.drop(TARGET_COLUMN, axis=1)
y = train[TARGET_COLUMN]

mlflow.set_experiment("tourism_model")

model = XGBRegressor()
model.fit(X, y)

os.makedirs("tourism_project/models", exist_ok=True)
MODEL_PATH = "tourism_project/models/xgboost_model.joblib"
joblib.dump(model, MODEL_PATH)

print("Training completed! Model saved to:", MODEL_PATH)
