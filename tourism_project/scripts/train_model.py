import pandas as pd
import joblib
from xgboost import XGBRegressor
import mlflow
import os

# Correct path to cleaned training data
TRAIN_CLEAN_PATH = "tourism_project/data/processed/train_clean.csv"

# Load cleaned data
train = pd.read_csv(TRAIN_CLEAN_PATH)

# Make sure the target column name is correct
TARGET_COLUMN = "target"  # change if your file uses a different column name

X = train.drop(TARGET_COLUMN, axis=1)
y = train[TARGET_COLUMN]

# MLflow experiment
mlflow.set_experiment("tourism_model")

# Train model
model = XGBRegressor()
model.fit(X, y)

# Save trained model
MODEL_DIR = "tourism_project/model_building"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = f"{MODEL_DIR}/xgboost_model.joblib"
joblib.dump(model, MODEL_PATH)

# Log to MLflow
mlflow.log_artifact(MODEL_PATH)

print("Model training completed successfully!")
