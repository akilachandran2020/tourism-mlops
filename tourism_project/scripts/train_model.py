import pandas as pd
import joblib
from xgboost import XGBRegressor
import mlflow
import os

TRAIN_PATH = "tourism_project/data/train.csv"

train = pd.read_csv(TRAIN_PATH)

TARGET_COLUMN = "ProdTaken"

X = train.drop(TARGET_COLUMN, axis=1)
y = train[TARGET_COLUMN]

mlflow.set_experiment("tourism_model")

model = XGBRegressor()
model.fit(X, y)

os.makedirs("tourism_project/models", exist_ok=True)
MODEL_PATH = "tourism_project/models/xgboost_model.joblib"
joblib.dump(model, MODEL_PATH)

mlflow.log_artifact(MODEL_PATH)

print("Model training completed successfully!")
