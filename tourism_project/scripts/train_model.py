import mlflow
import joblib
import pandas as pd
from xgboost import XGBRegressor

mlflow.set_experiment("tourism_model")

train = pd.read_csv("train_clean.csv")

X = train.drop("target", axis=1)
y = train["target"]

model = XGBRegressor()
model.fit(X, y)

joblib.dump(model, "XGBoost_best_model.joblib")

mlflow.log_artifact("XGBoost_best_model.joblib")

print("Model training completed!")
