import pandas as pd
import joblib
from xgboost import XGBClassifier
import mlflow
import os

TRAIN_PATH = "tourism_project/data/train.csv"
TARGET_COLUMN = "ProdTaken"

# Set experiment once
mlflow.set_experiment("tourism_model")

def main():
    train = pd.read_csv(TRAIN_PATH)

    X = train.drop(TARGET_COLUMN, axis=1)
    y = train[TARGET_COLUMN]

    # Encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Define model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    # Start MLflow run
    with mlflow.start_run():
        model.fit(X, y)

        os.makedirs("tourism_project/models", exist_ok=True)
        model_path = "tourism_project/models/xgboost_model.joblib"
        joblib.dump(model, model_path)

        # log artifact
        mlflow.log_artifact(model_path)

    print("Model training completed successfully!")

if __name__ == "__main__":
    main()
