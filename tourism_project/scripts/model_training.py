import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_DIR = "tourism_project/data"
MODEL_DIR = "tourism_project/model_building"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"{TRAIN_PATH} not found. Run data_preparation.py first.")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]
X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f" Test Accuracy: {acc:.4f}")

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
joblib.dump(clf, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
