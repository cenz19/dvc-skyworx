# src/evaluate.py
import pandas as pd
import joblib
import json

from sklearn.metrics import roc_auc_score, classification_report

TEST_PATH = "data/test.csv"
MODEL_PATH = "models/scorecard.pkl"
METRICS_PATH = "metrics.json"

TARGET = "target"

df = pd.read_csv(TEST_PATH)

y_true = df[TARGET]
X = df.drop(columns=[TARGET])

model = joblib.load(MODEL_PATH)

y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_true, y_proba)

report = classification_report(y_true, y_pred, output_dict=True)

metrics = {
    "auc": auc,
    "classification_report": report
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to metrics.json")