# src/retrain_if_drift.py
import json
import subprocess

DECISION_FILE = "drift_reports/retrain_flag.json"

with open(DECISION_FILE) as f:
    decision = json.load(f)

if decision.get("retrain", False):
    print("Retraining triggered")
    subprocess.run(["python", "src/train_models.py"], check=True)
else:
    print("Retraining skipped (no drift)")
