# src/should_retrain.py
import json
import os

DRIFT_FILE = "drift_reports/drift_summary.json"
DECISION_FILE = "drift_reports/retrain_flag.json"

def main():
    with open(DRIFT_FILE) as f:
        drift = json.load(f)

    retrain = any(
        v.get("drift_flag", False) for v in drift.values()
    )

    decision = {"retrain": retrain}

    with open(DECISION_FILE, "w") as f:
        json.dump(decision, f, indent=4)

    if retrain:
        print("Drift detected → retraining required")
    else:
        print("No drift detected → retraining not required")

if __name__ == "__main__":
    main()
