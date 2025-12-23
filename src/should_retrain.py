# src/should_retrain.py
import json
import os
import pandas as pd

# ---------------- CONFIG ----------------
DRIFT_FILE = "drift_reports/drift_summary.json"
NEW_DATA_FILE = "data/processed/new_sentiment.csv"
DECISION_FILE = "drift_reports/retrain_flag.json"

MIN_NEW_ROWS = 50   # threshold for retraining based on new data
# ----------------------------------------


def check_drift():
    if not os.path.exists(DRIFT_FILE):
        return False

    with open(DRIFT_FILE) as f:
        drift = json.load(f)

    return any(
        v.get("drift_flag", False) for v in drift.values()
    )


def check_new_data_volume():
    if not os.path.exists(NEW_DATA_FILE):
        return False

    df = pd.read_csv(NEW_DATA_FILE)
    return len(df) >= MIN_NEW_ROWS


def main():
    drift_trigger = check_drift()
    data_trigger = check_new_data_volume()

    retrain = drift_trigger or data_trigger

    decision = {
        "retrain": retrain,
        "reason": {
            "drift_detected": drift_trigger,
            "new_data_threshold_met": data_trigger
        }
    }

    os.makedirs("drift_reports", exist_ok=True)
    with open(DECISION_FILE, "w") as f:
        json.dump(decision, f, indent=4)

    # ---- Console output (important for viva/demo) ----
    if retrain:
        print("Retraining required")
        if drift_trigger:
            print("→ Reason: feature drift detected")
        if data_trigger:
            print("→ Reason: sufficient new tweet/news data")
    else:
        print("No retraining required")
        print("→ No drift and insufficient new data")


if __name__ == "__main__":
    main()
