# src/drift_detection.py
import os
import json
import pandas as pd
from datetime import timedelta
from river.drift import ADWIN, PageHinkley

DATA_PATH = "data/processed/merged_features.csv"
OUT_DIR = "drift_reports"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["return_lag1", "volume_lag1", "sentiment_lag1"]
CURRENT_DAYS = 30

def main():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    cutoff = df["Date"].max() - timedelta(days=CURRENT_DAYS)
    recent = df[df["Date"] >= cutoff]

    if recent.empty:
        raise RuntimeError("No recent data for drift detection")

    results = {}

    for f in FEATURES:
        adwin = ADWIN()
        ph = PageHinkley()

        drift_points = []

        for val in recent[f].dropna():
            adwin.update(val)
            ph.update(val)

            if adwin.drift_detected or ph.drift_detected:
                drift_points.append(True)
            else:
                drift_points.append(False)

        results[f] = {
            "adwin_drift": adwin.drift_detected,
            "page_hinkley_drift": ph.drift_detected,
            "drift_flag": adwin.drift_detected or ph.drift_detected
        }

    out_path = os.path.join(OUT_DIR, "drift_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("River drift detection completed")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
