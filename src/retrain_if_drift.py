import json
import subprocess
from pathlib import Path

DECISION_FILE = Path("drift_reports/retrain_flag.json")

def main():
    if not DECISION_FILE.exists():
        print("No retrain decision file found → skipping retrain")
        return

    with open(DECISION_FILE) as f:
        decision = json.load(f)

    if decision.get("retrain", False):
        print("Retraining triggered due to detected drift")

        # Call training as a subprocess (DVC-safe)
        subprocess.run(
            ["python", "src/train_models.py"],
            check=True
        )

    else:
        print("Retraining skipped (no drift detected)")

if __name__ == "__main__":
    main()
