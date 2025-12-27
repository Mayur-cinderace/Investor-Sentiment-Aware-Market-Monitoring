import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def main():
    os.makedirs("metrics", exist_ok=True)  # ✅ ADD THIS LINE

    df = pd.read_csv("data/processed/merged_features.csv")

    results = {}

    for ticker in df["Ticker"].unique():
        df_t = df[df["Ticker"] == ticker].copy()

        if len(df_t) < 50:
            continue

        X = df_t[["return_lag1","volume_lag1","sentiment_lag1"]].values
        y = df_t["Return"].shift(-1).dropna().values
        X = X[:-1]

        model = joblib.load(f"models/{ticker}/rf.joblib")
        preds = model.predict(X)

        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        results[ticker] = {"rmse": rmse}

    with open("metrics/evaluation.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Saved metrics/evaluation.json")

if __name__ == "__main__":
    main()
