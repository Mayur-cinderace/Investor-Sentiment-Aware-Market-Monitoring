# src/evaluate_models.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    df = pd.read_csv("data/processed/merged_features.csv")
    metrics = {}

    for t in df["Ticker"].unique():
        mdir = f"models/{t}"
        rf = joblib.load(f"{mdir}/rf.joblib")
        sx = joblib.load(f"{mdir}/scaler_x.joblib")
        sy = joblib.load(f"{mdir}/scaler_y.joblib")

        df_t = df[df["Ticker"] == t].copy()
        X = df_t[["return_lag1","volume_lag1","sentiment_lag1"]].values
        y = df_t["Return"].shift(-1).dropna().values
        X = X[:-1]

        Xs = sx.transform(X)
        preds = sy.inverse_transform(rf.predict(Xs).reshape(-1,1)).flatten()

        metrics[t] = {
            "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
            "MAE": float(mean_absolute_error(y, preds))
        }

    with open("metrics/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved evaluation.json")

if __name__ == "__main__":
    main()
