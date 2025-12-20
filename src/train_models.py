# src/train_models.py
import os
import joblib
import mlflow
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------
# MLflow setup
# ------------------------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Investor-Sentiment-Aware-Models")

# ------------------------------------------------------------------
# Ensure models directory exists
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------
# Simple MLP model
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------
# Train models for a single ticker
# ------------------------------------------------------------------
def train_ticker(df, ticker):
    df_t = df[df["Ticker"] == ticker].copy()

    # Feature matrix
    X = df_t[["return_lag1", "volume_lag1", "sentiment_lag1"]].values
    y = df_t["Return"].shift(-1).dropna().values

    # Align X with shifted y
    X = X[:-1]

    if len(X) < 20:
        raise ValueError(f"Not enough samples after lagging for {ticker}")

    # Scale
    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y.reshape(-1, 1)).flatten()

    split = int(0.8 * len(Xs))
    Xtr, Xte = Xs[:split], Xs[split:]
    ytr, yte = ys[:split], ys[split:]

    ticker_dir = f"models/{ticker}"
    os.makedirs(ticker_dir, exist_ok=True)

    with mlflow.start_run(run_name=ticker):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("train_samples", len(Xtr))
        mlflow.log_param("test_samples", len(Xte))

        # -------------------------------
        # Random Forest
        # -------------------------------
        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        rf.fit(Xtr, ytr)

        preds_rf = rf.predict(Xte)
        rmse_rf = np.sqrt(mean_squared_error(yte, preds_rf))

        joblib.dump(rf, f"{ticker_dir}/rf.joblib")
        mlflow.sklearn.log_model(rf, "rf")
        mlflow.log_metric("rf_rmse", rmse_rf)

        # -------------------------------
        # MLP
        # -------------------------------
        mlp = MLP(X.shape[1])
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)

        for epoch in range(50):
            optimizer.zero_grad()
            loss = loss_fn(mlp(Xtr_t), ytr_t)
            loss.backward()
            optimizer.step()

        mlp.eval()
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        preds_mlp = mlp(Xte_t).detach().numpy().flatten()
        rmse_mlp = np.sqrt(mean_squared_error(yte, preds_mlp))

        torch.save(mlp.state_dict(), f"{ticker_dir}/mlp.pth")
        mlflow.pytorch.log_model(mlp, "mlp")
        mlflow.log_metric("mlp_rmse", rmse_mlp)

        # -------------------------------
        # Scalers
        # -------------------------------
        joblib.dump(sx, f"{ticker_dir}/scaler_x.joblib")
        joblib.dump(sy, f"{ticker_dir}/scaler_y.joblib")

        print(
            f"[{ticker}] RF RMSE={rmse_rf:.6f}, "
            f"MLP RMSE={rmse_mlp:.6f}"
        )

# ------------------------------------------------------------------
# Main entry point (DVC stage)
# ------------------------------------------------------------------
def main():
    df = pd.read_csv("data/processed/merged_features.csv")

    print("Rows in merged features:", len(df))
    print("Tickers found:", df["Ticker"].unique())

    trained_any = False

    for ticker in df["Ticker"].unique():
        df_t = df[df["Ticker"] == ticker]

        if len(df_t) < 50:
            print(f"Skipping {ticker}: insufficient data ({len(df_t)} rows)")
            continue

        print(f"Training models for {ticker}")
        train_ticker(df, ticker)
        trained_any = True

    if not trained_any:
        raise RuntimeError(
            "No models were trained — check feature generation or data volume."
        )

    print("Training stage completed successfully.")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
