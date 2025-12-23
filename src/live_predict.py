import joblib
import torch
import numpy as np
from pathlib import Path

# -----------------------------
# MLP definition (same as train)
# -----------------------------
class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Load models for a ticker
# -----------------------------
def load_models(ticker: str):
    model_dir = Path("models") / ticker.upper()

    if not model_dir.exists():
        raise FileNotFoundError(f"No trained models found for ticker {ticker}")

    rf = joblib.load(model_dir / "rf.joblib")
    sx = joblib.load(model_dir / "scaler_x.joblib")
    sy = joblib.load(model_dir / "scaler_y.joblib")

    mlp = MLP(input_dim=3)
    mlp.load_state_dict(torch.load(model_dir / "mlp.pth", map_location="cpu"))
    mlp.eval()

    return rf, mlp, sx, sy


# -----------------------------
# Live user input
# -----------------------------
def get_live_input():
    return {
        "return_lag1": float(input("Previous day return: ")),
        "volume_lag1": float(input("Previous day volume: ")),
        "sentiment_lag1": float(input("Sentiment score (-1 to 1): "))
    }


# -----------------------------
# Prediction
# -----------------------------
def predict(features, rf, mlp, sx, sy):
    X = np.array([[features["return_lag1"],
                   features["volume_lag1"],
                   features["sentiment_lag1"]]])

    X_scaled = sx.transform(X)

    rf_pred = sy.inverse_transform(
        rf.predict(X_scaled).reshape(-1, 1)
    )[0, 0]

    mlp_pred = sy.inverse_transform(
        mlp(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy()
    )[0, 0]

    return rf_pred, mlp_pred


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ticker = input("Enter ticker (AAPL / GOOGL / TSLA): ").upper()

    rf, mlp, sx, sy = load_models(ticker)
    features = get_live_input()

    rf_out, mlp_out = predict(features, rf, mlp, sx, sy)

    print("\n================ LIVE INFERENCE ================")
    print(f"Ticker: {ticker}")
    print("Input features:", features)
    print(f"RF predicted return : {rf_out:.6f}")
    print(f"MLP predicted return: {mlp_out:.6f}")
    print("================================================")
