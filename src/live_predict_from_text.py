import joblib
import torch
import numpy as np
from pathlib import Path
from sentiment import compute_sentiment


class MLP(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_models(ticker):
    base = Path("models") / ticker
    rf = joblib.load(base / "rf.joblib")
    sx = joblib.load(base / "scaler_x.joblib")
    sy = joblib.load(base / "scaler_y.joblib")

    mlp = MLP(3)
    mlp.load_state_dict(torch.load(base / "mlp.pth", map_location="cpu"))
    mlp.eval()

    return rf, mlp, sx, sy


if __name__ == "__main__":
    ticker = input("Ticker (AAPL / GOOGL / TSLA): ").upper()

    prev_return = float(input("Previous day return: "))
    prev_volume = float(input("Previous day volume: "))

    text = input("Enter tweet/news sentence: ")
    sentiment = compute_sentiment(text)

    print(f"Computed sentiment score: {sentiment:.4f}")

    rf, mlp, sx, sy = load_models(ticker)

    X = np.array([[prev_return, prev_volume, sentiment]])
    Xs = sx.transform(X)

    rf_pred = sy.inverse_transform(
        rf.predict(Xs).reshape(-1, 1)
    )[0, 0]

    mlp_pred = sy.inverse_transform(
        mlp(torch.tensor(Xs, dtype=torch.float32)).detach().numpy()
    )[0, 0]

    print("\n====== LIVE PREDICTION ======")
    print(f"RF predicted return : {rf_pred:.6f}")
    print(f"MLP predicted return: {mlp_pred:.6f}")
