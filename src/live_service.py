# src/live_service.py
import time
import joblib
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x): return self.net(x)

def load_models(ticker):
    base = f"models/{ticker}"
    rf = joblib.load(f"{base}/rf.joblib")
    sx = joblib.load(f"{base}/scaler_x.joblib")
    sy = joblib.load(f"{base}/scaler_y.joblib")

    mlp = MLP(3)
    mlp.load_state_dict(torch.load(f"{base}/mlp.pth"))
    mlp.eval()

    return rf, mlp, sx, sy

MODELS = {t: load_models(t) for t in ["AAPL","GOOGL","TSLA"]}

while True:
    for t, (rf, mlp, sx, sy) in MODELS.items():
        df = yf.download(t, period="1d", interval="1m", progress=False)
        if df.empty: continue
        last = df.iloc[-1]
        X = np.array([[0, last["Volume"], 0]])
        Xs = sx.transform(X)

        pred_rf = sy.inverse_transform(rf.predict(Xs).reshape(-1,1))[0][0]
        pred_mlp = sy.inverse_transform(
            mlp(torch.tensor(Xs, dtype=torch.float32)).detach().numpy()
        )[0][0]

        print(f"{t} → RF:{pred_rf:.6f} MLP:{pred_mlp:.6f}")

    time.sleep(60)
