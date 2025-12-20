# live.py
import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
import subprocess
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import yfinance as yf
from gnewsclient import gnewsclient

# === DVC CONFIG ===
DVC_REMOTE = "origin"  # or your remote name
PUSH_AFTER_CYCLE = True  # Set False to disable auto-push
PULL_AT_START = True

# === SETTINGS ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="fuzzywuzzy")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Investor-Sentiment-Aware-Models")

os.makedirs("./saved_models", exist_ok=True)
os.makedirs("./data", exist_ok=True)

FETCH_INTERVAL_SECONDS = 60
SEQ_LENGTH = 10

# === DVC HELPERS ===
def dvc_pull():
    print("Pulling latest data and models from DVC remote...")
    try:
        subprocess.run(["dvc", "pull", "-r", DVC_REMOTE], check=True)
        print("DVC pull completed.")
    except Exception as e:
        print(f"DVC pull failed: {e}")

def dvc_add_and_push(path):
    if not PUSH_AFTER_CYCLE:
        return
    try:
        print(f"Adding {path} to DVC...")
        subprocess.run(["dvc", "add", path], check=True)
        subprocess.run(["git", "add", f"{path}.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Update {path}"], check=True)
        subprocess.run(["dvc", "push", "-r", DVC_REMOTE], check=True)
        print(f"Pushed {path}")
    except Exception as e:
        print(f"DVC push failed for {path}: {e}")

# === 1. LOAD STOCK DATA (TZ-AWARE, DVC) ===
def load_stock_data(path="data/stock_prices.csv"):
    if not os.path.exists(path):
        print(f"{path} not found → empty DF")
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "High", "Low", "Open", "Volume", "Return"])

    df = pd.read_csv(path, low_memory=False)
    date_col = 'date' if 'date' in df.columns else 'Date'
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[df['Date'].dt.year >= 2000].copy()
    df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

    for col in ['Close', 'High', 'Low', 'Open', 'Volume', 'Return']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[['Date', 'Ticker', 'Close', 'High', 'Low', 'Open', 'Volume', 'Return']]
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    print(f"Loaded {len(df)} valid rows (post-2000, tz-aware)")
    return df

# === 2. LOAD TEXT DATA ===
def load_text_data(paths=["data/news_articles.csv", "data/gnews_data.csv"]):
    dfs = []
    for p, src in zip(paths, ["news", "gnews"]):
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df = df.rename(columns={"content": "text"})
        df["source"] = src
        df = df[["text", "publishedAt", "source"]]
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["text", "publishedAt", "source", "date"])
    txt = pd.concat(dfs, ignore_index=True)
    txt["text"] = txt["text"].astype(str).str.lower()
    txt["text"] = txt["text"].str.replace(r"http\S+|www\S+", "", regex=True)
    txt["text"] = txt["text"].str.replace(r"[^a-zA-Z\s]", " ", regex=True)
    txt["text"] = txt["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    txt["date"] = pd.to_datetime(txt["publishedAt"], errors='coerce').dt.date
    txt = txt.dropna(subset=["date"])
    return txt

# === 3. SENTIMENT ===
POS_WORDS = ["good", "buy", "up", "rise", "gain", "positive", "bull", "strong", "profit", "growth", "high", "best", "win", "success", "pump", "moon", "rocket"]
NEG_WORDS = ["bad", "sell", "down", "fall", "loss", "negative", "bear", "weak", "decline", "low", "worst", "fail", "crash", "risk", "dump", "scam"]

def simple_sentiment(text):
    words = text.split()
    pos_count = sum(1 for word in words if word in POS_WORDS)
    neg_count = sum(1 for word in words if word in NEG_WORDS)
    total = pos_count + neg_count
    return (pos_count - neg_count) / total if total else 0.0

# === 4. LIVE FETCH STOCKS ===
def fetch_live_stocks(tickers=["AAPL", "GOOGL", "TSLA"], period="1d", interval="1m"):
    rows = []
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False, threads=False, prepost=True)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            df.columns = [str(col).lower().strip() for col in df.columns]
            df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df = df[['Open','High','Low','Close','Volume']].reset_index()
            df['Date'] = df['Datetime'] if 'Datetime' in df.columns else df['Date']
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('America/New_York')
            df['Ticker'] = t
            rows.append(df[['Date','Ticker','Open','High','Low','Close','Volume']])
        except Exception as e: print(f"yfinance {t}: {e}")
    if not rows: return pd.DataFrame()
    new = pd.concat(rows, ignore_index=True)
    new = new.sort_values(['Ticker','Date']).reset_index(drop=True)
    new['Return'] = new.groupby('Ticker')['Close'].pct_change().fillna(0)
    print(f"Fetched {len(new)} live rows (latest: {new['Date'].max()})")
    return new

# === 5. LIVE FETCH NEWS ===
def fetch_live_news(max_results=15):
    try:
        client = gnewsclient.NewsClient(language="en", location="us", topic="Business", max_results=max_results)
        items = client.get_news()
        if not items: return pd.DataFrame()
        df = pd.DataFrame(items)
        print(f"Got {len(df)} news items")
        df['text'] = df['title'].fillna('') + " " + df.get('description', '').fillna('')
        date_col = next((c for c in ['published','pubDate','publishedAt','date'] if c in df.columns), None)
        if not date_col: return pd.DataFrame()
        df['publishedAt'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['publishedAt'])
        df['source'] = 'gnews'
        df['date'] = df['publishedAt'].dt.date
        return df[['text','publishedAt','source','date']]
    except Exception as e:
        print(f"News error: {e}")
        return pd.DataFrame()

# === 6. STREAM LIVE ===
def stream_live(df_prices: pd.DataFrame, df_text: pd.DataFrame, persist=False):
    cycle = 0
    while True:
        cycle += 1
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Cycle {cycle}")

        # === FETCH STOCKS ===
        new_stk = fetch_live_stocks()
        if not new_stk.empty:
            if not df_prices.empty:
                latest_old = df_prices['Date'].max()
                new_stk = new_stk[new_stk['Date'] > latest_old]
            if not new_stk.empty:
                old_count = len(df_prices)
                df_prices = pd.concat([df_prices, new_stk])
                df_prices = df_prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
                print(f"Stocks: {old_count} → {len(df_prices)} (+{len(new_stk)} new)")
                if persist:
                    df_prices.to_csv("data/stock_prices.csv", index=False)
                    dvc_add_and_push("data/stock_prices.csv")

        # === FETCH NEWS ===
        new_news = fetch_live_news()
        if not new_news.empty:
            new_news["sentiment"] = new_news["text"].apply(simple_sentiment)
            old_txt = len(df_text)
            df_text = pd.concat([df_text, new_news]).drop_duplicates(subset=["publishedAt"]).sort_values("publishedAt").reset_index(drop=True)
            print(f"News: {old_txt} → {len(df_text)}")
            if persist:
                df_text.to_csv("data/live_news_data.csv", index=False)
                dvc_add_and_push("data/live_news_data.csv")

        # === PREDICTION ===
        if len(df_prices) > 0:
            df_prices["date"] = df_prices["Date"].dt.date
            daily_tot = df_text.groupby("date")["sentiment"].mean().reset_index() if len(df_text)>0 else pd.DataFrame()
            merged = df_prices.copy()
            if not daily_tot.empty:
                daily_tot["date"] = pd.to_datetime(daily_tot["date"]).dt.date
                merged = merged.merge(daily_tot, on="date", how="left")
            merged["sentiment"] = merged["sentiment"].ffill().fillna(0)
            merged["sentiment_lag1"] = merged.groupby("Ticker")["sentiment"].shift(1).bfill().fillna(0)

            for t in ["AAPL", "GOOGL", "TSLA"]:
                if t not in model_info: continue
                latest = merged[merged["Ticker"] == t].sort_values("Date").iloc[-1]
                point = {"return_lag1": latest["Return"], "volume_lag1": latest["Volume"], "sentiment_lag1": latest["sentiment"]}
                live_predict(t, point, model_info[t])
        else:
            print("No data to predict")

        time.sleep(FETCH_INTERVAL_SECONDS)

# === 7. TORCH HELPERS ===
class TSDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_torch(model, loader, epochs=50):
    crit, opt = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb.unsqueeze(1))
            loss.backward()
            opt.step()

def predict_torch(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.extend(model(xb).squeeze().numpy())
    return np.array(preds)

# === 8. MODELS ===
class MLPModel(nn.Module):
    def __init__(self, input_size): super().__init__(); self.net = nn.Sequential(nn.Linear(input_size,50),nn.ReLU(),nn.Linear(50,25),nn.ReLU(),nn.Linear(25,1))
    def forward(self, x): return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])

def create_sequences(X, y, L): xs, ys = [], []; [xs.append(X[i:i+L]) or ys.append(y[i+L]) for i in range(len(X)-L)]; return np.array(xs), np.array(ys)

# === 9. TRAIN ONE TICKER ===
def run_models_for_ticker(ticker, df_merged, seq_length=10):
    with mlflow.start_run(run_name=f"{ticker}_models"):
        df_t = df_merged[df_merged['Ticker'] == ticker].copy().sort_values('Date')
        df_t['return_lag1'] = df_t['Return'].shift(1); df_t['volume_lag1'] = df_t['Volume'].shift(1); df_t.dropna(inplace=True)
        df_t['target_return'] = df_t['Return'].shift(-1); df_t.dropna(inplace=True)
        X, y = df_t[['return_lag1','volume_lag1','sentiment_lag1']].values, df_t['target_return'].values
        sx, sy = MinMaxScaler(), MinMaxScaler()
        Xs, ys = sx.fit_transform(X), sy.fit_transform(y.reshape(-1,1)).flatten()
        split = int(0.8 * len(Xs))
        Xtr, Xte, ytr, yte = Xs[:split], Xs[split:], ys[:split], ys[split:]

        # RF
        rf = RandomForestRegressor(n_estimators=200, random_state=42); rf.fit(Xtr, ytr)
        rf_path = f'saved_models/{ticker}_rf.joblib'; joblib.dump(rf, rf_path)
        with mlflow.start_run(run_name=f"{ticker}_RF", nested=True): mlflow.sklearn.log_model(rf, "rf")

        # MLP
        mlp = MLPModel(X.shape[1]); train_torch(mlp, DataLoader(TSDataset(Xtr, ytr), batch_size=32, shuffle=False))
        mlp_path = f'saved_models/{ticker}_mlp.pth'; torch.save(mlp.state_dict(), mlp_path)
        with mlflow.start_run(run_name=f"{ticker}_MLP", nested=True): mlflow.pytorch.log_model(mlp, "mlp")

        # LSTM
        lstm_path = None
        Xseq, yseq = create_sequences(Xs, ys, seq_length)
        if len(Xseq) > 10:
            split_seq = int(0.8 * len(Xseq))
            lstm = LSTMModel(X.shape[1]); train_torch(lstm, DataLoader(TSDataset(Xseq[:split_seq], yseq[:split_seq]), batch_size=32, shuffle=False))
            lstm_path = f'saved_models/{ticker}_lstm.pth'; torch.save(lstm.state_dict(), lstm_path)
            with mlflow.start_run(run_name=f"{ticker}_LSTM", nested=True): mlflow.pytorch.log_model(lstm, "lstm")

        # Save models to DVC
        dvc_add_and_push("saved_models")

        return {"scaler_X": sx, "scaler_y": sy, "rf_path": rf_path, "mlp_path": mlp_path, "lstm_path": lstm_path, "input_size": X.shape[1]}

# === 10. LIVE PREDICT ===
def live_predict(ticker, point, info):
    Xnew = np.array([[point["return_lag1"], point["volume_lag1"], point["sentiment_lag1"]]])
    Xs = info["scaler_X"].transform(Xnew)
    rf = joblib.load(info["rf_path"])
    pred_rf = info["scaler_y"].inverse_transform(rf.predict(Xs).reshape(-1,1)).flatten()[0]
    mlp = MLPModel(info["input_size"]); mlp.load_state_dict(torch.load(info["mlp_path"])); mlp.eval()
    pred_mlp = info["scaler_y"].inverse_transform(mlp(torch.tensor(Xs, dtype=torch.float32)).detach().numpy().reshape(-1,1)).flatten()[0]
    pred_lstm = np.nan
    if info["lstm_path"]:
        lstm = LSTMModel(info["input_size"]); lstm.load_state_dict(torch.load(info["lstm_path"])); lstm.eval()
        seq = np.repeat(Xs, SEQ_LENGTH, axis=0).reshape(1, SEQ_LENGTH, -1)
        pred_lstm = info["scaler_y"].inverse_transform(lstm(torch.tensor(seq, dtype=torch.float32)).detach().numpy().reshape(-1,1)).flatten()[0]
    print(f"{ticker} - RF: {pred_rf:.6f}, MLP: {pred_mlp:.6f}, LSTM: {pred_lstm:.6f}")

# === MAIN ===
if __name__ == "__main__":
    if PULL_AT_START: dvc_pull()

    df_prices = load_stock_data()
    df_text = load_text_data()
    df_text['sentiment'] = df_text['text'].apply(simple_sentiment)
    df_prices['date'] = df_prices['Date'].dt.date
    daily_sent_total = df_text.groupby('date')['sentiment'].mean().reset_index()
    daily_sent_total['date'] = pd.to_datetime(daily_sent_total['date']).dt.date
    df_merged = df_prices.merge(daily_sent_total, on='date', how='left')
    df_merged['sentiment'] = df_merged['sentiment'].ffill().fillna(0)
    df_merged = df_merged.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df_merged['sentiment_lag1'] = df_merged.groupby('Ticker')['sentiment'].shift(1).bfill().fillna(0)

    model_info = {}
    for t in ['AAPL', 'GOOGL', 'TSLA']:
        print(f"\n=== TRAINING {t} ===")
        res = run_models_for_ticker(t, df_merged)
        if res: model_info[t] = res

    print("\n=== LIVE STREAM STARTED ===")
    stream_live(df_prices, df_text, persist=True)