# live.py
import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
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

# 0. SETTINGS
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="fuzzywuzzy")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Investor-Sentiment-Aware-Models")

os.makedirs("./saved_models", exist_ok=True)
os.makedirs("./data", exist_ok=True)

FETCH_INTERVAL_SECONDS = 60
SEQ_LENGTH = 10

# 1. LOAD STOCK DATA
def load_stock_data(path="data/stock_prices.csv"):
    if not os.path.exists(path):
        print(f"{path} not found → empty DF")
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "High", "Low", "Open", "Volume", "Return"])

    df = pd.read_csv(path, low_memory=False)
    
    # Use 'date' if exists, else 'Date'
    date_col = 'date' if 'date' in df.columns else 'Date'
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    # DROP bogus 1970 dates
    df = df[df['Date'].dt.year >= 2000].copy()
    
    # FORCE TZ: America/New_York
    df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    
    # Clean numeric
    for col in ['Close', 'High', 'Low', 'Open', 'Volume', 'Return']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[['Date', 'Ticker', 'Close', 'High', 'Low', 'Open', 'Volume', 'Return']]
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print(f"Loaded {len(df)} valid rows (post-2000, tz-aware)")
    return df

# 2. LOAD TEXT DATA
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

# 3. SENTIMENT
POS_WORDS = ["good", "buy", "up", "rise", "gain", "positive", "bull", "strong", "profit", "growth", "high", "best", "win", "success", "pump", "moon", "rocket"]
NEG_WORDS = ["bad", "sell", "down", "fall", "loss", "negative", "bear", "weak", "decline", "low", "worst", "fail", "crash", "risk", "dump", "scam"]

def simple_sentiment(text):
    words = text.split()
    pos_count = sum(1 for word in words if word in POS_WORDS)
    neg_count = sum(1 for word in words if word in NEG_WORDS)
    total = pos_count + neg_count
    if total == 0:
        return 0
    return (pos_count - neg_count) / total

# 4. LIVE FETCH STOCKS
def fetch_live_stocks(tickers=["AAPL", "GOOGL", "TSLA"], period="1d", interval="1m"):
    rows = []
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False, threads=False, prepost=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.columns = [str(col).lower().strip() for col in df.columns]
            df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df = df[['Open','High','Low','Close','Volume']].reset_index()
            df['Date'] = df['Datetime'] if 'Datetime' in df.columns else df['Date']
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('America/New_York')
            df['Ticker'] = t
            rows.append(df[['Date','Ticker','Open','High','Low','Close','Volume']])
        except Exception as e:
            print(f"yfinance {t}: {e}")
    if not rows:
        return pd.DataFrame()
    new = pd.concat(rows, ignore_index=True)
    new = new.sort_values(['Ticker','Date']).reset_index(drop=True)
    new['Return'] = new.groupby('Ticker')['Close'].pct_change().fillna(0)
    print(f"Fetched {len(new)} live rows (latest: {new['Date'].max()})")
    return new

# 5. LIVE FETCH NEWS
def fetch_live_news(max_results=15):
    try:
        client = gnewsclient.NewsClient(language="en", location="us", topic="Business", max_results=max_results)
        items = client.get_news()
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        print(f"Got {len(df)} news items")
        df['text'] = ''
        for idx, row in df.iterrows():
            text_parts = []
            if 'title' in row and pd.notna(row['title']):
                text_parts.append(str(row['title']))
            if 'description' in row and pd.notna(row['description']):
                text_parts.append(str(row['description']))
            elif 'content' in row and pd.notna(row['content']):
                text_parts.append(str(row['content'])[:200])
            df.at[idx, 'text'] = ' '.join(text_parts)
        date_col = next((c for c in ['published', 'pubDate', 'publishedAt', 'date'] if c in df.columns), None)
        if not date_col:
            print("No date column found in news")
            return pd.DataFrame()
        df['publishedAt'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['publishedAt'])
        df['source'] = 'gnews'
        df['date'] = df['publishedAt'].dt.date
        return df[['text', 'publishedAt', 'source', 'date']]
    except Exception as e:
        print(f"News error: {e}")
        return pd.DataFrame()

# 6. STREAM LIVE
def stream_live(df_prices: pd.DataFrame, df_text: pd.DataFrame, persist=False):
    cycle = 0
    while True:
        cycle += 1
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Cycle {cycle}")

        new_stk = fetch_live_stocks()
        if not new_stk.empty:
            if not df_prices.empty:
                latest_old = df_prices['Date'].max()
                # Ensure both are tz-aware
                new_stk['Date'] = pd.to_datetime(new_stk['Date'])
                latest_old = pd.to_datetime(latest_old)
                new_stk = new_stk[new_stk['Date'] > latest_old]

            if not new_stk.empty:
                old_count = len(df_prices)
                df_prices = pd.concat([df_prices, new_stk])
                df_prices = df_prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
                print(f"Stocks: {old_count} → {len(df_prices)} (+{len(new_stk)} new)")
                if persist:
                    df_prices.to_csv("data/stock_prices.csv", index=False)
            else:
                print("No newer stock data")

        # === NEWS & PREDICTION ===
        new_news = fetch_live_news()
        if not new_news.empty:
            new_news["sentiment"] = new_news["text"].apply(simple_sentiment)
            df_text = pd.concat([df_text, new_news]).drop_duplicates(subset=["publishedAt"]).sort_values("publishedAt").reset_index(drop=True)
            if persist:
                df_text.to_csv("data/live_news_data.csv", index=False)

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
                if t not in model_info:
                    continue
                latest = merged[merged["Ticker"] == t].sort_values("Date").iloc[-1]
                point = {
                    "return_lag1": latest["Return"],
                    "volume_lag1": latest["Volume"],
                    "sentiment_lag1": latest["sentiment"]
                }
                live_predict(t, point, model_info[t])
        else:
            print("No data")

        time.sleep(FETCH_INTERVAL_SECONDS)

# 7. TORCH HELPERS
class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_torch(model, loader, epochs=50):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            opt.zero_grad()
            outputs = model(batch_x)
            loss = crit(outputs, batch_y.unsqueeze(1))
            loss.backward()
            opt.step()

def predict_torch(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            outputs = model(batch_x)
            preds.extend(outputs.squeeze().numpy())
    return np.array(preds)

# 8. MODELS
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data_X, data_y, seq_length):
    xs, ys = [], []
    for i in range(len(data_X) - seq_length):
        x = data_X[i:i+seq_length]
        y = data_y[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 9. TRAIN MODELS FOR TICKER
def run_models_for_ticker(ticker, df_merged, seq_length=10):
    with mlflow.start_run(run_name=f"{ticker}_models"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("seq_length", seq_length)
        
        df_t = df_merged[df_merged['Ticker'] == ticker].copy()
        df_t = df_t.sort_values('Date')
        df_t['return_lag1'] = df_t['Return'].shift(1)
        df_t['volume_lag1'] = df_t['Volume'].shift(1)
        df_t.dropna(inplace=True)
        df_t['target_return'] = df_t['Return'].shift(-1)
        df_t.dropna(inplace=True)
        features_t = ['return_lag1', 'volume_lag1', 'sentiment_lag1']
        X_t = df_t[features_t].values
        y_t = df_t['target_return'].values
        scaler_X_t = MinMaxScaler()
        scaler_y_t = MinMaxScaler()
        Xs = scaler_X_t.fit_transform(X_t)
        ys = scaler_y_t.fit_transform(y_t.reshape(-1,1)).flatten()
        train_size_t = int(len(Xs) * 0.8)
        X_train_t, X_test_t = Xs[:train_size_t], Xs[train_size_t:]
        y_train_t, y_test_t = ys[:train_size_t], ys[train_size_t:]

        if len(X_train_t) == 0 or len(X_test_t) == 0:
            print(f"Not enough data for ticker {ticker}")
            return None

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train_t, y_train_t)
        y_rf_scaled = rf.predict(X_test_t)
        mse_rf = np.mean((y_test_t - y_rf_scaled)**2)
        rf_path = f'saved_models/{ticker}_rf.joblib'
        joblib.dump(rf, rf_path)
        with mlflow.start_run(run_name=f"{ticker}_RandomForest", nested=True):
            mlflow.log_metric("mse", mse_rf)
            mlflow.sklearn.log_model(rf, artifact_path=f"{ticker}_rf")
        print(f"{ticker} - RandomForest MSE: {mse_rf:.6f} saved to {rf_path}")

        input_size = X_t.shape[1]
        mlp = MLPModel(input_size)
        train_ds = TSDataset(X_train_t, y_train_t)
        train_loader_t = DataLoader(train_ds, batch_size=32, shuffle=False)
        train_torch(mlp, train_loader_t)
        test_ds = TSDataset(X_test_t, y_test_t)
        test_loader_t = DataLoader(test_ds, batch_size=32, shuffle=False)
        y_mlp_scaled = predict_torch(mlp, test_loader_t)
        mse_mlp = np.mean((y_test_t - y_mlp_scaled)**2)
        mlp_path = f'saved_models/{ticker}_mlp.pth'
        torch.save(mlp.state_dict(), mlp_path)
        with mlflow.start_run(run_name=f"{ticker}_MLP", nested=True):
            mlflow.log_metric("mse", mse_mlp)
            mlflow.pytorch.log_model(mlp, artifact_path=f"{ticker}_mlp")
        print(f"{ticker} - MLP MSE: {mse_mlp:.6f} saved to {mlp_path}")

        X_seq_t, y_seq_t = create_sequences(Xs, ys, seq_length)
        if len(X_seq_t) > 0:
            train_size_seq_t = int(len(X_seq_t) * 0.8)
            X_train_seq_t, X_test_seq_t = X_seq_t[:train_size_seq_t], X_seq_t[train_size_seq_t:]
            y_train_seq_t, y_test_seq_t = y_seq_t[:train_size_seq_t], y_seq_t[train_size_seq_t:]
            train_seq_ds = TSDataset(X_train_seq_t, y_train_seq_t)
            train_seq_loader = DataLoader(train_seq_ds, batch_size=32, shuffle=False)
            hidden_size = 50
            num_layers = 2
            lstm = LSTMModel(input_size, hidden_size, num_layers)
            train_torch(lstm, train_seq_loader)
            test_seq_ds = TSDataset(X_test_seq_t, y_test_seq_t)
            test_seq_loader = DataLoader(test_seq_ds, batch_size=32, shuffle=False)
            y_lstm_scaled = predict_torch(lstm, test_seq_loader)
            mse_lstm = np.mean((y_test_seq_t - y_lstm_scaled)**2)
            lstm_path = f'saved_models/{ticker}_lstm.pth'
            torch.save(lstm.state_dict(), lstm_path)
            with mlflow.start_run(run_name=f"{ticker}_LSTM", nested=True):
                mlflow.log_metric("mse", mse_lstm)
                mlflow.pytorch.log_model(lstm, artifact_path=f"{ticker}_lstm")
            print(f"{ticker} - LSTM MSE: {mse_lstm:.6f} saved to {lstm_path}")
        else:
            lstm_path = None
        return {
            "scaler_X": scaler_X_t, "scaler_y": scaler_y_t,
            "rf_path": rf_path, "mlp_path": mlp_path, "lstm_path": lstm_path,
            "input_size": input_size
        }

# 10. LIVE PREDICT
def live_predict(ticker, point, info):
    Xnew = np.array([[point["return_lag1"], point["volume_lag1"], point["sentiment_lag1"]]])
    Xs = info["scaler_X"].transform(Xnew)
    rf = joblib.load(info["rf_path"])
    pred_rf = info["scaler_y"].inverse_transform(rf.predict(Xs).reshape(-1,1)).flatten()[0]
    mlp = MLPModel(info["input_size"])
    mlp.load_state_dict(torch.load(info["mlp_path"]))
    mlp.eval()
    pred_mlp = info["scaler_y"].inverse_transform(mlp(torch.tensor(Xs, dtype=torch.float32)).squeeze().detach().numpy().reshape(-1,1)).flatten()[0]
    pred_lstm = np.nan
    if info["lstm_path"]:
        lstm = LSTMModel(info["input_size"], 50, 2)
        lstm.load_state_dict(torch.load(info["lstm_path"]))
        lstm.eval()
        X_seq_new = np.repeat(Xs, SEQ_LENGTH, axis=0).reshape(1, SEQ_LENGTH, -1)
        pred_lstm = info["scaler_y"].inverse_transform(lstm(torch.tensor(X_seq_new, dtype=torch.float32)).squeeze().detach().numpy().reshape(-1,1)).flatten()[0]
    print(f"{ticker} - RF: {pred_rf:.6f}, MLP: {pred_mlp:.6f}, LSTM: {pred_lstm:.6f}")
    return pred_rf, pred_mlp, pred_lstm

# MAIN
if __name__ == "__main__":
    df_prices = load_stock_data()
    df_text = load_text_data()
    df_text['sentiment'] = df_text['text'].apply(simple_sentiment)
    df_prices['date'] = df_prices['Date'].dt.date
    daily_sent = df_text.groupby(['date', 'source'])['sentiment'].mean().reset_index()
    daily_sent_total = daily_sent.groupby('date')['sentiment'].mean().reset_index()
    daily_sent_total['date'] = pd.to_datetime(daily_sent_total['date']).dt.date
    df_merged = df_prices.merge(daily_sent_total, on='date', how='left')
    df_merged['sentiment'] = df_merged['sentiment'].ffill().fillna(0)
    df_merged = df_merged.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df_merged['sentiment_lag1'] = df_merged.groupby('Ticker')['sentiment'].shift(1).bfill().fillna(0)
    model_info = {}
    for t in ['AAPL', 'GOOGL', 'TSLA']:
        print(f"\n=== TRAINING {t} ===")
        res = run_models_for_ticker(t, df_merged)
        if res is not None:
            model_info[t] = res
    print("\n=== LIVE STREAM STARTED ===")
    streamer = stream_live(df_prices, df_text, persist=True)
    for updated_merged in streamer:
        for t in ['AAPL', 'GOOGL', 'TSLA']:
            if t in model_info:
                latest = updated_merged[updated_merged['Ticker'] == t].sort_values('Date').iloc[-1]
                point = {
                    'return_lag1': latest['Return'],
                    'volume_lag1': latest['Volume'],
                    'sentiment_lag1': latest['sentiment']
                }
                live_predict(t, point, model_info[t])