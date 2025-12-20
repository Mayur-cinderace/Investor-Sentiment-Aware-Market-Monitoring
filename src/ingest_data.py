# src/ingest_data.py
import os
import shutil
import pandas as pd
import yfinance as yf
from datetime import datetime

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

TICKERS = ["AAPL", "GOOGL", "TSLA"]
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

def fetch_stock_data():
    frames = []
    for t in TICKERS:
        df = yf.download(t, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            continue
        df.reset_index(inplace=True)
        df["Ticker"] = t
        df["Return"] = df["Close"].pct_change()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(f"{RAW_DIR}/stock_prices.csv", index=False)
    print("Saved stock_prices.csv")

def copy_news_files():
    """
    Move news data from data/ → data/raw/ so DVC owns it
    """
    source_dir = "data"
    target_dir = "data/raw"
    os.makedirs(target_dir, exist_ok=True)

    source_files = [
        "news_articles.csv",
        "gnews_data.csv",
        "reddit_data.csv"
    ]

    for f in source_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(target_dir, f)

        if not os.path.exists(src):
            print(f"Warning: {src} not found")
            continue

        if os.path.abspath(src) == os.path.abspath(dst):
            print(f"Skipping {f} (already in target location)")
            continue

        shutil.copy(src, dst)
        print(f"Copied {src} → {dst}")


if __name__ == "__main__":
    fetch_stock_data()
    copy_news_files()
