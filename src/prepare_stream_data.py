import pandas as pd
import re
from pathlib import Path

INPUT_FILE = "news_articles_1.csv"
OUTPUT_FILE = "data/streaming/news_stream.csv"

TICKER_KEYWORDS = {
    "AAPL": ["apple", "iphone", "ipad", "tim cook"],
    "GOOGL": ["google", "alphabet", "youtube"],
    "TSLA": ["tesla", "elon", "musk"]
}

def infer_ticker(text):
    text = text.lower()
    for ticker, keywords in TICKER_KEYWORDS.items():
        if any(k in text for k in keywords):
            return ticker
    return None


def main():
    df = pd.read_csv(INPUT_FILE)

    # ---- unify text field ----
    if "content" in df.columns:
        df["text"] = df["content"]
    elif "description" in df.columns:
        df["text"] = df["description"]
    elif "title" in df.columns:
        df["text"] = df["title"]
    else:
        raise ValueError("No usable text column found")

    # ---- date ----
    if "publishedAt" in df.columns:
        df["date"] = pd.to_datetime(df["publishedAt"], errors="coerce").dt.date
    else:
        df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.date

    # ---- ticker inference ----
    df["ticker"] = df["text"].apply(infer_ticker)

    # ---- cleanup ----
    df = df.dropna(subset=["date", "ticker", "text"])
    df = df[["date", "ticker", "text"]]

    Path("data/streaming").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Streaming-ready file saved → {OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()
