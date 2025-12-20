# src/build_features.py
import pandas as pd
import os

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)

POS_WORDS = {"good", "buy", "up", "rise", "gain", "bull", "profit", "growth"}
NEG_WORDS = {"bad", "sell", "down", "fall", "loss", "bear", "risk", "crash"}

# ------------------------------------------------------------------
# Simple rule-based sentiment
# ------------------------------------------------------------------
def simple_sentiment(text):
    if not isinstance(text, str):
        return 0.0
    words = text.lower().split()
    pos = sum(w in POS_WORDS for w in words)
    neg = sum(w in NEG_WORDS for w in words)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0

# ------------------------------------------------------------------
# Load & normalize news data
# ------------------------------------------------------------------
def load_news():
    dfs = []

    for fname in ["news_articles.csv", "gnews_data.csv", "reddit_data.csv"]:
        path = f"data/raw/{fname}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        print("⚠ No news files found — sentiment will be zero")
        return pd.DataFrame(columns=["date", "sentiment"])

    news = pd.concat(dfs, ignore_index=True)

    # Normalize text column
    if "content" in news.columns:
        news["text"] = news["content"]
    elif "text" not in news.columns:
        raise ValueError("No text/content column found in news data")

    # Normalize datetime column
    if "publishedAt" not in news.columns:
        raise ValueError("No publishedAt column found in news data")

    news["publishedAt"] = pd.to_datetime(news["publishedAt"], errors="coerce")
    news = news.dropna(subset=["publishedAt"])

    news["date"] = news["publishedAt"].dt.date
    news["sentiment"] = news["text"].apply(simple_sentiment)

    # Daily aggregated sentiment
    daily_sent = (
        news.groupby("date")["sentiment"]
        .mean()
        .reset_index()
    )

    return daily_sent

# ------------------------------------------------------------------
# Main feature pipeline
# ------------------------------------------------------------------
def main():
    # -------------------------------
    # Load stock prices
    # -------------------------------
    prices = pd.read_csv("data/raw/stock_prices.csv")
    prices = prices.dropna(subset=["Ticker"])

    prices["Date"] = pd.to_datetime(prices["Date"], utc=True)
    prices["date"] = prices["Date"].dt.date

    # Ensure numeric columns (CRITICAL FIX)
    for col in ["Close", "Volume", "Return"]:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")

    # -------------------------------
    # Load sentiment
    # -------------------------------
    daily_sent = load_news()

    # -------------------------------
    # Merge prices + sentiment
    # -------------------------------
    merged = prices.merge(daily_sent, on="date", how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0)

    merged = merged.sort_values(["Ticker", "Date"])

    # -------------------------------
    # Lag features
    # -------------------------------
    merged["return_lag1"] = merged.groupby("Ticker")["Return"].shift(1)
    merged["volume_lag1"] = merged.groupby("Ticker")["Volume"].shift(1)
    merged["sentiment_lag1"] = merged.groupby("Ticker")["sentiment"].shift(1)

    # -------------------------------
    # Coerce lagged columns to numeric
    # -------------------------------
    merged["return_lag1"] = pd.to_numeric(
        merged["return_lag1"], errors="coerce"
    ).fillna(0)

    merged["volume_lag1"] = pd.to_numeric(
        merged["volume_lag1"], errors="coerce"
    )

    # Compute per-ticker median lagged volume
    median_volume = merged.groupby("Ticker")["volume_lag1"].median()

    # Map median volume back to rows (vectorized, NaN-safe)
    merged["volume_lag1"] = merged["volume_lag1"].fillna(
        merged["Ticker"].map(median_volume)
    )

    # Final fallback if still NaN (e.g., ticker itself missing)
    merged["volume_lag1"] = merged["volume_lag1"].fillna(0)


    merged["sentiment_lag1"] = merged["sentiment_lag1"].fillna(0)

    # -------------------------------
    # Final sanity filter
    # -------------------------------
    merged = merged[merged["Ticker"].notna()]

    # -------------------------------
    # Save output
    # -------------------------------
    merged.to_csv("data/processed/merged_features.csv", index=False)

    print("Saved data/processed/merged_features.csv")
    print("Rows:", len(merged))
    print("Tickers:", merged["Ticker"].unique())

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
