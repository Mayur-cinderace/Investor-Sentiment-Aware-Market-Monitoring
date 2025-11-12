# src/data_loader.py
import pandas as pd
from pathlib import Path
from typing import Dict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_stock_prices(file_path: str = "stock_prices.csv") -> Dict[str, pd.DataFrame]:
    """
    Load stock price data in multi-index column format like:
    ('Close', 'AAPL'), ('Close', 'GOOGL'), etc.
    Returns dict of DataFrames indexed by datetime.
    """
    fp = DATA_DIR / file_path
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found")

    # Read both header rows
    raw = pd.read_csv(fp, header=[0, 1])

    # Drop the bogus row containing "Date"
    raw = raw[raw[("Price", "Ticker")] != "Date"]

    # Convert the first column to datetime
    dates = pd.to_datetime(raw[("Price", "Ticker")], format="%d-%m-%y", dayfirst=True)
    raw = raw.drop(columns=[("Price", "Ticker")])

    # Auto-detect tickers from second level of column MultiIndex
    tickers = sorted(set(raw.columns.get_level_values(1)))
    price_dfs: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        # Select all metrics for this ticker
        sub = raw.xs(ticker, level=1, axis=1)
        sub.columns = sub.columns.str.lower()

        # Convert all columns to numeric
        sub = sub.apply(pd.to_numeric, errors="coerce")

        df = sub.copy()
        df.index = dates
        df = df.dropna(how="all").sort_index()

        price_dfs[ticker] = df

    return price_dfs

def load_news_sentiment_data(
    news_file: str = "gnews_data.csv", reddit_file: str = "reddit_data.csv"
):
    news_fp = DATA_DIR / news_file
    reddit_fp = DATA_DIR / reddit_file

    news = pd.read_csv(news_fp)
    reddit = pd.read_csv(reddit_fp)

    news["publishedAt"] = pd.to_datetime(news["publishedAt"], utc=True)
    reddit["publishedAt"] = pd.to_datetime(reddit["publishedAt"], utc=True)

    news["text"] = (
        news["title"].fillna("") + " "
        + news["description"].fillna("") + " "
        + news["content"].fillna("")
    )
    reddit["text"] = reddit["title"].fillna("") + " " + reddit["content"].fillna("")

    return news[["publishedAt", "text"]], reddit[["publishedAt", "text"]]


# --------------------------------------------------------------
# Quick sanity-check when executed directly
# --------------------------------------------------------------
# --------------------------------------------------------------
# TEST: load_news_sentiment_data() with dummy files
# --------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Test stock prices (you already saw this works)
    # ------------------------------------------------------------------
    prices = load_stock_prices()
    print("LOADED SUCCESSFULLY!")
    for ticker, df in prices.items():
        print(f"\n{ticker}: {len(df)} rows")
        print(df.head(2).round(2))

    # ------------------------------------------------------------------
    # 2. Create dummy news & reddit CSVs (only if they don't exist)
    # ------------------------------------------------------------------
    import json
    from datetime import datetime

    dummy_news = [
        {
            "title": "AAPL hits all-time high",
            "description": "Apple stock surges after earnings",
            "content": "Full article about AAPL...",
            "publishedAt": "2025-01-03T12:00:00Z"
        },
        {
            "title": "TSLA robotaxi delay",
            "description": "Elon says 2026 now",
            "content": "Investors react negatively...",
            "publishedAt": "2025-01-04T09:30:00Z"
        }
    ]

    dummy_reddit = [
        {
            "title": "GOOGL moonshot?",
            "content": "AI search will 10x revenue",
            "publishedAt": "2025-01-05T15:22:00Z"
        },
        {
            "title": "TSLA FSD v13 is insane",
            "content": "Just drove 100 miles no touch",
            "publishedAt": "2025-01-06T18:45:00Z"
        }
    ]

    def write_dummy_csv(path: Path, data: list, columns: list):
        if not path.exists():
            pd.DataFrame(data)[columns].to_csv(path, index=False)
            print(f"Created dummy file: {path}")

    write_dummy_csv(DATA_DIR / "gnews_data.csv", dummy_news,
                    ["title", "description", "content", "publishedAt"])
    write_dummy_csv(DATA_DIR / "reddit_data.csv", dummy_reddit,
                    ["title", "content", "publishedAt"])

    # ------------------------------------------------------------------
    # 3. Test sentiment loader
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TESTING load_news_sentiment_data()")
    print("="*60)

    try:
        news_df, reddit_df = load_news_sentiment_data()
        print(f"News rows: {len(news_df)}")
        print(news_df.head(2))
        print(f"\nReddit rows: {len(reddit_df)}")
        print(reddit_df.head(2))
        print("\nload_news_sentiment_data() WORKS!")
    except Exception as e:
        print(f"load_news_sentiment_data() FAILED: {e}")
        raise