import os
import shutil
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(exist_ok=True)

STOCK_FILE = RAW_DIR / "stock_prices.csv"

def fetch_stock_data():
    """
    In cloud environments (Codespaces), Yahoo Finance is blocked.
    If stock_prices.csv already exists, reuse it safely.
    """
    if STOCK_FILE.exists():
        print("Using existing stock_prices.csv (no external fetch)")
        return

    raise RuntimeError(
        "stock_prices.csv not found. "
        "Place it manually in data/raw when running in Codespaces."
    )

def copy_news_files():
    source_dir = Path("data")
    target_dir = RAW_DIR

    files = ["news_articles.csv", "gnews_data.csv", "reddit_data.csv"]

    for f in files:
        src = source_dir / f
        dst = target_dir / f

        if not src.exists():
            print(f"[WARN] {src} not found")
            continue

        if src.resolve() == dst.resolve():
            continue

        shutil.copy(src, dst)
        print(f"Copied {src} → {dst}")

if __name__ == "__main__":
    fetch_stock_data()
    copy_news_files()
