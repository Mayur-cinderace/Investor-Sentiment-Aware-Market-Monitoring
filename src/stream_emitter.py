import pandas as pd
import time
from pathlib import Path

SOURCE = "data/streaming/news_stream.csv"
SINK = "data/processed/new_sentiment.csv"

BATCH_SIZE = 5
SLEEP_SECONDS = 5


def main():
    df = pd.read_csv(SOURCE)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]

        if Path(SINK).exists():
            batch.to_csv(SINK, mode="a", header=False, index=False)
        else:
            batch.to_csv(SINK, index=False)

        print(f"Streamed rows {i} → {i + len(batch)}")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
