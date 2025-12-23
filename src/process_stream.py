import pandas as pd
from sentiment import compute_sentiment

def process_stream(path="data/streaming/new_tweets.csv"):
    df = pd.read_csv(path)

    df["sentiment"] = df["text"].apply(compute_sentiment)

    # Aggregate daily sentiment
    daily = (
        df.groupby(["date", "ticker"])["sentiment"]
        .mean()
        .reset_index()
    )

    daily.to_csv("data/processed/new_sentiment.csv", index=False)
    print("Processed new sentiment data")


if __name__ == "__main__":
    process_stream()
