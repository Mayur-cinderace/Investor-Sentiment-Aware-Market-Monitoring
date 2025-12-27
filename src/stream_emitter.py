import pandas as pd
import time
import requests

STREAM_FILE = "data/streaming/news_stream.csv"
API_URL = "http://localhost:8000/predict"
WAIT = 3  # seconds

def main():
    df = pd.read_csv(STREAM_FILE)

    print("📡 Starting stream...")
    print("-" * 50)

    while True:  # infinite streaming
        for _, row in df.iterrows():
            payload = {
                "sentence": row["text"]
            }

            try:
                r = requests.post(API_URL, json=payload, timeout=5)
                out = r.json()

                print(f"[{row['ticker']}]")
                print("Text      :", row["text"])
                print("Sentiment :", round(out["sentiment_score"], 3))
                print("Prediction:", round(out["predicted_return"], 4))
                print("-" * 50)

            except Exception as e:
                print("❌ API error:", e)

            time.sleep(WAIT)

if __name__ == "__main__":
    main()
