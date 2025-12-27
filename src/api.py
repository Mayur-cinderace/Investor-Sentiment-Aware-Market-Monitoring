from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import time

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from fastapi.responses import Response

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Investor Sentiment Inference API")

# -----------------------------
# Prometheus metrics
# -----------------------------
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

SENTIMENT_DISTRIBUTION = Histogram(
    "sentiment_score_distribution",
    "Distribution of sentiment scores",
    buckets=(-1, -0.5, 0, 0.5, 1)
)

# -----------------------------
# Load model + scaler
# -----------------------------
MODEL_PATH = "models/AAPL"
model = joblib.load(f"{MODEL_PATH}/rf.joblib")
scaler_x = joblib.load(f"{MODEL_PATH}/scaler_x.joblib")

# -----------------------------
# Sentiment logic
# -----------------------------
POS_WORDS = {"good", "buy", "up", "rise", "gain", "bull", "profit", "growth", "bullish"}
NEG_WORDS = {"bad", "sell", "down", "fall", "loss", "bear", "risk", "crash", "bearish"}

def simple_sentiment(text: str) -> float:
    words = text.lower().split()
    pos = sum(w in POS_WORDS for w in words)
    neg = sum(w in NEG_WORDS for w in words)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0

# -----------------------------
# Input schema
# -----------------------------
class InputText(BaseModel):
    sentence: str

# -----------------------------
# Market context
# -----------------------------
def get_latest_market_context():
    df = pd.read_csv("data/processed/merged_features.csv")
    last = df[df["Ticker"] == "AAPL"].iloc[-1]
    return last["return_lag1"], last["volume_lag1"]

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InputText):
    start_time = time.time()
    REQUEST_COUNT.inc()

    sentiment = simple_sentiment(data.sentence)
    SENTIMENT_DISTRIBUTION.observe(sentiment)

    return_lag1, volume_lag1 = get_latest_market_context()

    X = np.array([[return_lag1, volume_lag1, sentiment]])
    Xs = scaler_x.transform(X)
    pred = model.predict(Xs)[0]

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "sentence": data.sentence,
        "sentiment_score": sentiment,
        "predicted_return": float(pred)
    }

# -----------------------------
# Prometheus scrape endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -----------------------------
# Health check (very important)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
