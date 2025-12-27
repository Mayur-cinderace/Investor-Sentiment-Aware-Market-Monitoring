import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Investor Sentiment Monitor",
    layout="centered"
)

st.title("📈 Investor Sentiment–Aware Market Monitoring")
st.caption("Live sentiment-based return prediction using deployed ML model")

st.markdown("---")

# ---------------------------
# Input
# ---------------------------
sentence = st.text_area(
    "Enter a market-related sentence (tweet/news):",
    placeholder="Tesla stock looks bullish after earnings...",
    height=120
)

predict_btn = st.button("🔍 Predict")

# ---------------------------
# Output
# ---------------------------
if predict_btn:
    if not sentence.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Analyzing sentiment and predicting return..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"sentence": sentence},
                    timeout=5
                )
                result = response.json()

                st.success("Prediction successful")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Sentiment Score",
                        value=round(result["sentiment_score"], 3)
                    )

                with col2:
                    st.metric(
                        label="Predicted Return",
                        value=round(result["predicted_return"], 4)
                    )

                st.markdown("#### Model Output")
                st.json(result)

            except Exception as e:
                st.error(f"API error: {e}")

st.markdown("---")
st.caption("Backend: FastAPI • Monitoring: Prometheus • MLOps: DVC + MLflow")
