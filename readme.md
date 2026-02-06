# 📊 Investor Sentiment-Aware Market Monitoring

A machine learning system that monitors financial markets using investor sentiment analysis from news articles and social media (Reddit). This project integrates real-time data ingestion, sentiment analysis, predictive modeling, data drift detection, and automated model retraining.

## 🎯 Features

- **Multi-Source Data Ingestion**: Collect financial news from Google News, stock prices from Yahoo Finance, and discussions from Reddit
- **Sentiment Analysis**: Extract investor sentiment from unstructured text data
- **Predictive Modeling**: Train multiple ML models (Random Forest, MLP) to predict market movements
- **Real-Time Monitoring**: Live prediction capability for market sentiment analysis
- **Data Drift Detection**: Automatically detect distribution shifts in production data
- **Automated Retraining**: Intelligently retrain models when data drift is detected
- **REST API**: FastAPI-based service for real-time predictions
- **Interactive UI**: Streamlit dashboard for monitoring and visualization
- **Experiment Tracking**: MLflow integration for reproducible ML workflows
- **Metrics & Monitoring**: Prometheus-compatible metrics collection

## 📈 Project Architecture

```
Data Sources (News, Stocks, Reddit)
    ↓
Ingestion (gnews.py, yfin.py, reddit_data.py)
    ↓
Sentiment Analysis (sentiment.py)
    ↓
Feature Engineering (build_features.py)
    ↓
Model Training (train_models.py) ← MLFlow Tracking
    ↓
Model Evaluation (evaluate_models.py)
    ↓
Drift Detection (drift_detection.py)
    ↓
Retraining Decision (should_retrain.py)
    ↓
Live Predictions (api.py, live_predict.py)
    ↓
Dashboard (ui/app.py)
```

## 📋 Data Pipeline

The project uses **DVC (Data Version Control)** for reproducible ML pipelines:

- **Ingest**: Collect raw stock prices and news data
- **Features**: Build merged feature set with sentiment scores
- **Train**: Train ML models on prepared features
- **Evaluate**: Generate performance metrics
- **Drift Detection**: Monitor data and model drift
- **Retrain Decision**: Determine if retraining is necessary
- **Retrain**: Automatically retrain models when drift is detected

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git & DVC
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mayur-cinderace/Investor-Sentiment-Aware-Market-Monitoring.git
   cd Investor-Sentiment-Aware-Market-Monitoring
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC**
   ```bash
   dvc pull  # Pull data from DVC storage
   ```

### Configuration

Edit `params.yaml` to customize:
- Stock tickers (AAPL, GOOGL, TSLA, etc.)
- Training parameters (epochs, batch size, test split)
- Sentiment keywords
- Sequence length for time-series modeling

```yaml
tickers:
  - AAPL
  - GOOGL
  - TSLA

training:
  test_split: 0.2
  epochs: 50
  batch_size: 32
```

## 📊 Usage

### 1. Data Ingestion & Processing

```bash
# Ingest fresh data
python src/ingest_data.py

# Build features with sentiment analysis
python src/build_features.py
```

### 2. Model Training

```bash
# Train models using DVC pipeline
dvc repro

# Or run MLFlow-tracked training directly
python src/train_models.py
```

### 3. Monitor Model Performance

```bash
# Launch MLFlow UI to view experiments
mlflow ui  # Visit http://localhost:5000

# View evaluation metrics
python src/evaluate_models.py
```

### 4. Detect & Handle Data Drift

```bash
# Check for data drift
python src/drift_detection.py

# Decide if retraining is needed
python src/should_retrain.py

# Automatically retrain if drift is detected
python src/retrain_if_drift.py
```

### 5. Live Predictions

**REST API**:
```bash
# Start the FastAPI server
python src/api.py

# API will be available at http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

**Command Line**:
```bash
# Make predictions from text input
python src/live_predict_from_text.py --text "Apple stock surges on positive earnings"
```

**Interactive Dashboard**:
```bash
# Launch Streamlit UI
streamlit run ui/app.py
```

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── ingest_data.py           # Data collection
│   ├── sentiment.py             # Sentiment analysis
│   ├── build_features.py        # Feature engineering
│   ├── train_models.py          # Model training (RF, MLP)
│   ├── evaluate_models.py       # Performance evaluation
│   ├── drift_detection.py       # Data/model drift detection
│   ├── should_retrain.py        # Retraining decision logic
│   ├── retrain_if_drift.py      # Automated retraining
│   ├── api.py                   # FastAPI service
│   ├── live_predict.py          # Real-time predictions
│   └── live_predict_from_text.py # Text-based predictions
│
├── ui/                           # User interface
│   └── app.py                   # Streamlit dashboard
│
├── data/
│   ├── raw/                     # Original data
│   ├── processed/               # Cleaned & processed data
│   └── streaming/               # Real-time data streams
│
├── mlruns/                      # MLFlow experiment tracking
├── metrics/                     # Performance metrics
├── drift_reports/               # Drift detection reports
│
├── params.yaml                  # Pipeline configuration
├── dvc.yaml                     # DVC pipeline definition
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── prometheus.yml               # Prometheus metrics config
└── README.md                    # This file
```

## 🔧 Technologies & Libraries

| Component | Technology |
|-----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **ML Models** | Scikit-learn (Random Forest), TensorFlow (MLP) |
| **Sentiment Analysis** | NLTK |
| **Experiment Tracking** | MLFlow |
| **REST API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Data Versioning** | DVC |
| **Monitoring** | Prometheus |
| **Containerization** | Docker |

## 📊 Models

The system trains multiple models:
- **Random Forest**: Fast, interpretable predictions on tabular features
- **MLP (Neural Network)**: Deep learning approach for complex pattern recognition

Models are tracked with MLFlow and stored in `mlruns/` directory.

## ⚠️ Drift Detection & Retraining

The system automatically monitors for:
- **Data Drift**: Changes in input feature distributions
- **Concept Drift**: Changes in the relationship between features and target
- **Model Performance Drift**: Degradation in prediction accuracy

When drift is detected, the system can automatically trigger retraining to maintain model performance.

## 📈 Monitoring & Metrics

View real-time metrics:
```bash
# Prometheus metrics available at http://localhost:9090 (if running)
curl http://localhost:8000/metrics
```

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t sentiment-monitor .

# Run container
docker run -p 8000:8000 -p 8501:8501 sentiment-monitor
```

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Yahoo Finance for stock data
- Google News API for news articles
- NLTK for sentiment analysis tools
- MLFlow for experiment tracking
- FastAPI & Streamlit communities

---
**Last Updated**: February 2026
