import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def load_stock_data():
    df_raw = pd.read_csv(r'data\stock_prices.csv', skiprows=1, header=0, na_values=[''])
    df_raw = df_raw.rename(columns={df_raw.columns[0]: 'Date'})
    first_row = pd.read_csv(r'data\stock_prices.csv', nrows=1, header=None).iloc[0]
    metric_names = first_row[1:].tolist()
    rename_dict = {col: metric for col, metric in zip(df_raw.columns[1:], metric_names)}
    df_raw = df_raw.rename(columns=rename_dict)

    df_long = pd.melt(df_raw, id_vars=['Date'], value_vars=df_raw.columns[1:], var_name='Metric', value_name='Value')
    df_long['Ticker'] = np.tile(['AAPL', 'GOOGL', 'TSLA'] * 5, len(df_raw))
    df_prices = df_long.pivot_table(index=['Date', 'Ticker'], columns='Metric', values='Value', aggfunc='first').reset_index()
    df_prices['Date'] = pd.to_datetime(df_prices['Date'], format='%d-%m-%y')
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    df_prices[numeric_cols] = df_prices[numeric_cols].astype(float)
    df_prices['Return'] = df_prices.groupby('Ticker')['Close'].pct_change()
    return df_prices

def load_text_data():
    df_reddit = pd.read_csv(r'data\reddit_data.csv')
    df_news = pd.read_csv(r'data\news_articles.csv')
    df_gnews = pd.read_csv(r'data\gnews_data.csv')

    for df, src in [(df_reddit, 'reddit'), (df_news, 'news'), (df_gnews, 'gnews')]:
        df.rename(columns={'content': 'text'}, inplace=True)
        df['source'] = src
        df = df[['text', 'publishedAt', 'source']]

    df_text = pd.concat([df_reddit, df_news, df_gnews], ignore_index=True)
    df_text['text'] = df_text['text'].astype(str).str.lower()
    df_text['text'] = df_text['text'].str.replace(r'http\S+|www\S+', '', regex=True)
    df_text['text'] = df_text['text'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    df_text['date'] = pd.to_datetime(df_text['publishedAt'], errors='coerce').dt.date
    df_text = df_text.dropna(subset=['date'])
    return df_text

df_prices = load_stock_data()
df_text = load_text_data()

positive_words = ['good', 'buy', 'up', 'rise', 'gain', 'positive', 'bull', 'strong', 'profit', 'growth', 'high', 'best', 'win', 'success', 'pump', 'moon', 'rocket']
negative_words = ['bad', 'sell', 'down', 'fall', 'loss', 'negative', 'bear', 'weak', 'decline', 'low', 'worst', 'fail', 'crash', 'risk', 'dump', 'scam']

def simple_sentiment(text):
    words = text.split()
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    total = pos_count + neg_count
    if total == 0:
        return 0
    return (pos_count - neg_count) / total

df_text['sentiment'] = df_text['text'].apply(simple_sentiment)
daily_sent = df_text.groupby(['date', 'source'])['sentiment'].mean().reset_index()
daily_sent_total = daily_sent.groupby('date')['sentiment'].mean().reset_index()

df_prices['date'] = df_prices['Date'].dt.date
daily_sent_total['date'] = pd.to_datetime(daily_sent_total['date']).dt.date

df_merged = df_prices.merge(daily_sent_total, on='date', how='left')
df_merged['sentiment'] = df_merged['sentiment'].ffill().fillna(0)
df_merged = df_merged.sort_values(['Ticker', 'Date']).reset_index(drop=True)
df_merged['sentiment_lag1'] = df_merged.groupby('Ticker')['sentiment'].shift(1).bfill().fillna(0)

ticker = 'TSLA'
df_ticker = df_merged[df_merged['Ticker'] == ticker].copy()
df_ticker = df_ticker.sort_values('Date')

df_ticker['return_lag1'] = df_ticker['Return'].shift(1)
df_ticker['volume_lag1'] = df_ticker['Volume'].shift(1)
df_ticker.dropna(inplace=True)

df_ticker['target_return'] = df_ticker['Return'].shift(-1)
df_ticker.dropna(inplace=True)

features = ['return_lag1', 'volume_lag1', 'sentiment_lag1']
X = df_ticker[features].values
y = df_ticker['target_return'].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

train_size = int(len(X) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_model(model, loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

def predict_model(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            outputs = model(batch_x)
            preds.extend(outputs.squeeze().numpy())
    return np.array(preds)

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

input_size = X.shape[1]
linear_model = LinearModel(input_size)
train_model(linear_model, train_loader)
y_pred_linear_scaled = predict_model(linear_model, test_loader)
y_pred_linear = scaler_y.inverse_transform(y_pred_linear_scaled.reshape(-1,1)).flatten()
mse_linear = np.mean((y_test - y_pred_linear_scaled)**2)
mae_linear = np.mean(np.abs(y_test - y_pred_linear_scaled))
print(f"Linear Regression - MSE: {mse_linear:.6f}, MAE: {mae_linear:.6f}")

class MLPModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp_model = MLPModel(input_size)
train_model(mlp_model, train_loader)
y_pred_mlp_scaled = predict_model(mlp_model, test_loader)
y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1,1)).flatten()
mse_mlp = np.mean((y_test - y_pred_mlp_scaled)**2)
mae_mlp = np.mean(np.abs(y_test - y_pred_mlp_scaled))
print(f"MLP - MSE: {mse_mlp:.6f}, MAE: {mae_mlp:.6f}")

def create_sequences(data_X, data_y, seq_length):
    xs, ys = [], []
    for i in range(len(data_X) - seq_length):
        x = data_X[i:i+seq_length]
        y = data_y[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

train_size_seq = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:train_size_seq], X_seq[train_size_seq:]
y_train_seq, y_test_seq = y_seq[:train_size_seq], y_seq[train_size_seq:]

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_seq_dataset = SeqDataset(X_train_seq, y_train_seq)
test_seq_dataset = SeqDataset(X_test_seq, y_test_seq)
train_seq_loader = DataLoader(train_seq_dataset, batch_size=32, shuffle=False)
test_seq_loader = DataLoader(test_seq_dataset, batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

hidden_size = 50
num_layers = 2
lstm_model = LSTMModel(input_size, hidden_size, num_layers)
train_model(lstm_model, train_seq_loader)
y_pred_lstm_scaled = predict_model(lstm_model, test_seq_loader)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1,1)).flatten()
mse_lstm = np.mean((y_test_seq - y_pred_lstm_scaled)**2)
mae_lstm = np.mean(np.abs(y_test_seq - y_pred_lstm_scaled))
print(f"LSTM - MSE: {mse_lstm:.6f}, MAE: {mae_lstm:.6f}")

dates_test = df_ticker['Date'].iloc[train_size:train_size+len(y_test)].values

plt.figure(figsize=(14, 8))
plt.plot(dates_test, scaler_y.inverse_transform(y_scaled[train_size:train_size+len(y_test)].reshape(-1,1)), label='Actual', color='black')
plt.plot(dates_test, y_pred_linear, label='Linear Regression', color='blue')
plt.plot(dates_test, y_pred_mlp, label='MLP', color='green')
plt.plot(dates_test[:len(y_pred_lstm)], y_pred_lstm, label='LSTM', color='red')
plt.title(f'{ticker} Next Day Return Predictions (Sentiment-Aware)')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_predictions.png')
plt.show()
