import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

def load_stock_data():
    df_raw = pd.read_csv(r'data/stock_prices.csv', skiprows=1, header=0, na_values=[''])
    df_raw = df_raw.rename(columns={df_raw.columns[0]: 'Date'})
    first_row = pd.read_csv(r'data/stock_prices.csv', nrows=1, header=None).iloc[0]
    metric_names = first_row[1:].tolist()
    rename_dict = {col: metric for col, metric in zip(df_raw.columns[1:], metric_names)}
    df_raw = df_raw.rename(columns=rename_dict)

    df_long = pd.melt(df_raw, id_vars=['Date'], value_vars=df_raw.columns[1:], var_name='Metric', value_name='Value')
    df_long['Ticker'] = np.tile(['AAPL', 'GOOGL', 'TSLA'] * 5, len(df_raw))
    df_prices = df_long.pivot_table(index=['Date', 'Ticker'], columns='Metric', values='Value', aggfunc='first').reset_index()
    df_prices['Date'] = pd.to_datetime(df_prices['Date'], format='%Y-%m-%d')
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    df_prices[numeric_cols] = df_prices[numeric_cols].astype(float)
    df_prices['Return'] = df_prices.groupby('Ticker')['Close'].pct_change()
    return df_prices

def load_text_data():
    df_reddit = pd.read_csv(r'data/reddit_data.csv')
    df_news   = pd.read_csv(r'data/news_articles.csv')
    df_gnews  = pd.read_csv(r'data/gnews_data.csv')

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

print("Loading data...")
df_prices = load_stock_data()
df_text = load_text_data()

analyzer = SentimentIntensityAnalyzer()
df_text['sentiment'] = df_text['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
daily_sent = df_text.groupby(['date', 'source'])['sentiment'].mean().reset_index()
daily_sent_total = daily_sent.groupby('date')['sentiment'].mean().reset_index()

df_prices['date'] = df_prices['Date'].dt.date
daily_sent_total['date'] = pd.to_datetime(daily_sent_total['date']).dt.date

df_merged = df_prices.merge(daily_sent_total, on='date', how='left')
df_merged['sentiment'] = df_merged['sentiment'].ffill().fillna(0)
df_merged = df_merged.sort_values(['Ticker', 'Date']).reset_index(drop=True)
df_merged['sentiment_lag1'] = df_merged.groupby('Ticker')['sentiment'].shift(1).bfill().fillna(0)

print("\nComputing correlation matrix...")
corr_features = df_merged[['Close', 'Return', 'Volume', 'sentiment', 'sentiment_lag1']]
corr_matrix = corr_features.corr()
print("Correlation Matrix:\n", corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.png')
plt.show()

ticker = 'TSLA'
start_date = df_prices['Date'].min().date()
end_date = df_prices['Date'].max().date()

df_plot = df_merged[
    (df_merged['Ticker'] == ticker) &
    (df_merged['Date'].dt.date >= start_date) &
    (df_merged['Date'].dt.date <= end_date)
]

print("Generating Price vs Lagged Sentiment plot...")
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Close'], name='Price ($)', line=dict(color='purple')), secondary_y=False)
fig1.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['sentiment_lag1'], name='Sentiment (t-1)', line=dict(color='orange')), secondary_y=True)
fig1.update_layout(title=f"{ticker} Price vs Lagged Sentiment", height=400, hovermode='x unified')
fig1.update_yaxes(title_text="Price ($)", secondary_y=False)
fig1.update_yaxes(title_text="Sentiment", secondary_y=True)
fig1.write_image('price_vs_sentiment.png')
fig1.show()

print("Generating Cumulative Returns plot...")
fig2 = go.Figure()
for t in ['AAPL', 'GOOGL', 'TSLA']:
    sub = df_merged[df_merged['Ticker'] == t]
    cum = (1 + sub['Return']).cumprod() - 1
    fig2.add_trace(go.Scatter(x=sub['Date'], y=cum * 100, name=f"{t} (+{cum.iloc[-1]*100:.1f}%)"))
fig2.update_layout(title="Cumulative Returns", height=400, yaxis_title="Return (%)", hovermode='x unified')
fig2.write_image('cumulative_returns.png')
fig2.show()

print("Generating Sentiment by Source plot...")
fig3 = px.box(df_text, x='source', y='sentiment', color='source', title="Sentiment Distribution")
fig3.update_layout(height=400, showlegend=False)
fig3.write_image('sentiment_by_source.png')
fig3.show()

print("Generating Volume & Sentiment plot...")
vol_sent = df_merged.groupby('Date').agg({'Volume': 'sum', 'sentiment': 'mean'}).reset_index()
fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_trace(go.Bar(x=vol_sent['Date'], y=vol_sent['Volume']/1e6, name='Volume (M)', marker_color='green'), secondary_y=False)
fig4.add_trace(go.Scatter(x=vol_sent['Date'], y=vol_sent['sentiment'], name='Sentiment', line=dict(color='red')), secondary_y=True)
fig4.update_layout(title="Volume & Sentiment", height=400, barmode='overlay', hovermode='x unified')
fig4.update_yaxes(title_text="Volume (M)", secondary_y=False)
fig4.update_yaxes(title_text="Sentiment", secondary_y=True)
fig4.write_image('volume_sentiment.png')
fig4.show()

print("Generating Static Word Cloud...")
news_text = ' '.join(df_text[df_text['source'] == 'news']['text'].dropna())
if not news_text.strip():
    print("No news text available.")
else:
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        colormap='viridis',
        stopwords={'the', 'a', 'an', 'and', 'of', 'to', 'in', 'for', 'on', 'with', 'is', 'are', 'it', 'that', 'this', 'as'}
    ).generate(news_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("NewsAPI Word Cloud")
    plt.savefig('word_cloud.png')
    plt.show()
