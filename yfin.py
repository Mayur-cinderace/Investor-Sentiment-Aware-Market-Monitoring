import yfinance as yf

tickers = ["AAPL", "TSLA", "GOOGL"]
data = yf.download(tickers, start="2023-01-01", end="2025-10-01")
data.to_csv("stock_prices.csv")
