import pandas as pd
from pathlib import Path

file_path = Path("data/stock_prices.csv")  # adjust path if needed
df = pd.read_csv(file_path, header=[0, 1])
print("Shape:", df.shape)
print("Columns (first few):")
print(df.columns.tolist()[:10])
print("\nFirst few rows:")
print(df.head())
