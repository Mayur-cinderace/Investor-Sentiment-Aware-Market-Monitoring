import requests
import pandas as pd
import time

# Parameters
BASE_URL = "https://api.pullpush.io/reddit/search/comment/"
params = {"subreddit": "wallstreetbets", "q": "stock", "size": 100}

resp = requests.get(BASE_URL, params=params)
data = resp.json()["data"]

# Map Reddit fields to your CSV schema
records = []
for c in data:
    records.append({
        "source": "reddit",                  # All come from Reddit
        "author": c.get("author"),           # Reddit username
        "title": None,                       # Reddit comments don't have a title
        "description": None,                 # Optional
        "url": f"https://reddit.com{c.get('permalink','')}",  # link to comment
        "publishedAt": pd.to_datetime(c.get("created_utc"), unit='s'),
        "content": c.get("body")             # actual comment text
    })

# Create DataFrame with exact column order
df = pd.DataFrame(records, columns=["source","author","title","description","url","publishedAt","content"])

# Save to CSV
df.to_csv("reddit_data.csv", index=False, encoding="utf-8")
print(f"✅ Saved {len(df)} Reddit comments to reddit_data.csv")
print(df.head())
