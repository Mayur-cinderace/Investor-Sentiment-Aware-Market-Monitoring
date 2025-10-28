import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load .env
load_dotenv()
key = os.getenv("NEWSAPI_KEY")
if not key:
    raise SystemExit("Set NEWSAPI_KEY in env or .env")

# Fetch news data
url = "https://newsapi.org/v2/everything"
params = {
    "q": "stock market",
    "language": "en",
    "pageSize": 100,
}
headers = {"Authorization": key}

resp = requests.get(url, params=params, headers=headers)
resp.raise_for_status()
data = resp.json()
articles = data.get("articles", [])

print(f"Fetched {len(articles)} articles")

# Convert to DataFrame
if articles:
    df = pd.DataFrame(articles)

    # Select useful fields
    df = df[["source", "author", "title", "description", "url", "publishedAt", "content"]]
    # Flatten nested 'source' dicts
    df["source"] = df["source"].apply(lambda s: s.get("name") if isinstance(s, dict) else s)

    # Save to CSV
    df.to_csv("news_articles.csv", index=False, encoding="utf-8")
    print("✅ Saved to news_articles.csv")
else:
    print("⚠️ No articles found.")
