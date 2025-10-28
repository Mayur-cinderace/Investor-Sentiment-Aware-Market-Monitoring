import requests
import pandas as pd

# GNews API endpoint and parameters
url = "https://gnews.io/api/v4/search"
params = {
    "q": "stock market OR investing",
    "lang": "en",
    "country": "us",
    "max": 100,
    "apikey": "b948e6632c5e66f210a2c19a6757f4b1"  # <-- your key
}

# Send request
response = requests.get(url, params=params)
data = response.json()

# Extract articles safely
articles = data.get("articles", [])

# Normalize and extract fields
records = []
for a in articles:
    records.append({
        "source": a.get("source", {}).get("name"),
        "author": a.get("author"),
        "title": a.get("title"),
        "description": a.get("description"),
        "url": a.get("url"),
        "publishedAt": a.get("publishedAt"),
        "content": a.get("content")
    })

# Convert to DataFrame
df = pd.DataFrame(records, columns=[
    "source", "author", "title", "description", "url", "publishedAt", "content"
])

# Save to CSV
df.to_csv("gnews_data.csv", index=False, encoding="utf-8")

print(f"✅ Saved {len(df)} articles to gnews_data.csv")
print(df.head())
