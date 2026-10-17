"""
fetch_news.py — Pull news headlines for tickers from GDELT (free, no API key).
Usage: python fetch_news.py
"""
import sqlite3
import time
import requests

TICKERS = {
    "AAPL": '("Apple" OR "AAPL")',
    "TSLA": '("Tesla" OR "TSLA")',
    "MSFT": '("Microsoft" OR "MSFT")',
    "NVDA": '("Nvidia" OR "NVDA")',
    "JPM":  '("JPMorgan" OR "JPM")',
}
TIMESPAN = "6months"  # change as needed: 1week, 1month, 3months, 6months, 1year
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def init_db():
    conn = sqlite3.connect("stock_nlp.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            title TEXT,
            url TEXT,
            seen_date TEXT,
            UNIQUE(ticker, url)
        )""")
    return conn

def fetch_for_ticker(ticker, query):
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "ArtList",
        "maxrecords": 250,
        "format": "json",
        "timespan": TIMESPAN,
        "sort": "datedesc",
    }
    r = requests.get(GDELT_URL, params=params, timeout=30)
    if r.status_code != 200:
        print(f"[{ticker}] GDELT error {r.status_code}")
        return []
    try:
        return r.json().get("articles", [])
    except Exception:
        print(f"[{ticker}] bad JSON response")
        return []

def main():
    conn = init_db()
    for ticker, query in TICKERS.items():
        articles = fetch_for_ticker(ticker, query)
        inserted = 0
        for a in articles:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO news (ticker, title, url, seen_date) VALUES (?,?,?,?)",
                    (ticker, a.get("title", ""), a.get("url", ""), a.get("seendate", "")),
                )
                inserted += 1
            except Exception:
                pass
        conn.commit()
        print(f"[{ticker}] fetched {len(articles)}, new rows: {inserted}")
        time.sleep(5)  # be polite to GDELT
    total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    print(f"Done. Total headlines in DB: {total}")

if __name__ == "__main__":
    main()
