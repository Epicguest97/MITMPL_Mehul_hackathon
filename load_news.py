"""
load_news.py — Load historical news from HuggingFace into stock_nlp.db
(instead of fetching live from GDELT).
Usage: python load_news.py
"""
import sqlite3
import json
import os
import re
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

# Which sources to load. Start with benzinga_6000stocks — ticker-mapped + minute-level.
SOURCES = ["benzinga_6000stocks"]

TICKER_RE = re.compile(r"\$[A-Z]{1,5}\b")  # $AAPL style cashtags

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

def extract_tickers(extra):
    """Get tickers from extra_fields JSON, fall back to $CASHTAGS in text."""
    tickers = set()
    for key in ("stocks", "tickers"):
        vals = extra.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        tickers.update(str(v).upper() for v in vals)
    if not tickers:
        tickers.update(t.upper().strip("$") for t in TICKER_RE.findall(extra.get("_text", "")))
    return tickers

def main():
    conn = init_db()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    for source in SOURCES:
        print(f"Loading {source} (may download a few hundred MB)...")
        # datasets loads all subsets; filter by extra_fields.dataset below
        try:
            ds = load_dataset(
                "Brianferrell787/financial-news-multisource",
                split="train",
                token=hf_token,
            )
        except DatasetNotFoundError as exc:
            raise RuntimeError(
                "This dataset is gated. Request access on its Hugging Face page, "
                "then authenticate with `hf auth login` or set HF_TOKEN before "
                "running load_news.py."
            ) from exc
        rows = 0
        for r in ds:
            extra = json.loads(r["extra_fields"])
            if extra.get("dataset") != source:
                continue
            extra["_text"] = r["text"]
            tickers = extract_tickers(extra)
            for t in tickers:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO news (ticker, title, url, seen_date) VALUES (?,?,?,?)",
                        (t, r["text"][:500], extra.get("url", ""), r["date"]),
                    )
                    rows += 1
                except Exception:
                    pass
            if rows and rows % 100000 == 0:
                conn.commit()
                print(f"  {rows} rows so far...")
        conn.commit()
        print(f"{source}: {rows} ticker-headline rows")
    total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM news").fetchone()[0]
    print(f"Done. {total} rows across {tickers} tickers.")

if __name__ == "__main__":
    main()
