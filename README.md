# Stock NLP v1 — News Sentiment Backtest

Predicts next-day stock movement from news sentiment (FinBERT).

## Setup
    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install transformers torch yfinance pandas vectorbt requests

## Run (in order)
1. `python fetch_news.py`   — pulls 6 months of headlines from GDELT into stock_nlp.db
2. `python score_news.py`   — FinBERT scores every headline (first run downloads ~400MB model)
3. `python backtest.py`     — joins scores with prices, runs vectorbt backtest, prints stats + correlation

## What to look for
- corr(score, next-day return) per ticker: consistently positive = signal exists
- Strategy total return vs buy & hold baseline
- If corr hovers near 0, sentiment alone isn't predictive on your data — tune thresholds
  or add features before scaling anything.

## Files
- fetch_news.py — GDELT news ingestion (edit TICKERS / TIMESPAN at top)
- score_news.py — FinBERT inference into SQLite
- backtest.py   — vectorbt backtest + correlation sanity check
