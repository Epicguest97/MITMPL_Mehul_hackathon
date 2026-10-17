# Stock NLP v1 — News Sentiment Backtest

Predicts next-day stock movement from news sentiment (FinBERT).

## Setup
    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install transformers torch yfinance pandas vectorbt requests datasets

## Run (in order)
1. `python fetch_news.py`   — pulls 6 months of headlines from GDELT into stock_nlp.db
2. `python score_news.py`   — FinBERT scores every headline (first run downloads ~400MB model)
3. `python backtest.py`     — joins scores with prices, runs vectorbt backtest, prints stats + correlation

### Loading the historical Hugging Face dataset

`load_news.py` uses the gated `Brianferrell787/financial-news-multisource`
dataset. Request access on its Hugging Face page, then authenticate before
running the loader:

    hf auth login
    python load_news.py

Alternatively, set a Hugging Face access token in `HF_TOKEN` for that command.
Do not commit the token to the repository.

## What to look for
- corr(score, next-day return) per ticker: consistently positive = signal exists
- Strategy total return vs buy & hold baseline
- If corr hovers near 0, sentiment alone isn't predictive on your data — tune thresholds
  or add features before scaling anything.

## Files
- fetch_news.py — GDELT news ingestion (edit TICKERS / TIMESPAN at top)
- score_news.py — FinBERT inference into SQLite
- backtest.py   — vectorbt backtest + correlation sanity check
