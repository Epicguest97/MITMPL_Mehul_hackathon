# Stock NLP v2 — Full-corpus DuckDB pipeline

Upgrade from v1: stores the entire multi-source news corpus in DuckDB,
uses backtest-safe trading timestamps, and dedups syndicated headlines.

## Setup
    pip install duckdb datasets huggingface_hub transformers torch yfinance pandas vectorbt
    huggingface-cli login            # after accepting dataset terms

## Run in order
1. `python db.py`                      -> creates news.duckdb schema
2. `python load_duckdb.py`             -> streams all SUBSETS (edit list in file) into DuckDB
                                          (or: python load_duckdb.py benzinga_6000stocks)
3. `python explore.py`                 -> data-quality report: coverage, date ranges, URL gaps
4. `python score_news.py`              -> FinBERT scores everything un-scored
5. `python backtest.py 10`             -> backtest top-10-coverage tickers

## Schema (news.duckdb)
- raw_news: subset, event_ts (minute UTC), trading_ts (next NYSE open — USE THIS
  for backtest alignment), time_precision, text, url, source, extra (raw JSON)
- news_tickers: (news_id, ticker) many-to-many
- scores: (news_id, pos, neg, neu, score)  score = pos - neg

## Why trading_ts and not date
Day-level rows store midnight UTC in `date` — trading on "today's news" with that
timestamp is lookahead bias. `trading_ts` is the next NYSE session open: scores
attached to it are tradable. backtest.py aligns on trading_ts then shifts 1 day.

## Tuning knobs
- load_duckdb.py: SUBSETS list (all 24 available; finance-relevant preselected)
- backtest.py: BUY/SELL_THRESHOLD, MIN_HEADLINES, MAX_ROWS_PER_TICKER_DAY

## Validation step (before trusting backtest)
Score ~1000 rows of subset `finsen_us_2007_2023` — its extra_fields carry human
sentiment labels. Compare FinBERT score sign vs label: want >=75% agreement.
