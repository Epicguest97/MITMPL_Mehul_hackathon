"""
db.py — DuckDB schema for the full multi-source news corpus.
All timestamps stored as TIMESTAMP (UTC). Use `trading_ts` for backtest alignment
(NYSE next-session open) and `event_ts` for intra-day work.
"""
import duckdb

DB_PATH = "news.duckdb"

DDL = """
CREATE SEQUENCE IF NOT EXISTS news_seq START 1;

CREATE TABLE IF NOT EXISTS raw_news (
    news_id     BIGINT PRIMARY KEY DEFAULT nextval('news_seq'),
    subset      VARCHAR NOT NULL,        -- e.g. benzinga_6000stocks
    event_ts    TIMESTAMP,               -- minute-level UTC publish time (nullable)
    trading_ts  TIMESTAMP,               -- backtest-safe anchor: next NYSE open in UTC
    time_precision VARCHAR,              -- 'minute' | 'day'
    text        VARCHAR NOT NULL,
    url         VARCHAR,
    source      VARCHAR,                 -- publisher
    extra       VARCHAR                  -- full original extra_fields JSON
);

CREATE TABLE IF NOT EXISTS news_tickers (
    news_id BIGINT REFERENCES raw_news(news_id),
    ticker  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    news_id BIGINT PRIMARY KEY REFERENCES raw_news(news_id),
    pos DOUBLE, neg DOUBLE, neu DOUBLE, score DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_tickers ON news_tickers(ticker);
CREATE INDEX IF NOT EXISTS idx_news_trading ON raw_news(trading_ts);
CREATE INDEX IF NOT EXISTS idx_news_subset ON raw_news(subset);
"""

def get_con(db_path=DB_PATH):
    con = duckdb.connect(db_path)
    con.execute(DDL)
    return con

if __name__ == "__main__":
    con = get_con()
    print("Schema ready:", DB_PATH)
