"""
explore.py — Data-quality report: what do we actually have? Run after load_duckdb.py.
Usage: python explore.py
"""
from db import get_con

def main():
    con = get_con()
    print("== Rows per subset ==")
    print(con.execute("SELECT subset, COUNT(*) n, MIN(trading_ts) first, MAX(trading_ts) last FROM raw_news GROUP BY 1 ORDER BY n DESC").df().to_string())
    print("\n== Ticker coverage (top 25 by headline count) ==")
    print(con.execute("""
        SELECT ticker, COUNT(*) n, COUNT(DISTINCT subset) subsets,
               MIN(CAST(trading_ts AS DATE)) first, MAX(CAST(trading_ts AS DATE)) last
        FROM news_tickers t JOIN raw_news n USING(news_id)
        GROUP BY 1 ORDER BY n DESC LIMIT 25
    """).df().to_string())
    print("\n== Tickers with 500+ headlines ==")
    print(con.execute("""
        SELECT COUNT(*) FROM (
          SELECT ticker FROM news_tickers GROUP BY 1 HAVING COUNT(*) >= 500
        )
    """).df().to_string())
    print("\n== Time precision mix ==")
    print(con.execute("SELECT subset, time_precision, COUNT(*) FROM raw_news GROUP BY 1,2 ORDER BY 1,3 DESC").df().to_string())
    print("\n== URLs missing (dedup key will rely on text+date) ==")
    print(con.execute("SELECT subset, COUNT(*) FILTER (WHERE url='' OR url IS NULL) missing, COUNT(*) total FROM raw_news GROUP BY 1").df().to_string())

if __name__ == "__main__":
    main()
