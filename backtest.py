"""
backtest.py — Per-ticker daily sentiment vs next-day returns (DuckDB + yfinance + vectorbt).

Dedup note: the corpus warns of cross-subset duplicates (syndicated headlines).
We dedup on (ticker, trading day, normalized text) inside the query.
Usage: python backtest.py [N_TOP_TICKERS]   (default 10, uses tickers with most headlines)
"""
import sys
import pandas as pd
import yfinance as yf
import vectorbt as vbt
from db import get_con

BUY_THRESHOLD = 0.1
SELL_THRESHOLD = -0.1
MIN_HEADLINES = 300          # ignore tickers with thin coverage
MAX_ROWS_PER_TICKER_DAY = 50 # cap: a flood of articles on one day shouldn't dominate

def daily_scores(con, n_top):
    return con.execute(f"""
        WITH dedup AS (
            SELECT DISTINCT ticker, CAST(trading_ts AS DATE) d,
                   trim(lower(text)) AS norm_text, score
            FROM news_tickers t
            JOIN raw_news n USING(news_id)
            JOIN scores s USING(news_id)
        ),
        capped AS (
            SELECT *, row_number() OVER (PARTITION BY ticker, d ORDER BY score DESC) rn
            FROM dedup
        )
        SELECT ticker, d, AVG(score) avg_score, COUNT(*) n_news
        FROM capped WHERE rn <= {MAX_ROWS_PER_TICKER_DAY}
        GROUP BY 1, 2
        HAVING COUNT(*) >= 2  -- need at least 2 articles that day to trust the average
    """).df()

def main():
    n_top = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    con = get_con()
    ds = daily_scores(con, n_top)

    counts = ds.groupby("ticker")["n_news"].sum().sort_values(ascending=False)
    tickers = [t for t in counts.index if counts[t] >= MIN_HEADLINES][:n_top]
    print(f"Backtesting: {tickers}")
    if not tickers:
        print("No tickers pass MIN_HEADLINES — lower it or run load/score first.")
        return

    prices = yf.download(tickers, start=str(ds["d"].min()), auto_adjust=True,
                         progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.ffill()

    sig = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    for t in tickers:
        s = ds[ds["ticker"] == t].set_index("d")["avg_score"]
        sig[t] = s.reindex(prices.index).ffill().fillna(0.0)
    sig = sig.shift(1)  # trade next day — no lookahead

    pf = vbt.Portfolio.from_signals(
        prices, entries=sig > BUY_THRESHOLD, exits=sig < SELL_THRESHOLD,
        init_cash=100_000, fees=0.001,
    )
    print(pf.stats())
    print("\n--- Buy & hold baseline ---")
    print(vbt.Portfolio.from_holding(prices, init_cash=100_000).stats())

    rets = prices.pct_change()
    print("\n--- corr(avg_score, next-day return) ---")
    for t in tickers:
        al = pd.concat([sig[t], rets[t]], axis=1).dropna()
        print(f"{t}: n={len(al):5d}  corr={al.iloc[:,0].corr(al.iloc[:,1]):+.4f}")

if __name__ == "__main__":
    main()
