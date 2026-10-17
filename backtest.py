"""
backtest.py — Join daily avg sentiment scores with next-day returns via yfinance.
Usage: python backtest.py
"""
import sqlite3
import pandas as pd
import yfinance as yf
import vectorbt as vbt

BUY_THRESHOLD = 0.1
SELL_THRESHOLD = -0.1

def load_scores():
    conn = sqlite3.connect("stock_nlp.db")
    df = pd.read_sql_query("""
        SELECT n.ticker, n.seen_date, s.score
        FROM news n JOIN scores s ON n.id = s.news_id
    """, conn)
    # GDELT: "20260415T083000Z" | HF datasets: ISO "2026-04-15T08:30:00Z"
    iso = pd.to_datetime(df["seen_date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df["date"] = iso.fillna(pd.to_datetime(df["seen_date"], utc=True, errors="coerce")).dt.tz_localize(None)
    return df.dropna(subset=["date"])

def daily_scores(df):
    # Average score per ticker per day
    daily = df.groupby([df["ticker"], df["date"].dt.date])["score"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

def run():
    scored = daily_scores(load_scores())
    tickers = sorted(scored["ticker"].unique())
    print(f"Tickers: {tickers}")

    prices = yf.download(tickers, start=scored["date"].min().strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.ffill()

    # Build signal matrix (score reshaped into a price-like DataFrame, forward-filled)
    sig = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for t in tickers:
        s = scored[scored["ticker"] == t].set_index("date")["score"]
        sig[t] = s.reindex(prices.index).ffill().fillna(0.0)

    # Shift signals by 1 day: trade on next day's open-ish (close-to-close approximation)
    sig = sig.shift(1)

    entries = sig > BUY_THRESHOLD
    exits = sig < SELL_THRESHOLD

    pf = vbt.Portfolio.from_signals(
        prices, entries=entries, exits=exits,
        init_cash=100_000, fees=0.001,  # 0.1% commission
    )
    print(pf.stats())
    print("\n--- Baseline comparison (buy & hold) ---")
    bh = vbt.Portfolio.from_holding(prices, init_cash=100_000)
    print(bh.stats())

    # Quick sanity: correlation of score vs next-day return
    rets = prices.pct_change()
    for t in tickers:
        aligned = pd.concat([sig[t], rets[t]], axis=1).dropna()
        aligned.columns = ["score", "ret"]
        corr = aligned["score"].corr(aligned["ret"])
        print(f"{t}: corr(score, next-day return) = {corr:.4f}")

if __name__ == "__main__":
    run()
