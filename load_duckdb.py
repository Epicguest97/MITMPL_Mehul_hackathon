"""
load_duckdb.py — Stream the gated HF dataset into DuckDB, all finance-relevant subsets.

Prereqs: HF account, accepted dataset terms, `huggingface-cli login`.
Usage:  python load_duckdb.py                 # load everything in SUBSETS
        python load_duckdb.py benzinga_6000stocks   # one subset only
"""
import sys
import json
import re
import duckdb
import pandas as pd
from datasets import load_dataset
from db import get_con

# The finance/ticker-mapped subsets worth your disk. Add/remove freely.
SUBSETS = [
    "benzinga_6000stocks",        # ~6k tickers, analyst ratings + headlines, minute-level
    # "yahoo_finance_felixdrinkall",# 2017-2023, full articles, mentioned_companies
    # "fnspid_news",                # 1999-2023, headline+article, tickers
    # "finsen_us_2007_2023",        # 2007-2023, HAS SENTIMENT LABELS (validation gold)
    # "sentarl_combined",           # 20 assets, headline+subhead+abstract, minute-level
    # "cnbc_headlines",             # 2017-2020 headlines + preview
    # "yahoo_finance_articles",     # 2025 feed, day-level with trading anchor
    # "sp500_daily_headlines",      # 2008-2024 S&P 500 daily headlines
]

TICKER_RE = re.compile(r"^[A-Z][A-Z\.-]{0,6}$")
# Index/ETF junk that shows up in stocks fields but isn't tradable equity
BLOCKLIST = {"INDU", "SPX", "COMP", "NDX", "RUT", "VIX", "DJIA", "SPY",
             "QQQ", "IWM", "DIA", "ES", "YM", "NQ", "CL", "GC", "SI", "HG",
             "BRENT", "WTI", "NG", "ZB", "ZN", "ZW", "ZC", "ZS", "SB", "KC", "CT"}

def clean_tickers(extra):
    raw = extra.get("stocks") or []
    if isinstance(raw, str):
        raw = [raw]
    out = set()
    for t in raw:
        if not isinstance(t, str):
            continue
        t = t.upper().lstrip("$").strip()
        if TICKER_RE.match(t) and t not in BLOCKLIST:
            out.add(t)
    return sorted(out)

def parse_batch(batch):
    rows, tick_pairs = [], []
    for text, date, ef in zip(batch["text"], batch["date"], batch["extra_fields"]):
        extra = json.loads(ef)
        rows.append({
            "subset": extra.get("dataset", "?"),
            "event_ts": date if extra.get("time_precision") == "minute" else None,
            "trading_ts": extra.get("date_trading") or date,
            "time_precision": extra.get("time_precision", "day"),
            "text": text,
            "url": extra.get("url") or extra.get("link") or "",
            "source": extra.get("source") or extra.get("publisher") or "",
            "extra": ef,
        })
        tick_pairs.append(clean_tickers(extra))
    return rows, tick_pairs

def load_subset(con, subset):
    print(f"=== {subset} ===")
    ds = load_dataset(
        "Brianferrell787/financial-news-multisource",
        data_files=f"data/{subset}/*.parquet",
        split="train",
        streaming=True,
    )
    total_rows, total_pairs = 0, 0
    for batch in ds.iter(batch_size=5000):
        rows, tick_pairs = parse_batch(batch)
        df = pd.DataFrame(rows)
        con.register("_b", df)
        con.execute("INSERT INTO raw_news (subset,event_ts,trading_ts,time_precision,text,url,source,extra) SELECT subset,event_ts,trading_ts,time_precision,text,url,source,extra FROM _b")
        ids = con.execute("SELECT news_id FROM raw_news WHERE subset=? ORDER BY news_id DESC LIMIT ?", [subset, len(df)]).df()["news_id"].tolist()
        # ids come back newest-first; batch row i (oldest-first) needs reversed mapping
        ids = list(reversed(ids))
        pairs = [(ids[i], t) for i, ts in enumerate(tick_pairs) for t in ts]
        if pairs:
            pdf = pd.DataFrame(pairs, columns=["news_id", "ticker"])
            con.register("_p", pdf)
            con.execute("INSERT INTO news_tickers SELECT news_id, ticker FROM _p")
        total_rows += len(df)
        total_pairs += len(pairs)
        if total_rows % 100000 < 5000:
            print(f"  {total_rows} articles, {total_pairs} ticker tags")
    print(f"  DONE {subset}: {total_rows} articles, {total_pairs} ticker tags")

def main():
    subsets = sys.argv[1:] or SUBSETS
    con = get_con()
    for s in subsets:
        try:
            load_subset(con, s)
        except Exception as e:
            print(f"  FAILED {s}: {e} -- skipping")
    print(con.execute("SELECT subset, COUNT(*) FROM raw_news GROUP BY 1 ORDER BY 2 DESC").df())

if __name__ == "__main__":
    main()
