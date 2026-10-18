"""CLI entrypoint.

Usage
-----
# Run the full pipeline once and print a summary
python -m newsengine

# Run the pipeline and start the API server
python -m newsengine --serve

# Run backtests using data from a CSV file (columns: date, AAPL, MSFT, …)
python -m newsengine --backtest prices.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("newsengine")


def _run_pipeline_and_store() -> None:
    from .pipeline import run_pipeline
    from .store import Store

    log.info("Fetching and processing news …")
    arts = run_pipeline()
    Store().upsert(arts)
    log.info("Stored %d articles", len(arts))

    # Print brief summary to stdout
    from collections import Counter
    events = Counter(a.event_type for a in arts)
    tickers = sorted({t for a in arts for t in a.tickers})
    print(json.dumps({"processed": len(arts), "events": dict(events), "tickers": tickers}, indent=2))


def _run_backtest(prices_csv: str) -> None:
    import pandas as pd

    from .backtest import run_backtest, summarize
    from .portfolio import RiskLimits
    from .store import Store

    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    if prices.index.tz is None:
        import pytz
        prices.index = prices.index.tz_localize(pytz.utc)

    arts = Store().articles()
    if not arts:
        log.warning("No articles in store – run without --backtest first to populate the store.")
        sys.exit(1)

    df = run_backtest(arts, prices, RiskLimits())
    stats = summarize(df)
    print(json.dumps(stats, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Financial News Intelligence Engine")
    p.add_argument("--serve", action="store_true", help="Start the FastAPI server after pipeline run")
    p.add_argument("--backtest", metavar="PRICES_CSV", help="Run backtest with supplied price CSV")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if args.backtest:
        _run_backtest(args.backtest)
        return

    _run_pipeline_and_store()

    if args.serve:
        import uvicorn
        from .api import app  # noqa: F401 – imported for side-effects (route registration)
        uvicorn.run("newsengine.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
