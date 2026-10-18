"""Step 9: backtest with no look-ahead and transaction costs.

Timing: at each rebalance time t, signals use ONLY articles with published <= t.
Weights decided at t are applied to the return from the NEXT bar (t -> t+1), so a
signal never earns a return that was already realised when it was computed.
Costs: cost_bps charged on turnover each rebalance.
prices: DataFrame indexed by tz-aware UTC timestamps (bar times), columns = tickers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Article
from .portfolio import RiskLimits, target_weights
from .signals import aggregate_by_ticker


def run_backtest(articles: list[Article], prices: pd.DataFrame,
                 limits: RiskLimits = RiskLimits(), cost_bps: float = 5.0) -> pd.DataFrame:
    rets = prices.pct_change().fillna(0.0)
    times = list(prices.index)
    weights: dict[str, float] = {}
    rows = []
    for i, t in enumerate(times[:-1]):
        visible = [a for a in articles if a.published <= t]  # no look-ahead
        sig = aggregate_by_ticker(visible, t.to_pydatetime())
        sig = {k: v for k, v in sig.items() if k in prices.columns}
        new_w = target_weights(sig, limits, weights)
        names = set(new_w) | set(weights)
        turnover = sum(abs(new_w.get(n, 0.0) - weights.get(n, 0.0)) for n in names)
        cost = turnover * cost_bps / 1e4
        nxt = rets.iloc[i + 1]
        gross_ret = sum(w * nxt[n] for n, w in new_w.items())
        rows.append({"time": times[i + 1], "gross_return": gross_ret, "cost": cost,
                     "net_return": gross_ret - cost, "turnover": turnover})
        weights = new_w
    df = pd.DataFrame(rows).set_index("time")
    df["equity"] = (1 + df["net_return"]).cumprod()
    return df


def summarize(df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    if df.empty:
        return {}
    r = df["net_return"]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    dd = (df["equity"] / df["equity"].cummax() - 1).min()
    return {"total_return": float(df["equity"].iloc[-1] - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd else 0.0,
            "max_drawdown": float(dd), "total_cost": float(df["cost"].sum()),
            "avg_turnover": float(df["turnover"].mean())}
