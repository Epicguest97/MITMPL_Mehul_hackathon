"""Stock Index Rebalancer Module.

Maintains a 20-stock blue-chip equity index and dynamically executes
rebalancing decisions (REMOVE, ADD, OVERWEIGHT, UNDERWEIGHT, HOLD)
driven by real-time NLP sentiment analysis, DeBERTa event classification,
and market impact severity scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from .models import Article

log = logging.getLogger(__name__)

# ── Baseline 20-Stock Index Universe (Equal Weight: 5.0% each) ─────────────────
STOCK_METADATA: dict[str, dict[str, str]] = {
    "AAPL":  {"name": "Apple Inc.",               "sector": "Technology"},
    "MSFT":  {"name": "Microsoft Corporation",    "sector": "Technology"},
    "NVDA":  {"name": "NVIDIA Corporation",       "sector": "Technology"},
    "AMZN":  {"name": "Amazon.com Inc.",          "sector": "Consumer Discretionary"},
    "GOOGL": {"name": "Alphabet Inc.",            "sector": "Communication Services"},
    "META":  {"name": "Meta Platforms Inc.",      "sector": "Communication Services"},
    "TSLA":  {"name": "Tesla Inc.",               "sector": "Consumer Discretionary"},
    "BRK.B": {"name": "Berkshire Hathaway Inc.",  "sector": "Financials"},
    "JPM":   {"name": "JPMorgan Chase & Co.",     "sector": "Financials"},
    "V":     {"name": "Visa Inc.",                "sector": "Financials"},
    "UNH":   {"name": "UnitedHealth Group Inc.",  "sector": "Healthcare"},
    "JNJ":   {"name": "Johnson & Johnson",        "sector": "Healthcare"},
    "XOM":   {"name": "Exxon Mobil Corp.",        "sector": "Energy"},
    "PG":    {"name": "Procter & Gamble Co.",     "sector": "Consumer Staples"},
    "HD":    {"name": "Home Depot Inc.",          "sector": "Consumer Discretionary"},
    "MA":    {"name": "Mastercard Inc.",          "sector": "Financials"},
    "LLY":   {"name": "Eli Lilly and Co.",        "sector": "Healthcare"},
    "COST":  {"name": "Costco Wholesale Corp.",   "sector": "Consumer Staples"},
    "ABBV":  {"name": "AbbVie Inc.",              "sector": "Healthcare"},
    "AVGO":  {"name": "Broadcom Inc.",            "sector": "Technology"},
    # Reserve / Candidate replacements
    "AMD":   {"name": "Advanced Micro Devices",   "sector": "Technology"},
    "NFLX":  {"name": "Netflix Inc.",             "sector": "Communication Services"},
    "WMT":   {"name": "Walmart Inc.",             "sector": "Consumer Staples"},
    "BAC":   {"name": "Bank of America Corp.",    "sector": "Financials"},
    "CRM":   {"name": "Salesforce Inc.",          "sector": "Technology"},
    "ORCL":  {"name": "Oracle Corporation",       "sector": "Technology"},
    "CSCO":  {"name": "Cisco Systems Inc.",       "sector": "Technology"},
    "QCOM":  {"name": "Qualcomm Inc.",            "sector": "Technology"},
    "CVX":   {"name": "Chevron Corporation",      "sector": "Energy"},
    "GS":    {"name": "Goldman Sachs Group Inc.", "sector": "Financials"},
}

DEFAULT_INDEX_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK.B", "JPM", "V",
    "UNH", "JNJ", "XOM", "PG", "HD",
    "MA", "LLY", "COST", "ABBV", "AVGO",
]

RESERVE_CANDIDATES = [
    "AMD", "NFLX", "WMT", "BAC", "CRM",
    "ORCL", "CSCO", "QCOM", "CVX", "GS",
]

INITIAL_WEIGHT = 1.0 / len(DEFAULT_INDEX_STOCKS)  # 0.05 (5.0%)


@dataclass
class RebalanceDecision:
    ticker: str
    company_name: str
    sector: str
    action: str          # "REMOVE", "ADD", "OVERWEIGHT", "UNDERWEIGHT", "HOLD"
    initial_weight: float
    new_weight: float
    weight_change: float
    signal_score: float
    sentiment_label: str
    event_type: str
    reason: str
    triggering_news: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "action": self.action,
            "initial_weight": round(self.initial_weight, 4),
            "new_weight": round(self.new_weight, 4),
            "weight_change": round(self.weight_change, 4),
            "signal_score": round(self.signal_score, 4),
            "sentiment_label": self.sentiment_label,
            "event_type": self.event_type,
            "reason": self.reason,
            "triggering_news": self.triggering_news,
        }


def _article_to_trigger_summary(a: Article) -> dict[str, Any]:
    return {
        "title": a.title,
        "source": a.source,
        "url": a.url,
        "published": a.published.isoformat() if a.published else "",
        "sentiment_label": a.sentiment_label,
        "sentiment_score": round(a.sentiment_score, 4) if a.sentiment_score is not None else 0.0,
        "event_type": a.event_type or "other",
        "event_confidence": round(a.event_confidence, 2) if a.event_confidence is not None else 0.0,
        "impact": round(a.impact, 1) if a.impact is not None else 1.0,
        "impact_reasons": a.impact_reasons[:3] if a.impact_reasons else [],
    }


def compute_ticker_signal(
    ticker: str,
    articles: list[Article],
    as_of: datetime,
    lookback_days: float = 7.0,
    half_life_hours: float = 24.0,
) -> tuple[float, list[Article]]:
    """Compute time-decayed, impact-weighted sentiment signal for a ticker."""
    cutoff = as_of - timedelta(days=lookback_days)
    relevant = [
        a for a in articles
        if ticker in a.tickers and a.published and a.published >= cutoff and a.published <= as_of
    ]
    if not relevant:
        return 0.0, []

    total_weight = 0.0
    weighted_sentiment = 0.0

    for a in relevant:
        age_hours = max(0.0, (as_of - a.published).total_seconds() / 3600.0)
        recency = 0.5 ** (age_hours / half_life_hours)
        impact_factor = (a.impact or 1.0) / 5.0
        conf_factor = max(0.4, (a.event_confidence or 0.5))
        w = recency * impact_factor * conf_factor
        score = a.sentiment_score or 0.0

        total_weight += w
        weighted_sentiment += score * w

    if total_weight <= 0:
        return 0.0, []

    blended = weighted_sentiment / total_weight
    # Sort articles by |sentiment * impact| descending to show primary drivers
    relevant.sort(
        key=lambda a: abs((a.sentiment_score or 0.0) * (a.impact or 1.0)),
        reverse=True,
    )
    return round(blended, 4), relevant


def rebalance_index(
    articles: list[Article],
    initial_stocks: list[str] | None = None,
    candidate_pool: list[str] | None = None,
    as_of: datetime | None = None,
    remove_threshold: float = -0.30,
    severe_impact_threshold: float = 6.0,
    overweight_threshold: float = 0.18,
    underweight_threshold: float = -0.12,
) -> dict[str, Any]:
    """Execute index rebalancing on a 20-stock portfolio based on NLP news signals.

    Rules
    -----
    1. REMOVE: Stock has severe negative signal (< remove_threshold) OR
       high-impact negative event (e.g. credit default, bankruptcy, fraud probe with impact >= 6).
    2. ADD: For every stock removed, the top candidate from the reserve pool
       with the highest positive sentiment signal is promoted into the index.
    3. OVERWEIGHT: Stock with strong positive sentiment (>= overweight_threshold)
       receives an increased weight (5.0% -> ~7.5%).
    4. UNDERWEIGHT: Stock with moderate negative sentiment receives a reduced
       weight (5.0% -> ~2.5%).
    5. HOLD: Baseline weight maintained (5.0%).
    Final weights are normalized to sum to exactly 1.000 (100.0%).
    """
    as_of = as_of or datetime.now(timezone.utc)
    stocks = list(initial_stocks or DEFAULT_INDEX_STOCKS)
    candidates = list(candidate_pool or RESERVE_CANDIDATES)

    # Initial equal weights
    initial_weights = {t: INITIAL_WEIGHT for t in stocks}
    decisions: list[RebalanceDecision] = []

    raw_weights: dict[str, float] = {}
    removed: list[str] = []

    # ── Phase 1: Evaluate existing index constituents ─────────────────────────
    for ticker in stocks:
        meta = STOCK_METADATA.get(ticker, {"name": ticker, "sector": "General"})
        sig, relevant = compute_ticker_signal(ticker, articles, as_of)
        trigger_summaries = [_article_to_trigger_summary(a) for a in relevant[:3]]

        # Determine dominant event and sentiment
        dominant_event = relevant[0].event_type if relevant else "neutral"
        dominant_sentiment = (
            "positive" if sig > 0.1 else ("negative" if sig < -0.1 else "neutral")
        )

        # Check for catastrophic trigger:
        has_severe_negative_event = any(
            (a.sentiment_score or 0.0) <= -0.35 and (a.impact or 1.0) >= severe_impact_threshold
            for a in relevant
        )

        # Rule 1: REMOVE
        if sig < remove_threshold or has_severe_negative_event:
            removed.append(ticker)
            raw_weights[ticker] = 0.0
            worst_art = relevant[0] if relevant else None
            reason = (
                f"REMOVED: Severe negative news signal ({sig:+.2f}). "
                f"Triggered by {dominant_event} news"
            )
            if worst_art:
                reason += f" with impact {worst_art.impact:.1f}/10 ('{worst_art.title[:60]}...')."

            decisions.append(RebalanceDecision(
                ticker=ticker,
                company_name=meta["name"],
                sector=meta["sector"],
                action="REMOVE",
                initial_weight=INITIAL_WEIGHT,
                new_weight=0.0,
                weight_change=-INITIAL_WEIGHT,
                signal_score=sig,
                sentiment_label="negative",
                event_type=dominant_event,
                reason=reason,
                triggering_news=trigger_summaries,
            ))

        # Rule 3: OVERWEIGHT
        elif sig >= overweight_threshold:
            boost = min(0.035, sig * 0.05)
            w = INITIAL_WEIGHT + boost
            raw_weights[ticker] = w
            top_art = relevant[0] if relevant else None
            reason = f"OVERWEIGHT: Bullish news momentum ({sig:+.2f}) on {dominant_event}"
            if top_art:
                reason += f" ('{top_art.title[:60]}...')"

            decisions.append(RebalanceDecision(
                ticker=ticker,
                company_name=meta["name"],
                sector=meta["sector"],
                action="OVERWEIGHT",
                initial_weight=INITIAL_WEIGHT,
                new_weight=w,
                weight_change=w - INITIAL_WEIGHT,
                signal_score=sig,
                sentiment_label="positive",
                event_type=dominant_event,
                reason=reason,
                triggering_news=trigger_summaries,
            ))

        # Rule 4: UNDERWEIGHT
        elif sig <= underweight_threshold:
            cut = min(0.025, abs(sig) * 0.04)
            w = max(0.015, INITIAL_WEIGHT - cut)
            raw_weights[ticker] = w
            top_art = relevant[0] if relevant else None
            reason = f"UNDERWEIGHT: Negative news pressure ({sig:+.2f}) on {dominant_event}"
            if top_art:
                reason += f" ('{top_art.title[:60]}...')"

            decisions.append(RebalanceDecision(
                ticker=ticker,
                company_name=meta["name"],
                sector=meta["sector"],
                action="UNDERWEIGHT",
                initial_weight=INITIAL_WEIGHT,
                new_weight=w,
                weight_change=w - INITIAL_WEIGHT,
                signal_score=sig,
                sentiment_label="negative",
                event_type=dominant_event,
                reason=reason,
                triggering_news=trigger_summaries,
            ))

        # Rule 5: HOLD
        else:
            raw_weights[ticker] = INITIAL_WEIGHT
            reason = (
                f"HOLD: Stable baseline weight (5.0%). Neutral news sentiment ({sig:+.2f})."
                if relevant else "HOLD: Stable baseline weight (5.0%). No significant news."
            )
            decisions.append(RebalanceDecision(
                ticker=ticker,
                company_name=meta["name"],
                sector=meta["sector"],
                action="HOLD",
                initial_weight=INITIAL_WEIGHT,
                new_weight=INITIAL_WEIGHT,
                weight_change=0.0,
                signal_score=sig,
                sentiment_label=dominant_sentiment,
                event_type=dominant_event,
                reason=reason,
                triggering_news=trigger_summaries,
            ))

    # ── Phase 2: Add Replacements for Removed Stocks ──────────────────────────
    added: list[str] = []
    if removed:
        # Score reserve candidates to find the best replacement
        candidate_scores: list[tuple[float, str, list[Article]]] = []
        for cand in candidates:
            if cand in stocks:
                continue
            sig, relevant = compute_ticker_signal(cand, articles, as_of)
            candidate_scores.append((sig, cand, relevant))

        # Sort by signal descending
        candidate_scores.sort(key=lambda x: x[0], reverse=True)

        for i, removed_ticker in enumerate(removed):
            if i < len(candidate_scores):
                cand_sig, cand_ticker, cand_rel = candidate_scores[i]
            else:
                cand_ticker = candidates[i % len(candidates)]
                cand_sig = 0.0
                cand_rel = []

            meta = STOCK_METADATA.get(cand_ticker, {"name": cand_ticker, "sector": "General"})
            trigger_summaries = [_article_to_trigger_summary(a) for a in cand_rel[:3]]
            dominant_event = cand_rel[0].event_type if cand_rel else "growth"

            raw_weights[cand_ticker] = INITIAL_WEIGHT
            added.append(cand_ticker)

            reason = (
                f"ADDED: Promoted into 20-stock index to replace {removed_ticker}. "
                f"Supported by positive news signal ({cand_sig:+.2f}) on {dominant_event}."
            )
            if cand_rel:
                reason += f" Top headline: '{cand_rel[0].title[:60]}...'."

            decisions.append(RebalanceDecision(
                ticker=cand_ticker,
                company_name=meta["name"],
                sector=meta["sector"],
                action="ADD",
                initial_weight=0.0,
                new_weight=INITIAL_WEIGHT,
                weight_change=INITIAL_WEIGHT,
                signal_score=cand_sig,
                sentiment_label="positive" if cand_sig >= 0 else "neutral",
                event_type=dominant_event,
                reason=reason,
                triggering_news=trigger_summaries,
            ))

    # ── Phase 3: Normalize weights to 100.0% ──────────────────────────────────
    active_weights = {t: w for t, w in raw_weights.items() if w > 0}
    gross_active = sum(active_weights.values()) or 1.0

    normalized_portfolio: dict[str, float] = {
        t: round(w / gross_active, 4) for t, w in active_weights.items()
    }

    # Synchronize decisions with final normalized weights
    final_decisions: list[dict[str, Any]] = []
    for d in decisions:
        if d.action == "REMOVE":
            d.new_weight = 0.0
            d.weight_change = -d.initial_weight
        else:
            d.new_weight = normalized_portfolio.get(d.ticker, 0.0)
            d.weight_change = round(d.new_weight - d.initial_weight, 4)
        final_decisions.append(d.to_dict())

    # Sort decisions: REMOVE, ADD, OVERWEIGHT, UNDERWEIGHT, HOLD
    action_order = {"REMOVE": 0, "ADD": 1, "OVERWEIGHT": 2, "UNDERWEIGHT": 3, "HOLD": 4}
    final_decisions.sort(key=lambda x: (action_order.get(x["action"], 5), -abs(x["weight_change"])))

    # Compute total portfolio turnover
    turnover = sum(abs(d["weight_change"]) for d in final_decisions) / 2.0

    return {
        "as_of": as_of.isoformat(),
        "initial_universe_count": len(stocks),
        "initial_portfolio": {t: round(w, 4) for t, w in initial_weights.items()},
        "initial_stocks_detail": [
            {
                "ticker": t,
                "name": STOCK_METADATA.get(t, {}).get("name", t),
                "sector": STOCK_METADATA.get(t, {}).get("sector", "General"),
                "initial_weight": 0.05,
            }
            for t in stocks
        ],
        "rebalanced_portfolio": normalized_portfolio,
        "removed_stocks": removed,
        "added_stocks": added,
        "decisions": final_decisions,
        "summary": {
            "total_initial_stocks": len(stocks),
            "total_rebalanced_stocks": len(normalized_portfolio),
            "removed_count": len(removed),
            "added_count": len(added),
            "overweight_count": sum(1 for d in final_decisions if d["action"] == "OVERWEIGHT"),
            "underweight_count": sum(1 for d in final_decisions if d["action"] == "UNDERWEIGHT"),
            "hold_count": sum(1 for d in final_decisions if d["action"] == "HOLD"),
            "turnover_pct": round(turnover * 100.0, 2),
        },
    }


def rebalance_weight_history(
    articles: list[Article],
    as_of: datetime | None = None,
    days: int = 7,
    points: int = 7,
) -> dict[str, Any]:
    """Calculate the index weights progression across multiple timestamps over time.

    Returns
    -------
    timestamps       : list of formatted time strings (e.g. Day -6 to Day 0)
    dates            : list of short date labels (e.g. 'Oct 12', 'Oct 13', ...)
    all_tickers      : list of all tickers ever present in the index
    weights_by_ticker: {ticker: [w_t0, w_t1, ...]}
    history_points   : list of dicts with timestamp, weights, and decisions
    """
    as_of = as_of or datetime.now(timezone.utc)
    step = timedelta(days=days / max(1, points - 1))
    time_points = [as_of - step * (points - 1 - i) for i in range(points)]

    all_seen_tickers: set[str] = set(DEFAULT_INDEX_STOCKS)
    history_points = []

    for t in time_points:
        res = rebalance_index(articles, as_of=t)
        p_weights = res["rebalanced_portfolio"]
        all_seen_tickers.update(p_weights.keys())
        history_points.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M"),
            "date": t.strftime("%b %d"),
            "weights": p_weights,
            "removed": res["removed_stocks"],
            "added": res["added_stocks"],
        })

    sorted_tickers = sorted(all_seen_tickers)
    weights_by_ticker = {
        ticker: [hp["weights"].get(ticker, 0.0) for hp in history_points]
        for ticker in sorted_tickers
    }

    return {
        "days": days,
        "timestamps": [hp["timestamp"] for hp in history_points],
        "dates": [hp["date"] for hp in history_points],
        "all_tickers": sorted_tickers,
        "weights_by_ticker": weights_by_ticker,
        "history_points": history_points,
    }
