"""Step 6: advanced signal fusion.

Formula (per article)
---------------------
  raw_signal = sentiment_score
             × impact_weight        (= impact / 10)
             × recency              (= 0.5 ^ (age_h / HALF_LIFE_H))
             × reliability          (= mean source reliability in cluster)
             × corroboration        (= 1 + 0.15 × log2(n_unique_sources), ≤ 1.5)
             × confidence_weight    (= 0.6 + 0.4 × event_confidence)

  article_signal = clip(raw_signal, -1, 1)

All five multipliers are stored in ``Article.signal_breakdown`` for auditability.

Per-ticker aggregation
----------------------
Three time windows are computed independently and blended:

  window_signal(ticker, window_h) = tanh(Σ article_signals over window)
  ticker_signal = blend([6h, 24h, 72h] windows with weights [0.5, 0.3, 0.2])

This gives a momentum-aware composite:
  - The 6 h window captures breaking news impact.
  - The 24 h window provides medium-term alignment.
  - The 72 h window provides persistence / trend confirmation.

Agreement bonus
---------------
If ≥ 3 unique sources report the same directional signal (all positive or
all negative), the blended signal is boosted by up to 15 % before final
tanh squashing.

Output format  (aggregate_signals)
-----------------------------------
{
  ticker: {
    "signal":         float,    # final fused value in (-1, 1)
    "windows": {
      "6h":  float,             # short-term window contribution
      "24h": float,             # medium-term
      "72h": float,             # long-term / trend
    },
    "article_count":  int,
    "agreement":      float,    # fraction of sources that agree on direction
    "top_articles":   list[dict] – top 3 articles by |signal| for that ticker
  }
}
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta

from .models import Article

# ── Tunable parameters ────────────────────────────────────────────────────────
HALF_LIFE_H = 6.0
DEFAULT_RELIABILITY = 0.5
RELIABILITY: dict[str, float] = {
    "cnbc": 0.85,
    "marketwatch": 0.80,
    "yahoo_finance": 0.70,
    "gdelt": 0.50,
}

WINDOWS_H = {"6h": 6.0, "24h": 24.0, "72h": 72.0}
WINDOW_WEIGHTS = {"6h": 0.50, "24h": 0.30, "72h": 0.20}

AGREEMENT_MIN_SOURCES = 3
AGREEMENT_BOOST = 0.15   # max boost when all sources agree


# ── Source reliability ────────────────────────────────────────────────────────

def source_reliability(source: str) -> float:
    if source in RELIABILITY:
        return RELIABILITY[source]
    if source.startswith("gdelt:"):
        domain = source.split(":", 1)[1]
        if any(d in domain for d in ("reuters", "bloomberg", "wsj", "ft.com", "apnews")):
            return 0.90
        return RELIABILITY["gdelt"]
    return DEFAULT_RELIABILITY


# ── Per-article signal ────────────────────────────────────────────────────────

def article_signal(a: Article, now: datetime, store_breakdown: bool = False) -> float:
    """Compute and optionally store the full per-article signal decomposition.

    Parameters
    ----------
    a               : Article (must have sentiment_score, impact, event_confidence set)
    now             : reference timestamp (UTC-aware)
    store_breakdown : if True, write decomposition dict into a.signal_breakdown

    Returns
    -------
    signal : float in [-1, 1]
    """
    age_h = max(0.0, (now - a.published).total_seconds() / 3600)
    recency = 0.5 ** (age_h / HALF_LIFE_H)
    impact_weight = a.impact / 10.0

    srcs = a.cluster_sources or [a.source]
    reliability = sum(source_reliability(s) for s in srcs) / len(srcs)
    corroboration = min(1.5, 1 + 0.15 * math.log2(max(1, len(set(srcs)))))

    # Confidence weight: articles with high classifier confidence get more weight.
    # Range: [0.60, 1.00].  An article with confidence=1.0 gets full weight;
    # one with confidence=0.0 is down-weighted to 0.60 (still included, but muted).
    conf = getattr(a, "event_confidence", 0.5)
    confidence_weight = 0.60 + 0.40 * conf

    raw = (
        a.sentiment_score
        * impact_weight
        * recency
        * reliability
        * corroboration
        * confidence_weight
    )
    signal = max(-1.0, min(1.0, raw))

    if store_breakdown:
        a.signal_breakdown = {
            "sentiment":          round(a.sentiment_score, 4),
            "impact_weight":      round(impact_weight, 4),
            "recency":            round(recency, 4),
            "reliability":        round(reliability, 4),
            "corroboration":      round(corroboration, 4),
            "confidence_weight":  round(confidence_weight, 4),
            "raw":                round(raw, 6),
            "signal":             round(signal, 6),
        }

    return signal


# ── Windowed aggregation per ticker ──────────────────────────────────────────

def _articles_in_window(articles: list[Article], now: datetime, hours: float) -> list[Article]:
    cutoff = now - timedelta(hours=hours)
    return [a for a in articles if cutoff <= a.published <= now]


def _ticker_window_signal(
    articles: list[Article],
    ticker: str,
    now: datetime,
    hours: float,
) -> float:
    """tanh-squashed sum of article signals for a ticker within a time window."""
    acc = 0.0
    for a in _articles_in_window(articles, now, hours):
        if ticker in a.tickers:
            acc += article_signal(a, now)
    return math.tanh(acc)


def _agreement_factor(articles: list[Article], ticker: str, now: datetime, hours: float) -> float:
    """Fraction of unique sources that agree on direction within the window.

    Returns a value in [0, 1]. 1.0 = perfect agreement, 0.0 = no articles.
    """
    window = _articles_in_window(articles, now, hours)
    relevant = [a for a in window if ticker in a.tickers]
    if not relevant:
        return 0.5  # neutral when no data

    source_sigs: dict[str, float] = {}
    for a in relevant:
        src = (a.cluster_sources or [a.source])[0]
        source_sigs[src] = source_sigs.get(src, 0.0) + article_signal(a, now)

    if not source_sigs:
        return 0.5

    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in source_sigs.values()]
    dominant = max(signs.count(1), signs.count(-1))
    return dominant / len(signs)


# ── Public API ────────────────────────────────────────────────────────────────

def aggregate_signals(
    articles: list[Article],
    now: datetime,
    store_breakdown: bool = False,
) -> dict[str, dict]:
    """Full signal fusion per ticker.

    Returns a dict keyed by ticker with rich signal metadata::

        {
          "AAPL": {
            "signal":        0.42,
            "windows":       {"6h": 0.51, "24h": 0.38, "72h": 0.21},
            "article_count": 7,
            "agreement":     0.83,
            "top_articles":  [...]
          }
        }
    """
    # Step 1: collect all tickers visible up to `now`
    all_tickers: set[str] = set()
    for a in articles:
        if a.published <= now:
            all_tickers.update(a.tickers)
            if store_breakdown:
                article_signal(a, now, store_breakdown=True)

    result: dict[str, dict] = {}

    for ticker in sorted(all_tickers):
        # Per-window signals
        windows: dict[str, float] = {
            name: _ticker_window_signal(articles, ticker, now, hours)
            for name, hours in WINDOWS_H.items()
        }

        # Blended signal
        blended = sum(windows[w] * WINDOW_WEIGHTS[w] for w in windows)

        # Agreement check (over 24h window)
        agreement = _agreement_factor(articles, ticker, now, hours=24.0)
        n_sources_24h = len({
            (a.cluster_sources or [a.source])[0]
            for a in _articles_in_window(articles, now, 24.0)
            if ticker in a.tickers
        })
        if n_sources_24h >= AGREEMENT_MIN_SOURCES and agreement >= 0.75:
            direction = 1.0 if blended > 0 else -1.0
            boost = direction * AGREEMENT_BOOST * (agreement - 0.75) / 0.25
            blended = max(-1.0, min(1.0, blended + boost))

        # Article count
        ticker_arts = [a for a in articles if a.published <= now and ticker in a.tickers]
        article_count = len(ticker_arts)

        # Top 3 articles by |signal| magnitude
        scored = sorted(
            ticker_arts,
            key=lambda a: abs(article_signal(a, now)),
            reverse=True,
        )[:3]
        top_articles = [
            {
                "title":           a.title,
                "source":          a.source,
                "published":       a.published.isoformat(),
                "sentiment_label": a.sentiment_label,
                "sentiment_score": round(a.sentiment_score, 4),
                "event_type":      a.event_type,
                "impact":          a.impact,
                "signal":          round(article_signal(a, now), 4),
            }
            for a in scored
        ]

        result[ticker] = {
            "signal":        round(blended, 4),
            "windows":       {k: round(v, 4) for k, v in windows.items()},
            "article_count": article_count,
            "agreement":     round(agreement, 4),
            "top_articles":  top_articles,
        }

    return result


def aggregate_by_ticker(articles: list[Article], now: datetime) -> dict[str, float]:
    """Backward-compatible simple aggregation.  Returns {ticker: signal_float}."""
    rich = aggregate_signals(articles, now)
    return {t: v["signal"] for t, v in rich.items()}
