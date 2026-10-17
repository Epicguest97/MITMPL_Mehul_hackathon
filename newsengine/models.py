"""Data models for the Financial News Intelligence Engine.

Changelog
---------
* Added ``event_confidence`` – classifier confidence in [0, 1]
* Added ``impact_breakdown`` – dict of named sub-scores from the impact estimator
* Added ``signal_breakdown`` – dict of named signal components from fusion
* Added ``sentiment_scores_raw`` – raw FinBERT probability dict {positive, negative, neutral}
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class Article:
    title: str
    text: str
    published: datetime          # timezone-aware UTC
    source: str
    url: str
    tickers: list[str] = field(default_factory=list)

    # ── Step 3: FinBERT sentiment ──────────────────────────────────────────
    sentiment_label: str = "neutral"
    sentiment_score: float = 0.0          # P(pos) - P(neg) in [-1, 1]
    sentiment_scores_raw: dict = field(default_factory=dict)
    # e.g. {"positive": 0.72, "negative": 0.08, "neutral": 0.20}

    # ── Step 4: event classification ──────────────────────────────────────
    event_type: str = "other"
    event_confidence: float = 0.0         # classifier confidence in [0, 1]

    # ── Step 5: market impact ─────────────────────────────────────────────
    impact: float = 1.0                   # composite score in [1, 10]
    impact_breakdown: dict = field(default_factory=dict)
    # e.g. {"base": 5, "severity": 2, "magnitude": 1, "urgency": 1,
    #        "relevance": 1, "geographic_scope": 0}
    impact_reasons: list[str] = field(default_factory=list)  # human-readable

    # ── Step 6: signal fusion ─────────────────────────────────────────────
    signal_breakdown: dict = field(default_factory=dict)
    # e.g. {"sentiment": 0.6, "impact_weight": 0.8, "recency": 0.7,
    #        "reliability": 0.85, "corroboration": 1.15, "raw": 0.28}

    # ── deduplication metadata ────────────────────────────────────────────
    duplicate_count: int = 1
    cluster_sources: list[str] = field(default_factory=list)

    # ── identifier ────────────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return hashlib.sha1(self.url.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        d["published"] = self.published.isoformat()
        return d
