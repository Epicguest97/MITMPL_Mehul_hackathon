"""Step 7 (exposure): FastAPI service.

Run:
    uvicorn newsengine.api:app --reload

Endpoints
---------
GET  /health              liveness probe
POST /refresh             fetch → pipeline → store
GET  /articles            filtered article list  (ticker, event_type, limit)
GET  /articles/{id}       single article by SHA-1 id
GET  /signals             per-ticker scalar signal  (tanh-squashed, backward-compat)
GET  /signals/rich        full multi-window decomposition with agreement & top articles
GET  /portfolio           signals + target weights under RiskLimits
POST /backtest            run Step 9 backtest on stored articles + supplied prices
GET  /summary             aggregate stats over all stored articles
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .backtest import run_backtest, summarize
from .events import classify_event, load_event_classifier, EnsembleEventClassifier
from .impact import estimate_impact
from .normalize import extract_tickers
from .models import Article
from .pipeline import run_pipeline
from .portfolio import RiskLimits, target_weights
from .rebalancer import (
    rebalance_index,
    rebalance_weight_history,
    DEFAULT_INDEX_STOCKS,
    RESERVE_CANDIDATES,
    STOCK_METADATA,
)
from .sentiment import load_model as load_sentiment_model, score_articles
from .signals import aggregate_by_ticker, aggregate_signals
from .store import Store
from .stress_test import (
    DEFAULT_WHOLESALE_PORTFOLIO,
    run_portfolio_stress_test,
    get_shock_profile_for_event,
)


app = FastAPI(
    title="Financial News Intelligence Engine",
    description=(
        "Real-time financial news pipeline: fetch → normalise → deduplicate → "
        "FinBERT sentiment → event classification → market impact → composite signal → "
        "portfolio weights → backtest."
    ),
    version="1.0.0",
)

# Allow all origins in development; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


# Lazy singletons for analyze endpoint (avoids re-loading models on every call)
_sentiment_model = None
_event_classifier = None


def _get_sentiment_model():
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = load_sentiment_model()
    return _sentiment_model


def _get_event_classifier():
    global _event_classifier
    if _event_classifier is None:
        _event_classifier = load_event_classifier()
    return _event_classifier


# ── Pydantic request bodies ────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Analyze arbitrary text through the full NLP stack."""
    text: str
    tickers: list[str] = []   # optional: supply known tickers to skip extraction


class BacktestRequest(BaseModel):
    """
    prices: {ticker: {iso_timestamp: price}, ...}
    e.g.  {"AAPL": {"2024-01-02T00:00:00Z": 185.5, ...}}
    cost_bps: transaction cost in basis points per rebalance (default 5)
    max_position: per-ticker weight cap (default 0.10)
    """
    prices: dict[str, dict[str, float]]
    cost_bps: float = 5.0
    max_position: float = 0.10
    max_gross: float = 1.0
    max_net: float = 0.5
    min_signal: float = 0.10
    allow_short: bool = True


class SimulateRebalanceRequest(BaseModel):
    headline: str
    ticker: str = ""


class SentimentRequest(BaseModel):
    text: str


class ImpactRequest(BaseModel):
    text: str
    event_type: str | None = None
    tickers: list[str] = []
    confidence: float | None = None


class StressTestSimulateRequest(BaseModel):
    headline: str = ""
    event_type: str = ""
    sentiment_score: float | None = None
    impact_score: float | None = None
    trigger_threshold: float = 7.0




# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/refresh", tags=["ops"])
def refresh() -> dict[str, Any]:
    """Fetch live news, run full pipeline, persist to store."""
    arts = run_pipeline()
    store().upsert(arts)
    return {
        "processed": len(arts),
        "event_breakdown": _count_events(arts),
        "tickers_seen": sorted({t for a in arts for t in a.tickers}),
    }


@app.post("/analyze", tags=["analyze"])
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """Run the full NLP stack on arbitrary text and return structured outputs.

    Returns
    -------
    sentiment  : label, score in [-1,1], full probability distribution
    event      : category label + NLP confidence score
    impact     : composite score [1-10], named sub-score breakdown, reasons list
    tickers    : tickers extracted from the text
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    # ── Step 3: sentiment ────────────────────────────────────────────────────
    sm = _get_sentiment_model()
    label, score, raw_probs = sm.predict([text[:1000]])[0]

    # ── Step 4: event classification ────────────────────────────────────────
    clf = _get_event_classifier()
    event_type, event_confidence = classify_event(text, clf)

    # ── Step 2 (partial): ticker extraction ─────────────────────────────────
    tickers = req.tickers or extract_tickers(text)

    # ── Step 5: impact estimation ────────────────────────────────────────────
    impact_score, impact_breakdown, impact_reasons = estimate_impact(
        text, event_type, tickers, event_confidence
    )

    return {
        "input_text": text,
        "tickers": tickers,
        "sentiment": {
            "label": label,
            "score": round(score, 4),
            "raw_probs": {k: round(v, 4) for k, v in raw_probs.items()},
        },
        "event": {
            "type": event_type,
            "confidence": round(event_confidence, 4),
        },
        "impact": {
            "score": round(impact_score, 2),
            "breakdown": impact_breakdown,
            "reasons": impact_reasons,
        },
    }


@app.post("/sentiment", tags=["analyze"])
def get_sentiment(req: SentimentRequest) -> dict[str, Any]:
    """Run standalone FinBERT financial sentiment analysis on text."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    sm = _get_sentiment_model()
    label, score, raw_probs = sm.predict([text[:1000]])[0]
    return {
        "text": text,
        "sentiment_label": label,
        "sentiment_score": round(score, 4),
        "probabilities": {k: round(v, 4) for k, v in raw_probs.items()},
    }


@app.post("/impact", tags=["analyze"])
def get_impact(req: ImpactRequest) -> dict[str, Any]:
    """Estimate standalone market impact score (1-10) with full sub-factor breakdown."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    ev_type = req.event_type
    ev_conf = req.confidence
    if not ev_type:
        clf = _get_event_classifier()
        ev_type, ev_conf_inferred = classify_event(text, clf)
        ev_conf = ev_conf if ev_conf is not None else ev_conf_inferred
    elif ev_conf is None:
        ev_conf = 0.80

    tickers = req.tickers or extract_tickers(text)
    impact_score, impact_breakdown, impact_reasons = estimate_impact(
        text, ev_type, tickers, ev_conf
    )

    return {
        "text": text,
        "event_type": ev_type,
        "tickers": tickers,
        "confidence": round(ev_conf, 4),
        "impact_score": round(impact_score, 2),
        "breakdown": impact_breakdown,
        "reasons": impact_reasons,
    }


@app.get("/sources", tags=["news"])
def get_sources() -> dict[str, Any]:
    """Return the financial news ingestion sources and RSS feed endpoints."""
    from .fetch import RSS_FEEDS, REDDIT_FEEDS, GDELT_URL
    sources_list = [
        {"name": name, "url": url, "type": "RSS Financial Wire", "status": "active"}
        for name, url in RSS_FEEDS.items()
    ]
    for name, url in REDDIT_FEEDS.items():
        sources_list.append({"name": name, "url": url, "type": "Reddit Social Sentiment Feed", "status": "active"})
    sources_list.append({"name": "GDELT Global News", "url": GDELT_URL, "type": "Global Macro Stream", "status": "active"})

    return {
        "total_sources": len(sources_list),
        "sources": sources_list,
    }



@app.get("/articles", tags=["news"])
def articles(
    ticker: str | None = None,
    event_type: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    """Return enriched articles, optionally filtered by ticker or event type."""
    return store().query(ticker=ticker, event_type=event_type, limit=limit)


@app.get("/articles/{article_id}", tags=["news"])
def article(article_id: str) -> dict:
    """Return a single enriched article by its SHA-1 id."""
    rows = store().query(limit=500)
    for r in rows:
        if r["id"] == article_id:
            return r
    raise HTTPException(status_code=404, detail="Article not found")


@app.get("/signals", tags=["signals"])
def signals(
    min_age_h: float = Query(0.0, description="Ignore articles older than this many hours (0 = keep all)"),
) -> dict[str, float]:
    """Per-ticker scalar composite signal in (−1, 1). Backward-compatible."""
    now = datetime.now(timezone.utc)
    arts = store().articles()
    if min_age_h > 0:
        from datetime import timedelta
        cutoff = now - timedelta(hours=min_age_h)
        arts = [a for a in arts if a.published >= cutoff]
    return aggregate_by_ticker(arts, now)


@app.get("/signals/rich", tags=["signals"])
def signals_rich(
    min_age_h: float = Query(0.0, description="Ignore articles older than this many hours (0 = keep all)"),
    store_breakdown: bool = Query(False, description="Write signal_breakdown into each article in DB"),
) -> dict[str, Any]:
    """Full multi-window signal decomposition per ticker.

    Returns per-ticker dict with:
    - signal: blended scalar in (-1, 1)
    - windows: {6h, 24h, 72h} window contributions
    - article_count: articles contributing to this ticker
    - agreement: fraction of sources agreeing on direction
    - top_articles: top 3 articles by |signal| magnitude
    """
    now = datetime.now(timezone.utc)
    arts = store().articles()
    if min_age_h > 0:
        from datetime import timedelta
        cutoff = now - timedelta(hours=min_age_h)
        arts = [a for a in arts if a.published >= cutoff]
    return aggregate_signals(arts, now, store_breakdown=store_breakdown)


class ClassifyRequest(BaseModel):
    text: str


@app.post("/classify", tags=["analyze"])
def classify(req: ClassifyRequest) -> dict[str, Any]:
    """Run the ensemble event classifier and return the full per-model breakdown.

    Returns
    -------
    event_type  : winning category
    confidence  : ensemble confidence in [0, 1]
    scores      : per-category breakdown with NLI, embedding, keyword sub-scores
    model_votes : which category each sub-model voted for
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    clf = _get_event_classifier()
    if hasattr(clf, "classify_detailed"):
        return clf.classify_detailed(text)
    # keyword-only fallback
    evt, conf = clf.classify(text)
    return {"event_type": evt, "confidence": conf, "scores": {}, "model_votes": {"keyword": evt}}


@app.get("/portfolio", tags=["signals"])

def portfolio(
    max_position: float = Query(0.10, ge=0.01, le=1.0),
    max_gross: float = Query(1.0, ge=0.0),
    max_net: float = Query(0.5, ge=0.0),
    min_signal: float = Query(0.10, ge=0.0, le=1.0),
    allow_short: bool = Query(True),
) -> dict[str, Any]:
    """Signals + deterministic target weights under configurable risk limits."""
    now = datetime.now(timezone.utc)
    sig = aggregate_by_ticker(store().articles(), now)
    limits = RiskLimits(
        max_position=max_position,
        max_gross=max_gross,
        max_net=max_net,
        min_signal=min_signal,
        allow_short=allow_short,
    )
    w = target_weights(sig, limits)
    return {
        "as_of": now.isoformat(),
        "signals": sig,
        "weights": w,
        "risk_limits": {
            "max_position": limits.max_position,
            "max_gross": limits.max_gross,
            "max_net": limits.max_net,
            "min_signal": limits.min_signal,
            "allow_short": limits.allow_short,
        },
    }


@app.get("/rebalance", tags=["rebalancer"])
def get_rebalance(
    remove_threshold: float = Query(-0.30, description="Threshold below which a stock is removed"),
    overweight_threshold: float = Query(0.18, description="Threshold above which a stock is overweight"),
    underweight_threshold: float = Query(-0.12, description="Threshold below which a stock is underweight"),
) -> dict[str, Any]:
    """Execute dynamic stock index rebalancing for the 20-stock blue-chip index based on stored news."""
    arts = store().articles()
    return rebalance_index(
        arts,
        remove_threshold=remove_threshold,
        overweight_threshold=overweight_threshold,
        underweight_threshold=underweight_threshold,
    )


@app.get("/rebalance/history", tags=["rebalancer"])
def get_rebalance_history(
    days: int = Query(7, ge=1, le=90, description="Lookback window in days"),
    points: int = Query(7, ge=2, le=30, description="Number of observation time steps"),
) -> dict[str, Any]:
    """Return historical stock index constituent weights over time."""
    arts = store().articles()
    return rebalance_weight_history(arts, days=days, points=points)


@app.post("/rebalance/simulate", tags=["rebalancer"])
def simulate_rebalance(req: SimulateRebalanceRequest) -> dict[str, Any]:
    """Simulate a breaking news event and observe real-time index rebalance decisions."""
    text = req.headline.strip()
    if not text:
        raise HTTPException(status_code=422, detail="headline must not be empty")

    now = datetime.now(timezone.utc)
    ticker = req.ticker.strip().upper()
    tickers = [ticker] if ticker else extract_tickers(text)
    if not tickers:
        tickers = ["TSLA"] if "tesla" in text.lower() else ["AAPL"]

    # Full NLP enrichment on simulated headline
    sm = _get_sentiment_model()
    label, score, raw_probs = sm.predict([text[:1000]])[0]

    clf = _get_event_classifier()
    event_type, event_confidence = classify_event(text, clf)

    impact_score, impact_breakdown, impact_reasons = estimate_impact(
        text, event_type, tickers, event_confidence
    )

    import hashlib
    art_id = hashlib.sha1(f"sim:{text}:{now.isoformat()}".encode()).hexdigest()
    synthetic_art = Article(
        title=text,
        url=f"https://breakingnews.internal/sim/{art_id[:8]}",
        published=now,
        source="Breaking Wire (Simulated)",
        text=text,
        tickers=tickers,
        sentiment_label=label,
        sentiment_score=score,
        sentiment_scores_raw=raw_probs,
        event_type=event_type,
        event_confidence=event_confidence,
        impact=impact_score,
        impact_breakdown=impact_breakdown,
        impact_reasons=impact_reasons,
    )

    all_arts = [synthetic_art] + store().articles()
    result = rebalance_index(all_arts, as_of=now)
    result["simulated_article"] = {
        "title": text,
        "tickers": tickers,
        "sentiment_label": label,
        "sentiment_score": round(score, 4),
        "event_type": event_type,
        "event_confidence": round(event_confidence, 2),
        "impact": round(impact_score, 1),
        "impact_reasons": impact_reasons,
    }
    result["history"] = rebalance_weight_history(all_arts, as_of=now, days=7, points=7)
    return result


# ── Module B: Strategic Portfolio Stress Testing ──────────────────────────────

@app.get("/stress-test/portfolio", tags=["stress_test"])
def get_stress_test_portfolio() -> dict[str, Any]:
    """Return the synthetic portfolio of wholesale banking assets ($1B notional)."""
    base_val = sum(a["base_value"] for a in DEFAULT_WHOLESALE_PORTFOLIO)
    by_class: dict[str, float] = {}
    for a in DEFAULT_WHOLESALE_PORTFOLIO:
        by_class[a["asset_class"]] = by_class.get(a["asset_class"], 0.0) + a["base_value"]

    return {
        "portfolio_name": "Global Wholesale Banking Assets Portfolio",
        "total_notional": sum(a["notional"] for a in DEFAULT_WHOLESALE_PORTFOLIO),
        "total_base_value": base_val,
        "total_base_value_formatted": f"${base_val / 1e6:.1f}M",
        "asset_count": len(DEFAULT_WHOLESALE_PORTFOLIO),
        "class_breakdown": {k: round(v, 2) for k, v in by_class.items()},
        "assets": DEFAULT_WHOLESALE_PORTFOLIO,
    }


@app.post("/stress-test/simulate", tags=["stress_test"])
def simulate_stress_test(req: StressTestSimulateRequest) -> dict[str, Any]:
    """Execute strategic portfolio stress test for wholesale banking assets.

    Can be triggered via:
    1. Direct real-world headline text (NLP stack determines sentiment, event category, impact).
    2. Overriding event_type, sentiment_score, and impact_score directly.
    """
    headline = req.headline.strip()
    event_type = req.event_type.strip()
    sentiment_score = req.sentiment_score
    impact_score = req.impact_score

    nlp_details = {}

    if headline:
        # Full NLP pipeline
        sm = _get_sentiment_model()
        label, score, raw_probs = sm.predict([headline[:1000]])[0]
        clf = _get_event_classifier()
        ev_type, ev_conf = classify_event(headline, clf)
        tickers = extract_tickers(headline)
        imp_score, imp_breakdown, imp_reasons = estimate_impact(
            headline, ev_type, tickers, ev_conf
        )

        event_type = event_type or ev_type
        sentiment_score = sentiment_score if sentiment_score is not None else score
        impact_score = impact_score if impact_score is not None else imp_score

        nlp_details = {
            "headline": headline,
            "tickers": tickers,
            "sentiment_label": label,
            "sentiment_score": round(score, 4),
            "event_type": ev_type,
            "event_confidence": round(ev_conf, 2),
            "impact_score": round(imp_score, 2),
            "impact_reasons": imp_reasons,
        }
    else:
        event_type = event_type or "geopolitical"
        sentiment_score = -0.75 if sentiment_score is None else sentiment_score
        impact_score = 8.5 if impact_score is None else impact_score

    report = run_portfolio_stress_test(
        event_type=event_type,
        sentiment_score=sentiment_score,
        impact_score=impact_score,
        trigger_threshold=req.trigger_threshold,
        article_title=headline or f"{event_type.title()} Event Shock",
    )
    report["nlp_analysis"] = nlp_details
    return report


@app.get("/stress-test/events", tags=["stress_test"])
def get_stress_test_events(
    min_impact: float = Query(6.5, ge=1.0, le=10.0, description="Minimum impact score threshold"),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Return stored news articles qualifying as high-impact triggers for stress testing."""
    arts = store().articles()
    high_impact = [a for a in arts if (a.impact or 0.0) >= min_impact]
    high_impact.sort(key=lambda a: -(a.impact or 0.0))

    out = []
    for a in high_impact[:limit]:
        out.append({
            "id": a.id,
            "title": a.title,
            "published": a.published.isoformat() if a.published else "",
            "source": a.source,
            "event_type": a.event_type,
            "event_confidence": round(a.event_confidence or 0.0, 2),
            "sentiment_label": a.sentiment_label,
            "sentiment_score": round(a.sentiment_score or 0.0, 3),
            "impact": round(a.impact or 0.0, 1),
            "impact_reasons": a.impact_reasons,
            "triggers_stress_test": (a.impact or 0.0) >= 7.0,
        })
    return out


# ── UI Page Routes ────────────────────────────────────────────────────────────

@app.get("/", tags=["ui"], response_class=FileResponse)
@app.get("/setup", tags=["ui"], response_class=FileResponse)
@app.get("/setup.html", tags=["ui"], response_class=FileResponse)
@app.get("/api_setup", tags=["ui"], response_class=FileResponse)
@app.get("/api_setup.html", tags=["ui"], response_class=FileResponse)
@app.get("/index.html", tags=["ui"], response_class=FileResponse)
def serve_api_setup():
    """Serve Page 1: API Setup & NLP Risk Engine."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_setup.html")
    if os.path.exists(path):
        return FileResponse(path)
    idx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    return FileResponse(idx_path)


@app.get("/module_a", tags=["ui"], response_class=FileResponse)
@app.get("/module_a.html", tags=["ui"], response_class=FileResponse)
def serve_module_a():
    """Serve Page 2: Module A - Stock Index Rebalancer & Historical Weight Trajectory Dashboard."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "module_a.html")
    if os.path.exists(path):
        return FileResponse(path)
    return FileResponse("module_a.html")


@app.get("/module_b", tags=["ui"], response_class=FileResponse)
@app.get("/module_b.html", tags=["ui"], response_class=FileResponse)
def serve_module_b():
    """Serve Page 3: Module B - Strategic Portfolio Stress Testing."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "module_b.html")
    if os.path.exists(path):
        return FileResponse(path)
    return FileResponse("module_b.html")




@app.post("/backtest", tags=["backtest"])
def backtest(req: BacktestRequest) -> dict[str, Any]:
    """
    Run the Step-9 backtest on stored articles + caller-supplied price data.

    The backtest enforces no-look-ahead: signals at each bar only use articles
    published **before** that bar timestamp.  Transaction costs (``cost_bps``)
    are charged on turnover at every rebalance.

    Returns per-bar rows **and** summary statistics (Sharpe, max drawdown, …).
    """
    if not req.prices:
        raise HTTPException(status_code=422, detail="prices must not be empty")

    # Build price DataFrame (UTC-aware index, sorted)
    try:
        df_prices = pd.DataFrame(req.prices).sort_index()
        df_prices.index = pd.to_datetime(df_prices.index, utc=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid prices format: {exc}") from exc

    if len(df_prices) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 price bars for backtest")

    arts = store().articles()
    limits = RiskLimits(
        max_position=req.max_position,
        max_gross=req.max_gross,
        max_net=req.max_net,
        min_signal=req.min_signal,
        allow_short=req.allow_short,
    )
    df = run_backtest(arts, df_prices, limits, cost_bps=req.cost_bps)
    stats = summarize(df)

    rows = df.reset_index().rename(columns={"time": "timestamp"})
    rows["timestamp"] = rows["timestamp"].map(str)

    return {
        "summary": stats,
        "rows": rows.to_dict(orient="records"),
    }


@app.get("/summary", tags=["signals"])
def summary() -> dict[str, Any]:
    """High-level aggregate stats over all stored articles."""
    rows = store().all()
    if not rows:
        return {"count": 0}

    sentiments = [r["sentiment_label"] for r in rows]
    events = [r["event_type"] for r in rows]
    scores = [r["sentiment_score"] for r in rows]

    return {
        "count": len(rows),
        "sentiment_distribution": _tally(sentiments),
        "event_distribution": _tally(events),
        "sentiment_score": {
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        },
        "top_tickers": _top_tickers(rows, n=10),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tally(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def _count_events(arts) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in arts:
        out[a.event_type] = out.get(a.event_type, 0) + 1
    return out


def _top_tickers(rows: list[dict], n: int = 10) -> list[dict]:
    counts: dict[str, int] = {}
    for r in rows:
        for t in r.get("tickers", []):
            counts[t] = counts.get(t, 0) + 1
    return [{"ticker": t, "mentions": c}
            for t, c in sorted(counts.items(), key=lambda x: -x[1])[:n]]
