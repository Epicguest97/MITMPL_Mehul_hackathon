"""
Comprehensive test suite covering all 9 pipeline steps + 4 NLP components.

Steps / components tested
--------------------------
1  fetch        – mocked (network-free)
2  normalize    – canonical URL, HTML cleanup, ticker extraction, dedup
3  sentiment    – LexiconSentiment returns (label, score, raw_probs) triple
4  events       – KeywordEventClassifier returns (type, confidence) tuple
5  impact       – 7 sub-scores, bounds, transparency, structured breakdown
6  signals      – recency decay, corroboration, momentum windows, agreement
7  store        – DuckDB upsert + query (round-trip, indexed columns)
   api          – FastAPI TestClient smoke tests for all endpoints incl. /signals/rich
8  portfolio    – deterministic weights, hard risk limits, turnover
9  backtest     – no look-ahead, transaction costs, summarize()
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from newsengine.backtest import run_backtest, summarize
from newsengine.events import KeywordEventClassifier, classify_event
from newsengine.impact import estimate_impact
from newsengine.models import Article
from newsengine.normalize import canonical_url, deduplicate, extract_tickers, normalize
from newsengine.pipeline import enrich
from newsengine.portfolio import RiskLimits, target_weights
from newsengine.sentiment import LexiconSentiment
from newsengine.signals import aggregate_by_ticker, aggregate_signals, article_signal
from newsengine.store import Store

NOW = datetime(2024, 1, 10, 12, tzinfo=timezone.utc)

# Always use keyword classifier in tests (no network / model downloads)
_kw_clf = KeywordEventClassifier()


# ── Helpers ───────────────────────────────────────────────────────────────────

def art(title, src="cnbc", url=None, age_h=0, text=""):
    return Article(
        title=title, text=text,
        published=NOW - timedelta(hours=age_h),
        source=src, url=url or f"https://x.com/{abs(hash(title + src))}",
    )


def art_with(a):
    b = art("w")
    b.sentiment_score, b.impact = a.sentiment_score, a.impact
    return b


# ── Step 2: normalize & deduplicate ───────────────────────────────────────────

def test_dedup_and_tickers():
    a = art("Apple beats earnings estimates", "cnbc", "https://www.a.com/1?utm_source=x", 2)
    b = art("Apple beats earnings estimates!", "yahoo_finance", "https://b.com/9", 1)
    c = art("Fed holds rates steady", "cnbc", "https://c.com/3", 1)
    out = deduplicate(normalize([a, b, c]))
    assert len(out) == 2
    apple = next(x for x in out if "AAPL" in x.tickers)
    assert apple.duplicate_count == 2 and set(apple.cluster_sources) == {"cnbc", "yahoo_finance"}
    assert canonical_url("https://www.A.com/1/?utm_source=x") == "https://a.com/1"
    assert "TSLA" in extract_tickers("Shares of $TSLA and Nvidia rose")


def test_normalize_cleans_html():
    a = art("<b>Apple</b> &amp; Google rally", text="<p>gains of 5%</p>")
    normed = normalize([a])
    assert "<" not in normed[0].title
    assert "&amp;" not in normed[0].title
    assert "AAPL" in normed[0].tickers
    assert "GOOGL" in normed[0].tickers


# ── Step 3: sentiment ─────────────────────────────────────────────────────────

def test_lexicon_sentiment_returns_triple():
    model = LexiconSentiment()
    results = model.predict(["Tesla surges on record profit", "Company files for bankruptcy"])
    assert len(results) == 2
    for label, score, raw in results:
        assert label in ("positive", "negative", "neutral")
        assert -1.0 <= score <= 1.0
        assert set(raw.keys()) == {"positive", "negative", "neutral"}
        assert abs(sum(raw.values()) - 1.0) < 0.01  # probabilities sum to ~1

def test_lexicon_positive():
    model = LexiconSentiment()
    label, score, raw = model.predict(["Tesla surges on record profit breakthrough"])[0]
    assert label == "positive" and score > 0
    assert raw["positive"] > raw["negative"]

def test_lexicon_negative():
    model = LexiconSentiment()
    label, score, raw = model.predict(["Company files for bankruptcy after fraud probe"])[0]
    assert label == "negative" and score < 0
    assert raw["negative"] > raw["positive"]

def test_score_articles_writes_raw_probs():
    from newsengine.sentiment import score_articles
    a = art("Apple beats earnings")
    score_articles([a], LexiconSentiment())
    assert -1 <= a.sentiment_score <= 1
    assert isinstance(a.sentiment_scores_raw, dict)
    assert "positive" in a.sentiment_scores_raw


# ── Step 4: event classification ──────────────────────────────────────────────

def test_keyword_classifier_returns_tuple():
    evt, conf = _kw_clf.classify("Company files for bankruptcy after default")
    assert evt == "credit_event"
    assert 0.0 < conf <= 0.6   # keyword fallback caps at 0.6

def test_event_categories():
    cases = [
        ("Company files for bankruptcy after default", "credit_event"),
        ("Microsoft to acquire startup in billion deal", "mergers_acquisitions"),
        ("Fed raises interest rate amid inflation fears", "macroeconomic"),
        ("Apple launches new iPhone product line", "product_launch"),
        ("Quarterly earnings beat estimates on strong revenue", "earnings"),
        ("Sanctions imposed following missile invasion", "geopolitical"),
    ]
    for text, expected in cases:
        evt, conf = _kw_clf.classify(text)
        assert evt == expected, f"Expected {expected}, got {evt} for: {text}"
        assert conf > 0.0

def test_event_other():
    evt, conf = _kw_clf.classify("Random unrelated headline with no keywords")
    assert evt == "other" and conf == 0.0

def test_classify_event_uses_classifier_arg():
    """classify_event() should accept a classifier kwarg."""
    evt, conf = classify_event("Company files for bankruptcy", _kw_clf)
    assert evt == "credit_event"


# ── Step 5: impact estimation ─────────────────────────────────────────────────

def test_impact_returns_three_tuple():
    result = estimate_impact("Apple bankruptcy default", "credit_event", ["AAPL"], 0.9)
    assert len(result) == 3
    score, breakdown, reasons = result
    assert 1.0 <= score <= 10.0
    assert isinstance(breakdown, dict)
    assert isinstance(reasons, list)

def test_impact_breakdown_keys():
    _, breakdown, _ = estimate_impact("Fed raises rates", "macroeconomic", [], 0.5)
    for key in ("base", "severity", "magnitude", "urgency", "relevance",
                 "geographic_scope", "event_confidence_bonus", "total"):
        assert key in breakdown, f"Missing key: {key}"

def test_impact_bounds_and_severity_cap():
    hi, _, _ = estimate_impact("bankruptcy default chapter 11", "credit_event", ["AAPL"], 0.95)
    lo, _, _ = estimate_impact("new gadget product launch", "product_launch", [], 0.3)
    assert 1 <= lo < hi <= 10

def test_impact_keyword_stuffing_cap():
    stuffed = " ".join(["bankruptcy"] * 30)
    score, _, _ = estimate_impact(stuffed, "credit_event", ["AAPL"])
    assert score <= 10.0

def test_impact_financial_magnitude():
    _, bd_t, _ = estimate_impact("$500 trillion deal announced", "mergers_acquisitions", ["AAPL"])
    _, bd_m, _ = estimate_impact("$50 million deal announced", "mergers_acquisitions", ["AAPL"])
    assert bd_t["magnitude"] > bd_m["magnitude"]

def test_impact_urgency():
    _, bd_u, _ = estimate_impact("BREAKING: emergency rate cut", "macroeconomic", [])
    assert bd_u["urgency"] > 0.0

def test_impact_geographic_scope():
    _, bd_g, _ = estimate_impact("global recession fears spread worldwide", "macroeconomic", [])
    assert bd_g["geographic_scope"] > 0.0

def test_impact_confidence_bonus():
    _, bd_hi, _ = estimate_impact("earnings beat", "earnings", ["AAPL"], event_confidence=0.95)
    _, bd_lo, _ = estimate_impact("earnings beat", "earnings", ["AAPL"], event_confidence=0.2)
    assert bd_hi["event_confidence_bonus"] > bd_lo["event_confidence_bonus"]


# ── Step 6: signal fusion ─────────────────────────────────────────────────────

def test_article_signal_recency_decay():
    a = art("x", age_h=0); a.sentiment_score, a.impact = 0.8, 8
    old = art("y", age_h=12); old.sentiment_score, old.impact = 0.8, 8
    assert article_signal(a, NOW) > article_signal(old, NOW) > 0

def test_article_signal_corroboration():
    single = art("z"); single.sentiment_score, single.impact = 0.8, 8
    multi = art("z2"); multi.sentiment_score, multi.impact = 0.8, 8
    multi.cluster_sources = ["cnbc", "marketwatch", "yahoo_finance", "gdelt"]
    assert article_signal(multi, NOW) > article_signal(single, NOW)

def test_article_signal_stores_breakdown():
    a = art("x"); a.sentiment_score, a.impact, a.event_confidence = 0.7, 7, 0.9
    article_signal(a, NOW, store_breakdown=True)
    assert isinstance(a.signal_breakdown, dict)
    for key in ("sentiment", "impact_weight", "recency", "reliability",
                 "corroboration", "confidence_weight", "raw", "signal"):
        assert key in a.signal_breakdown

def test_aggregate_signals_structure():
    a = art("Apple surges"); a.tickers = ["AAPL"]
    a.sentiment_score, a.impact, a.event_confidence = 0.8, 8, 0.9
    rich = aggregate_signals([a], NOW)
    assert "AAPL" in rich
    r = rich["AAPL"]
    assert "signal" in r and "windows" in r and "article_count" in r
    assert "agreement" in r and "top_articles" in r
    assert set(r["windows"].keys()) == {"6h", "24h", "72h"}
    assert -1 < r["signal"] < 1

def test_no_lookahead_in_aggregate_signals():
    future = art("AAPL moonshot")
    future.published = NOW + timedelta(hours=1)
    future.tickers = ["AAPL"]; future.sentiment_score = 0.9; future.impact = 9
    rich = aggregate_signals([future], NOW)
    assert "AAPL" not in rich   # future article must not appear

def test_aggregate_by_ticker_backward_compat():
    a = art("Apple surges"); a.tickers = ["AAPL"]
    a.sentiment_score, a.impact = 0.8, 8
    sigs = aggregate_by_ticker([a], NOW)
    assert "AAPL" in sigs
    assert -1 < sigs["AAPL"] < 1


# ── Step 7: store ─────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_store(tmp_path):
    return Store(str(tmp_path / "test.duckdb"))


def _make_article(title="Test", tickers=None, event_type="earnings",
                  score=0.5, event_confidence=0.7) -> Article:
    return Article(
        title=title, text="body text",
        published=NOW,
        source="cnbc", url=f"https://example.com/{abs(hash(title))}",
        tickers=tickers or ["AAPL"],
        sentiment_label="positive", sentiment_score=score,
        sentiment_scores_raw={"positive": 0.75, "negative": 0.05, "neutral": 0.20},
        event_type=event_type, event_confidence=event_confidence,
        impact=7.0,
        impact_breakdown={"base": 4.0, "severity": 2.0, "magnitude": 1.0,
                          "urgency": 0.0, "relevance": 1.0, "geographic_scope": 0.0,
                          "event_confidence_bonus": 0.0, "total": 7.0},
        impact_reasons=["base=4"], duplicate_count=1, cluster_sources=["cnbc"],
    )


def test_store_upsert_and_roundtrip(tmp_store):
    a = _make_article()
    tmp_store.upsert([a])
    rows = tmp_store.all()
    assert len(rows) == 1
    assert rows[0]["tickers"] == ["AAPL"]
    assert rows[0]["event_type"] == "earnings"
    assert rows[0]["event_confidence"] == pytest.approx(0.7)
    assert isinstance(rows[0]["impact_breakdown"], dict)
    arts = tmp_store.articles()
    assert arts[0].sentiment_score == pytest.approx(0.5)


def test_store_upsert_idempotent(tmp_store):
    a = _make_article()
    tmp_store.upsert([a])
    tmp_store.upsert([a])
    assert len(tmp_store.all()) == 1


def test_store_query_by_ticker(tmp_store):
    a = _make_article(title="Apple news", tickers=["AAPL"])
    b = _make_article(title="Tesla news", tickers=["TSLA"])
    tmp_store.upsert([a, b])
    rows = tmp_store.query(ticker="AAPL")
    assert len(rows) == 1 and "AAPL" in rows[0]["tickers"]


def test_store_query_by_event_type(tmp_store):
    a = _make_article(title="Earnings", event_type="earnings")
    b = _make_article(title="War", event_type="geopolitical")
    tmp_store.upsert([a, b])
    geo = tmp_store.query(event_type="geopolitical")
    assert len(geo) == 1 and geo[0]["event_type"] == "geopolitical"


# ── Step 7: API ───────────────────────────────────────────────────────────────

@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import newsengine.api as api_mod

    fresh_store = Store(str(tmp_path / "api_test.duckdb"))
    monkeypatch.setattr(api_mod, "_store", fresh_store)
    return TestClient(api_mod.app)


def test_api_health(api_client):
    assert api_client.get("/health").json()["status"] == "ok"


def test_api_articles_empty(api_client):
    assert api_client.get("/articles").json() == []


def test_api_signals_empty(api_client):
    assert api_client.get("/signals").json() == {}


def test_api_signals_rich_empty(api_client):
    r = api_client.get("/signals/rich")
    assert r.status_code == 200
    assert r.json() == {}


def test_api_signals_rich_structure(api_client, monkeypatch):
    import newsengine.api as api_mod
    a = _make_article(title="Apple surges on earnings beat", tickers=["AAPL"])
    a.sentiment_score = 0.8; a.impact = 8; a.event_confidence = 0.85
    api_mod._store.upsert([a])
    r = api_client.get("/signals/rich")
    assert r.status_code == 200
    data = r.json()
    assert "AAPL" in data
    aapl = data["AAPL"]
    assert "signal" in aapl
    assert "windows" in aapl
    assert set(aapl["windows"].keys()) == {"6h", "24h", "72h"}
    assert "agreement" in aapl
    assert "article_count" in aapl
    assert "top_articles" in aapl


def test_api_portfolio_empty(api_client):
    data = api_client.get("/portfolio").json()
    assert "signals" in data and "weights" in data


def test_api_summary_empty(api_client):
    assert api_client.get("/summary").json()["count"] == 0


def test_api_articles_filter(api_client, monkeypatch):
    import newsengine.api as api_mod
    a = _make_article(title="Apple report", tickers=["AAPL"], event_type="earnings")
    b = _make_article(title="War news", tickers=["XOM"], event_type="geopolitical")
    api_mod._store.upsert([a, b])
    r = api_client.get("/articles?ticker=AAPL")
    assert all("AAPL" in row["tickers"] for row in r.json())
    r2 = api_client.get("/articles?event_type=geopolitical")
    assert all(row["event_type"] == "geopolitical" for row in r2.json())


def test_api_article_not_found(api_client):
    assert api_client.get("/articles/nonexistent_id").status_code == 404


def test_api_backtest(api_client, monkeypatch):
    import newsengine.api as api_mod
    idx = pd.date_range(NOW - timedelta(days=5), periods=6, freq="D")
    early = _make_article(title="Apple surges big")
    early.published = idx[3].to_pydatetime().replace(tzinfo=timezone.utc)
    early.tickers = ["AAPL"]; early.sentiment_score = 0.9; early.impact = 9
    api_mod._store.upsert([early])
    prices = {"AAPL": {ts.isoformat(): float(100 if i < 4 else 120)
                       for i, ts in enumerate(idx)}}
    r = api_client.post("/backtest", json={"prices": prices, "cost_bps": 10})
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data and "rows" in data


def test_api_backtest_invalid(api_client):
    assert api_client.post("/backtest", json={"prices": {}}).status_code == 422
    r = api_client.post("/backtest", json={"prices": {"AAPL": {"2024-01-01T00:00:00Z": 100}}})
    assert r.status_code == 422


# ── Step 8: portfolio ─────────────────────────────────────────────────────────

def test_portfolio_limits_deterministic():
    sig = {f"T{i}": 1.0 for i in range(20)}
    lim = RiskLimits()
    w = target_weights(sig, lim)
    assert w == target_weights(sig, lim)
    assert all(abs(x) <= lim.max_position + 1e-9 for x in w.values())
    assert sum(abs(x) for x in w.values()) <= lim.max_gross + 1e-9
    assert abs(sum(w.values())) <= lim.max_net + 1e-9
    assert target_weights({"A": -1.0}, RiskLimits(allow_short=False)) == {}


def test_portfolio_turnover_limit():
    current = {"AAPL": 0.08}
    lim = RiskLimits(max_turnover=0.01)
    w = target_weights({"AAPL": 1.0}, lim, current=current)
    names = set(w) | set(current)
    turnover = sum(abs(w.get(t, 0) - current.get(t, 0)) for t in names)
    assert turnover <= lim.max_turnover + 1e-9


# ── Step 9: backtest ──────────────────────────────────────────────────────────

def test_backtest_no_lookahead_and_costs():
    idx = pd.date_range(NOW - timedelta(days=5), periods=6, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 100, 100, 100, 100, 120]}, index=idx)
    news = art("Apple surges")
    news.published = idx[-1] + timedelta(hours=1)
    news.tickers, news.sentiment_score, news.impact = ["AAPL"], 0.9, 9
    df = run_backtest([news], prices)
    assert df["gross_return"].abs().sum() == 0 and df["cost"].sum() == 0

    early = art("Apple surges")
    early.published = idx[3]
    early.tickers, early.sentiment_score, early.impact = ["AAPL"], 0.9, 9
    prices2 = pd.DataFrame({"AAPL": [100, 100, 100, 100, 120, 120]}, index=idx)
    df2 = run_backtest([early], prices2, cost_bps=10)
    assert df2["gross_return"].sum() > 0 and df2["cost"].sum() > 0
    assert summarize(df2)["total_cost"] > 0


def test_backtest_summarize_keys():
    idx = pd.date_range(NOW - timedelta(days=3), periods=4, freq="D")
    prices = pd.DataFrame({"AAPL": [100.0, 102.0, 101.0, 104.0]}, index=idx)
    early = art("Apple bullish")
    early.published = idx[0]
    early.tickers, early.sentiment_score, early.impact = ["AAPL"], 0.7, 7
    df = run_backtest([early], prices)
    stats = summarize(df)
    for key in ("total_return", "sharpe", "max_drawdown", "total_cost", "avg_turnover"):
        assert key in stats


# ── Full pipeline round-trip (Steps 3+4+5 together) ──────────────────────────

def test_pipeline_enrich_lexicon():
    articles = enrich(
        deduplicate(normalize([art("Tesla stock surges on record profit")])),
        sentiment_model=LexiconSentiment(),
        event_classifier=_kw_clf,
    )
    a = articles[0]
    assert a.sentiment_label == "positive"
    assert -1 <= a.sentiment_score <= 1
    assert isinstance(a.sentiment_scores_raw, dict)
    assert a.event_type in ("earnings", "product_launch", "other")
    assert isinstance(a.event_confidence, float)
    assert 1 <= a.impact <= 10
    assert isinstance(a.impact_breakdown, dict)
    assert "total" in a.impact_breakdown


def test_api_analyze(api_client):
    r = api_client.post("/analyze", json={"text": "Apple surges on record earnings beat"})
    assert r.status_code == 200
    d = r.json()
    assert "sentiment" in d
    assert "score" in d["sentiment"]
    assert "event" in d
    assert "type" in d["event"]
    assert "impact" in d
    assert "score" in d["impact"]
    assert "AAPL" in d["tickers"]


def test_api_analyze_empty(api_client):
    assert api_client.post("/analyze", json={"text": ""}).status_code == 422


def test_api_classify(api_client):
    r = api_client.post("/classify", json={"text": "Company files for bankruptcy and defaults on bonds"})
    assert r.status_code == 200
    d = r.json()
    assert d["event_type"] == "credit_event"
    assert "confidence" in d
    assert "model_votes" in d


# ── Stock Index Rebalancer Tests ──────────────────────────────────────────────

def test_rebalance_initial_composition():
    from newsengine.rebalancer import rebalance_index, DEFAULT_INDEX_STOCKS
    result = rebalance_index([])
    assert result["initial_universe_count"] == 20
    assert len(result["initial_stocks_detail"]) == 20
    assert len(result["rebalanced_portfolio"]) == 20
    assert abs(sum(result["rebalanced_portfolio"].values()) - 1.0) < 1e-3


def test_rebalance_remove_on_severe_negative_news():
    from newsengine.rebalancer import rebalance_index
    # Bad news on TSLA
    bad = art("Tesla files for bankruptcy after default")
    bad.published = NOW
    bad.tickers = ["TSLA"]
    bad.sentiment_score = -0.85
    bad.impact = 8.5
    bad.event_type = "credit_event"
    bad.event_confidence = 0.9

    result = rebalance_index([bad], as_of=NOW)
    assert "TSLA" in result["removed_stocks"]
    assert "TSLA" not in result["rebalanced_portfolio"]
    assert len(result["added_stocks"]) == 1
    # Check that decisions contain REMOVE and ADD
    actions = {d["ticker"]: d["action"] for d in result["decisions"]}
    assert actions["TSLA"] == "REMOVE"
    # Weights still sum to 1.0
    assert abs(sum(result["rebalanced_portfolio"].values()) - 1.0) < 1e-3


def test_rebalance_overweight_on_positive_news():
    from newsengine.rebalancer import rebalance_index
    good = art("Apple beats earnings with record revenue")
    good.published = NOW
    good.tickers = ["AAPL"]
    good.sentiment_score = 0.80
    good.impact = 7.0
    good.event_type = "earnings"
    good.event_confidence = 0.85

    result = rebalance_index([good], as_of=NOW)
    actions = {d["ticker"]: d["action"] for d in result["decisions"]}
    assert actions["AAPL"] == "OVERWEIGHT"
    assert result["rebalanced_portfolio"]["AAPL"] > 0.05


def test_api_rebalance(api_client):
    r = api_client.get("/rebalance")
    assert r.status_code == 200
    d = r.json()
    assert "initial_stocks_detail" in d
    assert "rebalanced_portfolio" in d
    assert "decisions" in d
    assert len(d["initial_stocks_detail"]) == 20


def test_api_rebalance_simulate(api_client):
    r = api_client.post("/rebalance/simulate", json={
        "headline": "Tesla plunges on massive credit default and debt bankruptcy",
        "ticker": "TSLA"
    })
    assert r.status_code == 200
    d = r.json()
    assert "simulated_article" in d
    assert "TSLA" in d["removed_stocks"]
    assert len(d["added_stocks"]) >= 1


# ── Ingestion & Ticker Expansion Tests ────────────────────────────────────────

def test_ticker_expansion_100_plus():
    from newsengine.normalize import extract_tickers
    text = (
        "Palantir signs defense contract while CrowdStrike and Snowflake surge. "
        "Boeing faces FAA probe as Disney reports streaming gains. $COIN and $UBER rally."
    )
    tickers = extract_tickers(text)
    assert "PLTR" in tickers
    assert "CRWD" in tickers
    assert "SNOW" in tickers
    assert "BA" in tickers
    assert "DIS" in tickers
    assert "COIN" in tickers
    assert "UBER" in tickers


def test_reddit_clean_html():
    from newsengine.fetch import _clean_reddit_html
    raw = "<p>Check out this stock analysis submitted by /u/trader123 [link] [comments]</p>"
    cleaned = _clean_reddit_html(raw)
    assert "/u/trader123" not in cleaned
    assert "[link]" not in cleaned
    assert "Check out this stock analysis" in cleaned


def test_rebalance_weight_history_progression():
    from newsengine.rebalancer import rebalance_weight_history
    from datetime import datetime, timezone, timedelta

    t0 = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)
    bad_tsla = art("Tesla massive debt default and emergency liquidation")
    bad_tsla.published = t0 - timedelta(hours=2)
    bad_tsla.tickers = ["TSLA"]
    bad_tsla.sentiment_score = -0.90
    bad_tsla.impact = 9.0
    bad_tsla.event_type = "credit_event"
    bad_tsla.event_confidence = 0.95

    hist = rebalance_weight_history([bad_tsla], as_of=t0, days=7, points=5)
    assert len(hist["timestamps"]) == 5
    assert len(hist["dates"]) == 5
    assert "TSLA" in hist["all_tickers"]
    assert "weights_by_ticker" in hist
    assert len(hist["history_points"]) == 5

    # Prior to the bad news (at t0 - 7 days), TSLA weight is 0.05
    assert hist["weights_by_ticker"]["TSLA"][0] == 0.05
    # After the bad news (at t0), TSLA is removed and weight is 0.0
    assert hist["weights_by_ticker"]["TSLA"][-1] == 0.0

    # Replacement stock should enter index at t0
    added = hist["history_points"][-1]["added"]
    assert len(added) >= 1
    assert hist["weights_by_ticker"][added[0]][-1] > 0.0


def test_api_rebalance_history(api_client):
    r = api_client.get("/rebalance/history?days=7&points=5")
    assert r.status_code == 200
    d = r.json()
    assert "timestamps" in d
    assert "dates" in d
    assert "weights_by_ticker" in d
    assert "history_points" in d
    assert len(d["history_points"]) == 5


# ── Module B: Strategic Portfolio Stress Testing Tests ─────────────────────────

def test_stress_test_portfolio_composition():
    from newsengine.stress_test import DEFAULT_WHOLESALE_PORTFOLIO

    classes = {a["asset_class"] for a in DEFAULT_WHOLESALE_PORTFOLIO}
    assert "loans" in classes
    assert "bonds" in classes
    assert "derivatives" in classes
    assert "equities" in classes

    total_notional = sum(a["notional"] for a in DEFAULT_WHOLESALE_PORTFOLIO)
    assert total_notional == 1_000_000_000  # $1 Billion


def test_stress_test_geopolitical_shock():
    from newsengine.stress_test import run_portfolio_stress_test

    report = run_portfolio_stress_test(
        event_type="geopolitical",
        sentiment_score=-0.80,
        impact_score=8.8,
        trigger_threshold=7.0,
        article_title="Major Geopolitical Escalation and Chokepoint Disruption"
    )

    assert report["is_stress_triggered"] is True
    summary = report["portfolio_summary"]
    assert summary["total_pnl"] < 0
    assert summary["total_pnl_pct"] < 0
    assert "regulatory_capital" in report
    assert report["regulatory_capital"]["stressed_cet1_ratio"] < report["regulatory_capital"]["baseline_cet1_ratio"]

    # Verify all 4 asset classes are evaluated
    breakdown = report["asset_class_breakdown"]
    assert "loans" in breakdown
    assert "bonds" in breakdown
    assert "derivatives" in breakdown
    assert "equities" in breakdown


def test_stress_test_macroeconomic_rate_spike():
    from newsengine.stress_test import run_portfolio_stress_test

    report = run_portfolio_stress_test(
        event_type="macroeconomic",
        sentiment_score=-0.60,
        impact_score=8.0,
        trigger_threshold=7.0,
    )

    assert report["is_stress_triggered"] is True
    shocks = report["shocks_applied"]
    assert shocks["rate_shock_bps"] > 100.0  # rate shift upward


def test_stress_test_below_threshold():
    from newsengine.stress_test import run_portfolio_stress_test

    report = run_portfolio_stress_test(
        event_type="product_launch",
        sentiment_score=0.40,
        impact_score=4.5,
        trigger_threshold=7.0,
    )

    assert report["is_stress_triggered"] is False


def test_api_stress_test_endpoints(api_client):
    # GET /stress-test/portfolio
    r = api_client.get("/stress-test/portfolio")
    assert r.status_code == 200
    d = r.json()
    assert d["total_notional"] == 1_000_000_000
    assert len(d["assets"]) > 0

    # POST /stress-test/simulate with explicit parameters
    r_sim = api_client.post("/stress-test/simulate", json={
        "event_type": "credit_event",
        "sentiment_score": -0.85,
        "impact_score": 9.0,
        "trigger_threshold": 7.0
    })
    assert r_sim.status_code == 200
    d_sim = r_sim.json()
    assert d_sim["is_stress_triggered"] is True
    assert d_sim["portfolio_summary"]["total_pnl"] < 0

    # GET /stress-test/events
    r_ev = api_client.get("/stress-test/events")
    assert r_ev.status_code == 200
    assert isinstance(r_ev.json(), list)

    # GET /module_b
    r_mod = api_client.get("/module_b")
    assert r_mod.status_code == 200


def test_api_sentiment_endpoint(api_client):
    r = api_client.post("/sentiment", json={"text": "Apple beats earnings with record quarterly revenue"})
    assert r.status_code == 200
    d = r.json()
    assert "sentiment_label" in d
    assert "sentiment_score" in d
    assert "probabilities" in d
    assert d["sentiment_label"] in ("positive", "negative", "neutral")


def test_api_impact_endpoint(api_client):
    r = api_client.post("/impact", json={
        "text": "Boeing credit default and emergency FAA grounding of entire fleet",
        "event_type": "credit_event",
        "tickers": ["BA"],
    })
    assert r.status_code == 200
    d = r.json()
    assert "impact_score" in d
    assert "breakdown" in d
    assert "reasons" in d
    assert 1.0 <= d["impact_score"] <= 10.0


def test_api_sources_endpoint(api_client):
    r = api_client.get("/sources")
    assert r.status_code == 200
    d = r.json()
    assert "total_sources" in d
    assert "sources" in d
    assert d["total_sources"] >= 10


def test_api_page_routes(api_client):
    for path in [
        "/",
        "/setup",
        "/api_setup",
        "/api_setup.html",
        "/module_a",
        "/module_a.html",
        "/module_b",
        "/module_b.html",
    ]:
        r = api_client.get(path)
        assert r.status_code == 200




