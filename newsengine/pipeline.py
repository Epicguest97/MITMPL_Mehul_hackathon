"""End-to-end orchestration: fetch → normalize → dedupe → sentiment → event → impact."""
from __future__ import annotations

from .events import classify_event
from .fetch import fetch_all
from .impact import estimate_impact
from .models import Article
from .normalize import deduplicate, normalize
from .sentiment import score_articles


def enrich(articles: list[Article], sentiment_model=None, event_classifier=None) -> list[Article]:
    """Run Steps 3–5 on a pre-normalised, deduplicated article list.

    Parameters
    ----------
    articles          : list of Article objects (already normalised)
    sentiment_model   : pre-loaded FinBERT / LexiconSentiment instance (optional)
    event_classifier  : pre-loaded ZeroShot / Keyword classifier instance (optional)
    """
    if not articles:
        return []

    # Step 3 – FinBERT sentiment  (writes label, score, raw probs to each article)
    score_articles(articles, sentiment_model)

    for a in articles:
        text = f"{a.title}. {a.text}"

        # Step 4 – NLP event classification  (returns (type, confidence))
        a.event_type, a.event_confidence = classify_event(text, event_classifier)

        # Step 5 – Structured impact estimation (returns (score, breakdown, reasons))
        a.impact, a.impact_breakdown, a.impact_reasons = estimate_impact(
            text, a.event_type, a.tickers, a.event_confidence
        )

    return articles


def run_pipeline(
    raw: list[Article] | None = None,
    sentiment_model=None,
    event_classifier=None,
) -> list[Article]:
    raw = fetch_all() if raw is None else raw
    return enrich(deduplicate(normalize(raw)), sentiment_model, event_classifier)
