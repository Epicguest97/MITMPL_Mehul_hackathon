"""Step 1: Multi-source financial news & social ingestion (RSS + Reddit + GDELT).

Features
--------
* 10+ high-reliability live financial RSS streams (Google Finance, CNBC, WSJ,
  MarketWatch, Yahoo Finance, Investing.com, Benzinga, Seeking Alpha).
* Real-time Reddit social sentiment ingestion (r/stocks, r/wallstreetbets).
* GDELT fallback with tight timeout so it never stalls the pipeline.
* Parallel ThreadPoolExecutor fetcher (all sources fetched in ~2-3 seconds).
"""
from __future__ import annotations

import calendar
import concurrent.futures
import html
import logging
import re
from datetime import datetime, timezone
from typing import Any

from .models import Article

log = logging.getLogger(__name__)

# Standard browser user-agent to prevent 403 blocks from Reddit and news feeds
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36 FinancialNewsEngine/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── 10+ Verified High-Reliability Financial News Feeds ─────────────────────────
RSS_FEEDS = {
    "google_finance":        "https://news.google.com/rss/search?q=stocks+market+when:1d&hl=en-US&gl=US&ceid=US:en",
    "cnbc_markets":         "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "cnbc_top":             "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "wsj_markets":          "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "marketwatch_top":      "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "marketwatch_bulletins":"https://feeds.content.dowjones.io/public/rss/mw_bulletins",
    "yahoo_sp500":          "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC",
    "investing_stocks":     "https://www.investing.com/rss/stock.rss",
    "benzinga":             "https://www.benzinga.com/feed",
    "seeking_alpha":        "https://seekingalpha.com/market_currents.xml",
}

# ── Reddit Financial Subreddit Feeds ──────────────────────────────────────────
REDDIT_FEEDS = {
    "reddit_stocks": "https://www.reddit.com/r/stocks/.rss",
    "reddit_wsb":    "https://www.reddit.com/r/wallstreetbets/.rss",
}

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _clean_reddit_html(raw_html: str) -> str:
    """Strip Reddit RSS submission HTML artifacts and links."""
    text = re.sub(r"<[^>]+>", " ", raw_html or "")
    text = html.unescape(text)
    text = re.sub(r"submitted by\s+/u/\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[link\]\s*\[comments\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def fetch_rss(name: str, url: str, timeout: float = 6.0) -> list[Article]:
    """Fetch and parse a standard financial RSS feed with browser headers."""
    import feedparser
    import requests

    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            log.warning("RSS %s returned HTTP %d", name, r.status_code)
            return []
        feed = feedparser.parse(r.content)
    except Exception as exc:
        log.warning("RSS %s fetch failed: %s", name, exc)
        return []

    out = []
    for e in feed.entries:
        ts = e.get("published_parsed") or e.get("updated_parsed")
        published = (
            datetime.fromtimestamp(calendar.timegm(ts), timezone.utc)
            if ts else datetime.now(timezone.utc)
        )
        title = e.get("title", "").strip()
        summary = e.get("summary", "").strip()
        link = e.get("link", "").strip()
        if not title or not link:
            continue
        out.append(Article(
            title=title,
            text=summary,
            published=published,
            source=name,
            url=link,
        ))
    return out


def fetch_reddit(name: str, url: str, timeout: float = 6.0) -> list[Article]:
    """Fetch live market discussions from financial subreddits."""
    import feedparser
    import requests

    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            log.warning("Reddit %s returned HTTP %d", name, r.status_code)
            return []
        feed = feedparser.parse(r.content)
    except Exception as exc:
        log.warning("Reddit %s fetch failed: %s", name, exc)
        return []

    sub = name.replace("reddit_", "r/")
    out = []
    for e in feed.entries:
        ts = e.get("published_parsed") or e.get("updated_parsed")
        published = (
            datetime.fromtimestamp(calendar.timegm(ts), timezone.utc)
            if ts else datetime.now(timezone.utc)
        )
        title = e.get("title", "").strip()
        content = e.get("summary", "")
        clean_body = _clean_reddit_html(content)
        link = e.get("link", "").strip()
        if not title or not link:
            continue
        out.append(Article(
            title=title,
            text=clean_body[:500],
            published=published,
            source=f"reddit:{sub}",
            url=link,
        ))
    return out


def fetch_gdelt(
    query: str = "(stocks OR earnings OR economy OR merger)",
    max_records: int = 50,
    timespan: str = "6h",
    timeout: float = 4.0,
) -> list[Article]:
    """Fetch GDELT with a fast timeout so it never hangs the pipeline."""
    import requests

    try:
        r = requests.get(
            GDELT_URL,
            headers=HEADERS,
            timeout=timeout,
            params={
                "query": f"{query} sourcelang:english",
                "mode": "ArtList",
                "format": "json",
                "maxrecords": max_records,
                "timespan": timespan,
                "sort": "DateDesc",
            },
        )
        if r.status_code != 200:
            log.warning("GDELT returned HTTP %d", r.status_code)
            return []
        articles_json = r.json().get("articles", [])
    except Exception as exc:
        log.info("GDELT unavailable (%s); skipping without stalling", exc)
        return []

    out = []
    for a in articles_json:
        try:
            published = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            out.append(Article(
                title=a.get("title", "").strip(),
                text="",
                published=published,
                source=f"gdelt:{a.get('domain', 'unknown')}",
                url=a.get("url", ""),
            ))
        except Exception:
            continue
    return out


def fetch_all(min_sources: int = 2) -> list[Article]:
    """Fetch from all 12+ independent sources in parallel using ThreadPoolExecutor.

    Includes Google Finance, CNBC, WSJ, MarketWatch, Yahoo Finance,
    Reddit (r/stocks, r/wallstreetbets), Investing.com, Benzinga, and GDELT.
    A single slow or failing source never stalls the overall ingestion.
    """
    tasks: dict[str, Any] = {}
    articles: list[Article] = []
    sources_ok = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit RSS feeds
        for name, url in RSS_FEEDS.items():
            tasks[executor.submit(fetch_rss, name, url)] = name

        # Submit Reddit feeds
        for name, url in REDDIT_FEEDS.items():
            tasks[executor.submit(fetch_reddit, name, url)] = name

        # Submit GDELT (with fast timeout)
        tasks[executor.submit(fetch_gdelt)] = "gdelt"

        for future in concurrent.futures.as_completed(tasks):
            name = tasks[future]
            try:
                batch = future.result()
                if batch:
                    sources_ok += 1
                    articles.extend(batch)
                    log.info("Fetched %d articles from %s", len(batch), name)
            except Exception as exc:
                log.warning("Fetch task %s generated exception: %s", name, exc)

    log.info("Total fetched: %d articles from %d sources", len(articles), sources_ok)
    if sources_ok < min_sources:
        log.warning("Only %d sources returned data (wanted %d)", sources_ok, min_sources)

    return articles
