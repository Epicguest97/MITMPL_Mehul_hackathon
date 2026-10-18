"""Step 2: normalize, extract tickers, deduplicate / cluster."""
from __future__ import annotations

import html
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from .models import Article

# ── Comprehensive 100+ Ticker Master Universe across all 11 Market Sectors ────
COMPANIES: dict[str, list[str]] = {
    # ── Mega-Cap Tech & Semiconductors ───────────────────────────────────────
    "AAPL":  ["apple", "iphone", "ipad", "macbook", "tim cook"],
    "MSFT":  ["microsoft", "azure", "windows", "satya nadella"],
    "NVDA":  ["nvidia", "jensen huang", "geforce", "blackwell", "hopper"],
    "GOOGL": ["alphabet", "google", "sundar pichai", "youtube", "waymo"],
    "GOOG":  ["alphabet", "google"],
    "AMZN":  ["amazon", "aws", "andy jassy", "prime"],
    "META":  ["meta platforms", "facebook", "mark zuckerberg", "instagram", "whatsapp", "oculus"],
    "TSLA":  ["tesla", "elon musk", "cybertruck", "gigafactory"],
    "AVGO":  ["broadcom", "hock tan", "vmware"],
    "ORCL":  ["oracle", "larry ellison"],
    "CRM":   ["salesforce", "marc benioff"],
    "AMD":   ["advanced micro devices", "amd", "lisa su", "ryzen", "radeon"],
    "ADBE":  ["adobe", "photoshop"],
    "CSCO":  ["cisco", "cisco systems", "chuck robbins"],
    "QCOM":  ["qualcomm", "snapdragon"],
    "TXN":   ["texas instruments"],
    "INTC":  ["intel", "pat gelsinger"],
    "IBM":   ["ibm", "international business machines"],
    "AMAT":  ["applied materials"],
    "MU":    ["micron", "micron technology"],
    "LRCX":  ["lam research"],
    "ADI":   ["analog devices"],
    "NOW":   ["servicenow"],
    "PANW":  ["palo alto networks"],
    "SNOW":  ["snowflake"],
    "PLTR":  ["palantir", "alex karp"],
    "CRWD":  ["crowdstrike", "george kurtz"],
    "UBER":  ["uber", "dara khosrowshahi"],
    "ABNB":  ["airbnb", "brian chesky"],
    "DELL":  ["dell", "dell technologies", "michael dell"],

    # ── Financials ───────────────────────────────────────────────────────────
    "BRK.B": ["berkshire hathaway", "berkshire", "warren buffett", "buffett"],
    "JPM":   ["jpmorgan", "jp morgan", "chase", "jamie dimon"],
    "V":     ["visa"],
    "MA":    ["mastercard"],
    "BAC":   ["bank of america", "bofa", "brian moynihan"],
    "WFC":   ["wells fargo", "charlie scharf"],
    "C":     ["citigroup", "citi", "jane fraser"],
    "GS":    ["goldman sachs", "goldman", "david solomon"],
    "MS":    ["morgan stanley", "ted pick"],
    "BLK":   ["blackrock", "larry fink"],
    "SCHW":  ["charles schwab", "schwab"],
    "AXP":   ["american express", "amex"],
    "PYPL":  ["paypal"],
    "COIN":  ["coinbase", "brian armstrong"],

    # ── Healthcare & Pharmaceuticals ─────────────────────────────────────────
    "LLY":   ["eli lilly", "lilly", "mounjaro", "zepbound"],
    "UNH":   ["unitedhealth", "united health", "optum"],
    "JNJ":   ["johnson & johnson", "johnson and johnson", "j&j"],
    "ABBV":  ["abbvie", "humira", "skyrizi"],
    "MRK":   ["merck", "keytruda"],
    "TMO":   ["thermo fisher", "thermo fisher scientific"],
    "ABT":   ["abbott", "abbott laboratories"],
    "PFE":   ["pfizer", "albert bourla"],
    "DHR":   ["danaher"],
    "BMY":   ["bristol myers", "bristol-myers squibb"],
    "AMGN":  ["amgen"],
    "GILD":  ["gilead", "gilead sciences"],
    "ISRG":  ["intuitive surgical", "da vinci"],
    "VRTX":  ["vertex pharmaceuticals", "vertex"],
    "REGN":  ["regeneron"],
    "MRNA":  ["moderna"],
    "CVS":   ["cvs health", "cvs"],

    # ── Consumer Discretionary & Retail ──────────────────────────────────────
    "HD":    ["home depot"],
    "LOW":   ["lowe's", "lowes"],
    "NKE":   ["nike"],
    "MCD":   ["mcdonald's", "mcdonalds"],
    "SBUX":  ["starbucks"],
    "TGT":   ["target corp", "target"],
    "TJX":   ["tjx", "t.j. maxx", "marshalls"],
    "BKNG":  ["booking holdings", "booking.com"],
    "LULU":  ["lululemon"],
    "F":     ["ford", "ford motor"],
    "GM":    ["general motors", "gm", "mary barra"],

    # ── Consumer Staples ─────────────────────────────────────────────────────
    "WMT":   ["walmart", "sam walton"],
    "COST":  ["costco", "costco wholesale"],
    "PG":    ["procter & gamble", "procter and gamble", "p&g"],
    "KO":    ["coca-cola", "coca cola", "coke"],
    "PEP":   ["pepsico", "pepsi"],
    "PM":    ["philip morris"],
    "MO":    ["altria"],
    "CL":    ["colgate-palmolive", "colgate"],
    "MDLZ":  ["mondelez"],

    # ── Energy, Industrials & Defense ────────────────────────────────────────
    "XOM":   ["exxon", "exxonmobil", "darren woods"],
    "CVX":   ["chevron", "mike wirth"],
    "COP":   ["conocophillips"],
    "SLB":   ["schlumberger"],
    "OXY":   ["occidental petroleum", "occidental"],
    "CAT":   ["caterpillar"],
    "GE":    ["general electric", "ge aerospace"],
    "BA":    ["boeing"],
    "UNP":   ["union pacific"],
    "HON":   ["honeywell"],
    "LMT":   ["lockheed martin", "lockheed"],
    "RTX":   ["raytheon", "rtx corp"],
    "DE":    ["deere", "john deere"],
    "UPS":   ["united parcel service", "ups"],
    "FDX":   ["fedex"],
    "NOC":   ["northrop grumman"],
    "GD":    ["general dynamics"],

    # ── Communication, Media & Utilities ─────────────────────────────────────
    "NFLX":  ["netflix"],
    "DIS":   ["walt disney", "disney", "bob iger"],
    "CMCSA": ["comcast", "nbcuniversal"],
    "T":     ["at&t"],
    "VZ":    ["verizon"],
    "TMUS":  ["t-mobile"],
    "NEE":   ["nextera energy", "nextera"],
    "SO":    ["southern company"],
    "DUK":   ["duke energy"],
}

STOPCASH = {
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CNY", "INR",
    "GDP", "CPI", "ETF", "CEO", "CFO", "CTO", "COO", "IPO", "SEC", "FDA", "DOJ", "FED",
    "USA", "US", "UK", "EU", "AI", "EV", "PE", "EPS", "BPS", "YOY", "MOM", "ATH",
    "ALL", "NEW", "NOW", "FOR", "OUT", "THE", "AND", "NOT", "ONE", "TWO", "TOP", "BIG",
    "SEE", "HAS", "CAN", "BUT", "MAY", "BUY", "JOB", "PAY", "DAY", "WAR", "OIL", "GAS", "CAR"
}

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_CASHTAG = re.compile(r"(?<![A-Za-z0-9])\$([A-Z]{1,5})\b")
_TRACK = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}


def clean_text(s: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def canonical_url(url: str) -> str:
    p = urlsplit(url.strip())
    q = [(k, v) for k, v in parse_qsl(p.query) if k not in _TRACK]
    return urlunsplit((p.scheme.lower(), p.netloc.lower().removeprefix("www."),
                       p.path.rstrip("/"), urlencode(q), ""))


def extract_tickers(text: str) -> list[str]:
    found = {m for m in _CASHTAG.findall(text) if m not in STOPCASH}
    low = text.lower()
    for sym, names in COMPANIES.items():
        if any(re.search(rf"\b{re.escape(n)}\b", low) for n in names):
            found.add(sym)
        elif len(sym) >= 2 and sym not in STOPCASH and re.search(rf"\b{re.escape(sym)}\b", text):
            found.add(sym)
    return sorted(found)


def _tokens(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def normalize(raw: list[Article]) -> list[Article]:
    out = []
    for a in raw:
        if not a.url or not a.title.strip():
            continue
        a.title, a.text = clean_text(a.title), clean_text(a.text)
        a.url = canonical_url(a.url)
        a.tickers = extract_tickers(f"{a.title} {a.text}")
        out.append(a)
    return out


def deduplicate(articles: list[Article], threshold: float = 0.6) -> list[Article]:
    clusters: list[list[Article]] = []
    for a in sorted(articles, key=lambda x: x.published, reverse=True):
        toks = _tokens(a.title)
        merged = False
        for c in clusters:
            if _jaccard(toks, _tokens(c[0].title)) >= threshold:
                c.append(a)
                merged = True
                break
        if not merged:
            clusters.append([a])

    out = []
    for c in clusters:
        primary = c[0]
        primary.duplicate_count = len(c)
        primary.cluster_sources = sorted({x.source for x in c})
        all_tickers = {t for x in c for t in x.tickers}
        primary.tickers = sorted(all_tickers)
        out.append(primary)
    return out
