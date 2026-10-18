"""Step 5: structured market impact estimation (1–10).

Design
------
Impact is computed as a weighted sum of five independent sub-scores, each
with a named contribution that is stored in ``Article.impact_breakdown`` for
full transparency:

  impact = clip(base + severity + magnitude + urgency + relevance
               + geographic_scope + event_confidence_bonus, 1, 10)

Sub-score definitions
---------------------
base               event-type baseline (1–6)
severity           high-severity keywords (bankruptcy, invasion, …)  0–4
magnitude          financial scale detected ($Xbn vs $Xm)           0–2
urgency            breaking/emergency framing                        0–2
relevance          named tickers present / absent                   -1..+1
geographic_scope   global / nationwide language detected             0–1
event_confidence   bonus when NLP classifier is highly confident     0–1

All sub-scores are capped individually so keyword-stuffing or adversarial
inputs cannot push impact above 10.

Returns
-------
impact  : float in [1.0, 10.0]
breakdown : dict[str, float]   – named sub-scores for transparency
reasons : list[str]            – human-readable explanation strings
"""
from __future__ import annotations

import re

# ── Base scores by event type ─────────────────────────────────────────────────
_BASE: dict[str, float] = {
    "credit_event":         6.0,
    "geopolitical":         5.0,
    "macroeconomic":        5.0,
    "mergers_acquisitions": 5.0,
    "earnings":             4.0,
    "product_launch":       3.0,
    "other":                1.0,
}

# ── Severity keywords (keyword → raw boost) ────────────────────────────────────
_SEVERITY: dict[str, float] = {
    "bankruptcy":       3.0,
    "default":          3.0,
    "chapter 11":       3.0,
    "invasion":         3.0,
    "war":              2.5,
    "crash":            2.5,
    "recession":        2.5,
    "fraud":            2.5,
    "emergency":        2.0,
    "crisis":           2.0,
    "collapse":         2.0,
    "plunge":           1.5,
    "plunges":          1.5,
    "surge":            1.0,
    "surges":           1.0,
    "record":           1.0,
    "rate hike":        1.0,
    "rate cut":         1.0,
    "sanctions":        1.5,
    "downgrade":        1.5,
    "layoffs":          1.0,
    "recall":           1.0,
}
_SEVERITY_CAP = 4.0   # cap so keyword stuffing can't dominate

# ── Financial magnitude detection ─────────────────────────────────────────────
# Matches patterns like $500B, $1.2 trillion, 50 billion dollars, etc.
_TRILLION = re.compile(r"(\$[\d,.]+\s*(trillion|t\b)|\b[\d,.]+\s*trillion)", re.I)
_BILLION  = re.compile(r"(\$[\d,.]+\s*(billion|bn\b|\bb\b)|\b[\d,.]+\s*billion)", re.I)
_MILLION  = re.compile(r"(\$[\d,.]+\s*(million|mn\b|\bm\b)|\b[\d,.]+\s*million)", re.I)

# ── Urgency framing ───────────────────────────────────────────────────────────
_URGENCY = re.compile(
    r"\b(breaking|just in|flash|alert|urgent|immediate|emergency|developing)\b", re.I
)

# ── Geographic scope ──────────────────────────────────────────────────────────
_GLOBAL = re.compile(
    r"\b(global|worldwide|international|across the world|multi.country|cross.border)\b", re.I
)
_NATIONAL = re.compile(r"\b(national|nationwide|across the (us|uk|eu|china))\b", re.I)


def estimate_impact(
    text: str,
    event_type: str,
    tickers: list[str],
    event_confidence: float = 0.5,
) -> tuple[float, dict, list[str]]:
    """Estimate market impact.

    Parameters
    ----------
    text              : cleaned article text (title + body)
    event_type        : canonical event category from Step 4
    tickers           : list of ticker symbols extracted in Step 2
    event_confidence  : classifier confidence from Step 4 (used as bonus)

    Returns
    -------
    impact    : float in [1.0, 10.0]
    breakdown : dict of named sub-scores
    reasons   : list of human-readable explanation strings
    """
    low = text.lower()
    reasons: list[str] = []
    breakdown: dict[str, float] = {}

    # 1. Base score
    base = _BASE.get(event_type, 1.0)
    breakdown["base"] = base
    reasons.append(f"base[{event_type}] = {base}")

    # 2. Severity keywords (individual hits summed, then capped)
    sev_raw = 0.0
    hit_kws: list[str] = []
    for kw, boost in _SEVERITY.items():
        if kw in low:
            sev_raw += boost
            hit_kws.append(f"'{kw}' +{boost}")
    severity = round(min(sev_raw, _SEVERITY_CAP), 2)
    breakdown["severity"] = severity
    if hit_kws:
        reasons.append(f"severity keywords [{', '.join(hit_kws[:4])}] → capped at {severity}")

    # 3. Financial magnitude
    magnitude = 0.0
    if _TRILLION.search(text):
        magnitude = 2.0
        reasons.append("financial scale: trillion-dollar figure +2")
    elif _BILLION.search(text):
        magnitude = 1.0
        reasons.append("financial scale: billion-dollar figure +1")
    elif _MILLION.search(text):
        magnitude = 0.5
        reasons.append("financial scale: million-dollar figure +0.5")
    breakdown["magnitude"] = magnitude

    # 4. Urgency framing
    urgency_hits = _URGENCY.findall(text)
    urgency = min(2.0, len(urgency_hits) * 1.0)
    breakdown["urgency"] = urgency
    if urgency_hits:
        reasons.append(f"urgency framing [{', '.join(set(h.lower() for h in urgency_hits))}] +{urgency}")

    # 5. Ticker/company relevance
    if tickers:
        relevance = min(1.0, 0.5 + 0.25 * len(tickers))   # +0.5 base, +0.25 per extra ticker
        reasons.append(f"relevance: {len(tickers)} tracked ticker(s) +{relevance:.2f}")
    elif event_type in ("product_launch", "earnings", "mergers_acquisitions"):
        relevance = -1.0
        reasons.append("relevance: company-specific event without named ticker -1")
    else:
        relevance = 0.0
    breakdown["relevance"] = round(relevance, 2)

    # 6. Geographic scope
    if _GLOBAL.search(text):
        geo = 1.0
        reasons.append("geographic scope: global event +1")
    elif _NATIONAL.search(text):
        geo = 0.5
        reasons.append("geographic scope: national event +0.5")
    else:
        geo = 0.0
    breakdown["geographic_scope"] = geo

    # 7. Event-confidence bonus  (NLP classifier certainty)
    # Only award bonus when confidence is high (> 0.7) to avoid amplifying
    # uncertain classifications.
    conf_bonus = round(max(0.0, event_confidence - 0.7) * 2, 2)   # max +0.6
    breakdown["event_confidence_bonus"] = conf_bonus
    if conf_bonus > 0:
        reasons.append(f"event_confidence={event_confidence:.2f} → bonus +{conf_bonus}")

    # ── Composite ─────────────────────────────────────────────────────────────
    raw = base + severity + magnitude + urgency + relevance + geo + conf_bonus
    impact = round(max(1.0, min(10.0, raw)), 2)
    breakdown["total"] = impact

    return impact, breakdown, reasons
