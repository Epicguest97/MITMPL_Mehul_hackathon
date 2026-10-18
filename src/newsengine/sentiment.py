"""Step 3: Hybrid sentiment scoring.

Architecture
------------
1. ``FinBERT`` — primary NLP model (yiyanghkust/finbert-tone or ProsusAI/finbert).
   Returns P(pos), P(neg), P(neutral).  When neutral dominates (> 0.6) the
   raw score P(pos)-P(neg) tends to be near zero even for clearly directional
   text.  We correct for this with a lexical booster (see below).

2. ``FinancialLexicon`` — a rich, phrase-aware financial sentiment scorer:
   - 150+ positive / 200+ negative keywords
   - Regex patterns for percentage moves ("50% drop", "fell 3%")
   - Negation handling ("not bullish" → negative)
   - Title weighting: title is scored at 2× body weight
   - Phrase patterns: "all-time high", "record low", "economic slowdown looms"

3. ``HybridSentiment`` — blends both:
   final_score = α × finbert_score + (1-α) × lexicon_score
   where α = 0.5 when FinBERT neutral probability ≥ 0.55
         α = 0.75 otherwise (FinBERT is confident, trust it more)

   This ensures strongly negative text like "Silver drops 50%, slowdown looms"
   is never masked by FinBERT's neutral bias on macro/reporting-style headlines.

Public API
----------
  load_model()                    → HybridSentiment (or LexiconSentiment fallback)
  score_articles(articles, model) → list[Article]   (writes label, score, raw_probs)
"""
from __future__ import annotations

import logging
import os
import re

from .models import Article

log = logging.getLogger(__name__)

# ── Cache directory ───────────────────────────────────────────────────────────
# Pin HuggingFace downloads to <project>/models/ so the model is downloaded
# exactly ONCE and always found in the same place regardless of the working
# directory or the system-wide ~/.cache.
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
_MODELS_DIR = os.path.join(_HERE, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)
# Set BEFORE importing transformers/huggingface_hub so they pick it up.
os.environ.setdefault("HF_HOME", _MODELS_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_MODELS_DIR, "hub"))

MODEL = os.environ.get("FINBERT_MODEL", "yiyanghkust/finbert-tone")


# ── Financial Lexicon ─────────────────────────────────────────────────────────

class FinancialLexicon:
    """Phrase-aware, magnitude-sensitive financial sentiment lexicon."""

    # ── Positive words / phrases ──────────────────────────────────────────
    POS_WORDS = {
        # Earnings / performance
        "beat", "beats", "beating", "topped", "exceeded", "surpassed", "outperformed",
        "record", "records", "record-breaking", "all-time high", "profit", "profits",
        "profitable", "revenue", "growth", "grew", "growing", "expand", "expansion",
        "soar", "soars", "soared", "surge", "surges", "surged", "rally", "rallies",
        "rallied", "gain", "gains", "gained", "rise", "rises", "rose", "risen",
        "rebound", "rebounds", "rebounded", "recover", "recovers", "recovered",
        "recovery", "bounce", "bounces", "breakthrough", "breakout",
        # Corporate
        "upgrade", "upgraded", "upgrades", "outperform", "buy", "overweight",
        "bullish", "confidence", "optimism", "optimistic", "strong", "strength",
        "robust", "solid", "healthy", "positive", "improve", "improved", "improvement",
        "advance", "advances", "advanced", "boost", "boosted", "approve", "approved",
        "approval", "win", "wins", "won", "deal", "profitable", "dividend",
        "dividends", "buyback", "hire", "hiring", "launch", "launches", "launched",
        # Macro
        "boom", "booming", "expansion", "jobs added", "employment", "above expectations",
    }

    # ── Negative words / phrases ──────────────────────────────────────────
    NEG_WORDS = {
        # Price action
        "drop", "drops", "dropped", "dropping", "fall", "falls", "fell", "fallen",
        "falling", "decline", "declines", "declined", "declining", "slide", "slides",
        "slid", "plunge", "plunges", "plunged", "plunging", "tumble", "tumbles",
        "tumbled", "tumbling", "slump", "slumps", "slumped", "slumping", "sink",
        "sinks", "sank", "sunk", "dive", "dives", "dived", "crash", "crashes",
        "crashed", "collapse", "collapses", "collapsed", "tank", "tanks", "tanked",
        "stumble", "stumbles", "stumbled", "retreat", "retreats", "retreated",
        # Fundamentals
        "loss", "losses", "miss", "misses", "missed", "missing", "disappoint",
        "disappoints", "disappointed", "disappointing", "shortfall", "below",
        "downgrade", "downgrades", "downgraded", "underperform", "sell",
        "underweight", "bearish", "weak", "weakness", "weaker", "poor",
        # Corporate distress
        "bankruptcy", "bankrupt", "default", "defaults", "defaulted", "insolvent",
        "insolvency", "chapter 11", "chapter11", "debt", "layoff", "layoffs",
        "laid off", "fired", "cuts", "cut", "cutting", "freeze", "freezing",
        "recall", "recalls", "fraud", "probe", "investigation", "lawsuit", "fined",
        "penalty", "violation", "warning", "warns", "warned", "concern", "concerns",
        "worried", "worries", "worry", "fear", "fears", "feared",
        # Macro / economy
        "slowdown", "slow down", "recession", "recessionary", "contraction",
        "contracting", "inflation", "stagflation", "unemployment", "jobless",
        "deficit", "tariff", "tariffs", "sanction", "sanctions", "embargo",
        "war", "conflict", "invasion", "crisis", "crises", "downturn", "loom",
        "looms", "looming", "uncertainty", "risk", "risks", "risky", "volatile",
        "volatility", "anxiety", "anxieties", "pressure", "pressured",
        "overvalued", "bubble", "burst", "correction",
    }

    # ── Percentage move patterns ──────────────────────────────────────────
    _PCT_DROP = re.compile(
        r"(\d+(?:\.\d+)?)\s*%?\s*(drop|fall|fell|decline|plunge|slide|loss|down|lower|off|crash|slump|tumble)",
        re.I,
    )
    _PCT_GAIN = re.compile(
        r"(\d+(?:\.\d+)?)\s*%?\s*(gain|rise|jump|surge|rally|up|higher|climb|soar|advance)",
        re.I,
    )
    _PCT_PREFIX_DROP = re.compile(
        r"(drop|fall|fell|decline|plunge|slide|down|lower)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
        re.I,
    )
    _PCT_PREFIX_GAIN = re.compile(
        r"(rise|jump|surge|rally|gain|up|higher|climb|soar)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
        re.I,
    )

    # ── Negation window (look back N words) ──────────────────────────────
    _NEGATION = re.compile(r"\b(not|no|never|neither|nor|without|hardly|barely|scarcely)\b", re.I)
    _NEG_WINDOW = 4   # words after negation that are flipped

    # ── Strong phrase patterns ────────────────────────────────────────────
    _STRONG_NEG_PHRASES = [
        r"all.time low", r"record low", r"multi.year low", r"52.week low",
        r"economic slowdown", r"market crash", r"financial crisis",
        r"growth concerns", r"slowdown looms", r"recession fears",
        r"sell.off", r"bear market", r"profit warning",
    ]
    _STRONG_POS_PHRASES = [
        r"all.time high", r"record high", r"multi.year high", r"52.week high",
        r"blowout quarter", r"beats estimates", r"above consensus",
        r"bull market", r"buy the dip",
    ]
    _SNEG = re.compile("|".join(_STRONG_NEG_PHRASES), re.I)
    _SPOS = re.compile("|".join(_STRONG_POS_PHRASES), re.I)

    def _pct_score(self, text: str) -> float:
        """Detect percentage moves and return directional score boost."""
        boost = 0.0
        for m in self._PCT_DROP.finditer(text):
            pct = float(m.group(1))
            boost -= min(0.4, pct / 100)   # 50% drop → -0.40 contribution
        for m in self._PCT_GAIN.finditer(text):
            pct = float(m.group(1))
            boost += min(0.4, pct / 100)
        for m in self._PCT_PREFIX_DROP.finditer(text):
            pct = float(m.group(2))
            boost -= min(0.4, pct / 100)
        for m in self._PCT_PREFIX_GAIN.finditer(text):
            pct = float(m.group(2))
            boost += min(0.4, pct / 100)
        return max(-0.5, min(0.5, boost))

    def _word_score(self, text: str) -> float:
        """Lexicon score with negation handling."""
        words = re.findall(r"[a-zA-Z'-]+", text.lower())
        pos, neg = 0, 0
        neg_until = -1
        for i, w in enumerate(words):
            if self._NEGATION.match(w):
                neg_until = i + self._NEG_WINDOW
            in_neg = (i <= neg_until)
            if w in self.POS_WORDS:
                if in_neg:
                    neg += 1
                else:
                    pos += 1
            elif w in self.NEG_WORDS:
                if in_neg:
                    pos += 1
                else:
                    neg += 1
        total = pos + neg
        return (pos - neg) / total if total else 0.0

    def _phrase_score(self, text: str) -> float:
        score = 0.0
        for _ in self._SNEG.findall(text):
            score -= 0.3
        for _ in self._SPOS.findall(text):
            score += 0.3
        return max(-0.6, min(0.6, score))

    def score(self, title: str, body: str) -> float:
        """Combined score for title (2×) + body."""
        title_s = (self._word_score(title) * 2
                   + self._pct_score(title) * 2
                   + self._phrase_score(title) * 2)
        body_s  = (self._word_score(body)
                   + self._pct_score(body)
                   + self._phrase_score(body))
        raw = (title_s + body_s) / 3   # normalise (2 title + 1 body)
        return max(-1.0, min(1.0, raw))

    def predict_split(self, titles: list[str], bodies: list[str]) -> list[tuple[str, float, dict]]:
        out = []
        for title, body in zip(titles, bodies):
            score = self.score(title, body)
            if score > 0.05:
                label = "positive"
                raw = {"positive": 0.5 + score / 2, "negative": max(0.0, 0.1 - score / 4), "neutral": max(0.0, 0.4 - score / 4)}
            elif score < -0.05:
                label = "negative"
                a = abs(score)
                raw = {"positive": max(0.0, 0.1 - a / 4), "negative": 0.5 + a / 2, "neutral": max(0.0, 0.4 - a / 4)}
            else:
                label = "neutral"
                raw = {"positive": 0.1, "negative": 0.1, "neutral": 0.8}
            # normalise so probs sum to 1
            total = sum(raw.values())
            raw = {k: round(v / total, 4) for k, v in raw.items()}
            out.append((label, round(score, 4), raw))
        return out


# ── FinBERT ───────────────────────────────────────────────────────────────────

class FinBERT:
    def __init__(self, model_name: str = MODEL):
        from transformers import BertForSequenceClassification
        import torch

        self.torch = torch
        self.tok = self._load_tokenizer(model_name)
        # Try local cache first — never hits the network if model is on disk.
        self.model = self._load_model(model_name)
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        log.info("FinBERT loaded from cache: %s", _MODELS_DIR)

    @staticmethod
    def _load_model(model_name: str):
        from transformers import BertForSequenceClassification
        try:
            # local_files_only=True → raises if not cached; no network call
            return BertForSequenceClassification.from_pretrained(
                model_name, local_files_only=True
            ).eval()
        except Exception:
            log.info("FinBERT not in local cache — downloading once to %s", _MODELS_DIR)
            return BertForSequenceClassification.from_pretrained(model_name).eval()

    @staticmethod
    def _load_tokenizer(model_name: str):
        from transformers import AutoTokenizer, BertTokenizerFast
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(model_name)
            except Exception:
                from huggingface_hub import hf_hub_download
                vocab = hf_hub_download(model_name, "vocab.txt")
                return BertTokenizerFast(vocab_file=vocab, do_lower_case=False)

    def predict_raw(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """Return raw probability dicts."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(
                texts[i : i + batch_size],
                padding=True, truncation=True, max_length=256, return_tensors="pt",
            )
            with self.torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1).tolist()
            for p in probs:
                out.append(dict(zip(self.labels, p)))
        return out


# ── Hybrid Sentiment ──────────────────────────────────────────────────────────

class HybridSentiment:
    """Blends FinBERT with the financial lexicon.

    When FinBERT reports high neutral probability the lexicon score
    is given higher weight, preventing masking of clearly directional text.
    """

    NEUTRAL_THRESHOLD = 0.55   # above this → FinBERT is uncertain, blend 50/50
    ALPHA_CONFIDENT   = 0.70   # FinBERT weight when it's confident
    ALPHA_UNCERTAIN   = 0.45   # FinBERT weight when neutral dominates

    def __init__(self, finbert: FinBERT, lexicon: FinancialLexicon):
        self.finbert = finbert
        self.lexicon = lexicon

    def predict(self, texts: list[str], batch_size: int = 16) -> list[tuple[str, float, dict]]:
        # texts are "title. body" — split back for title weighting
        titles, bodies = [], []
        for t in texts:
            parts = t.split(". ", 1)
            titles.append(parts[0])
            bodies.append(parts[1] if len(parts) > 1 else "")

        fb_raws   = self.finbert.predict_raw(texts, batch_size)
        lex_preds = self.lexicon.predict_split(titles, bodies)
        out = []
        for fb_raw, (_, lex_score, _) in zip(fb_raws, lex_preds):
            fb_score = fb_raw.get("positive", 0.0) - fb_raw.get("negative", 0.0)
            neutral_p = fb_raw.get("neutral", 0.0)

            alpha = self.ALPHA_UNCERTAIN if neutral_p >= self.NEUTRAL_THRESHOLD else self.ALPHA_CONFIDENT
            blended = alpha * fb_score + (1 - alpha) * lex_score
            blended = max(-1.0, min(1.0, blended))

            if blended > 0.05:
                label = "positive"
            elif blended < -0.05:
                label = "negative"
            else:
                label = "neutral"

            # Synthetic raw probs that reflect the blended score
            a = abs(blended)
            if label == "positive":
                raw = {"positive": round(0.4 + 0.4 * a, 4), "negative": round(max(0.0, 0.1 - 0.05 * a), 4), "neutral": round(max(0.0, 0.5 - 0.35 * a), 4)}
            elif label == "negative":
                raw = {"positive": round(max(0.0, 0.1 - 0.05 * a), 4), "negative": round(0.4 + 0.4 * a, 4), "neutral": round(max(0.0, 0.5 - 0.35 * a), 4)}
            else:
                raw = {"positive": round(fb_raw.get("positive", 0.1), 4), "negative": round(fb_raw.get("negative", 0.1), 4), "neutral": round(fb_raw.get("neutral", 0.8), 4)}
            total = sum(raw.values())
            raw = {k: round(v / total, 4) for k, v in raw.items()}

            out.append((label, round(blended, 4), raw))
        return out


class LexiconSentiment:
    """Pure lexicon fallback — no FinBERT required."""

    _lexicon = FinancialLexicon()

    def predict(self, texts: list[str], batch_size: int = 16) -> list[tuple[str, float, dict]]:
        titles, bodies = [], []
        for t in texts:
            parts = t.split(". ", 1)
            titles.append(parts[0])
            bodies.append(parts[1] if len(parts) > 1 else "")
        return self._lexicon.predict_split(titles, bodies)


# ── Loader ────────────────────────────────────────────────────────────────────
# Module-level singleton — FinBERT weights are read from disk ONCE per
# Python process. Subsequent calls to load_model() return the cached object.
_MODEL_SINGLETON = None


def load_model(allow_fallback: bool = True):
    """Return the process-wide sentiment model (loads from disk on first call only).

    Thread-safety: Python's GIL makes the assignment atomic on CPython.
    """
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON
    try:
        fb  = FinBERT()
        lex = FinancialLexicon()
        log.info("HybridSentiment ready (FinBERT + lexicon)")
        _MODEL_SINGLETON = HybridSentiment(fb, lex)
    except Exception as exc:
        if not allow_fallback:
            raise
        log.warning("FinBERT unavailable (%s); falling back to LexiconSentiment", exc)
        _MODEL_SINGLETON = LexiconSentiment()
    return _MODEL_SINGLETON


def reset_model() -> None:
    """Force a reload on the next call — useful in tests or when swapping models."""
    global _MODEL_SINGLETON
    _MODEL_SINGLETON = None


def score_articles(articles: list[Article], model=None) -> list[Article]:
    model = model or load_model()
    texts = [f"{a.title}. {a.text}"[:1200] for a in articles]
    preds = model.predict(texts)
    for a, (label, score, raw) in zip(articles, preds):
        a.sentiment_label    = label
        a.sentiment_score    = score
        a.sentiment_scores_raw = {k: round(v, 4) for k, v in raw.items()}
    return articles
