"""Step 4: Three-model ensemble event classifier.

Architecture
============

Layer 1 – NLI Zero-Shot  (``DeBERTaZeroShotClassifier``)
---------------------------------------------------------
Model : ``MoritzLaurer/deberta-v3-small-zeroshot-v1``  (~180 MB)
         Fine-tuned DeBERTa-v3-small on NLI tasks, excellent zero-shot
         performance without requiring the 1.6 GB of bart-large-mnli.

Strategy: **multi-hypothesis aggregation**.
Each category is represented by 3 natural-language hypotheses.  The pipeline
scores every (article, hypothesis) pair and the entailment probability is
averaged across all hypotheses for a category before the argmax is taken.
This prevents a single misleading phrase from dominating.

Layer 2 – Sentence Embedding Similarity  (``EmbeddingEventClassifier``)
------------------------------------------------------------------------
Model : ``sentence-transformers/all-MiniLM-L6-v2``  (~90 MB)
         Loaded via raw transformers + mean-pooling (no separate library
         needed).

At **init time** the model encodes a set of reference sentences for each
category and stores the L2-normalised centroid embedding.

At **inference time** the article text is encoded in one forward pass and
compared to every centroid via cosine similarity.  Inference is ~5 ms/article
after init.

Layer 3 – Enhanced Keyword Classifier  (``EnhancedKeywordClassifier``)
-----------------------------------------------------------------------
No model download.  Improvements over the original:
• Bigrams and trigrams (e.g. "chapter 11", "rate cut")
• IDF-style weighting: rare/specific terms score higher than generic ones
• Per-category keyword precision scores (avoids "deal" matching everything)

Ensemble  (``EnsembleEventClassifier``)
----------------------------------------
Soft-votes: NLI × 0.45  +  embedding × 0.35  +  keyword × 0.20

The final confidence reflects both the winner's margin over the runner-up
and the cross-model agreement (all three agreeing → boosted confidence).

Public API
==========
  load_event_classifier()   → EnsembleEventClassifier
  classify_event(text, clf) → (event_type: str, confidence: float)
  reset_classifier()        → force reload
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ── Cache directory (mirrors sentiment.py) ────────────────────────────────────
_HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_HERE, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME",            _MODELS_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_MODELS_DIR, "hub"))

# Model name overrides via env vars
NLI_MODEL       = os.environ.get("EVENT_NLI_MODEL",   "MoritzLaurer/deberta-v3-small-zeroshot-v1")
EMBED_MODEL     = os.environ.get("EVENT_EMBED_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
CONF_THRESHOLD  = float(os.environ.get("EVENT_CONF_THRESHOLD", "0.30"))

# ── Canonical categories ──────────────────────────────────────────────────────
EVENT_TYPES = [
    "credit_event",
    "mergers_acquisitions",
    "geopolitical",
    "macroeconomic",
    "earnings",
    "product_launch",
    "other",
]

# ── Multi-hypothesis table for NLI ────────────────────────────────────────────
# 3 hypotheses per category – each phrased differently to cover varied wording
_NLI_HYPOTHESES: dict[str, list[str]] = {
    "credit_event": [
        "This text is about a company going bankrupt or defaulting on its debt.",
        "This article discusses a credit rating downgrade, insolvency or debt restructuring.",
        "This news covers missed bond payments, Chapter 11 filing or financial distress.",
    ],
    "mergers_acquisitions": [
        "This text is about a company being acquired, merging, or completing a corporate takeover.",
        "This article discusses a buyout, acquisition deal, or strategic merger announcement.",
        "This news covers a company bidding to purchase or spin off another business.",
    ],
    "geopolitical": [
        "This text is about a geopolitical conflict, war, international sanctions or political crisis.",
        "This article discusses military actions, trade embargo, diplomatic tensions or election results.",
        "This news covers international sanctions, missile strikes, invasion or ceasefire negotiations.",
    ],
    "macroeconomic": [
        "This text is about central bank interest rate decisions, inflation data or GDP growth.",
        "This article discusses unemployment figures, Federal Reserve policy or economic recession.",
        "This news covers CPI data, payroll reports, economic slowdown or monetary policy changes.",
    ],
    "earnings": [
        "This text is about a company reporting its quarterly financial results or earnings.",
        "This article discusses revenue figures, profit guidance, EPS beats or misses.",
        "This news covers a company's annual results, sales performance or updated financial outlook.",
    ],
    "product_launch": [
        "This text is about a company announcing or launching a new product or technology.",
        "This article discusses an innovation reveal, product unveiling or commercial rollout.",
        "This news covers a new service introduction, platform release or feature announcement.",
    ],
}

# ── Reference sentences for embedding similarity ──────────────────────────────
# 5–6 diverse examples per category give robust centroid embeddings
_EMBED_REFERENCES: dict[str, list[str]] = {
    "credit_event": [
        "The company filed for Chapter 11 bankruptcy protection after missing debt payments.",
        "Moody's downgraded the firm's credit rating to junk status amid rising default risk.",
        "The retailer has entered insolvency proceedings following months of financial distress.",
        "Creditors agreed to a debt restructuring plan to avoid a formal default.",
        "The bond issuer failed to make its scheduled coupon payment to investors.",
    ],
    "mergers_acquisitions": [
        "Microsoft announced a $69 billion deal to acquire Activision Blizzard.",
        "The pharmaceutical companies completed their merger to form the world's largest drugmaker.",
        "Amazon agreed to buy the grocery chain in a landmark retail acquisition.",
        "The board approved the hostile takeover bid from the rival conglomerate.",
        "The firm will spin off its consumer division into a separately listed company.",
    ],
    "geopolitical": [
        "Russia launched missile strikes on Ukrainian cities as fighting intensified.",
        "The United States imposed sweeping sanctions on Russian energy companies.",
        "Tensions escalated at the border as both nations mobilised troops.",
        "The trade war deepened after the administration imposed 25% tariffs on imports.",
        "World leaders met for emergency talks following the outbreak of armed conflict.",
    ],
    "macroeconomic": [
        "The Federal Reserve raised interest rates by 50 basis points to combat inflation.",
        "US CPI data came in hotter than expected at 4.2%, above the 4.0% consensus.",
        "GDP growth slowed to 1.1% in the fourth quarter, below forecasts.",
        "The unemployment rate fell to 3.5%, the lowest level in 50 years.",
        "The European Central Bank signalled further rate hikes to tame inflation.",
    ],
    "earnings": [
        "Apple reported quarterly earnings of $1.52 per share, beating estimates of $1.43.",
        "The retailer missed revenue expectations, posting $4.2 billion versus $4.8 billion expected.",
        "Management raised full-year guidance after a strong first quarter performance.",
        "The bank's net income fell 12% year-over-year, missing analyst forecasts.",
        "The company reported record quarterly revenue driven by strong cloud growth.",
    ],
    "product_launch": [
        "Apple unveiled its new iPhone 16 lineup with improved AI features.",
        "Tesla launched a new entry-level electric vehicle targeting mass-market consumers.",
        "The startup introduced its AI-powered platform at the annual technology conference.",
        "Google announced a major update to its search engine incorporating generative AI.",
        "The pharmaceutical company received FDA approval for its new cancer treatment.",
    ],
}


_MODEL_ALIASES: dict[str, str] = {
    # Moritz Laurer's DeBERTa-v3 small zero-shot model on Hugging Face
    "MoritzLaurer/deberta-v3-small-zeroshot-v1": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
}


def _resolve_model_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 – NLI Zero-Shot Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class DeBERTaZeroShotClassifier:
    """Multi-hypothesis NLI zero-shot classifier using DeBERTa-v3."""

    def __init__(self, model_name: str = NLI_MODEL):
        from transformers import pipeline as hf_pipeline
        resolved = _resolve_model_name(model_name)
        try:
            self._pipe = hf_pipeline(
                "zero-shot-classification",
                model=resolved,
                local_files_only=True,
            )
        except Exception:
            log.info("NLI model not in local cache — downloading %s to %s", resolved, _MODELS_DIR)
            self._pipe = hf_pipeline("zero-shot-classification", model=resolved)
        log.info("DeBERTaZeroShotClassifier loaded: %s (resolved=%s)", model_name, resolved)

    def scores(self, text: str) -> dict[str, float]:
        """Return {category: aggregated_entailment_score} for all categories."""
        all_hyps: list[str] = []
        category_idx: dict[str, list[int]] = {}
        for cat, hyps in _NLI_HYPOTHESES.items():
            category_idx[cat] = list(range(len(all_hyps), len(all_hyps) + len(hyps)))
            all_hyps.extend(hyps)

        result = self._pipe(text[:512], candidate_labels=all_hyps, hypothesis_template="{}", multi_label=True)
        label_to_score = dict(zip(result["labels"], result["scores"]))

        agg: dict[str, float] = {}
        for cat, idxs in category_idx.items():
            hyps = _NLI_HYPOTHESES[cat]
            agg[cat] = sum(label_to_score.get(h, 0.0) for h in hyps) / len(hyps)

        total = sum(agg.values()) or 1.0
        return {c: round(v / total, 6) for c, v in agg.items()}

    def classify(self, text: str) -> tuple[str, float]:
        """Classify event directly using DeBERTa zero-shot NLI.

        Returns (event_type, confidence in [0, 1]).
        """
        scores = self.scores(text)
        sorted_cats = sorted(scores, key=scores.get, reverse=True)
        winner = sorted_cats[0]
        winner_sc = scores[winner]
        if winner_sc < CONF_THRESHOLD:
            return "other", round(winner_sc, 4)
        runner_sc = scores[sorted_cats[1]] if len(sorted_cats) > 1 else 0.0
        margin = winner_sc - runner_sc
        confidence = round(min(1.0, winner_sc + margin * 0.5), 4)
        return winner, confidence

    def classify_detailed(self, text: str) -> dict:
        """Return full score breakdown for API use."""
        scores = self.scores(text)
        sorted_cats = sorted(scores, key=scores.get, reverse=True)
        winner, confidence = self.classify(text)
        return {
            "event_type": winner,
            "confidence": confidence,
            "scores": {cat: {"nli": round(scores[cat], 4)} for cat in sorted_cats},
            "model_votes": {"nli": winner},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 – Embedding Similarity Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingEventClassifier:
    """Sentence embedding + cosine similarity to pre-computed category centroids."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from transformers import AutoTokenizer, AutoModel

        try:
            self._tok   = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self._model = AutoModel.from_pretrained(model_name, local_files_only=True).eval()
        except Exception:
            log.info("Embedding model not in local cache — downloading to %s", _MODELS_DIR)
            self._tok   = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).eval()

        log.info("EmbeddingEventClassifier loaded: %s", model_name)
        # Pre-compute and store category centroids
        self._centroids: dict[str, torch.Tensor] = {}
        for cat, refs in _EMBED_REFERENCES.items():
            embs = self._encode(refs)          # (n_refs, dim)
            self._centroids[cat] = embs.mean(dim=0)   # (dim,)
        log.info("EmbeddingEventClassifier: category centroids ready")

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        enc = self._tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        out = self._model(**enc)
        # Mean pooling
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)

    def scores(self, text: str) -> dict[str, float]:
        """Return {category: cosine_similarity} — values in [0, 1] after relu."""
        emb = self._encode([text[:256]])[0]    # (dim,)
        sims: dict[str, float] = {}
        for cat, centroid in self._centroids.items():
            sims[cat] = float(F.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0)))

        # Shift and normalise (cosine can be negative for very dissimilar text)
        min_s = min(sims.values())
        shifted = {c: max(0.0, v - min_s) for c, v in sims.items()}
        total = sum(shifted.values()) or 1.0
        return {c: round(v / total, 6) for c, v in shifted.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 – Enhanced Keyword Classifier
# ═══════════════════════════════════════════════════════════════════════════════

# (term, weight) — higher weight = more discriminative signal
_KEYWORD_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "credit_event": [
        ("bankruptcy", 3.0), ("bankrupt", 3.0), ("chapter 11", 3.0), ("chapter11", 3.0),
        ("default", 2.5), ("defaulted", 2.5), ("insolvency", 2.5), ("insolvent", 2.5),
        ("debt restructuring", 2.5), ("credit downgrade", 2.5), ("downgrade", 1.5),
        ("missed payment", 2.0), ("bond default", 2.5), ("junk status", 2.0),
        ("creditor", 1.5), ("coupon payment", 1.5), ("distressed debt", 2.0),
    ],
    "mergers_acquisitions": [
        ("merger", 2.5), ("acquisition", 2.5), ("acquire", 2.0), ("acquires", 2.0),
        ("takeover", 2.5), ("buyout", 2.5), ("spin-off", 2.0), ("spin off", 2.0),
        ("due diligence", 2.0), ("hostile bid", 2.5), ("tender offer", 2.5),
        ("strategic deal", 2.0), ("to buy", 1.5), ("agreed to purchase", 2.0),
        ("all-stock deal", 2.0), ("billion deal", 1.5), ("corporate deal", 1.5),
    ],
    "geopolitical": [
        ("war", 2.0), ("invasion", 2.5), ("missile", 2.5), ("military", 2.0),
        ("sanctions", 2.5), ("sanction", 2.0), ("embargo", 2.5), ("tariff", 2.0),
        ("tariffs", 2.0), ("ceasefire", 2.5), ("conflict", 1.5), ("geopolitical", 2.5),
        ("diplomatic", 2.0), ("election", 1.5), ("troops", 2.0), ("nato", 2.0),
        ("trade war", 2.5), ("political crisis", 2.5),
    ],
    "macroeconomic": [
        ("federal reserve", 2.5), ("fed", 1.5), ("interest rate", 2.5), ("rate hike", 2.5),
        ("rate cut", 2.5), ("inflation", 2.5), ("cpi", 2.5), ("gdp", 2.5),
        ("recession", 2.5), ("unemployment", 2.5), ("payrolls", 2.5), ("jobs report", 2.5),
        ("central bank", 2.5), ("ecb", 2.5), ("monetary policy", 2.5), ("basis points", 2.0),
        ("economic slowdown", 2.5), ("stagflation", 2.5), ("consumer prices", 2.0),
    ],
    "earnings": [
        ("earnings", 2.5), ("quarterly results", 2.5), ("eps", 2.5), ("revenue", 2.0),
        ("profit", 1.5), ("net income", 2.5), ("operating income", 2.0), ("guidance", 2.0),
        ("beats estimates", 2.5), ("missed estimates", 2.5), ("above consensus", 2.5),
        ("full-year outlook", 2.0), ("raised guidance", 2.0), ("per share", 2.0),
        ("quarterly profit", 2.5), ("fiscal quarter", 2.0),
    ],
    "product_launch": [
        ("launches", 2.0), ("launched", 2.0), ("unveils", 2.5), ("unveiled", 2.5),
        ("introduces", 2.0), ("new product", 2.5), ("rollout", 2.0), ("fda approval", 2.5),
        ("approved by", 1.5), ("innovation", 1.5), ("new platform", 2.0),
        ("new feature", 1.5), ("technology announcement", 2.0), ("product reveal", 2.5),
    ],
}

# Precompile patterns at import time
_COMPILED: dict[str, list[tuple[re.Pattern, float]]] = {
    cat: [(re.compile(rf"\b{re.escape(term)}\b", re.I), w) for term, w in terms]
    for cat, terms in _KEYWORD_WEIGHTS.items()
}


class EnhancedKeywordClassifier:
    """Weighted keyword/bigram classifier with IDF-style term specificity."""

    def scores(self, text: str) -> dict[str, float]:
        low = text.lower()
        raw: dict[str, float] = {}
        for cat, patterns in _COMPILED.items():
            raw[cat] = sum(w for pat, w in patterns if pat.search(low))
        total = sum(raw.values()) or 1.0
        return {c: round(v / total, 6) for c, v in raw.items()}

    def classify(self, text: str) -> tuple[str, float]:
        sc = self.scores(text)
        best = max(sc, key=sc.get)
        if sc[best] == 0.0:
            return "other", 0.0
        # Confidence: margin between top-2, capped at 0.60
        sorted_vals = sorted(sc.values(), reverse=True)
        margin = sorted_vals[0] - (sorted_vals[1] if len(sorted_vals) > 1 else 0.0)
        conf = round(min(0.60, margin * 2), 4)
        return best, conf


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

_WEIGHTS = {"nli": 0.45, "embed": 0.35, "keyword": 0.20}


class EnsembleEventClassifier:
    """Weighted soft-vote ensemble of NLI + embedding + keyword classifiers."""

    def __init__(
        self,
        nli_clf: DeBERTaZeroShotClassifier,
        emb_clf: EmbeddingEventClassifier,
        kw_clf:  EnhancedKeywordClassifier,
    ):
        self._nli = nli_clf
        self._emb = emb_clf
        self._kw  = kw_clf

    def classify(self, text: str) -> tuple[str, float]:
        """Return (event_type, confidence).

        Confidence reflects both the winning category's margin and cross-model
        agreement (all three pointing to the same label → bonus).
        """
        nli_sc  = self._nli.scores(text)
        emb_sc  = self._emb.scores(text)
        kw_sc   = self._kw.scores(text)

        # Ensure all dicts have the same keys
        cats = list(_NLI_HYPOTHESES.keys())
        ensemble: dict[str, float] = {}
        for cat in cats:
            ensemble[cat] = (
                _WEIGHTS["nli"]     * nli_sc.get(cat, 0.0)
                + _WEIGHTS["embed"] * emb_sc.get(cat, 0.0)
                + _WEIGHTS["keyword"] * kw_sc.get(cat, 0.0)
            )

        # Winner and runner-up
        sorted_cats = sorted(ensemble, key=ensemble.get, reverse=True)
        winner      = sorted_cats[0]
        winner_sc   = ensemble[winner]
        runner_sc   = ensemble[sorted_cats[1]] if len(sorted_cats) > 1 else 0.0
        margin      = winner_sc - runner_sc

        # Below confidence threshold → "other"
        if winner_sc < CONF_THRESHOLD:
            return "other", round(winner_sc, 4)

        # Cross-model agreement bonus
        nli_winner  = max(nli_sc,  key=nli_sc.get)
        emb_winner  = max(emb_sc,  key=emb_sc.get)
        kw_winner   = max(kw_sc,   key=kw_sc.get)
        agree_count = sum([nli_winner == winner, emb_winner == winner, kw_winner == winner])
        agree_bonus = (agree_count - 1) * 0.05   # +0.05 per extra model agreeing

        confidence = round(min(1.0, winner_sc + margin * 0.5 + agree_bonus), 4)
        return winner, confidence

    def classify_detailed(self, text: str) -> dict:
        """Return full breakdown for debugging/API use."""
        nli_sc = self._nli.scores(text)
        emb_sc = self._emb.scores(text)
        kw_sc  = self._kw.scores(text)
        cats   = list(_NLI_HYPOTHESES.keys())

        ensemble: dict[str, float] = {}
        for cat in cats:
            ensemble[cat] = (
                _WEIGHTS["nli"]       * nli_sc.get(cat, 0.0)
                + _WEIGHTS["embed"]   * emb_sc.get(cat, 0.0)
                + _WEIGHTS["keyword"] * kw_sc.get(cat, 0.0)
            )

        sorted_cats = sorted(ensemble, key=ensemble.get, reverse=True)
        winner = sorted_cats[0]
        event_type, confidence = self.classify(text)

        return {
            "event_type":  event_type,
            "confidence":  confidence,
            "scores": {
                cat: {
                    "ensemble": round(ensemble[cat], 4),
                    "nli":      round(nli_sc.get(cat, 0.0), 4),
                    "embedding": round(emb_sc.get(cat, 0.0), 4),
                    "keyword":  round(kw_sc.get(cat, 0.0), 4),
                }
                for cat in sorted_cats
            },
            "model_votes": {
                "nli":     max(nli_sc, key=nli_sc.get),
                "embedding": max(emb_sc, key=emb_sc.get),
                "keyword": max(kw_sc, key=kw_sc.get),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Loader  (singleton pattern — models loaded once per process)
# ═══════════════════════════════════════════════════════════════════════════════

_CLASSIFIER_SINGLETON: DeBERTaZeroShotClassifier | EnsembleEventClassifier | EnhancedKeywordClassifier | None = None


def load_event_classifier(allow_fallback: bool = True):
    """Return the process-wide event classifier (loads models on first call only)."""
    global _CLASSIFIER_SINGLETON
    if _CLASSIFIER_SINGLETON is not None:
        return _CLASSIFIER_SINGLETON
    mode = os.environ.get("EVENT_CLASSIFIER_MODE", "nli").lower()
    try:
        if mode == "ensemble":
            nli = DeBERTaZeroShotClassifier()
            emb = EmbeddingEventClassifier()
            kw  = EnhancedKeywordClassifier()
            _CLASSIFIER_SINGLETON = EnsembleEventClassifier(nli, emb, kw)
            log.info("EnsembleEventClassifier ready (NLI + embedding + keyword)")
        else:
            _CLASSIFIER_SINGLETON = DeBERTaZeroShotClassifier()
            log.info("DeBERTaZeroShotClassifier ready (MoritzLaurer/deberta-v3 NLI zero-shot)")
    except Exception as exc:
        if not allow_fallback:
            raise
        log.warning("Primary event classifier unavailable (%s); using keyword fallback", exc)
        _CLASSIFIER_SINGLETON = EnhancedKeywordClassifier()
    return _CLASSIFIER_SINGLETON


# Backward-compatible aliases
ZeroShotEventClassifier = DeBERTaZeroShotClassifier
KeywordEventClassifier = EnhancedKeywordClassifier


def reset_classifier() -> None:
    """Force reload on next call — useful in tests or when switching models."""
    global _CLASSIFIER_SINGLETON
    _CLASSIFIER_SINGLETON = None


# ── Public classify_event function ────────────────────────────────────────────

def classify_event(text: str, classifier=None) -> tuple[str, float]:
    """Classify event type and return (event_type, confidence ∈ [0,1]).

    Parameters
    ----------
    text       : article title + body text
    classifier : optional pre-loaded classifier instance (bypasses singleton)
    """
    clf = classifier or load_event_classifier()
    return clf.classify(text)
