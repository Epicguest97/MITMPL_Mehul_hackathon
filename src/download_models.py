#!/usr/bin/env python
"""
Download FinBERT, DeBERTa NLI event classifier, and MiniLM sentence embedding
models to the local ./models/ directory exactly once.  After running this script,
the server never needs a network connection to load sentiment or event models.

Usage
-----
    source venv/bin/activate
    python src/download_models.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_HERE, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = _MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_MODELS_DIR, "hub")

FINBERT_MODEL = os.environ.get("FINBERT_MODEL", "yiyanghkust/finbert-tone")
NLI_MODEL = os.environ.get("EVENT_NLI_MODEL", "MoritzLaurer/deberta-v3-small-zeroshot-v1")
EMBED_MODEL = os.environ.get("EVENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


_MODEL_ALIASES: dict[str, str] = {
    "MoritzLaurer/deberta-v3-small-zeroshot-v1": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
}


def _resolve_model_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


def download_pipeline(model_name: str, task: str) -> None:
    resolved = _resolve_model_name(model_name)
    print(f"\n{'='*60}")
    print(f"Downloading pipeline: {model_name} -> {resolved}  (task={task})")
    print(f"Destination: {_MODELS_DIR}")
    print(f"{'='*60}")
    from transformers import pipeline as hf_pipeline
    _ = hf_pipeline(task, model=resolved)
    print(f"✓  {model_name} cached successfully")


def download_automodel(model_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Downloading AutoModel: {model_name}")
    print(f"Destination: {_MODELS_DIR}")
    print(f"{'='*60}")
    from transformers import AutoTokenizer, AutoModel
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModel.from_pretrained(model_name)
    print(f"✓  {model_name} cached successfully")


if __name__ == "__main__":
    try:
        download_pipeline(FINBERT_MODEL, "text-classification")
        download_pipeline(NLI_MODEL, "zero-shot-classification")
        download_automodel(EMBED_MODEL)
        print(f"\n✓  All 3 models cached in {_MODELS_DIR}")
        print("   The server will now start offline without any network access.\n")
    except Exception as e:
        print(f"\n✗  Download failed: {e}", file=sys.stderr)
        sys.exit(1)
