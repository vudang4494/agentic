"""Reranker (RRK) — cross-encoder relevance gate.

Tier 1 of the verify funnel (per-section):
  1. Receive top-20 candidate docs from RSR (dense + sparse fused).
  2. Score each (query, doc) pair with bge-reranker-v2-m3 cross-encoder.
  3. Return top-8 docs sorted by rerank_score [0,1], optionally gate below RELEVANCE_FLOOR.

Why cross-encoder over cosine bi-encoder:
  - Bi-encoder encodes query and doc independently → cosine only measures topical overlap.
  - Cross-encoder jointly encodes (query, doc) → captures precise query-doc alignment.
  - Anisotropy in bi-encoder embeddings makes absolute cosine values unreliable.

Source: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa ~568M, text-classification).
Runs via direct transformers (FlagReranker has tokenizer API breakage on transformers>=5.0).

DO NOT use Ollama — it has no /api/rerank endpoint.

Usage:
    from research.rerank import rerank, RERANKER_MODEL, RELEVANCE_FLOOR, TOP_K_FINAL

IMPORTANT: This module uses direct transformers AutoTokenizer + AutoModelForSequenceClassification
because FlagReranker.compute_score() breaks on transformers>=5.x (prepare_for_model removed).
The correct scoring for bge-reranker-v2-m3 is: sigmoid(single_scalar_logit) → [0, 1].
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .types import Source

# ---- Knobs (override via research/__init__.py or env) ----
TOP_K_RETRIEVE = 20   # candidates before reranking
TOP_K_FINAL = 8       # kept after reranking
RELEVANCE_FLOOR = 0.25   # [CALIBRATE] cross-encoder is now a real relevance GATE, not rank-only.
# 0.25 = the docstring-prescribed start (lower tertile of bge-reranker score dist).
# Safety: rerank() never hard-starves a section -- if fewer than min_keep docs clear
# the floor, the top min_keep are rescued regardless, so an on-topic-but-low-scoring
# section still gets evidence. Tune via precision@k on a labeled (query, doc, relevant) set.

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cpu"   # Apple Silicon MPS can be tried if torch.backends.mps available
RERANKER_USE_FP16 = True  # ~half memory, minimal quality loss on M-series

# ---- Reranker instance (load once, reuse across sections) ----
_reranker = None
_tokenizer = None


def _get_reranker():
    """Lazy-load the reranker model. Call once per process."""
    global _reranker, _tokenizer
    if _reranker is not None:
        return _reranker, _tokenizer

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        raise ImportError(
            "transformers or torch not installed. Run: pip install transformers torch. "
            "See requirements.txt"
        )

    _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    _reranker = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        dtype=torch.float16 if RERANKER_USE_FP16 else torch.float32,
    )
    if RERANKER_DEVICE == "mps":
        try:
            _reranker = _reranker.to("mps")
        except Exception:
            _reranker = _reranker.to("cpu")
            print("[research/rerank] MPS unavailable, falling back to CPU", flush=True)
    _reranker.eval()
    return _reranker, _tokenizer


def _obj_to_text(doc) -> str:
    """Extract text from a doc (dict or Source/dataclass)."""
    if isinstance(doc, dict):
        return doc.get("text", doc.get("excerpt", ""))
    return getattr(doc, "text", None) or getattr(doc, "excerpt", "")


def _score_pairs(query: str, docs: list) -> list:
    """Score (query, doc_text) pairs using bge-reranker-v2-m3.

    The model has num_labels=1 → outputs a single scalar logit.
    Correct scoring: sigmoid(logit) → [0, 1].
    Accepts dicts or Source/dataclass objects.
    """
    model, tokenizer = _get_reranker()
    import torch

    texts = [_obj_to_text(d) for d in docs]
    pairs = [[query, t] for t in texts]

    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().tolist()

    return scores


def _is_source(obj) -> bool:
    """True if obj is a Source dataclass instance."""
    return isinstance(obj, Source) or (
        hasattr(obj, "__dataclass_fields__")
    )


def _attach_score(obj, score: float):
    """Attach rerank_score to a Source/dataclass or dict."""
    if isinstance(obj, dict):
        obj["rerank_score"] = float(score)
    else:
        setattr(obj, "rerank_score", float(score))


def rerank(query: str, docs: list, top_k: int = TOP_K_FINAL, min_keep: int = 3) -> list:
    """Score candidate docs by relevance to query using cross-encoder.

    Args:
        query: The section prompt / sub-query to score against.
        docs: List of Source objects or dicts with at least
              {"excerpt": str} or {"text": str}.
        top_k: Maximum docs to return. Default 8.

    Returns:
        List of Source objects (same type as input) sorted by rerank_score
        descending. Each Source gains a "rerank_score" attribute.
        Docs below RELEVANCE_FLOOR are dropped before top_k selection.
        Compatible with notes.enrich_top_sources() downstream.
    """
    if not docs:
        return []

    scores = _score_pairs(query, docs)

    # Pair with scores, sort descending
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Gate to RELEVANCE_FLOOR (cross-encoder as a real filter), cap at top_k.
    result = []
    for doc, score in scored:
        if score >= RELEVANCE_FLOOR:
            _attach_score(doc, score)
            result.append(doc)
            if len(result) >= top_k:
                break

    # Rescue: never hard-starve a section. If fewer than min_keep cleared the
    # floor, keep the top min_keep highest-scored docs regardless (identity-based
    # skip so a strict floor can't drop an on-topic-but-low-scoring section to 0).
    if len(result) < min_keep:
        for doc, score in scored[:min_keep]:
            if any(doc is r for r in result):
                continue
            _attach_score(doc, score)
            result.append(doc)
            if len(result) >= min_keep:
                break

    return result


def rerank_with_sources(query: str, sources: list, top_k: int = TOP_K_FINAL) -> list:
    """Convenience wrapper — rerank() now handles Source objects directly."""
    return rerank(query, sources, top_k=top_k)


# ---- Smoke test ----
if __name__ == "__main__":
    test_docs = [
        {"id": "1", "text": "Transformers use self-attention to process input sequences in parallel."},
        {"id": "2", "text": "A recipe for chocolate cake with cocoa powder and frosting."},
        {"id": "3", "text": "Attention mechanisms allow models to weigh the importance of different parts of the input."},
        {"id": "4", "text": "The weather forecast predicts rain for tomorrow afternoon."},
    ]
    query = "How do Transformers process sequences with attention?"

    print(f"TOP_K_RETRIEVE={TOP_K_RETRIEVE}, TOP_K_FINAL={TOP_K_FINAL}, RELEVANCE_FLOOR={RELEVANCE_FLOOR}")
    result = rerank(query, test_docs, top_k=3)
    for d in result:
        print(f"  score={d['rerank_score']:.4f}  id={d['id']}  text={d['text'][:60]}...")
