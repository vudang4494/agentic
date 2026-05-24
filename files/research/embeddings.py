"""bge-m3 embeddings via Ollama /api/embed, used by notes.rank()."""
import math
from typing import List

import httpx

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "bge-m3:latest"
TIMEOUT = 60.0


def embed(texts: List[str], model: str = DEFAULT_MODEL) -> List[List[float]]:
    """Batch-embed a list of texts. Returns one vector per input, preserving order.

    Returns an empty list on any failure -- callers must handle the empty case
    (typically by falling back to lexical ranking).
    """
    if not texts:
        return []
    payload = {"model": model, "input": texts}
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post(f"{OLLAMA_BASE}/api/embed", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"[research/embeddings] WARN: embed call failed: {e}", flush=True)
        return []
    vectors = data.get("embeddings") or []
    # Newer Ollama returns {"embeddings": [[...], [...]]}; older may return {"embedding": [...]} for single.
    if isinstance(vectors, list) and vectors and not isinstance(vectors[0], list):
        vectors = [vectors]
    if len(vectors) != len(texts):
        print(f"[research/embeddings] WARN: got {len(vectors)} vectors for {len(texts)} inputs", flush=True)
    return vectors


def cosine(a: List[float], b: List[float]) -> float:
    """Plain-Python cosine similarity. Returns 0.0 if either vector is empty/degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
