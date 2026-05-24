"""Query generation: section prompt -> list of search queries.

Uses qwen3.5:4b by default (better JSON compliance than gemma3:4b in practice).
Strips Qwen3 thinking-mode <think>...</think> blocks defensively. Falls back to
a deterministic template if the model can't produce parseable JSON.
"""
import json
import re
import threading
import time
from typing import List, Optional

import httpx

from .types import Query

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_TIMEOUT = 90.0

QUERY_GEN_SYS = (
    "You are a research assistant. Given a section prompt, output a compact JSON array "
    "of 3 to 5 search queries optimized for academic search (arxiv) and Wikipedia. "
    "Each item is an object: {\"q\": \"...\", \"intent\": \"primary|supporting|definition|canonical\"}. "
    "Each query must be <= 12 words, specific, and avoid generic terms. "
    "Output ONLY the JSON array. No prose, no markdown fences, no explanation."
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{[^\[\]]*\}\s*,?\s*)+\]", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _parse_queries(raw: str, max_n: int = 5) -> List[Query]:
    raw = _strip_think(raw)
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        # Sometimes the model emits a single object instead of an array.
        single = re.search(r"\{[^{}]*\"q\"[^{}]*\}", raw)
        if single:
            try:
                obj = json.loads(single.group(0))
                if isinstance(obj, dict) and obj.get("q"):
                    return [Query(q=str(obj["q"]).strip()[:120],
                                  intent=str(obj.get("intent", ""))[:40])]
            except Exception:
                pass
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out: List[Query] = []
    for item in data[:max_n]:
        if isinstance(item, dict):
            q = str(item.get("q", "")).strip()[:120]
            intent = str(item.get("intent", "")).strip()[:40]
        elif isinstance(item, str):
            q, intent = item.strip()[:120], ""
        else:
            continue
        if q:
            out.append(Query(q=q, intent=intent))
    return out


def _fallback_queries(section_title: str, chapter_title: str) -> List[Query]:
    """Deterministic fallback when the model can't produce parseable JSON.

    These won't be as good as model-generated queries but ensure the pipeline
    still has SOMETHING to retrieve.
    """
    title = (section_title or chapter_title or "").strip()
    if not title:
        return []
    return [
        Query(q=f"{title} survey",      intent="primary"),
        Query(q=f"{title} 2024",        intent="recent"),
        Query(q=f"{title} introduction", intent="definition"),
    ]


_LOCK = threading.Lock()


def queries_for(section_prompt: str, chapter_title: str, section_title: str,
                client_base: str = OLLAMA_BASE, model: str = DEFAULT_MODEL,
                reviewer_hint: Optional[str] = None,
                timeout: float = DEFAULT_TIMEOUT) -> List[Query]:
    """Generate 3-5 search queries for a section. Always returns at least the fallback list."""
    user_prompt = (
        f"Chapter: {chapter_title}\n"
        f"Section: {section_title}\n"
        f"Section prompt:\n{section_prompt}\n\n"
    )
    if reviewer_hint:
        user_prompt += (
            f"REVIEWER FEEDBACK on the previous draft: {reviewer_hint}\n"
            "Bias queries toward addressing this gap.\n\n"
        )
    user_prompt += "Return the JSON array now."

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": QUERY_GEN_SYS},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {"temperature": 0.3, "num_predict": 400, "top_p": 0.9},
        "think": False,  # Qwen3 thinking-mode off (newer Ollama) -- silently ignored on older Ollama
    }

    raw = ""
    try:
        with _LOCK, httpx.Client(timeout=timeout) as c:
            t0 = time.time()
            r = c.post(f"{client_base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        raw = (data.get("message") or {}).get("content", "")
        print(f"[research/query_gen] {model} -> {len(raw)} chars in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[research/query_gen] WARN: model call failed ({e}); using fallback", flush=True)
        return _fallback_queries(section_title, chapter_title)

    parsed = _parse_queries(raw)
    if not parsed:
        print(f"[research/query_gen] WARN: could not parse JSON from model output; using fallback", flush=True)
        return _fallback_queries(section_title, chapter_title)
    return parsed
