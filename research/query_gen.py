"""Query generation: section prompt -> list of search queries.

Uses Gemma 4 12B for the research layer.
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
from .config import QUERY_GEN_MODEL

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = QUERY_GEN_MODEL
DEFAULT_TIMEOUT = 90.0

QUERY_GEN_SYS = (
    "You are a research assistant. Given a section prompt, output a compact JSON array "
    "of 3 to 5 search queries optimized for academic search (arxiv) and Wikipedia. "
    "Each item is an object: {\"q\": \"...\", \"intent\": \"primary|supporting|definition|canonical\"}. "
    "Each query must be <= 12 words, specific, and avoid generic terms. "
    "Output ONLY the JSON array. No prose, no markdown fences, no explanation."
)
QUERY_DECOMPOSE_SYS = (
    "You are a research planner. Given a section title and prompt, decompose it into "
    "2-3 ATOMIC sub-topics or aspects. An atomic sub-topic is a single specific concept, "
    "method, or fact that can be looked up independently. "
    "Return a JSON array of strings: [sub-topic A, sub-topic B, sub-topic C]. "
    "Each sub-topic must be <= 8 words. "
    "Output ONLY the JSON array. No prose, no markdown fences."
)
_JSON_STR_ARRAY_RE = re.compile(r'\[\s*(?:"[^"]+"\s*,?\s*)+\]', re.DOTALL)



_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{[^\[\]]*\}\s*,?\s*)+\]", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()



def _parse_subtopics(raw: str) -> List[str]:
    """Parse a JSON array of strings from decomposition output."""
    raw = _strip_think(raw)
    m = _JSON_STR_ARRAY_RE.search(raw)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return [str(s).strip()[:80] for s in data if s][:3]

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



def _decompose_and_query(section_prompt, chapter_title, section_title, model, client_base, timeout):
    """Two-stage: decompose into atomic sub-topics, then generate queries."""
    decompose_prompt = f"Section: {section_title}\nPrompt: {section_prompt[:300]}"
    subtopics = []
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{client_base}/api/chat", json={
                "model": model, "stream": False,
                "messages": [
                    {"role": "system", "content": QUERY_DECOMPOSE_SYS},
                    {"role": "user", "content": decompose_prompt},
                ],
                "options": {"temperature": 0.2, "num_predict": 200, "top_p": 0.9},
                "think": False,
            })
            r.raise_for_status()
        raw = _strip_think((r.json().get("message") or {}).get("content", ""))
        subtopics = _parse_subtopics(raw)
    except Exception:
        subtopics = []
    extra_queries = [Query(q=st, intent="atomic") for st in subtopics] if subtopics else []
    if subtopics:
        print(f"[query_gen] decomposed: {subtopics}", flush=True)
    user_prompt = f"Chapter: {chapter_title}\nSection: {section_title}\nSection prompt:\n{section_prompt}\n\nReturn the JSON array now."
    try:
        with httpx.Client(timeout=timeout) as c:
            t0 = time.time()
            r = c.post(f"{client_base}/api/chat", json={
                "model": model, "stream": False,
                "messages": [
                    {"role": "system", "content": QUERY_GEN_SYS},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": 0.3, "num_predict": 400, "top_p": 0.9},
                "think": False,
            })
            r.raise_for_status()
            data = r.json()
        raw = (data.get("message") or {}).get("content", "")
        print(f"[query_gen] {model} -> {len(raw)} chars in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[query_gen] WARN: model call failed ({e}); using fallback", flush=True)
        return _fallback_queries(section_title, chapter_title)
    parsed = _parse_queries(raw)
    if not parsed:
        print(f"[query_gen] WARN: could not parse JSON; using fallback", flush=True)
        return _fallback_queries(section_title, chapter_title)
    return extra_queries + parsed

def queries_for(section_prompt, chapter_title, section_title,
                client_base=OLLAMA_BASE, model=DEFAULT_MODEL,
                reviewer_hint=None, timeout=DEFAULT_TIMEOUT,
                prior_query_sigs=None,
                # When provided, skip archetype routing and use LLM fallback directly.
                domain_context=None):
    """Generate 3-5 search queries using understanding-based semantic routing.

    Delegates to query_router.queries_for() which uses:
      1. bge-m3 embedding routing to archetype query templates (0 LLM)
      2. LLM fallback (1 call) for novel/unmatched sections

    Kept for backward compatibility with callers that import query_gen.queries_for().
    """
    from . import query_router as _qr
    return _qr.queries_for(
        section_prompt, chapter_title, section_title,
        client_base=client_base, model=model,
        reviewer_hint=reviewer_hint, timeout=timeout,
        prior_query_sigs=prior_query_sigs,
        domain_context=domain_context,
    )
