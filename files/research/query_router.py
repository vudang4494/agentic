"""Semantic query routing: understand topic → return structured queries.

Stage A (understanding over prompting):
  - Tier 1: Canonical seeds (exact match)     → direct arxiv ID fetch (~0ms)
  - Tier 2: Semantic archetype routing        → cosine match → archetype queries (~1s)
  - Tier 3: LLM fallback (1 call, not 2)     → for novel/unmatched sections (~5s)

No LLM calls for routing decisions — bge-m3 handles topic understanding.
"""
import json
import math
import re
import threading
import time
from typing import List, Optional

import httpx

from .embeddings import embed as _embed, cosine as _cosine
from .types import Query
from .config import QUERY_GEN_MODEL

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = QUERY_GEN_MODEL
DEFAULT_TIMEOUT = 90.0

# ---- Semantic archetype registry ---------------------------------------------
# Each archetype maps a topic category to domain-appropriate search queries.
# Embeddings of `description` are precomputed at startup and cosine-matched
# against the section's embedding to select the right query set.
#
# Archetypes are COVERED enough for the LLM/Agentic 2026 topic:
#   Transformer, RLHF/DPO, Agentic AI, Tool Use, Reasoning/CoT,
#   RAG/Retrieval, Fine-tuning, Evaluation, Multimodal, Emerging.
#
# Adding a new topic: add an archetype with a distinctive description + queries.

_ARCHETYPES = [
    {
        "name": "transformer",
        "description": "Transformer architecture, self-attention mechanism, multi-head attention, positional encoding, feed-forward networks, layer normalization",
        "queries": [
            Query(q="Transformer architecture self-attention Vaswani 2017", intent="canonical"),
            Query(q="multi-head attention mechanism transformer", intent="primary"),
            Query(q="positional encoding transformer rotary RoPE", intent="supporting"),
        ],
    },
    {
        "name": "rlhf_dpo",
        "description": "Reinforcement learning from human feedback, RLHF, DPO direct preference optimization, reward model, alignment, PPO training",
        "queries": [
            Query(q="RLHF reinforcement learning human feedback InstructGPT", intent="canonical"),
            Query(q="DPO direct preference optimization language models", intent="canonical"),
            Query(q="reward model alignment language models 2023", intent="supporting"),
        ],
    },
    {
        "name": "agentic_ai",
        "description": "Agentic AI systems, autonomous agents, agentic workflows, AI agents 2025 2026, multi-agent systems, agent frameworks, reasoning agents",
        "queries": [
            Query(q="agentic AI autonomous agents 2025 2026 survey", intent="primary"),
            Query(q="AI agent frameworks architecture 2026", intent="primary"),
            Query(q="multi-agent systems LLM cooperation", intent="supporting"),
            Query(q="autonomous agentic reasoning benchmark", intent="supporting"),
        ],
    },
    {
        "name": "tool_use",
        "description": "Tool use in language models, function calling, tool-augmented LLM, ReAct reasoning acting, plugin systems",
        "queries": [
            Query(q="LLM tool use function calling 2025 2026", intent="primary"),
            Query(q="ReAct reasoning acting language models", intent="canonical"),
            Query(q="tool-augmented language models retrieval", intent="supporting"),
        ],
    },
    {
        "name": "reasoning",
        "description": "Chain-of-thought prompting, reasoning in language models, CoT, System 1 System 2 thinking, math reasoning, logical inference",
        "queries": [
            Query(q="chain-of-thought prompting reasoning LLMs Wei", intent="canonical"),
            Query(q="reasoning language models chain-of-thought 2025", intent="primary"),
            Query(q="System 1 System 2 thinking AI", intent="supporting"),
        ],
    },
    {
        "name": "rag_retrieval",
        "description": "Retrieval-augmented generation, RAG, knowledge retrieval, vector database, semantic search, chunking strategies",
        "queries": [
            Query(q="retrieval-augmented generation RAG Lewis 2020", intent="canonical"),
            Query(q="RAG vector database semantic search 2025 2026", intent="primary"),
            Query(q="knowledge retrieval language model augmented", intent="supporting"),
        ],
    },
    {
        "name": "finetuning",
        "description": "Fine-tuning language models, LoRA low-rank adaptation, instruction tuning, PEFT parameter-efficient, adapter methods",
        "queries": [
            Query(q="LoRA low-rank adaptation language models Hu", intent="canonical"),
            Query(q="instruction tuning language models 2025", intent="primary"),
            Query(q="PEFT parameter-efficient fine-tuning methods", intent="supporting"),
        ],
    },
    {
        "name": "evaluation",
        "description": "Evaluating language models, benchmark datasets, MMLU, HumanEval, HELM, perplexity, toxicity testing",
        "queries": [
            Query(q="LLM evaluation benchmark MMLU HumanEval 2025", intent="primary"),
            Query(q="perplexity language model evaluation", intent="supporting"),
            Query(q="HELM benchmark holistic language model evaluation", intent="supporting"),
        ],
    },
    {
        "name": "multimodal",
        "description": "Multimodal language models, vision-language models, GPT-4V, Claude vision, image understanding, audio models",
        "queries": [
            Query(q="multimodal language models vision-language 2025 2026", intent="primary"),
            Query(q="GPT-4V vision language model evaluation", intent="canonical"),
            Query(q="vision-language model training alignment", intent="supporting"),
        ],
    },
    {
        "name": "emerging_trends",
        "description": "Emerging trends LLMs 2025 2026, state of AI, LLM ecosystem, frontier models, GPT-5 Claude 4 Gemini, model releases",
        "queries": [
            Query(q="LLM trends 2025 2026 frontier models", intent="primary"),
            Query(q="GPT-5 Claude 4 Gemini AI model releases 2025", intent="primary"),
            Query(q="LLM landscape 2026 survey capabilities", intent="supporting"),
        ],
    },
    {
        "name": "scaling",
        "description": "Scaling laws, compute-optimal training, Chinchilla, model scaling, emergent capabilities,涌现能力",
        "queries": [
            Query(q="scaling laws language models Kaplan 2020", intent="canonical"),
            Query(q="Chinchilla compute-optimal training Hoffmann", intent="canonical"),
            Query(q="emergent capabilities large language models", intent="supporting"),
        ],
    },
    {
        "name": "context_window",
        "description": "Long context window, attention efficiency, sparse attention, FlashAttention, context length scaling, million token context",
        "queries": [
            Query(q="FlashAttention efficient transformer training Dao", intent="canonical"),
            Query(q="long context window language models million tokens 2025", intent="primary"),
            Query(q="sparse attention mechanism transformer efficiency", intent="supporting"),
        ],
    },
    {
        "name": "tokenization",
        "description": "Tokenization, BPE byte pair encoding, subword tokenization, sentencepiece, tokenizer training, vocabulary size",
        "queries": [
            Query(q="BPE byte pair encoding tokenization neural", intent="canonical"),
            Query(q="tokenizer vocabulary size LLM efficiency", intent="supporting"),
            Query(q="subword tokenization language models 2025", intent="supporting"),
        ],
    },
    {
        "name": "safety_alignment",
        "description": "AI safety, alignment problem, value alignment, helpful harmless honest, constitutional AI, red teaming, interpretability",
        "queries": [
            Query(q="AI safety alignment problem language models 2025", intent="primary"),
            Query(q="constitutional AI Anthropic CAI", intent="canonical"),
            Query(q="red teaming language model safety evaluation", intent="supporting"),
        ],
    },
    {
        "name": "open_source",
        "description": "Open source language models, Llama, Mistral, Gemma, open weights models, open-source LLM ecosystem 2025 2026",
        "queries": [
            Query(q="Llama 3 open source language model Meta 2024 2025", intent="canonical"),
            Query(q="Mistral open source LLM models 2025 2026", intent="primary"),
            Query(q="open source LLM ecosystem comparison 2026", intent="supporting"),
        ],
    },
    {
        "name": "prompting",
        "description": "Prompt engineering, few-shot prompting, in-context learning, prompt optimization, system prompts, prompt tuning",
        "queries": [
            Query(q="in-context learning few-shot prompting LLMs", intent="canonical"),
            Query(q="prompt engineering techniques 2025", intent="primary"),
            Query(q="prompt optimization tuning language models", intent="supporting"),
        ],
    },
]

# Cosine threshold for archetype match. Below this → LLM fallback.
_ROUTE_SIM_THRESHOLD = 0.50
# Number of top archetypes to consider when combining.
_COMBINE_TOP_K = 2

# ---- LLM fallback prompt (single-pass, not two-stage) -----------------------
_QUERY_GEN_SYS = (
    "You are a research assistant. Given a section prompt, output a compact JSON array "
    "of 3 to 5 search queries optimized for academic search (arxiv) and Wikipedia. "
    "Each item is an object: {\"q\": \"...\", \"intent\": \"primary|supporting|definition|canonical\"}. "
    "Each query must be <= 12 words, specific, and avoid generic terms. "
    "Output ONLY the JSON array. No prose, no markdown fences, no explanation."
)
_QUERY_DECOMPOSE_SYS = (
    "You are a research planner. Given a section title and prompt, decompose it into "
    "2-3 ATOMIC sub-topics or aspects. An atomic sub-topic is a single specific concept, "
    "method, or fact that can be looked up independently. "
    "Return a JSON array of strings: [sub-topic A, sub-topic B, sub-topic C]. "
    "Each sub-topic must be <= 8 words. "
    "Output ONLY the JSON array. No prose, no markdown fences."
)
_JSON_STR_ARRAY_RE = re.compile(r'\[\s*(?:"[^"]+"\s*,?\s*)+\]', re.DOTALL)
_THINK_RE = re.compile(r"<THINK>.*?</THINK>", re.DOTALL | re.IGNORECASE)
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{[^\[\]]*\}\s*,?\s*)+\]", re.DOTALL)
_EMBED_MODEL = "bge-m3:latest"  # #3 unify (was nomic, used without required prefix)
_LOCK = threading.Lock()


# ---- Precompute archetype embeddings ----------------------------------------
_archetype_vectors: Optional[List[List[float]]] = None
_archetype_names: Optional[List[str]] = None
_archetype_queries: Optional[List[List[Query]]] = None
_Initialized = False


def _ensure_init():
    global _archetype_vectors, _archetype_names, _archetype_queries, _Initialized
    if _Initialized:
        return
    with _LOCK:
        if _Initialized:
            return
        descs = [a["description"] for a in _ARCHETYPES]
        vecs = _embed(descs, model=_EMBED_MODEL)
        if not vecs:
            print("[query_router] WARN: archetype embedding failed; routing will use LLM for all sections",
                  flush=True)
            _archetype_vectors = []
            _archetype_names = [a["name"] for a in _ARCHETYPES]
            _archetype_queries = [a["queries"] for a in _ARCHETYPES]
        else:
            _archetype_vectors = vecs
            _archetype_names = [a["name"] for a in _ARCHETYPES]
            _archetype_queries = [a["queries"] for a in _ARCHETYPES]
            print(f"[query_router] loaded {len(_ARCHETYPES)} archetypes", flush=True)
        _Initialized = True


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _route_to_archetypes(section_prompt: str, section_title: str) -> tuple:
    """Cosine-match section to archetypes. Returns (matched_queries, archetype_name, cos)."""
    _ensure_init()
    if not _archetype_vectors:
        return [], None, 0.0

    # Build routing text: title weighted + prompt
    routing_text = f"{section_title}. {section_prompt[:500]}"
    vecs = _embed([routing_text], model=_EMBED_MODEL)
    if not vecs:
        return [], None, 0.0

    query_vec = vecs[0]
    best_idx = 0
    best_cos = 0.0
    for i, arch_vec in enumerate(_archetype_vectors):
        c = _cosine(query_vec, arch_vec)
        if c > best_cos:
            best_cos = c
            best_idx = i

    arch_name = _archetype_names[best_idx]
    queries = _archetype_queries[best_idx] if best_idx < len(_archetype_queries) else []
    return list(queries), arch_name, best_cos


# ---- LLM fallback (single-pass, replaces two-stage) ------------------------

def _parse_queries(raw: str, max_n: int = 5) -> List[Query]:
    raw = _strip_think(raw)
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
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
    title = (section_title or chapter_title or "").strip()
    if not title:
        return []
    return [
        Query(q=f"{title} survey",       intent="primary"),
        Query(q=f"{title} 2024 2025",    intent="recent"),
        Query(q=f"{title} introduction", intent="definition"),
    ]


def _llm_query_gen(section_prompt: str, chapter_title: str, section_title: str,
                   model: str, client_base: str, timeout: float) -> List[Query]:
    """Single-pass LLM query gen (replaces the old two-stage approach).

    Takes section_prompt + chapter/section titles → returns 3-5 structured queries.
    Uses a single LLM call instead of decompose-then-generate.
    """
    user_prompt = (
        f"Chapter: {chapter_title}\n"
        f"Section: {section_title}\n"
        f"Section prompt:\n{section_prompt}\n\n"
        "Return the JSON array now."
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _QUERY_GEN_SYS},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {"temperature": 0.3, "num_predict": 400, "top_p": 0.9},
        "think": False,
    }
    try:
        with httpx.Client(timeout=timeout) as c:
            t0 = time.time()
            r = c.post(f"{client_base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        raw = _strip_think((data.get("message") or {}).get("content", ""))
        print(f"[query_router] LLM fallback {model} -> {len(raw)} chars in {time.time()-t0:.1f}s",
              flush=True)
    except Exception as e:
        print(f"[query_router] WARN: LLM fallback failed ({e}); using deterministic fallback",
              flush=True)
        return _fallback_queries(section_title, chapter_title)

    parsed = _parse_queries(raw)
    if not parsed:
        print("[query_router] WARN: could not parse JSON from LLM; using deterministic fallback",
              flush=True)
        return _fallback_queries(section_title, chapter_title)
    return parsed


# ---- Public API (drop-in replacement for queries_for in query_gen.py) --------

def queries_for(
    section_prompt: str,
    chapter_title: str,
    section_title: str,
    client_base: str = OLLAMA_BASE,
    model: str = DEFAULT_MODEL,
    reviewer_hint: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    prior_query_sigs: Optional[set] = None,
    # When provided, skip archetype routing and use LLM fallback directly.
    # This ensures query generation is driven by the section's actual goal,
    # not a generic archetype match (e.g. "Self-Attention" → "reasoning" archetype is wrong).
    domain_context: Optional[str] = None,
) -> List[Query]:
    """Semantic query routing: understand topic → structured queries.

    Routing strategy:
      1. If domain_context is provided → skip archetypes, use LLM fallback directly
         (archetypes can't reliably disambiguate "Self-Attention (transformer)" from
          "Self-Attention (reasoning/CoT)" without domain context).
      2. Cosine-match section to archetypes via bge-m3 embedding (~1s, 0 LLM)
      3. If cos ≥ {threshold} → return archetype queries (0 LLM)
      4. If 0.30 < cos < threshold → combine archetype + LLM
      5. If cos ≤ 0.30 (novel/unmatched) → LLM fallback (1 call)

    When prior_query_sigs is provided (round > 1), filters out queries that are
    near-identical to ones already used in earlier rounds to diversify re-search.

    Returns a list of Query objects. Compatible with the old queries_for() signature.
    """
    _ensure_init()

    def _dedup(queries: List[Query], seen: set) -> List[Query]:
        """Filter queries that are too similar to previously-used ones."""
        if not seen:
            return queries
        out: List[Query] = []
        for q in queries:
            sig = (q.q.strip().lower()[:60], getattr(q, "intent", "unknown"))
            if sig not in seen:
                out.append(q)
        return out

    # Step 0: if domain context provided, skip archetype entirely
    # Inject domain context into section_prompt so LLM knows the broader topic
    if domain_context:
        print(f"[query_router] domain_context provided — skipping archetype routing, using LLM fallback",
              flush=True)
        # The LLM uses section_prompt to generate queries; prepend domain context there
        return _dedup(_llm_query_gen(
            f"[DOMAIN CONTEXT]\n{domain_context}\n\n{section_prompt}",
            chapter_title, section_title,
            model, client_base, timeout,
        ), prior_query_sigs)

    # Step 1: route to archetype
    arch_queries, arch_name, cos = _route_to_archetypes(section_prompt, section_title)

    print(f"[query_router] section='{section_title[:40]}' → archetype='{arch_name}' (cos={cos:.3f})",
          flush=True)

    # Step 2: decide routing tier
    if cos >= _ROUTE_SIM_THRESHOLD:
        # Tier 2a: high-confidence archetype match → use archetype directly
        return _dedup(list(arch_queries), prior_query_sigs)

    elif cos >= 0.30:
        # Tier 2b: moderate match → LLM generates additional queries
        # but archetype provides a solid base
        llm_queries = _llm_query_gen(
            section_prompt, chapter_title, section_title,
            model, client_base, timeout,
        )
        # Merge: archetype first (they're high-quality), then LLM (fills gaps)
        combined: List[Query] = list(arch_queries)
        seen_q = {q.q.lower() for q in combined}
        for q in llm_queries:
            if q.q.lower() not in seen_q:
                combined.append(q)
                seen_q.add(q.q.lower())
        return _dedup(combined[:5], prior_query_sigs)

    else:
        # Tier 3: novel section → LLM fallback
        return _dedup(_llm_query_gen(
            section_prompt, chapter_title, section_title,
            model, client_base, timeout,
        ), prior_query_sigs)
