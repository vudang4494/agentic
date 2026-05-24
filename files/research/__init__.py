"""Agentic Deep Research layer -- Stage 2 of the AgentDeepLearning roadmap.

Plugs into deep_research.py per-section loop:

    queries  = query_gen.queries_for(section_prompt, ch_t, pp_t, client)
    sources  = search.gather(queries, providers=PROVIDERS_DEFAULT)
    sources  = notes.rank(sources, section_prompt, top_k=TOP_K_DEFAULT)
    evidence = notes.format_for_prompt(sources)

Models used (configurable, default values are what the user has installed):
  - QUERY_GEN_MODEL  = qwen3.5:4b   (better JSON compliance than gemma3:4b)
  - EMBED_MODEL      = bge-m3       (dense retrieval ranking)
  - Writer / reviewer stay on whatever deep_research.MODEL is set to.

The package fails closed: if any provider errors out, sources just becomes a
shorter list. If the whole layer can't import (httpx missing, etc.), the main
pipeline detects RESEARCH_AVAILABLE=False and runs in legacy mode.
"""

from .types import Source, Query
from . import search, query_gen, notes, embeddings, fetch, verify, planner  # noqa: F401

# Provider order matters: cheap+reliable first (arxiv, wikipedia), then web sources.
# search.gather() filters out providers whose prerequisites aren't met (e.g. tavily
# without TAVILY_API_KEY, or once tavily auto-disabled on HTTP 432 rate-limit).
# Brave is added if BRAVE_API_KEY is set (2000/mo free at https://brave.com/search/api/).
# DDG is HTML-scraped, no key required -- best free web fallback after Tavily exhaustion.
PROVIDERS_DEFAULT = ("arxiv", "wikipedia", "brave", "ddg", "tavily")
TOP_K_DEFAULT = 8
FULL_TEXT_TOP_N = 2          # fetch full body for the top-N ranked sources
FULL_TEXT_MAX_WORDS = 350    # cap per source so the prompt doesn't blow up
QUERY_GEN_MODEL = "qwen3.5:4b"
JUDGE_MODEL = "qwen3.5:4b"   # citation grounding verifier (verify.verify_section)
EMBED_MODEL = "bge-m3:latest"

# Iterative-research thresholds (used by deep_research.run loop).
MIN_GROUNDING = 0.55     # below this -> trigger re-research + rewrite
MAX_RESEARCH_ROUNDS = 2  # cap on how many times we re-search per section

__all__ = [
    "Source", "Query",
    "search", "query_gen", "notes", "embeddings", "fetch", "verify", "planner",
    "PROVIDERS_DEFAULT", "TOP_K_DEFAULT", "FULL_TEXT_TOP_N", "FULL_TEXT_MAX_WORDS",
    "QUERY_GEN_MODEL", "JUDGE_MODEL", "EMBED_MODEL",
    "MIN_GROUNDING", "MAX_RESEARCH_ROUNDS",
]
