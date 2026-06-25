"""Agentic Deep Research layer -- Stage 2 of the AgentDeepLearning roadmap.

Plugs into deep_research.py per-section loop:

    queries  = query_gen.queries_for(section_prompt, ch_t, pp_t, client)
    sources  = search.gather(queries, providers=PROVIDERS_DEFAULT)
    sources  = notes.rank(sources, section_prompt, top_k=TOP_K_RETRIEVE)
    sources  = rerank.rerank(section_prompt, sources, top_k=TOP_K_FINAL)
    evidence = notes.format_for_prompt(sources)
    grounding = faithfulness.grounding_score(claims, sources)

Constants are defined in `research.config` and re-exported here for backward
compatibility so callers can keep using `research.XYZ` without creating circular
imports during package initialization.
"""

from .types import Source, Query
from .config import (
    PROVIDERS_DEFAULT,
    TOP_K_DEFAULT,
    TOP_K_RETRIEVE,
    TOP_K_FINAL,
    PRIMARY_FLOOR,
    FULL_TEXT_TOP_N,
    FULL_TEXT_MAX_WORDS,
    RELEVANCE_FLOOR,
    HHEM_SUPPORT,
    GROUND_UPPER,
    GROUND_LOWER,
    QUERY_GEN_MODEL,
    JUDGE_MODEL,
    DISCOVERY_MODEL,
    OUTLINE_MODEL,
    WRITER_MODEL,
    EMBED_MODEL,
    MIN_GROUNDING,
    MAX_RESEARCH_ROUNDS,
    AUTO_SUPPORT_COS,
    AUTO_UNRELATED_COS,
)
from . import (
    search, query_gen, query_router, notes, embeddings,
    fetch, verify, planner, canonical_seeds,
)  # noqa: F401

try:
    from . import rerank  # noqa: F401
    from . import faithfulness  # noqa: F401
    VFY_V2_AVAILABLE = True
except ImportError:
    rerank = None
    faithfulness = None
    VFY_V2_AVAILABLE = False

__all__ = [
    "Source", "Query",
    "search", "query_gen", "query_router", "notes", "embeddings",
    "fetch", "verify", "planner", "canonical_seeds",
    "rerank", "faithfulness",
    "VFY_V2_AVAILABLE",
    "PROVIDERS_DEFAULT",
    "TOP_K_DEFAULT", "TOP_K_RETRIEVE", "TOP_K_FINAL",
    "FULL_TEXT_TOP_N", "FULL_TEXT_MAX_WORDS",
    "PRIMARY_FLOOR",
    "RELEVANCE_FLOOR",
    "HHEM_SUPPORT",
    "GROUND_UPPER", "GROUND_LOWER",
    "QUERY_GEN_MODEL", "JUDGE_MODEL", "DISCOVERY_MODEL", "OUTLINE_MODEL", "WRITER_MODEL", "EMBED_MODEL",
    "MIN_GROUNDING", "MAX_RESEARCH_ROUNDS",
    "AUTO_SUPPORT_COS", "AUTO_UNRELATED_COS",
]
