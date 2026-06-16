"""Shared research-layer configuration constants.

Tách constants ra khỏi package root để tránh circular import khi các submodule
cần model/knob defaults trong lúc `research.__init__` vẫn đang khởi tạo.
"""

# Provider order matters: cheap+reliable first (arxiv, wikipedia), then web sources.
PROVIDERS_DEFAULT = ("arxiv", "wikipedia", "tavily", "ddg")

# ---- v1 Retrieval Knobs ----
TOP_K_DEFAULT = 8
PRIMARY_FLOOR = 3
FULL_TEXT_TOP_N = 2
FULL_TEXT_MAX_WORDS = 350

# ---- v2 Retrieval Knobs (RRK tier) ----
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 8
RELEVANCE_FLOOR = 0.001

# ---- v2 Grounding Knobs (HHEM tier) ----
HHEM_SUPPORT = 0.5

# ---- v2 CRAG Knobs (CRAG tier) ----
GROUND_UPPER = 0.80
GROUND_LOWER = 0.40

# ---- Model Defaults ----
QUERY_GEN_MODEL = "gemma4:e4b"
JUDGE_MODEL = "gemma4:e4b"
DISCOVERY_MODEL = "gemma4:e4b"
OUTLINE_MODEL = "gemma4:e4b"
WRITER_MODEL = "batiai/qwen3.6-35b:iq3"
EMBED_MODEL = "bge-m3:latest"  # #3 unify with verify-side (was nomic: asymmetric, needs search_query/document prefix the code never passed)

# Iterative-research thresholds
MIN_GROUNDING = 0.55
MAX_RESEARCH_ROUNDS = 2

# ---- v1 Cosine gates (backward compat) ----
AUTO_SUPPORT_COS = 0.75
AUTO_UNRELATED_COS = 0.30

__all__ = [
    "PROVIDERS_DEFAULT",
    "TOP_K_DEFAULT", "TOP_K_RETRIEVE", "TOP_K_FINAL",
    "PRIMARY_FLOOR", "FULL_TEXT_TOP_N", "FULL_TEXT_MAX_WORDS",
    "RELEVANCE_FLOOR",
    "HHEM_SUPPORT",
    "GROUND_UPPER", "GROUND_LOWER",
    "QUERY_GEN_MODEL", "JUDGE_MODEL", "DISCOVERY_MODEL", "OUTLINE_MODEL", "WRITER_MODEL", "EMBED_MODEL",
    "MIN_GROUNDING", "MAX_RESEARCH_ROUNDS",
    "AUTO_SUPPORT_COS", "AUTO_UNRELATED_COS",
]
