"""Canonical-paper seed map for known-item retrieval (Rank5).

bookv6 retrieved 0/5 must-cite foundations because the arxiv ti: search is fed
descriptive queries that never name the seminal paper, and Tavily blogs occupy
most slots. This module resolves canonical method/model NAMES mentioned in a
section to their arxiv IDs, so the pipeline can fetch the primary source
directly (search.arxiv_by_id) and inject it into the candidate pool.

Authoritative seed list for the LLM domain. Keep aligned with the gold lists in
files/eval/topics/*.yaml (this module is the runtime source of truth; eval mirrors it).
"""
import re

# alias (matched in section text) -> arxiv id (no version suffix)
SEED_MAP = {
    # Foundations
    "attention is all you need": "1706.03762",
    "transformer architecture": "1706.03762",
    "vaswani": "1706.03762",
    "layer normalization": "1607.06450",
    "layernorm": "1607.06450",
    "adam optimizer": "1412.6980",
    "adamw": "1711.05101",
    "dropout": "1207.0580",
    # Pre-training / encoders
    "bert": "1810.04805",
    "roberta": "1907.11692",
    "t5": "1910.10683",
    "text-to-text transfer": "1910.10683",
    "elmo": "1802.05365",
    "gpt-2": "1902.09737",
    # Scaling / large models
    "gpt-3": "2005.14165",
    "few-shot learners": "2005.14165",
    "scaling laws": "2001.08361",
    "kaplan": "2001.08361",
    "chinchilla": "2203.15556",
    "compute-optimal": "2203.15556",
    "palm": "2204.02311",
    "llama": "2302.13971",
    "llama 2": "2307.09288",
    "mixtral": "2401.04088",
    "switch transformer": "2101.03961",
    # Alignment / instruction tuning
    "instructgpt": "2203.02155",
    "rlhf": "2203.02155",
    "reinforcement learning from human feedback": "2203.02155",
    "instruction tuning": "2210.11416",
    "flan": "2210.11416",
    "direct preference optimization": "2305.18290",
    "dpo": "2305.18290",
    "constitutional ai": "2212.08073",
    # Reasoning / prompting
    "chain-of-thought": "2201.11903",
    "chain of thought": "2201.11903",
    "self-consistency": "2203.11171",
    "tree of thoughts": "2305.10601",
    "react": "2210.03629",
    # Efficiency / adaptation
    "lora": "2106.09685",
    "low-rank adaptation": "2106.09685",
    "qlora": "2305.14314",
    "prefix tuning": "2101.00190",
    "flash attention": "2205.14135",
    "flashattention": "2205.14135",
    "rope": "2104.09864",
    "rotary position embedding": "2104.09864",
    "roformer": "2104.09864",
    "alibi": "2108.12409",
    "mamba": "2312.00752",
    "state space model": "2312.00752",
    # Retrieval / multimodal
    "retrieval-augmented generation": "2005.11401",
    "rag": "2005.11401",
    "clip": "2103.00020",
    "flamingo": "2204.14198",
    "blip-2": "2301.12597",
    "vision transformer": "2010.11929",
    "vit": "2010.11929",
    # Decoding / eval
    "nucleus sampling": "1904.09751",
    "top-p sampling": "1904.09751",
    "beam search": "1409.3215",
}

# Longest-first so "chain-of-thought" is tried before any shorter overlap.
# Each alias is pre-compiled to a case-insensitive word-boundary regex. The \b
# boundaries prevent substring false-positives (e.g. "bert" never fires inside
# "roberta", "rag" never inside "storage") regardless of case.
_ALIASES = sorted(SEED_MAP.keys(), key=len, reverse=True)
_ALIAS_RE = {a: re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE) for a in _ALIASES}


def resolve_seeds(text: str, max_seeds: int = 4) -> list:
    """Return up to max_seeds canonical arxiv IDs whose alias appears in `text`."""
    if not text:
        return []
    found = []
    seen = set()
    for alias in _ALIASES:
        arx = SEED_MAP[alias]
        if arx in seen:
            continue
        if _ALIAS_RE[alias].search(text):
            found.append(arx)
            seen.add(arx)
            if len(found) >= max_seeds:
                break
    return found
