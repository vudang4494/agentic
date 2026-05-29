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
    # gpt-2 omitted: it was an OpenAI tech report, not a canonical arxiv paper
    # (the old id 1902.09737 was wrong). Don't seed a fabricated id.
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

# Rank8: canonical attribution facts for the most-cited papers, keyed by the
# alias the writer is likely to name. Used to flag confidently-wrong author/year
# attributions that citation-grounding (support-by-snippet) cannot catch -- e.g.
# bookv6 3.2 misattributed Chinchilla to "Muennighoff et al. (2023)"
# (real: Hoffmann et al., 2022). {alias_lower: (surname, year)}.
CANONICAL_FACTS = {
    "attention is all you need": ("Vaswani", 2017),
    "transformer architecture": ("Vaswani", 2017),
    "bert": ("Devlin", 2018),
    "roberta": ("Liu", 2019),
    "t5": ("Raffel", 2019),
    "gpt-3": ("Brown", 2020),
    "scaling laws": ("Kaplan", 2020),
    "chinchilla": ("Hoffmann", 2022),
    "compute-optimal": ("Hoffmann", 2022),
    "palm": ("Chowdhery", 2022),
    "llama": ("Touvron", 2023),
    "instructgpt": ("Ouyang", 2022),
    "rlhf": ("Ouyang", 2022),
    "chain-of-thought": ("Wei", 2022),
    "direct preference optimization": ("Rafailov", 2023),
    "dpo": ("Rafailov", 2023),
    "lora": ("Hu", 2021),
    "low-rank adaptation": ("Hu", 2021),
    "retrieval-augmented generation": ("Lewis", 2020),
    "rope": ("Su", 2021),
    "roformer": ("Su", 2021),
    "flash attention": ("Dao", 2022),
    "mamba": ("Gu", 2023),
    "clip": ("Radford", 2021),
    "vision transformer": ("Dosovitskiy", 2020),
    "adam optimizer": ("Kingma", 2014),
}

_FACT_ALIASES = sorted(CANONICAL_FACTS.keys(), key=len, reverse=True)
_FACT_RE = {a: re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE) for a in _FACT_ALIASES}
# An ORIGINATION attribution near the method: "introduced/proposed/developed/...
# by Surname (YYYY)". Gating on an origination verb avoids flagging a legit
# ADJACENT cite (e.g. "GPT-3 ... later analyzed by Wei (2022)") as a
# misattribution -- only claims that the wrong author CREATED the method fire.
_ORIGIN_ATTRIB_RE = re.compile(
    r"(?:introduc|propos|develop|pioneer|present|formulat|originat|creat|devis)\w*"
    r"[^.]{0,40}?\bby\s+([A-Z][a-zA-Z]+)\s*(?:et al\.?,?\s*)?\(?\s*(\d{4})\s*\)?",
    re.IGNORECASE,
)
# Possessive origination: "Muennighoff et al.'s (2023) work on <method>".
_POSSESSIVE_ATTRIB_RE = re.compile(
    r"([A-Z][a-zA-Z]+)\s+et al\.?[’']?s?\s*\(?\s*(\d{4})\s*\)?[’']?s?\s+"
    r"(?:work|paper|study|research|analysis|approach|method)\s+(?:on|in|of)",
    re.IGNORECASE,
)
_ATTRIB_PATTERNS = (_ORIGIN_ATTRIB_RE, _POSSESSIVE_ATTRIB_RE)


def check_attribution(content: str) -> list:
    """Flag confidently-wrong ORIGINATION attributions for canonical methods.

    Attribution-centric: for each origination claim ('introduced by Surname
    (YYYY)' or 'Surname et al.'s (YYYY) work on ...') found in the text, assign
    it to the NEAREST canonical method (within a tight +/-30-char window of the
    claim) and compare to the known truth. Anchoring to the nearest method avoids
    charging a far-away method in the same sentence (e.g. a 'work on Chinchilla'
    claim won't be blamed on a 'GPT-3' mentioned 40 chars later). Advisory only;
    grounding measures snippet support, this measures factual truth.
    """
    if not content:
        return []
    flags = []
    seen = set()
    for pat in _ATTRIB_PATTERNS:
        for am in pat.finditer(content):
            try:
                year = int(am.group(2))
            except (ValueError, IndexError):
                continue
            if year < 2010 or year > 2027:
                continue
            surname = am.group(1)
            # Which canonical method does this claim refer to? The nearest alias
            # within +/-30 chars of the claim span.
            win_start = max(0, am.start() - 30)
            window = content[win_start: am.end() + 30]
            best_alias, best_dist = None, 10 ** 9
            for alias in _FACT_ALIASES:
                am2 = _FACT_RE[alias].search(window)
                if not am2:
                    continue
                apos = win_start + am2.start()
                dist = min(abs(apos - am.start()), abs(apos - am.end()))
                if dist < best_dist:
                    best_dist, best_alias = dist, alias
            if best_alias is None:
                continue
            truth_surname, truth_year = CANONICAL_FACTS[best_alias]
            if (truth_surname, truth_year) in seen:
                continue
            if surname.lower() != truth_surname.lower() or abs(year - truth_year) > 1:
                flags.append({
                    "concept": best_alias, "found": f"{surname} ({year})",
                    "expected": f"{truth_surname} ({truth_year})",
                })
                seen.add((truth_surname, truth_year))
    return flags


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
