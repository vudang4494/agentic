"""
Pure metric functions for the research-quality eval harness.

Input: gold YAML dict + state.json dict from a pipeline run.
Output: per-section metric dicts + aggregate dict + pass/fail dict.

No I/O, no LLM calls -- so these functions are unit-testable and the eval
harness can re-run them against an archived state.json without re-running
the pipeline.
"""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse


# ----------------------------------------------------------------------------
# Source-level helpers
# ----------------------------------------------------------------------------

_ARXIV_URL_RE = re.compile(
    # /abs/, /pdf/, /html/ (new format), with optional v\d+ suffix
    r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
    re.IGNORECASE,
)


def arxiv_ids_in_sources(sources: list[dict]) -> set[str]:
    """Return the set of arxiv IDs (without version suffix) in `sources`.

    Recognizes BOTH:
      - native arxiv provider ids ("arxiv:1706.03762v2" -> "1706.03762")
      - Tavily/Brave/etc. URLs pointing at arxiv.org (/abs/, /pdf/, /html/)
    """
    out: set[str] = set()
    for s in sources or []:
        sid = (s.get("id") or "").lower()
        if sid.startswith("arxiv:"):
            raw = sid.split(":", 1)[1].strip()
            out.add(re.sub(r"v\d+$", "", raw))
        url = (s.get("url") or "").lower()
        m = _ARXIV_URL_RE.search(url)
        if m:
            out.add(m.group(1))
    return out


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())


def gold_aliases(gold_item: dict) -> list[str]:
    """Pull all match strings for a gold paper: arxiv id, title, aliases."""
    out: list[str] = []
    if gold_item.get("arxiv"):
        out.append(gold_item["arxiv"])
    if gold_item.get("title"):
        out.append(gold_item["title"])
    out.extend(gold_item.get("aliases", []) or [])
    return out


def gold_paper_hits(sources: list[dict], gold_papers: list[dict]) -> dict:
    """Return {arxiv_id: hit_kind} where hit_kind is "arxiv" (direct PDF) or
    "wiki" (Wikipedia article about the paper). Wiki credit is given when a
    source's title contains any alias of the gold paper AND the source's URL
    points at en.wikipedia.org (covers both wiki: native id AND Tavily/etc.
    sources whose URL is a wikipedia page).

    Why count wiki: a hit on `Attention_Is_All_You_Need` Wikipedia article IS
    retrieving Vaswani 2017 content, just via the Wikipedia surface instead
    of the PDF. The writer can ground claims with [N] -> Wikipedia source,
    which is a real citation. Refusing this credit understates retrieval
    quality.
    """
    found_arxiv = arxiv_ids_in_sources(sources)
    hits: dict[str, str] = {}
    # Wiki sources: either native "wiki:" id OR any source whose url is on en.wikipedia.org
    wiki_titles_norm: list[str] = []
    for s in sources or []:
        sid = (s.get("id") or "")
        url = (s.get("url") or "").lower()
        is_wiki = sid.startswith("wiki:") or "en.wikipedia.org/wiki/" in url
        if not is_wiki:
            continue
        # Build a normalized blob from id-tail + title for alias matching.
        tail = sid.replace("wiki:", "").replace("_", " ") if sid.startswith("wiki:") else ""
        blob = _norm_title(tail) + " " + _norm_title(s.get("title") or "") + " " + _norm_title(url)
        wiki_titles_norm.append(blob)
    for gp in gold_papers:
        arx = gp.get("arxiv")
        if arx and arx in found_arxiv:
            hits[arx] = "arxiv"
            continue
        aliases_norm = [_norm_title(a) for a in gold_aliases(gp) if a and not re.fullmatch(r"\d{4}\.\d{4,5}", a)]
        for wt in wiki_titles_norm:
            if any(a and a in wt for a in aliases_norm):
                if arx:
                    hits[arx] = "wiki"
                break
    return hits


def _host(url: str) -> str:
    """Lowercase host without a leading 'www.' prefix. ('' on parse failure).
    Uses removeprefix, NOT lstrip -- lstrip('www.') strips ANY leading w/./3 chars
    (e.g. 'wandb.ai' -> 'andb.ai'), which corrupted domain counts."""
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return h[4:] if h.startswith("www.") else h


def _host_suffix_match(host: str, domains) -> bool:
    """host == d or host endswith '.d' (suffix-match, not substring) -- so
    'labelbox.com'/'matrix.com' do NOT spuriously match 'x.com'."""
    return any(host == d or host.endswith("." + d) for d in domains)


def domains_in_sources(sources: list[dict]) -> Counter:
    """Count occurrences of each domain in a section's source list."""
    c: Counter = Counter()
    for s in sources or []:
        host = _host(s.get("url") or "")
        if host:
            c[host] += 1
    return c


# ----------------------------------------------------------------------------
# Output-level helpers
# ----------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_WORD_RE = re.compile(r"\b\w+\b")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def citations_in(text: str) -> list[int]:
    return [int(m.group(1)) for m in _CITATION_RE.finditer(text or "")]


def detect_looping(text: str, phrase_len: int = 5, min_repeats: int = 4) -> bool:
    """Heuristic: any `phrase_len`-word window that repeats >= `min_repeats` times.

    Catches degenerate writer behavior where the same phrase recurs (the most
    common 4B-model failure mode pre-W1).
    """
    if not text:
        return False
    words = text.split()
    if len(words) < phrase_len * min_repeats:
        return False
    seen: Counter = Counter()
    for i in range(len(words) - phrase_len + 1):
        phrase = " ".join(words[i : i + phrase_len])
        seen[phrase] += 1
        if seen[phrase] >= min_repeats:
            return True
    return False


def subtopic_coverage(book_text: str, expected: list[str]) -> tuple[float, list[str]]:
    """Return (coverage_ratio, missing_list). Case-insensitive substring match."""
    if not expected:
        return 1.0, []
    haystack = (book_text or "").lower()
    missing = [t for t in expected if t.lower() not in haystack]
    hit = len(expected) - len(missing)
    return hit / len(expected), missing


# ----------------------------------------------------------------------------
# Per-section metrics
# ----------------------------------------------------------------------------

def section_metrics(key: str, section: dict, gold: dict) -> dict:
    """Compute all per-section metrics for one entry in state['passes'][key]."""
    content = section.get("content", "") or ""
    sources = section.get("sources", []) or []
    verify = section.get("verify") or {}
    rounds = section.get("research_rounds") or []
    review = section.get("review") or {}

    must_papers = [it for it in gold.get("must_cite", []) if it.get("arxiv")]
    should_papers = [it for it in gold.get("should_cite", []) if it.get("arxiv")]
    must_ids = {p["arxiv"] for p in must_papers}
    forbidden = {d.lower() for d in gold.get("forbidden_domains", [])}

    must_hits = gold_paper_hits(sources, must_papers)
    should_hits = gold_paper_hits(sources, should_papers)
    domain_hits = domains_in_sources(sources)
    # Host-suffix match (not substring): 'labelbox.com' must NOT count as 'x.com'.
    forbidden_hits = {d: c for d, c in domain_hits.items() if _host_suffix_match(d, forbidden)}

    cites = citations_in(content)
    n_cites = len(cites)
    unique_cited = len(set(cites))
    n_sources = len(sources)
    n_unused = max(0, n_sources - unique_cited)
    wc = section.get("wc") or word_count(content)
    cites_per_1k = (n_cites * 1000.0 / wc) if wc > 0 else 0.0

    grounding = verify.get("grounding")
    zero_cite_red_flag = (n_cites == 0 and n_sources > 0)

    primary_like = 0
    secondary_like = 0
    for s in sources:
        sid = (s.get("id") or "").lower()
        host = _host(s.get("url") or "")
        is_primary = (
            sid.startswith("arxiv:")
            or sid.startswith("wiki:")
            or host in {"arxiv.org", "en.wikipedia.org", "wikipedia.org", "aclanthology.org", "openreview.net", "proceedings.mlr.press", "jmlr.org", "nature.com", "science.org", "semanticscholar.org"}
            or host.endswith(".wikipedia.org")
        )
        is_secondary = any(x in host for x in ("medium.com", "substack.com", "towardsdatascience.com", "analyticsvidhya.com", "blog")) if host else False
        if is_primary:
            primary_like += 1
        elif is_secondary:
            secondary_like += 1

    return {
        "key": key,
        "title": section.get("title") or section.get("pp_t") or "?",
        "retrieval": {
            "n_sources": n_sources,
            "must_cite_hits": sorted(must_hits.keys()),
            "must_cite_hit_kinds": dict(must_hits),   # {arxiv_id: "arxiv"|"wiki"}
            "must_cite_misses": sorted(must_ids - set(must_hits)),
            "should_cite_hits": sorted(should_hits.keys()),
            "should_cite_hit_kinds": dict(should_hits),
            "forbidden_domain_hits": forbidden_hits,
            "n_queries": len(section.get("queries") or []),
            "research_rounds": len(rounds),
            "primary_source_count": primary_like,
            "secondary_source_count": secondary_like,
            "primary_source_pct": round(primary_like / n_sources, 4) if n_sources else 0.0,
        },
        "grounding": {
            "score": grounding,
            "n_citations_in_text": n_cites,
            "unique_cited": unique_cited,
            "weak_citations": len(verify.get("weak_citations") or []),
            "zero_cite_red_flag": zero_cite_red_flag,
        },
        "output": {
            "word_count": wc,
            "citations_per_1000w": round(cites_per_1k, 2),
            "sources_ranked_but_unused": n_unused,
            "looping_detected": detect_looping(content),
        },
        "review": {
            "depth": review.get("depth"),
            "coherence": review.get("coherence"),
            "format": review.get("format"),
            "skipped": review.get("_skipped", False),
        } if review else None,
        "cost": {
            "tokens": section.get("tokens", 0),
            "tps": section.get("tps", 0),
        },
    }


# ----------------------------------------------------------------------------
# Aggregate + pass/fail
# ----------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def aggregate(per_section: list[dict], gold: dict, book_text: str) -> dict:
    """Reduce per-section metrics into a single aggregate dict."""
    must_ids = {item["arxiv"] for item in gold.get("must_cite", []) if item.get("arxiv")}
    should_ids = {item["arxiv"] for item in gold.get("should_cite", []) if item.get("arxiv")}

    # Pool hits across all sections; track the strongest kind per paper
    # (arxiv direct beats wiki credit if both surfaced anywhere).
    pooled_kinds: dict[str, str] = {}
    for s in per_section:
        for aid, kind in s["retrieval"].get("must_cite_hit_kinds", {}).items():
            if pooled_kinds.get(aid) != "arxiv":
                pooled_kinds[aid] = kind
        for aid, kind in s["retrieval"].get("should_cite_hit_kinds", {}).items():
            if pooled_kinds.get(aid) != "arxiv":
                pooled_kinds[aid] = kind
    pooled_found = set(pooled_kinds.keys())

    must_recall = (len(must_ids & pooled_found) / len(must_ids)) if must_ids else 1.0
    should_recall = (len(should_ids & pooled_found) / len(should_ids)) if should_ids else 1.0
    must_arxiv_direct = sum(1 for aid in (must_ids & pooled_found) if pooled_kinds[aid] == "arxiv")
    must_via_wiki = sum(1 for aid in (must_ids & pooled_found) if pooled_kinds[aid] == "wiki")

    groundings = [s["grounding"]["score"] for s in per_section]
    grounding_mean = _mean([g for g in groundings if g is not None])

    n_sections = len(per_section)
    n_zero = sum(1 for s in per_section if s["grounding"]["zero_cite_red_flag"])
    n_loop = sum(1 for s in per_section if s["output"]["looping_detected"])
    n_round2 = sum(1 for s in per_section if s["retrieval"]["research_rounds"] >= 2)
    forbidden_total = sum(
        sum(s["retrieval"]["forbidden_domain_hits"].values()) for s in per_section
    )

    words = [s["output"]["word_count"] for s in per_section]
    cites_per_k = [s["output"]["citations_per_1000w"] for s in per_section]
    tokens_total = sum(s["cost"]["tokens"] for s in per_section)
    primary_counts = [s["retrieval"].get("primary_source_count", 0) for s in per_section]
    secondary_counts = [s["retrieval"].get("secondary_source_count", 0) for s in per_section]
    total_sources = sum(s["retrieval"].get("n_sources", 0) for s in per_section)
    total_primary = sum(primary_counts)
    total_secondary = sum(secondary_counts)

    coverage, missing_subtopics = subtopic_coverage(
        book_text, gold.get("expected_subtopics", []),
    )

    return {
        "n_sections": n_sections,
        "must_cite_recall": round(must_recall, 3),
        "must_cite_missed": sorted(must_ids - pooled_found),
        "must_cite_arxiv_direct": must_arxiv_direct,   # PDF retrieved
        "must_cite_via_wiki": must_via_wiki,           # only wiki page about it
        "should_cite_recall": round(should_recall, 3),
        "grounding_mean": round(grounding_mean, 3),
        "zero_cite_section_count": n_zero,
        "loop_section_count": n_loop,
        "loop_section_pct": round(n_loop / n_sections, 3) if n_sections else 0.0,
        "research_round_2_count": n_round2,
        "research_round_2_rate": round(n_round2 / n_sections, 3) if n_sections else 0.0,
        "forbidden_domain_hits": forbidden_total,
        "subtopic_coverage": round(coverage, 3),
        "subtopic_missing": missing_subtopics,
        "median_words": _median([float(w) for w in words]),
        "mean_citations_per_1000w": round(_mean(cites_per_k), 2),
        "total_tokens": tokens_total,
        "primary_sources_total": total_primary,
        "secondary_sources_total": total_secondary,
        "primary_source_pct": round(total_primary / total_sources, 4) if total_sources else 0.0,
        "secondary_source_pct": round(total_secondary / total_sources, 4) if total_sources else 0.0,
    }


def pass_fail(agg: dict, thresholds: dict, *, is_partial: bool = False) -> dict:
    """Apply thresholds from gold.thresholds to the aggregate. Return {check: {target, actual, pass}}.

    When is_partial=True, scale the breadth-sensitive thresholds (should_cite_recall,
    subtopic_coverage) proportionally to how many sections were actually generated vs.
    the full-run expectation. This prevents misleading FAIL results on smoke/partial
    runs while leaving full-run benchmarks unchanged.
    """
    checks = []
    def chk(name: str, op: str, target, actual, notes: str = ""):
        cmp_ok = {
            ">=": actual >= target,
            "<=": actual <= target,
            "==": actual == target,
        }[op]
        checks.append({"check": name, "op": op, "target": target,
                       "actual": actual, "pass": bool(cmp_ok),
                       "note": notes if (is_partial and notes) else ""})

    n_expected = 1
    n_gen = agg.get("n_sections", 0)
    # Auto-detect partial: if fewer than 25% of expected sections were generated
    if not is_partial and n_gen > 0:
        # n_chapters/n_passes are not available here; caller sets is_partial explicitly
        pass

    if is_partial and n_gen > 0:
        # In partial mode, relax breadth-sensitive thresholds so that smoke/partial runs
        # don't fail on coverage/citation metrics that are structurally impossible to pass
        # with a small slice of the book. The factor 0.5 means half the full-run bar.
        eff_should_cite = round(thresholds.get("should_cite_recall_min", 0.0) * 0.5, 4)
        eff_coverage    = round(thresholds.get("subtopic_coverage_min", 0.0)  * 0.5, 4)
        note_sc = f"(partial-run mode; threshold relaxed from full-run bar)"
        note_cov = f"(partial-run mode; threshold relaxed from full-run bar)"
    else:
        eff_should_cite = thresholds.get("should_cite_recall_min", 0.0)
        eff_coverage    = thresholds.get("subtopic_coverage_min", 0.0)
        note_sc = note_cov = ""

    chk("must_cite_recall",     ">=", thresholds.get("must_cite_recall_min", 0.0),    agg["must_cite_recall"])
    chk("should_cite_recall",   ">=", eff_should_cite,                                 agg["should_cite_recall"], note_sc)
    chk("grounding_mean",       ">=", thresholds.get("grounding_mean_min", 0.0),      agg["grounding_mean"])
    chk("zero_cite_sections",   "<=", thresholds.get("zero_cite_section_max", 0),     agg["zero_cite_section_count"])
    chk("loop_section_pct",     "<=", thresholds.get("loop_section_pct_max", 1.0),    agg["loop_section_pct"])
    chk("research_round_2_rate","<=", thresholds.get("research_round_2_rate_max", 1.0), agg["research_round_2_rate"])
    chk("forbidden_domain_hits","<=", thresholds.get("forbidden_domain_hits_max", 0), agg["forbidden_domain_hits"])
    chk("subtopic_coverage",    ">=", eff_coverage,                                    agg["subtopic_coverage"], note_cov)
    chk("median_words_min",     ">=", thresholds.get("median_words_min", 0),          agg["median_words"])
    chk("median_words_max",     "<=", thresholds.get("median_words_max", 10000),      agg["median_words"])

    overall_pass = all(c["pass"] for c in checks)
    return {"checks": checks, "overall_pass": overall_pass,
            "n_passed": sum(c["pass"] for c in checks), "n_total": len(checks),
            "is_partial": is_partial}
