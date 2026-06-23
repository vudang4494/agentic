"""Citation-grounding verifier with cosine pre-filter + batch LLM judge.

Architecture (Stage A latency optimization):
  1. Cosine pre-filter: auto-resolve obvious supports (cos ≥ 0.75) and
     unrelated (cos ≤ 0.30) using bge-m3 embedding similarity. Zero LLM calls.
  2. Batch judge: remaining borderline citations → 1 LLM call per section
     (vs. 1 call per citation in the original sequential design).
  3. Impact: ~67% citations auto-resolved (no LLM); judge calls 1050 → ~350/run.

Result schema is identical to the original verify_section() so callers need zero changes.
"""
import json
import math
import re
from typing import Dict, List, Tuple

import httpx

from .embeddings import embed as _embed, cosine as _cosine
from .types import Source
from .config import JUDGE_MODEL

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = JUDGE_MODEL
DEFAULT_TIMEOUT = 60.0
MAX_CLAIM_CHARS = 600
MAX_EVIDENCE_CHARS = 1200

# Cosine thresholds for auto-resolution (conservative — reduces false-positive risk).
# cos ≥ AUTO_SUPPORT_COS  → verdict = "supports"     (no LLM call)
# cos ≤ AUTO_UNRELATED_COS → verdict = "unrelated"  (no LLM call)
# 0.30 < cos < 0.75       → QUEUE for LLM judge
AUTO_SUPPORT_COS = 0.75
AUTO_UNRELATED_COS = 0.30
EMBED_MODEL = "bge-m3:latest"

# Map verdicts to numeric grounding scores (0..1).
_VERDICT_SCORE = {
    "supports":    1.0,
    "partial":     0.5,
    "no_evidence": 0.3,    # missing data, not the writer's fault
    "unrelated":   0.0,
    "contradicts": 0.0,
}

# ---- Per-citation judge (original, kept for fallback / single-cite edge cases) ----
JUDGE_SYS = (
    "You are a strict citation-grounding judge. You receive a claim from a research book "
    "and the EVIDENCE that the author cites in support of that claim. Decide whether the "
    "evidence actually supports the claim and respond with ONLY a single JSON object on "
    "one line, no prose, no markdown fences:\n"
    '{"verdict":"supports|partial|contradicts|unrelated|no_evidence","reason":"<one short sentence>"}\n'
    "Verdicts:\n"
    "  supports     -- evidence clearly states or implies the claim\n"
    "  partial      -- evidence supports part of the claim but not all\n"
    "  contradicts  -- evidence says the opposite\n"
    "  unrelated    -- evidence is on a different topic\n"
    "  no_evidence  -- the evidence text is empty or too generic to judge\n"
    "Judge by MEANING, not exact wording: mark 'supports' when the evidence states, implies, or "
    "faithfully paraphrases the claim -- the surface words may differ. Reserve 'partial' for a "
    "claim only half-covered by the evidence, 'unrelated' for evidence about a different subject, "
    "and 'contradicts' for evidence asserting the opposite. Topical overlap with no claim-level "
    "support is NOT 'supports'. A faithful paraphrase IS 'supports'."
)

# ---- Batch judge (Stage A: N citations → 1 call) ----
JUDGE_BATCH_SYS = (
    "You are a strict citation-grounding judge. For each claim-evidence pair below, "
    "decide whether the evidence supports the claim. "
    "Return ONLY a JSON array (no prose, no markdown fences), one object per citation in order:\n"
    '[{"verdict":"supports|partial|contradicts|unrelated|no_evidence","reason":"<one short sentence>"}]\n'
    "Verdicts:\n"
    "  supports     -- evidence clearly states or implies the claim\n"
    "  partial      -- evidence supports part of the claim but not all\n"
    "  contradicts  -- evidence says the opposite\n"
    "  unrelated    -- evidence is on a different topic\n"
    "  no_evidence  -- the evidence text is empty or too generic to judge\n"
    "Judge by MEANING, not exact wording: mark 'supports' when the evidence states, implies, or "
    "faithfully paraphrases the claim -- the surface words may differ. Reserve 'partial' for a "
    "claim only half-covered by the evidence, 'unrelated' for evidence about a different subject, "
    "and 'contradicts' for evidence asserting the opposite. Topical overlap with no claim-level "
    "support is NOT 'supports'. A faithful paraphrase IS 'supports'."
)

_THINK_RE = re.compile(r"<THINK>.*?</THINK>", re.DOTALL | re.IGNORECASE)
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{[^\[\]]*\}\s*,?\s*)+\]", re.DOTALL)


# ============================================================================
# Helpers
# ============================================================================

def _strip_think(s: str) -> str:
    return _THINK_RE.sub("", s or "").strip()


def _extract_claim(text: str, marker_pos: int, window_chars: int = 400) -> str:
    """Pull the sentence(s) leading up to a `[N]` marker, up to ~window_chars."""
    start = max(0, marker_pos - window_chars)
    chunk = text[start:marker_pos]
    m = re.search(r"[.!?]\s+(?=[A-Z(\"'])", chunk)
    if m:
        chunk = chunk[m.end():]
    nn = chunk.rfind("\n\n")
    if nn >= 0:
        chunk = chunk[nn + 2:]
    return chunk.strip()[:MAX_CLAIM_CHARS]


# ============================================================================
# Cosine pre-filter (Stage A, Tầng 2)
# ============================================================================

def _cosine_prefilter(
    citations: List[Tuple[int, str, Source]]
) -> Tuple[List[Tuple[int, str, Source, float]], List[Tuple[int, str, Source]]]:
    """Split citations into (auto_resolved, queued_for_llm) based on cosine similarity.

    Auto-resolves:
      - cos ≥ 0.75 → "supports"
      - cos ≤ 0.30 → "unrelated"
      - 0.30 < cos < 0.75 → queued for LLM judge

    Returns (auto_resolved, queued) where each item in auto_resolved is
    (cite_num, claim, source, cosine_score) and queued is (cite_num, claim, source).
    """
    # Build texts for batch embedding: [claim, source_excerpt] per citation
    texts, citation_data = [], []
    for cite_num, claim, source in citations:
        evidence = (source.excerpt or "")[:MAX_EVIDENCE_CHARS]
        texts.extend([claim, evidence])
        citation_data.append((cite_num, claim, source))

    vectors = _embed(texts, model=EMBED_MODEL)
    if not vectors or len(vectors) < len(texts):
        # Embedding failed — queue everything for LLM
        return [], citations

    auto_resolved, queued = [], []
    for i, (cite_num, claim, source) in enumerate(citation_data):
        claim_vec = vectors[i * 2]
        ev_vec = vectors[i * 2 + 1]
        cos = _cosine(claim_vec, ev_vec)
        if cos >= AUTO_SUPPORT_COS:
            auto_resolved.append((cite_num, claim, source, cos))
        elif cos <= AUTO_UNRELATED_COS:
            auto_resolved.append((cite_num, claim, source, cos))
        else:
            queued.append((cite_num, claim, source))

    return auto_resolved, queued


def _build_batch_prompt(
    queued: List[Tuple[int, str, Source]]
) -> str:
    """Render a batch judge prompt from a list of queued (cite_num, claim, source)."""
    lines = []
    for i, (cite_num, claim, source) in enumerate(queued, start=1):
        evidence = (source.excerpt or "")[:MAX_EVIDENCE_CHARS]
        lines.append(f"CITATION [{cite_num}]:")
        lines.append(f"Claim: {claim}")
        lines.append(f"Evidence: {evidence}")
        lines.append("")
    return "\n".join(lines)


def _judge_batch(
    client: httpx.Client,
    model: str,
    queued: List[Tuple[int, str, Source]],
    num_predict: int,
) -> List[Dict]:
    """Call the judge model once for all queued citations. Returns list of verdict dicts."""
    prompt = _build_batch_prompt(queued)
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": JUDGE_BATCH_SYS},
            {"role": "user",   "content": prompt},
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "top_p": 0.9,
        },
        "think": False,
    }
    try:
        r = client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        raw = _strip_think((r.json().get("message") or {}).get("content", ""))
    except Exception as e:
        # Fallback: mark all queued as no_evidence
        return [{"verdict": "no_evidence", "reason": f"batch judge call failed: {e}", "_skipped": True}
                 for _ in queued]

    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        return [{"verdict": "no_evidence", "reason": "batch judge returned non-JSON array", "_skipped": True}
                for _ in queued]
    try:
        data = json.loads(m.group(0))
    except Exception:
        return [{"verdict": "no_evidence", "reason": "batch judge JSON parse failed", "_skipped": True}
                for _ in queued]

    results = []
    for item in data:
        if not isinstance(item, dict):
            results.append({"verdict": "no_evidence", "reason": "malformed item in batch response"})
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in _VERDICT_SCORE:
            verdict = "no_evidence"
        results.append({
            "verdict": verdict,
            "reason": str(item.get("reason", ""))[:200],
        })
    # FAIL-CLOSED: a truncated / short JSON array (model returned fewer objects than queued) must
    # NOT let the unreturned citations skip the G2 gate. Averaging over only the returned scores
    # would inflate cite_precision and let an unsupported section pass. Pad missing verdicts to
    # no_evidence so EVERY queued citation contributes a score; ignore any extras.
    while len(results) < len(queued):
        results.append({"verdict": "no_evidence",
                        "reason": "batch judge returned fewer items than queued", "_skipped": True})
    return results[:len(queued)]


# ============================================================================
# Main entry point
# ============================================================================

def verify_section(
    content: str,
    sources: List[Source],
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict:
    """Verify every `[N]` citation in content against its source.

    Two-tier strategy (Stage A):
      1. Cosine pre-filter: auto-resolve obvious supports (cos ≥ 0.75) and
         unrelated (cos ≤ 0.30) using bge-m3. Zero LLM calls.
      2. Batch judge: remaining borderline citations → 1 LLM call per section.

    Returns identical schema to the original implementation:
        {
          "grounding": float in [0,1]  -- average score across all citations,
          "n_citations": int,
          "verdicts": [{"n": int, "claim": str, "verdict": str, "reason": str}, ...],
          "weak_citations": [n for n where verdict is unrelated/contradicts],
          "weak_summary": str  -- aggregated reason string for re-research feedback
        }

    No-citation sections WITH sources score grounding=0.0 (writer dropped all citations);
    only sections with neither [N] markers nor sources score 1.0.
    """
    # ---- Step 1: extract all [N] markers ----
    markers: List[Tuple[int, int]] = []
    for m in re.finditer(r"\[(\d+)\]", content):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        markers.append((n, m.start()))

    if not markers:
        if sources:
            return {
                "grounding": 0.0,
                "n_citations": 0,
                "verdicts": [],
                "weak_citations": [],
                "weak_summary": (
                    "section dropped ALL citations despite evidence being provided -- "
                    "writer must use [N] markers anchored to specific factual claims"
                ),
            }
        return {"grounding": 1.0, "n_citations": 0, "verdicts": [],
                "weak_citations": [], "weak_summary": ""}

    # ---- Step 2: build citation data (claim + source) ----
    citations: List[Tuple[int, str, Source]] = []
    for n, pos in markers:
        if not (1 <= n <= len(sources)):
            # Out-of-range citation — auto-unrelated, no LLM needed
            continue
        claim = _extract_claim(content, pos)
        citations.append((n, claim, sources[n - 1]))

    # Also handle any out-of-range citations
    out_of_range = [(n, pos) for n, pos in markers if not (1 <= n <= len(sources))]

    # ---- Step 3: cosine pre-filter ----
    auto_resolved, queued = _cosine_prefilter(citations)

    n_auto = len(auto_resolved)
    n_queued = len(queued)
    if n_auto > 0:
        supported = sum(1 for _, _, _, c in auto_resolved if c >= AUTO_SUPPORT_COS)
        print(f"  [VERIFY cosine] auto: {supported}/{n_auto} supports, "
              f"{n_auto - supported}/{n_auto} unrelated; queued: {n_queued} for LLM",
              flush=True)

    # ---- Step 4: batch judge for queued ----
    verdicts: List[Dict] = []
    scores: List[float] = []

    # Add out-of-range as auto-unrelated
    for n, _ in out_of_range:
        verdicts.append({"n": n, "claim": "", "verdict": "unrelated",
                         "reason": f"citation [{n}] but only {len(sources)} sources available"})
        scores.append(0.0)

    # Add cosine-auto verdicts
    for cite_num, claim, source, cos in auto_resolved:
        if cos >= AUTO_SUPPORT_COS:
            verdict = "supports"
            reason = f"cosine-auto (cos={cos:.3f} ≥ {AUTO_SUPPORT_COS})"
        else:
            verdict = "unrelated"
            reason = f"cosine-auto (cos={cos:.3f} ≤ {AUTO_UNRELATED_COS})"
        verdicts.append({"n": cite_num, "claim": claim[:200], "verdict": verdict,
                         "reason": reason, "_cosine_auto": True})
        scores.append(_VERDICT_SCORE[verdict])

    # Add LLM-judged verdicts
    if queued:
        # num_predict scales with citation count: ~80 tokens/verdict + overhead
        num_predict = max(300, min(400 * len(queued) + 150, 4000))
        with httpx.Client(timeout=timeout) as client:
            results = _judge_batch(client, model, queued, num_predict)

        for (cite_num, claim, source), result in zip(queued, results):
            verdicts.append({"n": cite_num, "claim": claim[:200], **result})
            scores.append(_VERDICT_SCORE.get(result.get("verdict", "no_evidence"), 0.0))

    # Sort verdicts by citation number for consistent output
    verdicts.sort(key=lambda v: v["n"])

    grounding = sum(scores) / max(len(scores), 1)
    weak = [v["n"] for v in verdicts if v["verdict"] in ("unrelated", "contradicts")]
    weak_reasons = [v["reason"] for v in verdicts
                    if v["verdict"] in ("unrelated", "contradicts") and v.get("reason")]
    weak_summary = "; ".join(weak_reasons[:3])

    return {
        "grounding": round(grounding, 3),
        "n_citations": len(markers),
        "verdicts": verdicts,
        "weak_citations": weak,
        "weak_summary": weak_summary,
    }


def topic_relevance_check(
    section_title: str,
    goal: str,
    must_cover_terms: List[str],
    avoid_terms: List[str],
    content: str,
    prior_sections: List[dict] = None,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    llm_call_fn=None,
) -> Dict:
    """Check whether a section answers its intended goal and avoids topic drift.

    G4: when `llm_call_fn` (the product's LOCAL gemma judge via Ollama) is supplied,
    the quantized term-count heuristic is blended with a real reverse-question
    relevance judge (answer_relevance). LOCAL MODEL ONLY -- never Claude/external;
    falls back to the pure heuristic if llm_call_fn is None or fails.
    """
    prior_sections = prior_sections or []
    text = content or ""
    lowered = text.lower()
    must_cover_terms = [t for t in must_cover_terms if t]
    avoid_terms = [t for t in avoid_terms if t]
    missing_terms = [t for t in must_cover_terms if t.lower() not in lowered]
    drift_terms = [t for t in avoid_terms if t.lower() in lowered]

    overlap_warning = False
    if prior_sections and text:
        current_words = set(re.findall(r"[A-Za-z0-9]+", lowered))
        for ps in prior_sections[-3:]:
            prev = (ps.get("content", "") or "").lower()
            prev_words = set(re.findall(r"[A-Za-z0-9]+", prev))
            if not prev_words:
                continue
            overlap = len(current_words & prev_words) / max(1, len(current_words | prev_words))
            if overlap >= 0.55:
                overlap_warning = True
                break

    score = 1.0
    if must_cover_terms:
        score -= 0.5 * (len(missing_terms) / max(1, len(must_cover_terms)))
    if drift_terms:
        score -= min(0.3, 0.1 * len(drift_terms))
    if overlap_warning:
        score -= 0.2
    if len(text.split()) < 120:
        score -= 0.1
    heuristic_score = max(0.0, min(1.0, score))

    # G4: refine the quantized term-count heuristic with a real semantic judge.
    # answer_relevance() asks the LOCAL gemma judge "does the body answer the goal?" --
    # catches well-grounded prose that drifts off the question, which the term-count
    # heuristic (quantizes to {0.5,0.75,1.0}) cannot see. Local Ollama model only.
    ar_score = None
    score = heuristic_score
    if llm_call_fn is not None:
        try:
            ar = answer_relevance(content, f"{section_title}. {goal}", llm_call_fn)
            if ar and ar.get("verdict") not in (None, "unknown"):
                ar_score = float(ar.get("score", 0.5))
                score = 0.6 * ar_score + 0.4 * heuristic_score
        except Exception:
            score = heuristic_score
    score = max(0.0, min(1.0, score))

    # StageE protection: the judge ALONE must not push a fully term-covered, non-drifting
    # section below the 0.50 hard-block line (avoid false topic-drift blocks).
    if score < 0.50 and not missing_terms and not drift_terms:
        score = 0.50

    verdict = "relevant"
    if score < 0.50:
        verdict = "off_topic"
    elif score < 0.70:
        verdict = "partial"

    reason_parts = []
    if missing_terms:
        reason_parts.append("missing: " + ", ".join(missing_terms[:4]))
    if drift_terms:
        reason_parts.append("drift: " + ", ".join(drift_terms[:4]))
    if overlap_warning:
        reason_parts.append("overlaps prior sections")
    if not reason_parts:
        reason_parts.append("goal and topic alignment look acceptable")

    return {
        "topic_relevance": round(score, 3),
        "heuristic_score": round(heuristic_score, 3),
        "answer_relevance_score": ar_score,
        "verdict": verdict,
        "missing_terms": missing_terms,
        "drift_terms": drift_terms,
        "overlap_warning": overlap_warning,
        "reason": "; ".join(reason_parts),
    }


# ============================================================================
# GATE-6: Cross-Reference Verification
# ============================================================================

def verify_cross_references_v2(
    section_content: str,
    prior_sections: List[dict],
    min_refs: int = 2,
) -> dict:
    """Verify that section references prior sections by title.

    RULES Stage B: "Cac chapter co dang noi nhung dieu khac nhau that khong"
    GATE-6: Cross-references >= 2 per section for book coherence.

    Returns:
        {
            "pass": bool,
            "refs_found": int,
            "ref_titles": [str],
            "orphan": bool,
            "reason": str,
        }
    """
    if not prior_sections:
        # First section has no prior to reference - this is OK
        return {
            "pass": True,
            "refs_found": 0,
            "ref_titles": [],
            "orphan": False,
            "reason": "First section - no prior sections to reference",
        }

    content_lower = section_content.lower()
    prior_titles = []
    for ps in prior_sections:
        title = ps.get("title", "")
        if title:
            prior_titles.append(title)
            # Also check normalized form (without chapter prefix)
            normalized = re.sub(r"^\d+\.\d+\s+", "", title).strip()
            if normalized != title:
                prior_titles.append(normalized)

    # Count how many prior titles appear in content
    refs_found = 0
    ref_titles = []
    for title in prior_titles:
        title_lower = title.lower()
        # Check if title (or significant words from it) appear in content
        # Use word-based matching for partial title matches
        words = [w for w in re.findall(r"[A-Za-z0-9]+", title_lower) if len(w) > 3]
        if len(words) >= 2:
            # Require at least 2 significant words to match
            match_count = sum(1 for w in words if w in content_lower)
            if match_count >= min(2, len(words)):
                refs_found += 1
                ref_titles.append(title)
        elif title_lower in content_lower:
            refs_found += 1
            ref_titles.append(title)

    # Deduplicate ref_titles
    ref_titles = list(dict.fromkeys(ref_titles))

    pass_result = refs_found >= min_refs
    orphan = refs_found == 0 and len(prior_sections) > 0

    return {
        "pass": pass_result,
        "refs_found": refs_found,
        "ref_titles": ref_titles,
        "orphan": orphan,
        "reason": (
            f"Found {refs_found} cross-references"
            + (f": {', '.join(ref_titles[:3])}" if ref_titles else "")
            + (". First section - OK." if not prior_sections else "")
            + (" ❌ ORPHAN - no references to prior sections!" if orphan else "")
        ),
    }


# ============================================================================
# Self-RAG-lite (pure regex, no LLM calls — kept unchanged)
# ============================================================================

def _drop_citation_in_sentence(text: str, marker_pos: int) -> str:
    """Remove a [N] citation marker from within its sentence."""
    before = text[:marker_pos]
    after = text[marker_pos:]

    sentence_start = max(0, len(before) - 300)
    last_period = before.rfind(".")
    if last_period >= sentence_start:
        sentence_start = last_period + 1
    else:
        last_newline = before.rfind("\n")
        if last_newline >= 0:
            sentence_start = last_newline + 1

    end_paren = re.search(r"([.!?])\s+", after)
    if end_paren:
        sentence_end = marker_pos + end_paren.end()
    else:
        sentence_end = len(text)

    sentence = text[sentence_start:sentence_end]
    cleaned_sent = re.sub(r"\s*\[[\d,\s]+\]\s*", " ", sentence)
    cleaned_sent = re.sub(r"\s{2,}", " ", cleaned_sent).strip()

    content_words = [w for w in cleaned_sent.split() if len(w) > 3]
    if len(content_words) < 4:
        return (text[:sentence_start] + f" {cleaned_sent}." + text[sentence_end:]).strip()

    return text[:sentence_start] + cleaned_sent + text[sentence_end:]


_HEDGE_PATTERNS = [
    re.compile(r"according to\s+\[[\d,\s]+\]", re.IGNORECASE),
    re.compile(r"(?:research|studies|papers)\s+\[[\d,\s]+\]\s+", re.IGNORECASE),
    re.compile(r"(?:recent|work)\s+\[[\d,\s]+\]", re.IGNORECASE),
]


def scrub_unsupported_citations(content: str, sources: List[Source],
                                verify_res: Dict) -> tuple:
    """Self-RAG-lite: remove unsupported citations from content.

    Drops [N] markers for citations rated 'unrelated' or 'no_evidence'.
    Cosine-auto 'supports' verdicts are trusted and NOT scrubbed.
    """
    if not verify_res or not verify_res.get("verdicts"):
        return content, 0, ""

    verdicts = verify_res["verdicts"]
    bad_nums: set = set()
    for v in verdicts:
        # cosine-auto 'supports' verdicts are trusted
        if v.get("_cosine_auto") and v["verdict"] == "supports":
            continue
        if v["verdict"] in ("unrelated", "no_evidence"):
            bad_nums.add(v["n"])

    if not bad_nums:
        return content, 0, ""

    cleaned = content
    for n in sorted(bad_nums):
        cleaned = re.sub(r"\[\s*" + str(n) + r"\s*\]", "", cleaned)
        cleaned = re.sub(r"\[\s*[\d,\s]+" + str(n) + r"[\d,\s]*\]\s*", "", cleaned)

    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)

    n_dropped = len(bad_nums)
    total_citations = len(verdicts)
    abstain_note = ""
    if n_dropped >= total_citations * 0.8:
        abstain_note = (
            "\n\n> **Note:** This section's citations could not be verified against "
            "available sources. Claims should be treated as unverified until confirmed.\n"
        )

    return cleaned, n_dropped, abstain_note


# ============================================================================
# v2: CRAG 3-way decision + answer relevance guard
# ============================================================================

def crag_decision(grounding: float, round_idx: int,
                  upper: float = 0.80, lower: float = 0.40,
                  max_rounds: int = 2) -> str:
    """CRAG 3-way decision after HHEM grounding check.

    Args:
        grounding: HHEM grounding score [0, 1].
        round_idx: Current research round (0-indexed).
        upper: Accept threshold. Default 0.80.
        lower: Incorrect threshold. Default 0.40.
        max_rounds: Maximum research rounds. Default 2.

    Returns:
        "accept" | "ambiguous" | "incorrect"

    - accept: grounding >= upper → pass through
    - incorrect: grounding <= lower OR out of rounds → discard + re-search
    - ambiguous: lower < grounding < upper → blend: add search + rewrite weak claims
    """
    if round_idx >= max_rounds:
        return "accept"   # out of rounds → accept to avoid infinite loop

    if grounding >= upper:
        return "accept"

    if grounding <= lower:
        return "incorrect"

    return "ambiguous"


def answer_relevance(section_body: str, sub_query: str,
                     llm_call_fn=None) -> dict:
    """Reverse-question relevancy: does the section body actually answer sub_query?

    Guard against topic drift: WRT may write well-grounded prose that sidesteps
    the actual question. This catches that.

    Args:
        section_body: The markdown section body.
        sub_query: The original section prompt / question.
        llm_call_fn: LLM callable. If None, returns neutral score.

    Returns:
        dict: {score: float [0,1], verdict: str, reason: str}
    """
    if not llm_call_fn:
        return {"score": 0.5, "verdict": "unknown", "reason": "no LLM available"}

    prompt = (
        "Ban la mot truong kiem tra chat che. Doc cau hoi va cau tra loi.\n"
        "Quyet dinh cau tra loi co dap ung cau hoi hay khong.\n"
        "Chi tra ve mot object JSON tren 1 dong, khong co markdown fences:\n"
        '{"score": <0-1>, "verdict": "relevant|partial|off_topic", "reason": "<1 cau ngan thuyet minh>"}"\n\n'
        f"Cau hoi: {sub_query}\n\n"
        f"Cau tra loi:\n{section_body[:3000]}"
    )
    try:
        result = llm_call_fn(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        import json as _json2
        # Try to extract JSON
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            data = _json2.loads(m.group(0))
            return {
                "score": float(data.get("score", 0.5)),
                "verdict": str(data.get("verdict", "unknown")),
                "reason": str(data.get("reason", "")),
            }
    except Exception:
        pass

    return {"score": 0.5, "verdict": "unknown", "reason": "parse failed"}


def strip_refine(section_body: str, sources: list,
                  llm_call_fn=None) -> str:
    """Optional: strip-refine on ACCEPT path.

    Decompose top-8 into knowledge strips, rerank each strip with RRK,
    keep only relevant strips. CRAG decompose-then-recompose pattern.

    Args:
        section_body: The markdown body.
        sources: Top-8 sources.
        llm_call_fn: LLM callable for strip extraction.

    Returns:
        Refined body string (unchanged if llm_call_fn is None).
    """
    if not llm_call_fn:
        return section_body

    # Step 1: decompose body into strips (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", section_body)
                  if p.strip() and len(p.strip()) > 100]

    if not paragraphs:
        return section_body

    # Step 2: score each strip's relevance to the section title / first sentence
    # Use the first heading or first line as query
    first_line = section_body.strip().split("\n")[0][:200]

    try:
        from . import rerank as _rerank_v2
        scored = _rerank_v2.rerank(first_line, [{"id": f"s{i}", "text": p} for i, p in enumerate(paragraphs)],
                                    top_k=len(paragraphs))
        # Keep strips with score >= RELEVANCE_FLOOR
        kept = [p for p, d in zip(paragraphs, scored) if d.get("rerank_score", 0) >= 0.25]
        if kept and len(kept) < len(paragraphs):
            print(f"  [STRIP-REFINE] kept {len(kept)}/{len(paragraphs)} strips", flush=True)
            return "\n\n".join(kept)
    except Exception:
        pass

    return section_body


# ============================================================================
# v2: Main entry point using HHEM + CRAG
# ============================================================================

def verify_section_v2(
    content: str,
    sources: list,
    section_prompt: str = "",
    grounding_result: dict = None,
    round_idx: int = 0,
    max_rounds: int = 2,
    llm_call_fn=None,
) -> dict:
    """v2 verify: HHEM grounding + CRAG decision.

    Tier 1 (RRK) is handled by research.rerank BEFORE this is called.
    This function handles Tier 2 (HHEM) and Tier 3 (CRAG decision).

    Args:
        content: Section markdown body.
        sources: Top-8 sources (already reranked).
        section_prompt: Original section prompt (for answer_relevance guard).
        grounding_result: Pre-computed from faithfulness.grounding_score().
                           If None, computes inline.
        round_idx: Current research round (0-indexed).
        max_rounds: Max research rounds.
        llm_call_fn: LLM callable for answer_relevance and strip_refine.

    Returns:
        dict with keys:
          verify_version: "v2"
          crag_decision: "accept" | "ambiguous" | "incorrect"
          grounding: float [0,1]
          n_supported / n_partial / n_unsupported: int
          n_claims: int
          per_claim: list of (claim, score, verdict)
          answer_relevance: dict (from answer_relevance())
          weak_summary: str
          weak_citations: list (from per_claim unsupported)
          cite_precision: float [0,1] (# supported claims / # total claims)
    """
    # Step 1: compute grounding if not provided
    if grounding_result is None:
        try:
            from . import faithfulness as _f
            claims = _f.decompose_claims(content, llm_call_fn)
            grounding_result = _f.grounding_score(claims, sources)
        except ImportError:
            # Graceful fallback: use v1 verify
            return verify_section(content, sources)

    grounding = grounding_result.get("grounding", 0.0)

    # Step 2: CRAG decision
    decision = crag_decision(grounding, round_idx, max_rounds=max_rounds)

    # Step 3: answer relevance guard (if section_prompt provided)
    answer_rel = {}
    if section_prompt and llm_call_fn:
        answer_rel = answer_relevance(content, section_prompt, llm_call_fn)
        if answer_rel.get("score", 1.0) < 0.4:
            print(f"  [ANSWER-RELEVANCE] LOW score={answer_rel['score']:.3f} "
                  f"({answer_rel.get('reason', '')})", flush=True)

    # Step 4: strip-refine on accept path
    refined_content = content
    if decision == "accept" and llm_call_fn:
        refined_content = strip_refine(content, sources, llm_call_fn)

    # Step 5: build weak_citations from per_claim
    per_claim = grounding_result.get("per_claim", [])
    weak_citations = [
        claim for claim, score, verdict in per_claim
        if verdict == "unsupported"
    ]

    weak_summary_parts = [
        f"[{verdict}] {claim[:80]}"
        for claim, score, verdict in per_claim
        if verdict in ("unsupported", "partial")
    ][:3]
    weak_summary = "; ".join(weak_summary_parts)

    # Citation precision: ratio of supported claims to total claims
    n_claims = grounding_result.get("n_claims", len(per_claim))
    n_supported = grounding_result.get("n_supported", 0)
    cite_precision = n_supported / n_claims if n_claims > 0 else 0.0

    return {
        "verify_version": "v2",
        "crag_decision": decision,
        "grounding": round(grounding, 4),
        "n_supported": grounding_result.get("n_supported", 0),
        "n_partial": grounding_result.get("n_partial", 0),
        "n_unsupported": grounding_result.get("n_unsupported", 0),
        "n_claims": n_claims,
        "per_claim": per_claim,
        "cite_precision": round(cite_precision, 4),
        "answer_relevance": answer_rel,
        "weak_summary": weak_summary,
        "weak_citations": weak_citations,
        "content_refined": refined_content if refined_content != content else None,
    }

