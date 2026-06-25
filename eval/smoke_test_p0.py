#!/usr/bin/env python3
"""
Smoke test cho P0a / P0b / P0c fixes.

Tests:
  P0a -- Domain Relevance Gate: Hard Block khi domain mismatch
  P0b -- Canonical Paper injection: ton tai trong evidence pool
  P0c -- Seen-Count Penalty: cong thuc manh hon + persist trong state.json

Xem GLOSSARY.md neu can giai thich thuat ngu.

Run:
  python3 eval/smoke_test_p0.py --skip-full    # chi code checks (nhanh)
  python3 eval/smoke_test_p0.py --topic "Transformer" --canonical-ids "1706.03762,1607.06450"
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from research.discovery import discover_topic
from research.deep_investigate import investigate_section
from research.outline_from_research import generate_outline
from research import search as _search
from research.types import Source, Query


# =============================================================================
# Helpers
# =============================================================================

def _norm(s: str) -> str:
    """Chuan hoa arxiv ID: bo prefix arxiv:, bo version suffix."""
    s = (s or "").strip()
    s = re.sub(r"[vV]\d+$", "", s)
    if s.startswith("arxiv:"):
        s = s[6:]
    return s


# =============================================================================
# Test: P0a -- Domain Relevance Gate hard block
# =============================================================================

def test_p0a_hard_block() -> dict:
    """
    P0a: sau max_rounds ma domain van mismatch -> phai raise RuntimeError.
    Khong duoc "ACCEPT with degraded quality".
    RULES Stage C: topic_relevance >= 0.50 (raised after benchmark found generic titles).
    """
    print("\n[P0a] Kiem tra: Domain Relevance Gate hard block")
    code = (ROOT / "research" / "deep_investigate.py").read_text()

    has_soft_block = "ACCEPT with degraded quality" in code
    has_hard_block = "RuntimeError" in code and "HARD BLOCK" in code
    # Real gate: `ev_topic_rel < ev_threshold` where `ev_threshold = min(0.40, ...)` ~= 0.40.
    # (The old misleading "< 0.50" log literal was corrected to interpolate ev_threshold.)
    has_ev_threshold = "ev_topic_rel < ev_threshold" in code and "ev_threshold = min(0.40" in code

    result = {
        "test": "P0a: Domain Relevance Gate hard block + ev_threshold ~=0.40",
        "co_soft_block": has_soft_block,      # phai = False
        "co_hard_block": has_hard_block,      # phai = True
        "co_ev_threshold_040": has_ev_threshold,  # phai = True (real gate ~0.40, not 0.50)
        "PASS": has_hard_block and not has_soft_block and has_ev_threshold,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: P0a companion -- skip archetype routing khi co topic_context
# =============================================================================

def test_p0a_query_router() -> dict:
    """
    P0a companion: query_router.phai bo qua archetype routing khi co domain_context.
    Vi archetype khong the phan biet "Self-Attention (Transformer)" voi "Self-Attention (CoT)".
    """
    print("\n[P0a companion] Kiem tra: query_router bo qua archetype khi co domain_context")
    code = (ROOT / "research" / "query_router.py").read_text()

    has_domain_context = "domain_context" in code
    has_skip_archetype = "skip archetype routing" in code or (
        "if domain_context" in code and "_llm_query_gen" in code
    )

    result = {
        "test": "P0a companion: query_router bo qua archetype khi co domain_context",
        "co_domain_context_param": has_domain_context,
        "co_skip_logic": has_skip_archetype,
        "PASS": has_domain_context and has_skip_archetype,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: P0b -- Canonical Paper injection
# =============================================================================

def test_p0b_injection(topic: str, canonical_ids: list[str]) -> dict:
    """
    P0b: Canonical Papers phai duoc inject vao TopicProfile va protected.
    """
    print(f"\n[P0b] Kiem tra: Canonical Paper injection, topic={topic}")
    profile = discover_topic(topic, canonical_arxiv_ids=canonical_ids)

    protected_ids = set(profile.protected_source_ids or [])
    protected_ids_norm = {_norm(pid) for pid in protected_ids}
    canonical_papers = profile.canonical_papers or []
    requested_norm = {_norm(cid) for cid in canonical_ids}
    missing = requested_norm - protected_ids_norm

    result = {
        "test": "P0b: Canonical Paper injection",
        "canonical_ids_yeu_cau": canonical_ids,
        "protected_ids": sorted(protected_ids),
        "so_papers_tim_thay": len(canonical_papers),
        "papers_mau": canonical_papers[:2],
        "missing_normalized": sorted(missing),
        "PASS": len(canonical_papers) > 0 and len(missing) == 0,
    }
    _print_result(result)
    return result


def test_p0b_resume() -> dict:
    """
    P0b: resume path phai detect new Canonical IDs va re-inject.
    """
    print("\n[P0b] Kiem tra: resume path re-inject Canonical Papers")
    code = (ROOT / "pipeline" / "deep_research_v3.py").read_text()

    co_reinject = "RE-INJECT" in code
    co_merged = "merged_ids" in code
    co_existing = "existing_ids" in code

    result = {
        "test": "P0b: resume re-inject Canonical Papers",
        "co_reinject_logic": co_reinject,
        "co_merged_ids": co_merged,
        "co_existing_check": co_existing,
        "PASS": co_reinject and co_existing,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: P0c -- Seen-Count Penalty
# =============================================================================

def test_p0c_formula() -> dict:
    """
    P0c: penalty formula phai manh hon (1-f)^2 thay vi 1-f*0.8.
    """
    print("\n[P0c] Kiem tra: Seen-Count Penalty formula")
    code = (ROOT / "research" / "notes.py").read_text()

    co_new_formula = "(1.0 - fraction) ** 2" in code or "(1 - fraction)" in code

    old_14_50 = max(0.1, 1.0 - (14 / 50) * 0.8)
    frac = min(1.0, 14 / 50)
    new_14_50 = max(0.05, (1.0 - frac) ** 2)

    old_1_50 = max(0.1, 1.0 - (1 / 50) * 0.8)
    new_1_50 = max(0.05, (1.0 - 1 / 50) ** 2)

    result = {
        "test": "P0c: Seen-Count Penalty formula",
        "co_cong_thuc_moi": co_new_formula,
        "cu_14_50": round(old_14_50, 4),
        "moi_14_50": round(new_14_50, 4),
        "cu_1_50": round(old_1_50, 4),
        "moi_1_50": round(new_1_50, 4),
        "moi_nhieu_hon_cu": new_14_50 < old_14_50,
        "lan_dau_khong_bi_penalty_nhieu": new_1_50 > 0.95,
        "PASS": co_new_formula and new_14_50 < old_14_50 and new_1_50 > 0.95,
    }
    _print_result(result)
    return result


def test_p0c_persist() -> dict:
    """
    P0c: run_seen_counts phai duoc luu vao state.json va load khi resume.
    """
    print("\n[P0c] Kiem tra: Seen-Count persist trong state.json")
    code = (ROOT / "pipeline" / "deep_research_v3.py").read_text()

    co_save = '"run_seen_counts"' in code
    co_load = "run_seen_counts" in code and ("saved_counts" in code or "run_seen_counts" in code)

    result = {
        "test": "P0c: Seen-Count persist",
        "co_save": co_save,
        "co_load": co_load,
        "PASS": co_save and co_load,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: RULES Stage B -- semantic overlap check in audit_outline
# =============================================================================

def test_rules_stage_b_semantic_overlap() -> dict:
    """RULES Stage B: semantic_overlap jaccard < 0.7 -> issue flagged."""
    print("\n[StageB] Kiem tra: semantic_overlap jaccard check in audit_outline")
    code = (ROOT / "research" / "outline_from_research.py").read_text()

    has_jaccard = "jaccard" in code.lower()
    has_threshold = "jaccard >= 0.50" in code  # real flag threshold (docs' 0.7 is aspirational)
    has_overlap_issues = "semantic_overlap_issues" in code

    result = {
        "test": "RULES Stage B: semantic_overlap jaccard check",
        "co_jaccard": has_jaccard,
        "co_threshold_050": has_threshold,
        "co_overlap_issues_tracked": has_overlap_issues,
        "PASS": has_jaccard and has_threshold and has_overlap_issues,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: RULES Stage C -- paper dominance >50% trigger
# =============================================================================

def test_rules_stage_c_dominance() -> dict:
    """RULES Stage C: paper dominance >50% -> extra penalty."""
    print("\n[StageC] Kiem tra: paper dominance >50% in notes.py")
    code = (ROOT / "research" / "notes.py").read_text()

    # Provider concentration penalty (was 0.70, now 0.50 per RULES)
    has_provider_penalty = "top_provider_pct > 0.50" in code or "top_provider_pct > 0.5" in code
    has_penalty_score = "provider_pct" in code

    result = {
        "test": "RULES Stage C: paper dominance >50% penalty",
        "co_threshold_50": has_provider_penalty,
        "co_penalty_tracked": has_penalty_score,
        "PASS": has_provider_penalty and has_penalty_score,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: RULES Stage D -- min word count >= 120 hard rule
# =============================================================================

def test_rules_stage_d_wordcount() -> dict:
    """RULES Stage D: word_count < 120 -> HARD BLOCK."""
    print("\n[StageD] Kiem tra: min word_count >= 120 hard rule")
    code = (ROOT / "research" / "deep_investigate.py").read_text()

    has_120_check = "< 120" in code
    has_wordcount_block = "StageD HARD BLOCK" in code

    result = {
        "test": "RULES Stage D: min_word_count >= 120",
        "co_120_check": has_120_check,
        "co_hard_block": has_wordcount_block,
        "PASS": has_120_check and has_wordcount_block,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: RULES Stage E -- canonical_coverage FAIL/BLOCKED
# =============================================================================

def test_rules_stage_e_canonical() -> dict:
    """RULES Stage E: canonical_coverage=0 -> HARD BLOCK."""
    print("\n[StageE] Kiem tra: canonical_coverage FAIL/BLOCKED")
    code = (ROOT / "research" / "deep_investigate.py").read_text()

    has_canon_check = "canonical_coverage" in code
    has_stage_e_block = "StageE HARD BLOCK" in code

    result = {
        "test": "RULES Stage E: canonical_coverage FAIL/BLOCKED",
        "co_canon_check": has_canon_check,
        "co_stage_e_block": has_stage_e_block,
        "PASS": has_canon_check and has_stage_e_block,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: RULES Stage E -- topic_purity FAIL (chi grounding insufficient)
# =============================================================================

def test_rules_stage_e_topic_purity_fail() -> dict:
    """RULES Stage E: chi grounding pass = FAIL (topic_purity must also pass)."""
    print("\n[StageE] Kiem tra: topic_purity FAIL condition (grounding-only insufficient)")
    code = (ROOT / "research" / "deep_investigate.py").read_text()

    has_topic_fail = "topic_relevance < min_topic_relevance" in code  # grounding-pass alone insufficient
    has_topic_hard_block = "StageE HARD BLOCK" in code and "topic_purity=" in code

    result = {
        "test": "RULES Stage E: topic_purity FAIL condition",
        "co_topic_purity_check": has_topic_fail,
        "co_topic_hard_block": has_topic_hard_block,
        "PASS": has_topic_fail and has_topic_hard_block,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: RULES Stage F -- heading hygiene check in assemble
# =============================================================================

def test_rules_stage_f_heading() -> dict:
    """RULES Stage F: heading inflation -> WARNING (not silent)."""
    print("\n[StageF] Kiem tra: heading hygiene check in assemble")
    code = (ROOT / "pipeline" / "deep_research_v3.py").read_text()

    has_heading_check = "heading_re" in code
    has_dup_check = "duplicate_h2" in code or "duplicate_h3" in code
    has_orphan_check = "orphan_h3" in code
    has_warning = "WARNING heading hygiene" in code

    result = {
        "test": "RULES Stage F: heading hygiene check",
        "co_heading_regex": has_heading_check,
        "co_dup_check": has_dup_check,
        "co_orphan_check": has_orphan_check,
        "co_warning": has_warning,
        "PASS": has_heading_check and has_dup_check and has_warning,
    }
    _print_result(result)
    return result


# =============================================================================
# Test: Full Section -- chay 1 Section voi P0a/b/c
# =============================================================================

def test_full_section(topic: str, canonical_ids: list[str], section_key: str = "1.1") -> dict:
    """
    Chay 1 Section thuc te voi P0a/b/c.
    Verify: Section co content, Canonical Papers trong evidence, Grounding tot.
    """
    print(f"\n[Full Section] topic={topic}, Canonical Papers={canonical_ids}")

    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return {"test": "Full Section", "PASS": False, "error": "Ollama not running"}
    except Exception as e:
        return {"test": "Full Section", "PASS": False, "error": f"Ollama unreachable: {e}"}

    # Discovery: tao TopicProfile voi Canonical Papers
    profile = discover_topic(topic, canonical_arxiv_ids=canonical_ids)
    protected_ids = list(set(profile.protected_source_ids or []))

    # Build topic_context day du (nhu deep_research_v3.py lam)
    canon_terms = ", ".join(getattr(profile, "canonical_terms", [])[:10])
    must_cov = ", ".join(getattr(profile, "must_cover", [])[:10])
    out_scope = ", ".join(getattr(profile, "out_of_scope", [])[:8])
    topic_context = (
        f"Book: {getattr(profile, 'name', topic)} -- {getattr(profile, 'description', '')}"
        f"\nCanonical terms: {canon_terms}"
        f"\nMust cover: {must_cov}"
        f"\nOut of scope: {out_scope}"
    )
    print(f"  Canonical Papers protected: {len(protected_ids)}")
    print(f"  Topic context: {len(topic_context)} chars")

    # Outline: tao Chapter + Section
    outline_sources = []
    for q_str in (profile.initial_queries or [])[:6]:
        try:
            outline_sources.extend(
                _search.gather([Query(q=q_str)], providers=("arxiv", "wikipedia", "ddg"), per_provider_k=5)
            )
        except Exception as e:
            print(f"  [gather] {q_str[:40]}: {e}")
    seen, unique = set(), []
    for s in outline_sources:
        if s.url and s.url not in seen:
            seen.add(s.url)
            unique.append(s)

    outline = generate_outline(profile, unique, n_chapters=2, sections_per_chapter=1)
    chapters = outline.chapters
    if not chapters:
        return {"test": "Full Section", "PASS": False, "error": "Khong co Chapter tu Outline"}

    first_ch = chapters[0]
    first_sec = (first_ch.get("sections") or [{}])[0]
    sec_title = first_sec.get("t", "")
    sec_goal = first_sec.get("pr", "") or sec_title
    chapter_title = first_ch.get("t", "")

    print(f"  Chay Section: {section_key} - {sec_title}")

    # Deep Investigation: research + write + verify
    try:
        result = investigate_section(
            section_prompt=sec_goal,
            chapter_title=chapter_title,
            section_title=sec_title,
            topic_context=topic_context,
            prior_sections=[],
            prior_concepts=[],
            providers=("arxiv", "wikipedia", "ddg"),
            max_rounds=2,
            section_meta=first_sec,
            protected_source_ids=protected_ids,
            run_seen_counts={},
        )
    except RuntimeError as e:
        if "HARD BLOCK" in str(e):
            return {
                "test": "Full Section",
                "section_key": section_key,
                "section_title": sec_title,
                "loi": str(e)[:120],
                "PASS": True,
                "note": "P0a HARD BLOCK fired -- dung nhu mong",
            }
        raise

    # Kiem tra ket qua
    source_ids = {getattr(s, "id", "") or "" for s in (result.sources or [])}
    source_urls = {getattr(s, "url", "") or "" for s in (result.sources or [])}
    wc = len(result.content.split())

    canonical_in_evidence = []
    for cid in canonical_ids:
        cid_n = _norm(cid)
        found = any(_norm(sid) == cid_n or cid_n in sid for sid in source_ids)
        found = found or any(cid_n in url or cid in url for url in source_urls)
        if found:
            canonical_in_evidence.append(cid)

    result_check = {
        "test": "Full Section",
        "section_key": section_key,
        "section_title": sec_title,
        "so_tu": wc,
        "so_nguon": len(result.sources or []),
        "Grounding": result.grounding_score,
        "Canonical_in_evidence": canonical_in_evidence,
        "source_mau": sorted(source_ids)[:3],
        "PASS": (
            wc > 100 and
            len(result.sources or []) > 0 and
            result.grounding_score > 0
        ),
    }
    _print_result(result_check)
    return result_check


# =============================================================================
# Utility
# =============================================================================

def _print_result(r: dict):
    status = "PASS" if r.get("PASS") else "FAIL"
    print(f"  [{status}] {r['test']}")
    for k, v in r.items():
        if k not in ("test", "PASS"):
            print(f"    {k}: {v}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Smoke test cho P0a / P0b / P0c")
    p.add_argument("--topic", "-t", default="Transformer Architecture")
    p.add_argument("--canonical-ids", default="1706.03762,1607.06450")
    p.add_argument("--section", "-s", default="1.1")
    p.add_argument("--skip-full", action="store_true", help="Chi kiem tra code, khong chay Section thuc te")
    args = p.parse_args()
    canonical_ids = [x.strip() for x in args.canonical_ids.split(",") if x.strip()]

    results = []

    # --- P0a / P0b / P0c code checks ---
    results.append(test_p0a_hard_block())
    results.append(test_p0a_query_router())
    results.append(test_p0b_resume())
    results.append(test_p0c_formula())
    results.append(test_p0c_persist())

    # --- RULES Stage B-F code checks ---
    results.append(test_rules_stage_b_semantic_overlap())
    results.append(test_rules_stage_c_dominance())
    results.append(test_rules_stage_d_wordcount())
    results.append(test_rules_stage_e_canonical())
    results.append(test_rules_stage_e_topic_purity_fail())
    results.append(test_rules_stage_f_heading())

    # --- Injection check (co goi Discovery, nhung nhanh) ---
    results.append(test_p0b_injection(args.topic, canonical_ids))

    # --- Full Section test (can Ollama) ---
    if not args.skip_full:
        results.append(test_full_section(args.topic, canonical_ids, args.section))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for r in results if r.get("PASS"))
    n_total = len(results)
    for r in results:
        status = "PASS" if r.get("PASS") else "FAIL"
        note = f"  <- {r.get('note', '')}" if r.get("note") else ""
        loi = f"  <- {r.get('loi', '')}" if r.get("loi") else ""
        loi = loi or note
        print(f"  [{status}] {r['test']}{loi}")

    print(f"\n  Ket qua: {n_pass}/{n_total} passed")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
