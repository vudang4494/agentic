#!/usr/bin/env python3
"""Post-process book to fix all 4 known issues:
1. Markdown: `### ##` → `###` (writer model bug)
2. Verify: re-run v2 on all 24 fallback sections
3. Quality threshold: sync 0.45 → 0.55
4. Regenerate 2 sections with zero citations (4.6, 5.7)
"""
import json, re, time, sys
from pathlib import Path

sys.path.insert(0, "/Users/vudang/PythonLab/AgentDeepLearning/files")

# ---- Config ----
RUN_DIR = Path("/Users/vudang/PythonLab/AgentDeepLearning/output/runs/llm_trends_2026_2027")
STATE_PATH = RUN_DIR / "state.json"
MIN_GROUNDING = 0.55  # sync with pipeline's MIN_GROUNDING

# ---- 1. Fix markdown `### ##` → `###` ----
def fix_markdown(content: str) -> tuple:
    """Fix writer model markdown bug: `### ##` → `###`"""
    before = content
    # Fix ### ## pattern
    content = re.sub(r'^### ##\s+', '### ', content, flags=re.MULTILINE)
    # Fix ## ### pattern (wrong order)
    content = re.sub(r'^## ###\s+', '## ', content, flags=re.MULTILINE)
    # Fix any ## followed by ## (double header markers)
    content = re.sub(r'^(#{1,3}) \1\s+', r'\1 ', content, flags=re.MULTILINE)
    # Fix orphan #### without content (dangling level-4 headers)
    content = re.sub(r'^####\s*\n(?=\n|^#)', '', content, flags=re.MULTILINE)
    # Clean up multiple blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    n_fixed = before.count('### ##') + before.count('## ###')
    return content, n_fixed


# ---- 2. Re-verify with v2 on all sections ----
def re_verify_v2(state: dict) -> dict:
    """Re-run verify v2 on all sections that fell back to v1"""
    print("  [POST] Running verify v2 on all sections...")
    try:
        from research import faithfulness as _f_mod
        from research import verify as _v_mod
        print("  [POST] HHEM + verify_v2 loaded successfully")
    except ImportError as e:
        print(f"  [POST] verify v2 not available: {e}")
        return state

    passes = state["passes"]
    v2_count = 0
    fallback_count = 0

    for key, section in passes.items():
        content = section.get("content", "")
        sources = section.get("sources", [])
        verify = section.get("verify", {})

        # Check if it fell back to v1 (has n_citations but no verify_version or no n_claims)
        is_fallback = (
            verify.get("verify_version") != "v2"
            and verify.get("n_citations", 0) > 0
        )

        if not is_fallback:
            continue

        fallback_count += 1

        # Re-run v2 verify
        try:
            claims = _f_mod.decompose_claims(content, None)
            grounding_res = _f_mod.grounding_score(claims, sources)
            round_idx = 0

            verify_res = _v_mod.verify_section_v2(
                content, sources,
                section_prompt="",
                grounding_result=grounding_res,
                round_idx=round_idx,
                max_rounds=2,
                llm_call_fn=None,
            )

            crag = verify_res.get("crag_decision", "accept")
            g = verify_res.get("grounding", 0)
            n_claims = verify_res.get("n_claims", len(claims))
            n_supported = verify_res.get("n_supported", 0)

            # Update verify record
            verify["verify_version"] = "v2"
            verify["grounding"] = g
            verify["n_citations"] = n_claims
            verify["n_claims"] = n_claims
            verify["n_supported"] = n_supported
            verify["crag_decision"] = crag
            verify["cite_precision"] = verify_res.get("cite_precision", 0)
            verify["weak_summary"] = verify_res.get("weak_summary", "")
            verify["weak_citations"] = verify_res.get("weak_citations", [])

            v2_count += 1

            if v2_count % 5 == 0:
                print(f"  [POST] v2 verified {v2_count}/{fallback_count} fallback sections...")

        except Exception as e:
            print(f"  [POST] v2 failed for {key}: {e}")

    print(f"  [POST] v2 complete: {v2_count}/{fallback_count} sections re-verified")
    return state


# ---- 3. Re-compute quality marking with synced threshold ----
def recompute_quality(state: dict) -> dict:
    """Re-compute quality tag with synced MIN_GROUNDING threshold"""
    passes = state["passes"]
    degraded_before = sum(1 for v in passes.values() if v.get("quality") == "degraded")

    for key, section in passes.items():
        sources = section.get("sources", [])
        verify = section.get("verify", {})
        review = section.get("review", {})

        g = verify.get("grounding", 0)
        ncit = verify.get("n_citations", verify.get("n_claims", 0))
        has_sources = bool(sources)

        # Also check review issues for truncation/format problems
        issues = review.get("issues", "")
        truncation_keywords = ["truncation", "truncated", "broken", "incomplete",
                               "cut-off", "cutoff", "copy-paste"]
        has_truncation = any(k in issues.lower() for k in truncation_keywords)

        # Sync threshold: use MIN_GROUNDING (0.55), not 0.45
        if has_sources and (ncit == 0 or g < MIN_GROUNDING):
            quality = "degraded"
        elif has_truncation and g < MIN_GROUNDING:
            quality = "degraded"  # even if above threshold, truncation is a content issue
        else:
            quality = "ok"

        section["quality"] = quality

    degraded_after = sum(1 for v in passes.values() if v.get("quality") == "degraded")
    print(f"  [POST] Quality recomputed: degraded {degraded_before} → {degraded_after}")
    return state


# ---- 4. Regenerate 2 sections with zero citations ----
def regenerate_zero_cite_sections(state: dict) -> dict:
    """Re-generate sections that have 0 citations despite having sources"""
    zero_cite_keys = []
    for key, section in state["passes"].items():
        ncit = section.get("verify", {}).get("n_citations", 0)
        nsources = len(section.get("sources", []))
        if ncit == 0 and nsources > 0:
            zero_cite_keys.append(key)

    if not zero_cite_keys:
        print("  [POST] No zero-citation sections to regenerate")
        return state

    print(f"  [POST] Regenerating {len(zero_cite_keys)} zero-citation sections: {zero_cite_keys}")
    print("  [POST] NOTE: Full regeneration requires re-running the pipeline!")
    print("  [POST] Marking for regeneration in state.json...")

    for key in zero_cite_keys:
        section = state["passes"][key]
        section["_needs_regen"] = True
        section["quality"] = "degraded"
        print(f"  [POST] Flagged {key}: {section.get('title', '')}")

    return state


# ---- Main ----
def main():
    print("=" * 60)
    print("POST-PROCESSING: llm_trends_2026_2027")
    print("=" * 60)

    with open(STATE_PATH) as f:
        state = json.load(f)

    passes = state["passes"]

    # Quick stats
    degraded_before = sum(1 for v in passes.values() if v.get("quality") == "degraded")
    v2_before = sum(1 for v in passes.values() if v.get("verify", {}).get("verify_version") == "v2")
    broken_headers = sum(1 for s in passes.values() for l in s.get("content", "").split("\n") if "### ##" in l)

    print(f"\nBEFORE:")
    print(f"  Degraded: {degraded_before}/96")
    print(f"  v2 verified: {v2_before}/96")
    print(f"  Broken headers (### ##): {broken_headers}")
    print(f"  Total words: {state.get('total_words', 0):,}")

    # Step 1: Fix markdown
    print(f"\n[1/4] Fixing markdown `### ##` → `###`...")
    total_fixed = 0
    for key, section in passes.items():
        content = section.get("content", "")
        fixed, n_fixed = fix_markdown(content)
        section["content"] = fixed
        total_fixed += n_fixed
    print(f"  Fixed {total_fixed} markdown errors")

    # Step 2: Re-verify with v2
    print(f"\n[2/4] Re-running verify v2 on fallback sections...")
    state = re_verify_v2(state)

    # Step 3: Recompute quality with synced threshold
    print(f"\n[3/4] Recomputing quality (threshold 0.45 → {MIN_GROUNDING})...")
    state = recompute_quality(state)

    # Step 4: Flag zero-citation sections
    print(f"\n[4/4] Checking zero-citation sections...")
    state = regenerate_zero_cite_sections(state)

    # Save
    backup_path = RUN_DIR / "state.json.bak"
    with open(backup_path, "w") as f:
        json.dump(state, f)
    print(f"\nBackup saved: {backup_path}")

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"Updated: {STATE_PATH}")

    # Final stats
    degraded_after = sum(1 for v in state["passes"].values() if v.get("quality") == "degraded")
    v2_after = sum(1 for v in state["passes"].values() if v.get("verify", {}).get("verify_version") == "v2")
    broken_after = sum(1 for s in state["passes"].values() for l in s.get("content", "").split("\n") if "### ##" in l)

    print(f"\nAFTER:")
    print(f"  Degraded: {degraded_before} → {degraded_after}")
    print(f"  v2 verified: {v2_before} → {v2_after}")
    print(f"  Broken headers: {broken_headers} → {broken_after}")

    zero_cite = [(k, v) for k, v in state["passes"].items()
                 if v.get("verify", {}).get("n_citations", 0) == 0 and len(v.get("sources", [])) > 0]
    print(f"  Zero-citation: {len(zero_cite)} (flagged for regen)")
    for k, v in zero_cite:
        print(f"    {k}: {v.get('title', '')}")

    # Save markdown-only book
    print(f"\n[SAVE] Regenerating book.md...")
    chapters = {}
    for key, section in state["passes"].items():
        ch = key.split(".")[0]
        chapters.setdefault(ch, {})[key] = section

    lines = [
        "---",
        'title: "Large Language Models: A Comprehensive Handbook"',
        'subtitle: "Mathematics, Architecture, Training, Alignment, Deployment, and the Future of AI"',
        "author: Generated by Agentic Deep Research | WRT: qwen3.6-35b | RSR: gemma-4-12b | June 2026",
        "lang: en",
        'geometry: "margin=1.5in"',
        "fontsize: 11pt",
        "---",
        "",
        "# Large Language Models: A Comprehensive Handbook",
        "",
        "_A Comprehensive Handbook: Mathematics, Architecture, Training, Alignment, Deployment, and the Future of AI_",
        "",
        "---",
    ]

    for ch_num in sorted(chapters.keys(), key=lambda x: int(x)):
        ch_sections = chapters[ch_num]
        ch_t = list(ch_sections.values())[0].get("ch_t", f"Chapter {ch_num}")
        lines.append(f"\n# {ch_t}\n")
        for key in sorted(ch_sections.keys()):
            section = ch_sections[key]
            pp_t = section.get("title", "")
            content = section.get("content", "")
            quality = section.get("quality", "ok")
            g = section.get("verify", {}).get("grounding", 0)

            if quality == "degraded":
                lines.append(f"\n> **Note:** Section below quality threshold (grounding={g:.2f}). Consider regenerating.\n")

            # Fix any remaining markdown issues
            content = re.sub(r'### ##\s+', '### ', content, flags=re.MULTILINE)

            lines.append(f"\n## {pp_t}\n")
            lines.append(content)
            lines.append("")

    book_md = "\n".join(lines)
    book_path = RUN_DIR / "book.md"
    with open(book_path, "w") as f:
        f.write(book_md)
    print(f"Saved: {book_path} ({len(book_md):,} chars)")

    print("\n" + "=" * 60)
    print("POST-PROCESS COMPLETE")
    print("=" * 60)

    return {
        "degraded_before": degraded_before,
        "degraded_after": degraded_after,
        "v2_before": v2_before,
        "v2_after": v2_after,
        "broken_headers_before": broken_headers,
        "broken_headers_after": broken_after,
        "zero_citation_count": len(zero_cite),
    }


if __name__ == "__main__":
    results = main()
    sys.exit(0)
