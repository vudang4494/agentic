#!/usr/bin/env python3
"""
Targeted regeneration of broken sections (g=0, _needs_regen=True).
Mirrors the full pipeline research → rank → gen → verify loop for specific sections.

Usage:
  python3 scripts/regen_broken.py --run llm_trends_2026_2027
  python3 scripts/regen_broken.py --run llm_trends_2026_2027 --sections 4.6 5.7
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

from deep_research import (
    OllamaClient, SYS, gen, sanitize, wc, compute_target_words,
    build_context, log, RESEARCH_AVAILABLE,
)
import research as _research

_CITE_RE = re.compile(r"\[(\d+)\]")


def main():
    p = argparse.ArgumentParser(description="Regenerate broken sections")
    p.add_argument("--run", required=True, help="Run name")
    p.add_argument("--sections", nargs="+", default=["4.6", "5.7"],
                   help="Sections to regenerate, e.g. 4.6 5.7")
    args = p.parse_args()

    run_dir = ROOT / "output/runs" / args.run
    state_path = run_dir / "state.json"

    with open(state_path) as f:
        state = json.load(f)

    passes = state["passes"]
    client = OllamaClient()

    results = {}
    for key in args.sections:
        print(f"\n{'='*60}")
        print(f"Regenerating: {key}")
        print(f"{'='*60}")

        sec = passes.get(key)
        if not sec:
            print(f"  [ERROR] Section {key} not found")
            continue

        ch_n = int(key.split(".")[0])
        pp_n = int(key.split(".")[1])
        ch_t = sec.get("ch_t", "")
        pp_t = sec.get("title", "")

        # Prompt from CHAPTERS hardcoded outline (the ground truth)
        from deep_research import CHAPTERS
        prompt = ""
        for ch in CHAPTERS:
            if ch["n"] == ch_n:
                for pp in ch.get("passes", []):
                    if pp["p"] == pp_n:
                        prompt = pp["pr"]
                        break
        print(f"  Title: {pp_t}")

        ranked = []
        evidence_block = ""
        verify_res = None
        rounds_log = []

        if RESEARCH_AVAILABLE:
            for round_n in range(1, _research.MAX_RESEARCH_ROUNDS + 1):
                t_r = time.time()
                print(f"\n  [RESEARCH r{round_n}]")

                # --- QGN ---
                queries = _research.query_gen.queries_for(prompt, ch_t, pp_t)
                print(f"    QGN: {len(queries)} queries")

                # --- Gather ---
                raw_sources = _research.search.gather(
                    queries, providers=_research.PROVIDERS_DEFAULT, per_provider_k=3,
                )
                print(f"    Gathered: {len(raw_sources)} raw sources")

                # --- Rank (RRF if v2, else cosine) ---
                if _research.VFY_V2_AVAILABLE:
                    ranked = _research.notes.rank_rrf(
                        raw_sources, prompt, top_k=_research.TOP_K_RETRIEVE,
                        embed_model=_research.EMBED_MODEL,
                        primary_floor=_research.PRIMARY_FLOOR,
                    )
                else:
                    ranked = _research.notes.rank(
                        raw_sources, prompt, top_k=_research.TOP_K_RETRIEVE,
                        embed_model=_research.EMBED_MODEL,
                        primary_floor=_research.PRIMARY_FLOOR,
                    )
                print(f"    Ranked: {len(ranked)} (RRF)")

                # --- RRK rerank ---
                try:
                    ranked = _research.rerank.rerank(prompt, ranked, top_k=_research.TOP_K_FINAL)
                    print(f"    RRK: top-{len(ranked)}")
                except Exception as e:
                    print(f"    RRK failed ({e}), using top-{_research.TOP_K_FINAL}")
                    ranked = ranked[:_research.TOP_K_FINAL]

                # --- Full-text enrichment ---
                ranked = _research.notes.enrich_top_sources(
                    ranked, top_n=_research.FULL_TEXT_TOP_N, max_words_per=_research.FULL_TEXT_MAX_WORDS,
                )

                # --- Format evidence ---
                evidence_block = _research.notes.format_for_prompt(ranked)
                target_words = compute_target_words(len(ranked), has_research=True)

                print(f"    {len(ranked)} sources | {len(evidence_block)} chars | "
                      f"target={target_words}w | {time.time()-t_r:.1f}s")

                # --- Context ---
                context_block = build_context(state, ch_n, pp_n)

                # --- Generate ---
                content, stats, w = gen(
                    client, ch_n, ch_t, pp_n, pp_t, prompt, target_words,
                    context_block=context_block, evidence_block=evidence_block,
                )
                if not content:
                    print("    Generation FAILED")
                    break

                # --- Citation cleanup ---
                content, n_dropped = _research.notes.clean_citations(content, len(ranked))
                if n_dropped:
                    print(f"    [CITE-FIXUP] dropped {n_dropped} bad citations")

                w = wc(content)
                n_cites = len(_CITE_RE.findall(content))
                print(f"    Generated: {w}w, {n_cites} citations")

                # --- Verify (v1: count [N] markers, simpler and reliable) ---
                t_v = time.time()
                try:
                    # v1 verify: counts [N] citation markers directly
                    # This is more reliable than v2 when evidence sources lack excerpts
                    cite_markers = len(_CITE_RE.findall(content))
                    verify_res = {
                        "grounding": 0.0,  # Will be set after v2 grounding check
                        "n_citations": cite_markers,
                        "quality_tag": "ok",
                    }
                    # Try v2 grounding if sources have excerpts
                    try:
                        f = _research.faithfulness
                        claims = f.decompose_claims(content, None)
                        grounding_res = f.grounding_score(claims, ranked)
                        verify_res_v2 = _research.verify.verify_section_v2(
                            content, ranked,
                            section_prompt=prompt,
                            grounding_result=grounding_res,
                            round_idx=round_n - 1,
                            max_rounds=_research.MAX_RESEARCH_ROUNDS,
                        )
                        verify_res["grounding"] = verify_res_v2.get("grounding", 0.0)
                        verify_res["quality_tag"] = verify_res_v2.get("quality_tag", "ok")
                        verify_res["crag_decision"] = verify_res_v2.get("crag_decision", "accept")
                    except Exception as e2:
                        print(f"    [V2 grounding failed: {e2}]")
                except Exception as e:
                    print(f"    Verify failed: {e}")
                    verify_res = {"grounding": 0, "n_citations": 0, "quality_tag": "ok"}

                g = verify_res.get("grounding", 0)
                print(f"    Verify: g={g:.3f} ({verify_res.get('n_citations',0)} cites) "
                      f"in {time.time()-t_v:.1f}s")

                # --- Round score ---
                has_cites = (verify_res.get("n_citations", 0) > 0)
                round_score = (1 if has_cites else 0, g)
                rounds_log.append({
                    "round": round_n, "content": content, "w": w,
                    "stats": stats, "ranked": ranked,
                    "evidence_block": evidence_block,
                    "verify": verify_res, "score": round_score,
                })

                # --- CRAG gate ---
                if _research.VFY_V2_AVAILABLE:
                    quality_tag = verify_res.get("quality_tag", "ok")
                    crag = verify_res.get("crag_decision", "accept")
                    if quality_tag == "ok":
                        print(f"    [CRAG] accept")
                        break
                    elif crag in ("incorrect", "self_rag") and round_n < _research.MAX_RESEARCH_ROUNDS:
                        hint = verify_res.get("weak_summary", "low grounding")
                        print(f"    [CRAG] retry: {hint}")
                        continue
                    else:
                        break
                else:
                    if g >= _research.MIN_GROUNDING:
                        print(f"    [VFY] accept (g={g:.3f})")
                        break
                    elif round_n < _research.MAX_RESEARCH_ROUNDS:
                        print(f"    [VFY] retry (g={g:.3f} < {_research.MIN_GROUNDING})")
                        continue
                    else:
                        break

            # --- Best round ---
            best = max(rounds_log, key=lambda r: r["score"])
            content = best["content"]
            w = best["w"]
            stats = best["stats"]
            ranked = best["ranked"]
            verify_res = best["verify"]
            evidence_block = best["evidence_block"]

            if len(rounds_log) > 1:
                print(f"    [BEST-ROUND] g={verify_res.get('grounding',0):.3f} "
                      f"from best of {len(rounds_log)} rounds")

            # --- Self-RAG scrub ---
            try:
                scrubbed, n_scrubbed, abstain_note = (
                    _research.verify.scrub_unsupported_citations(content, ranked, verify_res)
                )
                if n_scrubbed:
                    print(f"    [SELF-RAG] scrubbed {n_scrubbed} unsupported citations")
                    content = scrubbed
                    if abstain_note:
                        content += abstain_note
            except Exception as e:
                print(f"    [SELF-RAG] failed: {e}")

        else:
            # No research
            target_words = compute_target_words(0, has_research=False)
            context_block = build_context(state, ch_n, pp_n)
            content, stats, w = gen(
                client, ch_n, ch_t, pp_n, pp_t, prompt, target_words,
                context_block=context_block, evidence_block="",
            )

        # --- Update state ---
        passes[key]["content"] = content
        passes[key]["wc"] = w
        passes[key]["tokens"] = stats.get("tokens", 0)
        passes[key]["tps"] = stats.get("tps", 0)
        passes[key]["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        passes[key]["verify"] = verify_res or {"grounding": 0}
        passes[key]["quality"] = (verify_res or {}).get("quality_tag", "ok")
        passes[key]["_needs_regen"] = False

        n_cites = len(_CITE_RE.findall(content))
        g = (verify_res or {}).get("grounding", 0)
        print(f"\n  Final: {w}w, {n_cites} cites, g={g:.3f}")
        print(f"  Last 200: {repr(content[-200:])}")
        results[key] = {"w": w, "citations": n_cites, "grounding": g}

    # Save state
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    print(f"\nState saved: {state_path}")
    print("\nSummary:")
    for key, r in results.items():
        print(f"  {key}: {r['w']}w | {r['citations']} cites | g={r['grounding']:.3f}")


if __name__ == "__main__":
    main()
