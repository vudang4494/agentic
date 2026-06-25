#!/usr/bin/env python3
"""
A/B Benchmark Harness — v1 (cosine gate + LLM judge) vs v2 (RRK + HHEM + CRAG).

DESIGN: Each section is processed ONCE:
  1. Gather search results (shared by both arms)
  2. Generate ONE body using v2's top-8 (higher relevance)
  3. Verify that SAME body through BOTH v1 and v2 arms

This isolates verify-layer quality — not confounded by rank or writer differences.

Metrics per arm:
  - grounding_rate      (mean supported/total claims)
  - cite_precision     (# supported claims / # total claims)
  - relevance_at_8      (median rerank_score, v2 only)
  - crag_branches      (accept/ambiguous/incorrect distribution)
  - llm_calls_per_section
  - wall_clock_per_section_s

Usage:
    python3 bench/bench_pipeline.py --topic "Transformers" --sections 12 --seed 42

Output:
    output/bench/<ts>/bench_report.json  (machine-readable)
    + stdout table (human-readable)

Requirements:
    pip install FlagEmbedding sentence-transformers transformers torch
    ollama pull gemma4:e4b batiai/qwen3.6-35b:iq3
    ollama pull bge-m3:latest
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import research as _research
    from research import rerank as _rerank
except ImportError as e:
    print(f"[bench] research layer unavailable: {e}")
    _research = None
    _rerank = None

BENCH_DIR = Path(__file__).parent.parent / "output" / "bench"
WRITER_MODEL = os.environ.get("DEEP_RESEARCH_WRITER_MODEL", "batiai/qwen3.6-35b:iq3")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
JUDGE_MODEL = "gemma4:e4b"
DEFAULT_TIMEOUT = 600
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def setup_run(ts: str) -> Path:
    run_dir = BENCH_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---- Ollama writer ----

def _ollama_generate(prompt: str, system: str = "", model: str = WRITER_MODEL,
                     temperature: float = 0.7, num_predict: int = 15000) -> tuple[str, dict]:
    import httpx
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": msgs,
        "options": {
            "temperature": temperature,
            "num_predict": min(num_predict, 15000),
            "top_p": 0.95,
            "top_k": 20,
            "repeat_penalty": 1.05,
        },
    }
    t0 = time.time()
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
        r = c.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    content = data.get("message", {}).get("content", "").strip()
    ec = data.get("eval_count", 0)
    ed = data.get("eval_duration", 0)
    tps = ec / (ed / 1e9) if ed > 0 else 0
    return content, {
        "tokens": ec,
        "tps": round(tps, 1),
        "elapsed": round(time.time() - t0, 1),
    }


SYS_PROMPT = (
    "You are a world-class technical book writer specializing in Large Language Models. "
    "You are writing ONE section of a larger book.\n\n"
    "STRICT OUTPUT RULES:\n"
    "1. Do NOT output any H1 (`#`) or H2 (`##`) heading.\n"
    "2. Do NOT start with meta-introductions. Open with a concrete fact, formula, or named method.\n"
    "3. Do NOT write a Conclusion or Summary section.\n"
    "4. Do NOT write a References section.\n"
    "5. CITATIONS (MANDATORY): place at least FIVE `[N]` citation markers, using real source numbers "
    "from the EVIDENCE block. Anchor each `[N]` on a specific factual claim.\n"
    "6. Write in scholarly, precise style. Include LaTeX math ($...$) where relevant.\n"
    "Output ONLY the section body Markdown -- nothing else."
)


def write_body(prompt: str, top_sources: list) -> tuple[str, dict]:
    evidence_lines = []
    for i, src in enumerate(top_sources[:8], 1):
        text = (src.excerpt if hasattr(src, "excerpt") else
                src.get("excerpt", src.get("text", "")))[:500]
        evidence_lines.append(f"[{i}] {text}")
    evidence_block = "\n\n".join(evidence_lines)
    full_prompt = (
        f"{SYS_PROMPT}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"SECTION PROMPT:\n{prompt}"
    )
    return _ollama_generate(full_prompt)


# ---- v1 arm ----

def arm_v1_top8(sources: list, query: str) -> list:
    if not _research:
        return sources[:8]
    try:
        ranked = _research.notes.rank(
            sources, query, top_k=8,
            embed_model=_research.EMBED_MODEL, precomputed=False,
        )
        return ranked[:8]
    except Exception as e:
        print(f"    [v1 rank] failed: {e}", flush=True)
        return sources[:8]


def arm_v1_verify(body_md: str, sources: list) -> dict:
    """v1: cosine pre-filter + batch LLM judge."""
    if not _research:
        return _empty_vfy()
    try:
        return _research.verify.verify_section(body_md, sources, model=JUDGE_MODEL)
    except Exception as e:
        print(f"    [v1 verify] failed: {e}", flush=True)
        return _empty_vfy()


# ---- v2 arm ----

def arm_v2_top8(sources: list, query: str) -> list:
    if _rerank is None or not (_research.VFY_V2_AVAILABLE if _research else False):
        return arm_v1_top8(sources, query)
    try:
        return _rerank.rerank(query, sources, top_k=8)
    except Exception as e:
        print(f"    [v2 rank] failed: {e}, fallback to v1", flush=True)
        return arm_v1_top8(sources, query)


def arm_v2_verify(body_md: str, sources: list, query: str) -> dict:
    """v2: HHEM grounding + CRAG."""
    if _rerank is None or not (_research.VFY_V2_AVAILABLE if _research else False):
        return arm_v1_verify(body_md, sources)
    try:
        claims = _research.faithfulness.decompose_claims(body_md, None)
        grounding_res = _research.faithfulness.grounding_score(claims, sources)
        result = _research.verify.verify_section_v2(
            body_md, sources,
            section_prompt=query,
            grounding_result=grounding_res,
            round_idx=0,
            max_rounds=_research.MAX_RESEARCH_ROUNDS,
            llm_call_fn=None,
        )
        return result
    except Exception as e:
        print(f"    [v2 verify] failed: {e}, fallback to v1", flush=True)
        return arm_v1_verify(body_md, sources)


# ---- Helpers ----

def _empty_vfy():
    return {"grounding": 0.0, "n_citations": 0, "n_supported": 0,
            "n_partial": 0, "n_unsupported": 0, "verdicts": []}


def _rel_scores(top8: list) -> list:
    scores = []
    for s in top8:
        sc = getattr(s, "rerank_score", None)
        if sc is None and isinstance(s, dict):
            sc = s.get("rerank_score")
        if sc is not None:
            scores.append(sc)
    return scores


def _median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return float(s[n // 2])


def _p10(values: list) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return float(sorted(values)[max(0, int(len(values) * 0.1) - 1)])


def _p90(values: list) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return float(sorted(values)[min(len(values) - 1, int(len(values) * 0.9))])


def summarize(values: list) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "n": 0}
    return {
        "mean": round(sum(values) / len(values), 4),
        "median": round(_median(values), 4),
        "p10": round(_p10(values), 4),
        "p90": round(_p90(values), 4),
        "n": len(values),
    }


# ---- Per-section run (generates ONE body, evaluates with BOTH arms) ----

def run_section(sec: dict) -> dict:
    query = sec.get("prompt", sec.get("title", ""))
    title = sec.get("title", "unknown")

    result = {
        "title": title,
        "query": query,
        "body_words": 0,
        "raw_sources_count": 0,
        # v1
        "v1": {"grounding": 0.0, "n_claims": 0, "n_supported": 0,
               "n_partial": 0, "n_unsupported": 0, "crag": "N/A",
               "llm_calls": 0, "wall_s": 0.0, "rel_scores": [], "rank_time_s": 0.0},
        # v2
        "v2": {"grounding": 0.0, "n_claims": 0, "n_supported": 0,
               "n_partial": 0, "n_unsupported": 0, "crag": "accept",
               "llm_calls": 0, "wall_s": 0.0, "rel_scores": [], "rank_time_s": 0.0},
    }

    # Step 1: gather (shared)
    t0 = time.time()
    try:
        raw_sources = _research.search.gather(
            [query], providers=_research.PROVIDERS_DEFAULT, per_provider_k=3,
        ) if _research else []
    except Exception as e:
        print(f"  [{title}] search failed: {e}", flush=True)
        raw_sources = []

    result["raw_sources_count"] = len(raw_sources)
    if not raw_sources:
        print(f"  [{title}] no sources, skipping", flush=True)
        return result

    # Step 2: rank (both arms)
    t_rank = time.time()

    # v1 top-8
    top8_v1 = arm_v1_top8(raw_sources, query)
    result["v1"]["rel_scores"] = _rel_scores(top8_v1)
    result["v1"]["rank_time_s"] = time.time() - t_rank

    # v2 top-8
    t_rank2 = time.time()
    top8_v2 = arm_v2_top8(raw_sources, query)
    result["v2"]["rel_scores"] = _rel_scores(top8_v2)
    result["v2"]["rank_time_s"] = time.time() - t_rank2

    # Use v2 top-8 for writing (higher relevance quality)
    write_sources = top8_v2

    # Step 3: write ONE body
    body = ""
    t_write = time.time()
    try:
        body, wmeta = write_body(query, write_sources)
        v1_llm = 1
        result["v1"]["wall_s"] += wmeta["elapsed"]
        result["v2"]["wall_s"] += wmeta["elapsed"]
    except Exception as e:
        print(f"  [{title}] writer failed: {e}", flush=True)
        body = f"# {title}\n\nContent for {query}."
        v1_llm = 0

    result["body_words"] = len(body.split())
    result["v1"]["llm_calls"] = v1_llm
    result["v2"]["llm_calls"] = v1_llm
    write_elapsed = time.time() - t_write

    # Step 4: v1 verify (LLM judge)
    t_v1 = time.time()
    v1_res = arm_v1_verify(body, top8_v1)
    result["v1"]["grounding"] = v1_res.get("grounding", 0.0)
    result["v1"]["n_claims"] = v1_res.get("n_citations", 0)
    n_sup = sum(1 for v in v1_res.get("verdicts", []) if v.get("verdict") == "supports")
    n_part = sum(1 for v in v1_res.get("verdicts", []) if v.get("verdict") == "partial")
    n_unr = sum(1 for v in v1_res.get("verdicts", []) if v.get("verdict") in ("unrelated", "contradicts"))
    result["v1"]["n_supported"] = n_sup
    result["v1"]["n_partial"] = n_part
    result["v1"]["n_unsupported"] = n_unr
    result["v1"]["llm_calls"] += 1  # LLM judge call
    result["v1"]["wall_s"] += time.time() - t_v1

    # Step 5: v2 verify (HHEM)
    t_v2 = time.time()
    v2_res = arm_v2_verify(body, top8_v2, query)
    result["v2"]["grounding"] = v2_res.get("grounding", 0.0)
    result["v2"]["n_claims"] = v2_res.get("n_claims", v2_res.get("n_citations", 0))
    result["v2"]["n_supported"] = v2_res.get("n_supported", 0)
    result["v2"]["n_partial"] = v2_res.get("n_partial", 0)
    result["v2"]["n_unsupported"] = v2_res.get("n_unsupported", 0)
    result["v2"]["crag"] = v2_res.get("crag_decision", "accept")
    # HHEM is NOT an LLM call
    result["v2"]["wall_s"] += time.time() - t_v2

    total_elapsed = time.time() - t0
    print(f"  [{title}] {result['body_words']}w | "
          f"v1: g={result['v1']['grounding']:.3f} sup={result['v1']['n_supported']}/{result['v1']['n_claims']} | "
          f"v2: g={result['v2']['grounding']:.3f} sup={result['v2']['n_supported']}/{result['v2']['n_claims']} "
          f"crag={result['v2']['crag']} | "
          f"total={total_elapsed:.0f}s", flush=True)

    return result


def run_benchmark(sections: list) -> dict:
    results = {"sections": [], "v1": {}, "v2": {}}

    v1_agg = {"grounding": [], "n_claims": [], "n_supported": [], "n_partial": [],
              "n_unsupported": [], "llm_calls": [], "wall_s": [], "rel_scores": []}
    v2_agg = {"grounding": [], "n_claims": [], "n_supported": [], "n_partial": [],
              "n_unsupported": [], "llm_calls": [], "wall_s": [], "rel_scores": [],
              "crag_accept": 0, "crag_ambiguous": 0, "crag_incorrect": 0}

    for sec in sections:
        r = run_section(sec)
        results["sections"].append(r)

        v1_agg["grounding"].append(r["v1"]["grounding"])
        v1_agg["n_claims"].append(r["v1"]["n_claims"])
        v1_agg["n_supported"].append(r["v1"]["n_supported"])
        v1_agg["n_partial"].append(r["v1"]["n_partial"])
        v1_agg["n_unsupported"].append(r["v1"]["n_unsupported"])
        v1_agg["llm_calls"].append(r["v1"]["llm_calls"])
        v1_agg["wall_s"].append(r["v1"]["wall_s"])
        v1_agg["rel_scores"].extend(r["v1"]["rel_scores"])

        v2_agg["grounding"].append(r["v2"]["grounding"])
        v2_agg["n_claims"].append(r["v2"]["n_claims"])
        v2_agg["n_supported"].append(r["v2"]["n_supported"])
        v2_agg["n_partial"].append(r["v2"]["n_partial"])
        v2_agg["n_unsupported"].append(r["v2"]["n_unsupported"])
        v2_agg["llm_calls"].append(r["v2"]["llm_calls"])
        v2_agg["wall_s"].append(r["v2"]["wall_s"])
        v2_agg["rel_scores"].extend(r["v2"]["rel_scores"])

        crag = r["v2"]["crag"]
        if "accept" in str(crag).lower():
            v2_agg["crag_accept"] += 1
        elif "ambiguous" in str(crag).lower():
            v2_agg["crag_ambiguous"] += 1
        elif "incorrect" in str(crag).lower():
            v2_agg["crag_incorrect"] += 1

    results["v1"] = v1_agg
    results["v2"] = v2_agg
    return results


def cite_precision_avg(sup, part, unr):
    precs = []
    for s, p, u in zip(sup, part, unr):
        t = s + p + u
        precs.append(s / t if t > 0 else 0.0)
    return round(sum(precs) / len(precs), 4) if precs else 0.0


def print_results(results: dict):
    v1 = results["v1"]
    v2 = results["v2"]

    print(f"\n{'=' * 110}")
    print(f"{'A/B BENCHMARK — v1 (cosine+LLM judge) vs v2 (RRK+HHEM+CRAG)':^110}")
    print(f"{'=' * 110}")

    # Citation precision
    v1_cp = cite_precision_avg(v1["n_supported"], v1["n_partial"], v1["n_unsupported"])
    v2_cp = cite_precision_avg(v2["n_supported"], v2["n_partial"], v2["n_unsupported"])

    # rel_at_8: median per-section of median rel score
    def section_rel_medians(rel_scores_all, n_sections):
        scores = []
        for sec in results["sections"]:
            rs = sec["v2"]["rel_scores"]
            if rs:
                scores.append(_median(rs))
        return round(_median(scores), 4) if scores else 0.0

    v1_rel = round(_median(v1["rel_scores"]), 4) if v1["rel_scores"] else 0.0
    v2_rel = round(_median(v2["rel_scores"]), 4) if v2["rel_scores"] else 0.0

    metrics = [
        ("grounding (mean)", [v1["grounding"], v2["grounding"]], True),
        ("grounding (median)", [v1["grounding"], v2["grounding"]], False),
        ("cite_precision", [[v1_cp], [v2_cp]], True),
        ("n_supported (mean)", [v1["n_supported"], v2["n_supported"]], True),
        ("n_partial (mean)", [v1["n_partial"], v2["n_partial"]], False),
        ("n_unsupported (mean)", [v1["n_unsupported"], v2["n_unsupported"]], False),
        ("n_claims (mean)", [v1["n_claims"], v2["n_claims"]], True),
        ("rel_at_8 (overall median)", [[v1_rel], [v2_rel]], True),
        ("rel_at_8 (per-sec median)", [[section_rel_medians(v2["rel_scores"], len(results["sections"]))],
                                        [section_rel_medians(v2["rel_scores"], len(results["sections"]))]], True),
        ("llm_calls/section (mean)", [v1["llm_calls"], v2["llm_calls"]], False),
        ("wall_s/section (mean)", [v1["wall_s"], v2["wall_s"]], False),
        ("body_words (mean)", [[sum(r["body_words"] for r in results["sections"]) / max(len(results["sections"]), 1)],
                                [sum(r["body_words"] for r in results["sections"]) / max(len(results["sections"]), 1)]], True),
    ]

    print(f"\n  {'Metric':<28} {'v1':>18} {'v2':>18} {'Delta':>12} {'Winner':>8}")
    print(f"  {'-' * 90}")

    for name, (v1_vals, v2_vals), higher_better in metrics:
        if isinstance(v1_vals[0], dict):
            v1_s = v1_vals[0]
            v2_s = v2_vals[0]
        else:
            v1_s = summarize(v1_vals)
            v2_s = summarize(v2_vals)
        mean1 = v1_s["mean"] if "mean" in v1_s else v1_s.get("mean", 0)
        mean2 = v2_s["mean"] if "mean" in v2_s else v2_s.get("mean", 0)
        delta = mean2 - mean1
        if higher_better:
            winner = "v2" if delta > 0 else ("v1" if delta < 0 else "tie")
        else:
            winner = "v2" if delta < 0 else ("v1" if delta > 0 else "tie")
        v1_str = f"{mean1:.4f}" + (f" ({v1_s.get('p10', 0):.2f}-{v1_s.get('p90', 0):.2f})" if "p10" in v1_s else "")
        v2_str = f"{mean2:.4f}" + (f" ({v2_s.get('p10', 0):.2f}-{v2_s.get('p90', 0):.2f})" if "p10" in v2_s else "")
        print(f"  {name:<28} {v1_str:>18} {v2_str:>18} {delta:>+12.4f} {winner:>8}")

    # CRAG distribution
    total = len(results["sections"])
    print(f"\n  {'CRAG Distribution (v2 only)':<28} {'v1=N/A':>18} {f'v2':>18}")
    print(f"  {'-' * 66}")
    for branch, label in [("accept", "accept"), ("ambiguous", "ambiguous"), ("incorrect", "incorrect")]:
        v1_str = "N/A"
        v2_count = v2.get(f"crag_{branch}", 0)
        v2_str = f"{v2_count}/{total}"
        print(f"  {label:<28} {v1_str:>18} {v2_str:>18}")

    # Per-section detail
    print(f"\n  Per-section detail:")
    print(f"  {'Section':<35} {'v1_g':>7} {'v1_sup/c':>8} {'v2_g':>7} {'v2_sup/c':>8} {'v2_crag':>12} {'rel8':>8} {'llm_v1':>7} {'llm_v2':>7}")
    print(f"  {'-' * 105}")
    for sec in results["sections"]:
        v1g = sec["v1"]["grounding"]
        v1s = f"{sec['v1']['n_supported']}/{sec['v1']['n_claims']}"
        v2g = sec["v2"]["grounding"]
        v2s = f"{sec['v2']['n_supported']}/{sec['v2']['n_claims']}"
        crag = sec["v2"]["crag"]
        rs = _median(sec["v2"]["rel_scores"])
        l1 = sec["v1"]["llm_calls"]
        l2 = sec["v2"]["llm_calls"]
        print(f"  {sec['title']:<35} {v1g:>7.3f} {v1s:>8} {v2g:>7.3f} {v2s:>8} {crag:>12} {rs:>8.4f} {l1:>7} {l2:>7}")

    print(f"\n{'=' * 110}\n")


def build_report(results: dict, ts: str, topic: str, seed: int) -> dict:
    v1 = results["v1"]
    v2 = results["v2"]
    n = len(results["sections"])

    v1_cp = cite_precision_avg(v1["n_supported"], v1["n_partial"], v1["n_unsupported"])
    v2_cp = cite_precision_avg(v2["n_supported"], v2["n_partial"], v2["n_unsupported"])

    def section_rel_medians(scores, n_sec):
        meds = []
        for sec in results["sections"]:
            rs = sec["v2"]["rel_scores"]
            if rs:
                meds.append(_median(rs))
        return round(_median(meds), 4) if meds else 0.0

    return {
        "ts": ts,
        "topic": topic,
        "n_sections": n,
        "seed": seed,
        "models": {
            "writer": WRITER_MODEL,
            "judge_v1": JUDGE_MODEL,
            "embed": _research.EMBED_MODEL if _research else None,
            "reranker": RERANKER_MODEL,
        },
        "vfy_v2_available": _research.VFY_V2_AVAILABLE if _research else False,
        "v1": {
            "grounding": summarize(v1["grounding"]),
            "cite_precision": round(v1_cp, 4),
            "n_supported": summarize(v1["n_supported"]),
            "n_partial": summarize(v1["n_partial"]),
            "n_unsupported": summarize(v1["n_unsupported"]),
            "n_claims": summarize(v1["n_claims"]),
            "llm_calls_per_section": summarize(v1["llm_calls"]),
            "wall_s_per_section": summarize(v1["wall_s"]),
            "rel_at_8_median": round(_median(v1["rel_scores"]), 4),
        },
        "v2": {
            "grounding": summarize(v2["grounding"]),
            "cite_precision": round(v2_cp, 4),
            "n_supported": summarize(v2["n_supported"]),
            "n_partial": summarize(v2["n_partial"]),
            "n_unsupported": summarize(v2["n_unsupported"]),
            "n_claims": summarize(v2["n_claims"]),
            "llm_calls_per_section": summarize(v2["llm_calls"]),
            "wall_s_per_section": summarize(v2["wall_s"]),
            "rel_at_8_median": round(_median(v2["rel_scores"]), 4),
            "crag_branches": {
                "accept": v2["crag_accept"],
                "ambiguous": v2["crag_ambiguous"],
                "incorrect": v2["crag_incorrect"],
            },
        },
        "per_section": [
            {
                "title": s["title"],
                "body_words": s["body_words"],
                "raw_sources": s["raw_sources_count"],
                "v1": {
                    "grounding": s["v1"]["grounding"],
                    "n_supported": s["v1"]["n_supported"],
                    "n_claims": s["v1"]["n_claims"],
                    "n_partial": s["v1"]["n_partial"],
                    "n_unsupported": s["v1"]["n_unsupported"],
                    "llm_calls": s["v1"]["llm_calls"],
                    "wall_s": round(s["v1"]["wall_s"], 1),
                    "rel_scores_median": round(_median(s["v1"]["rel_scores"]), 4),
                },
                "v2": {
                    "grounding": s["v2"]["grounding"],
                    "n_supported": s["v2"]["n_supported"],
                    "n_claims": s["v2"]["n_claims"],
                    "n_partial": s["v2"]["n_partial"],
                    "n_unsupported": s["v2"]["n_unsupported"],
                    "crag_decision": s["v2"]["crag"],
                    "llm_calls": s["v2"]["llm_calls"],
                    "wall_s": round(s["v2"]["wall_s"], 1),
                    "rel_scores_median": round(_median(s["v2"]["rel_scores"]), 4),
                },
            }
            for s in results["sections"]
        ],
    }


# ---- Default outline ----

DEFAULT_OUTLINE = [
    ("History and Evolution", "History and evolution of large language models from n-gram to transformers"),
    ("Mathematical Foundations", "Mathematical foundations of LLMs: MLE, cross-entropy, and attention"),
    ("Transformer Architecture", "Transformer architecture and self-attention mechanism in LLMs"),
    ("Pre-training and Scaling Laws", "Pre-training paradigm and scaling laws for LLMs"),
    ("Prompt Engineering", "Prompt engineering techniques for LLMs"),
    ("Fine-tuning Methods", "Fine-tuning methods: SFT, RLHF, DPO for LLMs"),
    ("Retrieval-Augmented Generation", "Retrieval-augmented generation (RAG) for LLMs"),
    ("Multimodal Models", "Multimodal large language models"),
    ("LLM Agents and Tool Use", "LLM agents and tool use"),
    ("Evaluation Benchmarks", "Evaluation benchmarks for LLMs"),
    ("Safety and Alignment", "Safety, alignment, and responsible AI for LLMs"),
    ("Deployment and Efficiency", "Deployment strategies and efficiency techniques for LLMs"),
]


def main():
    parser = argparse.ArgumentParser(description="A/B benchmark: v1 vs v2 verify arms")
    parser.add_argument("--topic", default="Large Language Models",
                        help="Topic / book title")
    parser.add_argument("--sections", type=int, default=12,
                        help="Number of sections (max 12)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--outline-file", type=str, default="",
                        help="JSON file with outline [{title, prompt}] or leave empty for default")
    args = parser.parse_args()

    random.seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = setup_run(ts)
    print(f"[bench] ==============================================")
    print(f"[bench] Run {ts}")
    print(f"[bench] Topic: {args.topic}")
    print(f"[bench] Sections: {args.sections}")
    print(f"[bench] Writer: {WRITER_MODEL}")
    print(f"[bench] Judge v1: {JUDGE_MODEL}")
    if _research:
        print(f"[bench] VFY_V2_AVAILABLE={_research.VFY_V2_AVAILABLE}")
        print(f"[bench] EMBED={_research.EMBED_MODEL}")
        print(f"[bench] RERANKER={RERANKER_MODEL}")
    else:
        print("[bench] WARNING: research layer unavailable")
    print(f"[bench] ==============================================")

    # Build sections
    if args.outline_file and Path(args.outline_file).exists():
        sections = json.loads(Path(args.outline_file).read_text())
        print(f"[bench] Loaded {len(sections)} sections from {args.outline_file}")
    else:
        sections = [
            {"title": title, "prompt": prompt}
            for title, prompt in DEFAULT_OUTLINE[:min(args.sections, len(DEFAULT_OUTLINE))]
        ]
        print(f"[bench] Using default outline ({len(sections)} sections)")

    print(f"[bench] Starting benchmark...", flush=True)
    t0 = time.time()
    results = run_benchmark(sections)
    total_wall = time.time() - t0

    print(f"\n[bench] Benchmark complete: {total_wall:.0f}s total wall time")

    print_results(results)

    report = build_report(results, ts, args.topic, args.seed)
    report_path = run_dir / "bench_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[bench] Report saved: {report_path}")


if __name__ == "__main__":
    main()
