#!/usr/bin/env python3
"""
Research-quality eval harness.

Usage:
    python3 files/eval/run_eval.py --topic transformer
    python3 files/eval/run_eval.py --topic transformer --skip-pipeline  # re-score archived run

Workflow:
    1. Load files/eval/topics/<name>.yaml (gold standard)
    2. Run deep_research.run() with the topic's n_chapters / n_passes
    3. Read files/output/eval_<name>_<ts>.state.json
    4. Compute per-section + aggregate + pass/fail metrics
    5. Write files/eval/reports/eval_<name>_<ts>.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml  # type: ignore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "files"))

from eval import metrics as M  # noqa: E402


def _load_gold(topic_slug: str) -> dict:
    path = HERE / "topics" / f"{topic_slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"gold topic not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _run_pipeline(gold: dict, out_name: str) -> dict:
    """Invoke deep_research.run() with the gold topic. Returns timings dict."""
    import deep_research  # noqa: WPS433 -- dynamic import after sys.path adjust

    t0 = time.time()
    print(f"\n=== EVAL: launching pipeline for topic={gold['topic']!r} ===")
    print(f"  n_chapters={gold.get('n_chapters', 12)}, n_passes={gold.get('n_passes', 8)}")
    print(f"  out_name={out_name}")
    print()
    deep_research.run(
        batch=1,
        start_ch=1, start_pp=1,
        end_ch=gold.get("n_chapters", 12),
        render=False,
        review=False,                # keep eval focused on retrieval+grounding
        research=True,
        topic=gold["topic"],
        n_chapters=gold.get("n_chapters", 12),
        n_passes=gold.get("n_passes", 8),
        out_name=out_name,
    )
    elapsed = time.time() - t0
    print(f"\n=== pipeline finished in {elapsed/60:.1f}min ===")
    return {"elapsed_sec": elapsed}


def _load_state(out_name: str) -> dict:
    state_path = ROOT / "files" / "output" / f"{out_name}.state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"state.json not found at {state_path}")
    with open(state_path) as f:
        return json.load(f)


def _assemble_book_text(state: dict) -> str:
    """Concat all section content in chapter+section order."""
    passes = state.get("passes", {})
    ordered = sorted(passes.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(".")))
    return "\n\n".join(v.get("content", "") for _, v in ordered)


def _score(state: dict, gold: dict) -> dict:
    passes = state.get("passes", {})
    per_section = [
        M.section_metrics(k, passes[k], gold)
        for k in sorted(passes.keys(), key=lambda x: tuple(int(p) for p in x.split(".")))
    ]
    book_text = _assemble_book_text(state)
    agg = M.aggregate(per_section, gold, book_text)
    pf = M.pass_fail(agg, gold.get("thresholds", {}))
    return {"per_section": per_section, "aggregate": agg, "pass_fail": pf}


def _write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_md_report(path: Path, payload: dict) -> None:
    g = payload["gold"]
    agg = payload["scores"]["aggregate"]
    pf = payload["scores"]["pass_fail"]
    per = payload["scores"]["per_section"]

    lines: list[str] = []
    lines.append(f"# Eval report -- {g['topic']}")
    lines.append("")
    lines.append(f"- Run ID: `{payload['run_id']}`")
    lines.append(f"- Difficulty: `{g.get('difficulty', '?')}`")
    lines.append(f"- Outline: {g.get('n_chapters')} chapters x {g.get('n_passes')} passes "
                 f"= {agg['n_sections']} sections actually generated")
    if payload.get("timings", {}).get("elapsed_sec"):
        lines.append(f"- Elapsed: {payload['timings']['elapsed_sec']/60:.1f} min")
    lines.append("")
    lines.append(f"## Overall: {'PASS' if pf['overall_pass'] else 'FAIL'} "
                 f"({pf['n_passed']}/{pf['n_total']} checks passed)")
    lines.append("")

    lines.append("| Check | Target | Actual | Pass |")
    lines.append("|---|---|---|---|")
    for c in pf["checks"]:
        mark = "OK" if c["pass"] else "FAIL"
        lines.append(f"| `{c['check']}` | {c['op']} {c['target']} | {c['actual']} | {mark} |")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k in ("must_cite_recall", "should_cite_recall", "grounding_mean",
              "zero_cite_section_count", "loop_section_count", "loop_section_pct",
              "research_round_2_rate", "forbidden_domain_hits",
              "subtopic_coverage", "median_words",
              "mean_citations_per_1000w", "total_tokens"):
        lines.append(f"| `{k}` | {agg[k]} |")
    lines.append("")
    if agg.get("must_cite_missed"):
        lines.append("**Missed must-cite arxiv IDs:** "
                     + ", ".join(f"`{x}`" for x in agg["must_cite_missed"]))
        lines.append("")
    if agg.get("subtopic_missing"):
        lines.append("**Missing expected subtopics:** "
                     + ", ".join(f"`{x}`" for x in agg["subtopic_missing"]))
        lines.append("")

    lines.append("## Per-section metrics")
    lines.append("")
    lines.append("| Sec | Title | Words | Cites | Grounding | Sources | Round | Zero? | Loop? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in per:
        g_score = s["grounding"]["score"]
        g_str = f"{g_score:.2f}" if isinstance(g_score, (int, float)) else "-"
        zero = "Y" if s["grounding"]["zero_cite_red_flag"] else "-"
        loop = "Y" if s["output"]["looping_detected"] else "-"
        title = (s["title"] or "")[:32]
        lines.append(f"| {s['key']} | {title} | {s['output']['word_count']} | "
                     f"{s['grounding']['n_citations_in_text']} | {g_str} | "
                     f"{s['retrieval']['n_sources']} | "
                     f"{s['retrieval']['research_rounds']} | "
                     f"{zero} | {loop} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description="Research-quality eval harness")
    p.add_argument("--topic", required=True,
                   help="topic slug under files/eval/topics/<slug>.yaml")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="re-score an existing state.json instead of running pipeline")
    p.add_argument("--state",
                   help="explicit state.json path (with --skip-pipeline)")
    p.add_argument("--out-name",
                   help="override the auto-generated --out-name for the pipeline run")
    args = p.parse_args()

    gold = _load_gold(args.topic)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"eval_{args.topic}_{run_id}"
    timings = {}
    if not args.skip_pipeline:
        timings = _run_pipeline(gold, out_name)
        state = _load_state(out_name)
    else:
        state_path = Path(args.state) if args.state else ROOT / "files" / "output" / f"{out_name}.state.json"
        with open(state_path) as f:
            state = json.load(f)

    scores = _score(state, gold)
    payload = {
        "run_id": run_id,
        "topic_slug": args.topic,
        "gold": gold,
        "timings": timings,
        "scores": scores,
    }
    json_path = HERE / "reports" / f"eval_{args.topic}_{run_id}.json"
    md_path = HERE / "reports" / f"eval_{args.topic}_{run_id}.md"
    _write_json_report(json_path, payload)
    _write_md_report(md_path, payload)

    print()
    print(f"=== EVAL DONE ===")
    print(f"  JSON report: {json_path}")
    print(f"  MD report:   {md_path}")
    print()
    pf = scores["pass_fail"]
    print(f"Overall: {'PASS' if pf['overall_pass'] else 'FAIL'} "
          f"({pf['n_passed']}/{pf['n_total']} checks)")
    for c in pf["checks"]:
        mark = "OK" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['check']:<28} {c['op']} {c['target']:<10} actual={c['actual']}")
    return 0 if pf["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
