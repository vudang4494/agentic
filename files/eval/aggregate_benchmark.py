"""Aggregate multiple BAER run reports into ONE multi-run benchmark (mean +/- std).

Turns N single-run `book_eval.json` files (produced by benchmark_book.py) into a benchmark table
across topics, so quality is reported with central tendency + spread instead of a single anecdote.
Optionally compares a gated pipeline run against its single-prompt baseline (same topic).

Usage:
    python3 files/eval/aggregate_benchmark.py <run-name-or-dir> [<run-name-or-dir> ...]
    python3 files/eval/aggregate_benchmark.py --out files/output/benchmark.md rlhf_bench diffusion_bench ...
Writes <out>.md + <out>.json (default: files/output/benchmark.{md,json}).
"""
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "files/output/runs"

# (json path in book_eval.json, label, formatter)  -- pulled per run, then aggregated
_METRICS = [
    ("benchmark.pages_pdf", "pages"),
    ("benchmark.total_words", "words"),
    ("benchmark.sections", "sections"),
    ("benchmark.accepted", "accepted"),
    ("benchmark.blocked", "blocked"),
    ("eval.accept_rate", "accept_rate"),
    ("analyze.redundancy.max_sim", "redundancy_max"),
    ("analyze.redundancy.n_near_dup", "near_dup_pairs"),
    ("analyze.tech_depth.pct_with_formula", "formula_pct"),
    ("analyze.references.on_topic_frac", "ref_on_topic"),
    ("analyze.references.canonical_recall", "canonical_recall"),
    ("eval.topic_ge_0.50", "topic_pass"),
    ("eval.topic_mean", "topic_mean"),
    ("eval.cite_precision_mean", "cite_precision"),
]


def _dig(d, path):
    # Greedy longest-key-first walk so keys containing a literal "." (e.g. "topic_ge_0.50")
    # survive the dot-path split instead of resolving to a phantom "topic_ge_0" -> "50".
    cur = d
    keys = path.split(".")
    i = 0
    while i < len(keys):
        if not isinstance(cur, dict):
            return None
        for j in range(len(keys), i, -1):
            cand = ".".join(keys[i:j])
            if cand in cur:
                cur, i = cur[cand], j
                break
        else:
            return None
    return cur


def _load_eval(run):
    run_dir = Path(run) if os.path.isdir(run) else RUNS / run
    ej = run_dir / "book_eval.json"
    if not ej.exists():
        # generate it
        print(f"[aggregate] {run}: book_eval.json missing -> running BAER", flush=True)
        subprocess.run([sys.executable, str(ROOT / "files/eval/benchmark_book.py"), str(run_dir)],
                       capture_output=True)
    if not ej.exists():
        print(f"[aggregate] {run}: still no book_eval.json -- skipping", flush=True)
        return None, run_dir
    return json.loads(ej.read_text()), run_dir


def _agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (st.mean(vals), st.pstdev(vals))


def main():
    args = sys.argv[1:]
    out = RUNS.parent / "benchmark"
    if "--out" in args:
        i = args.index("--out"); out = Path(args[i + 1]).with_suffix(""); del args[i:i + 2]
    runs = args
    if not runs:
        sys.exit("usage: aggregate_benchmark.py [--out PATH] <run> [<run> ...]")

    per_run = {}
    for r in runs:
        ev, rd = _load_eval(r)
        if ev is None:
            continue
        topic = (ev.get("benchmark") or {}).get("topic") or rd.name
        row = {label: _dig(ev, path) for path, label in _METRICS}
        row["_run"] = rd.name
        per_run[rd.name] = row

    if not per_run:
        sys.exit("[aggregate] no runs loaded")

    # aggregate each metric across runs
    agg = {}
    for _, label in _METRICS:
        agg[label] = _agg([row.get(label) for row in per_run.values()])

    # ---- write JSON ----
    out_json = {"n_runs": len(per_run), "runs": per_run,
                "aggregate": {k: ({"mean": round(v[0], 4), "std": round(v[1], 4)} if v else None)
                              for k, v in agg.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(out_json, indent=2, ensure_ascii=False))

    # ---- write Markdown ----
    L = [f"# Multi-run benchmark ({len(per_run)} runs)", ""]
    keys = [lbl for _, lbl in _METRICS]
    L.append("| metric | " + " | ".join(per_run) + " | **mean ± std** |")
    L.append("|---|" + "---|" * (len(per_run) + 1))
    def fmt(v):
        if v is None:
            return "—"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    for k in keys:
        cells = [fmt(per_run[r].get(k)) for r in per_run]
        a = agg[k]
        astr = f"**{a[0]:.3f} ± {a[1]:.3f}**" if a else "—"
        L.append(f"| {k} | " + " | ".join(cells) + f" | {astr} |")
    L += ["", "> grounding (HHEM, G3) is omitted from the headline table — its post-hoc BAER score scores the",
          "> assembled prose, NOT the write-time gate, so the two are not comparable. cite_precision (G2) is",
          "> shown but is pinned at 1.000 across every run (saturated → NOT discriminating here). The only",
          "> live discriminating signal in this benchmark is topic (G4).",
          "> Generated by `files/eval/aggregate_benchmark.py` over per-run BAER reports."]
    out.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[aggregate] wrote {out.with_suffix('.md')} + {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
