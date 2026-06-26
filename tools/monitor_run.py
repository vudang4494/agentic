#!/usr/bin/env python3
"""Clean live snapshot of a deep-research run + the anti-dup/grounding lever fires.

Reads output/runs/<name>/state.json for per-section outcomes and (optionally) the
stdout log for lever-fire counts, then prints one compact, noise-free report.

Usage:
    python3 tools/monitor_run.py <run-name> [--log <path>] [--target 6]
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "output" / "runs"

# log signal -> human label (counts how many times each lever fired)
LEVER_PATTERNS = {
    "seed-grounded auto-inject": "#2 seed-grounding inject",
    "cross-chapter relate": "#3 outline relate pass",
    "G6 NEAR-DUP": "#7 in-loop near-dup block",
    "G6 TRIM": "#7 paragraph trim",
    "verify-revise armed": "#4 writer verify-revise",
    "ReAct re-dispatch": "#5 re-dispatch tried",
    "RECOVERED the section": "#5 section recovered",
    "auto-disabled; routing academic queries via Tavily": "arxiv->tavily takeover",
}


def _pid_etime(run_name: str):
    try:
        out = subprocess.run(["pgrep", "-f", f"out-name {run_name}"], capture_output=True, text=True).stdout.split()
        if not out:
            return None, None
        pid = out[0]
        et = subprocess.run(["ps", "-o", "etime=", "-p", pid], capture_output=True, text=True).stdout.strip()
        return pid, et
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--log", default="")
    ap.add_argument("--target", type=int, default=0, help="expected total sections")
    args = ap.parse_args()

    run_dir = RUNS / args.run
    sj = run_dir / "state.json"
    pid, et = _pid_etime(args.run)
    alive = pid is not None

    print(f"=== RUN {args.run} | {'RUNNING pid '+pid+' etime '+et if alive else 'NOT running'} ===")

    sections = {}
    if sj.is_file():
        try:
            sections = json.loads(sj.read_text()).get("sections", {})
        except Exception:
            pass
    n = len(sections)
    tgt = args.target or n
    ok = sum(1 for s in sections.values() if s.get("quality") == "ok")
    deg = sum(1 for s in sections.values() if s.get("quality") == "degraded")
    blk = sum(1 for s in sections.values() if s.get("quality") == "BLOCKED")
    tot_w = sum(len((s.get("content") or "").split()) for s in sections.values())
    print(f"sections: {n}{'/'+str(tgt) if tgt else ''} done | ok={ok} degraded={deg} blocked={blk} | ~{tot_w} words")

    for k in sorted(sections.keys()):
        s = sections[k]
        q = s.get("quality", "?")
        mark = {"ok": "OK ", "degraded": "deg", "BLOCKED": "BLK"}.get(q, q[:3])
        w = len((s.get("content") or "").split())
        print(f"  [{mark}] {k} {(s.get('title','') or '')[:46]:46} | {w:4}w "
              f"cites={s.get('n_citations',0)} xref={s.get('cross_refs',0)} "
              f"topic={s.get('topic_relevance',0):.2f}")

    if args.log and os.path.isfile(args.log):
        try:
            txt = Path(args.log).read_text(errors="ignore")
        except Exception:
            txt = ""
        print("--- lever fires (from log) ---")
        for pat, label in LEVER_PATTERNS.items():
            c = txt.count(pat)
            if c:
                print(f"  {label}: {c}")
        # current activity: last non-noise line
        noise = re.compile(r"Loading weights|Materializing|it/s\]")
        last = [ln for ln in txt.replace("\r", "\n").splitlines() if ln.strip() and not noise.search(ln)]
        if last:
            print(f"--- now: {last[-1][:100]}")
    print(f"(snapshot {time.strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
