#!/usr/bin/env python3
"""Zero-duplicate verifier for a generated book (the no-dup target's acceptance test).

Reads a run's state.json (or book.md), then measures REAL content duplication with the
same local signal the pipeline uses (bge-m3 cosine) at two granularities:

  1. SECTION level  -- pairwise cosine of section bodies; pairs >= --sec-thresh (default
     0.85, the pipeline's dedup_cosine_max) are near-duplicate sections.
  2. PARAGRAPH level -- every paragraph vs every other section's paragraphs; pairs
     >= --para-thresh (default 0.92, the pipeline's trim threshold) are repeated content.

PASS = zero near-duplicate sections AND zero duplicate paragraphs. Exits non-zero on FAIL
so it can gate a run. LOCAL only (bge-m3 via Ollama); never calls an external API.

Usage:
    python3 eval/check_dedup.py <run-name-or-dir>
    python3 eval/check_dedup.py antidup_smoke --sec-thresh 0.85 --para-thresh 0.92
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research.embeddings import embed, cosine  # noqa: E402

RUNS = ROOT / "output" / "runs"


def _load_sections(run_dir: Path):
    """Return [(title, body)] from state.json (preferred) or book.md fallback."""
    sj = run_dir / "state.json"
    if sj.is_file():
        d = json.loads(sj.read_text())
        secs = d.get("sections", {})
        out = []
        for k in sorted(secs.keys()):
            s = secs[k]
            body = s.get("content", "") or ""
            if s.get("quality") == "BLOCKED" or body.lstrip().startswith("[BLOCKED"):
                continue
            out.append((s.get("title", k), body))
        return out
    bm = run_dir / "book.md"
    if bm.is_file():
        text = bm.read_text()
        parts = re.split(r"^##\s+(.+)$", text, flags=re.MULTILINE)
        out = []
        for i in range(1, len(parts), 2):
            out.append((parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else ""))
        return out
    raise FileNotFoundError(f"no state.json or book.md in {run_dir}")


def _paras(body: str):
    return [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name under output/runs/ or a path")
    ap.add_argument("--sec-thresh", type=float, default=0.85)
    ap.add_argument("--para-thresh", type=float, default=0.92)
    args = ap.parse_args()

    run_dir = Path(args.run) if os.path.sep in args.run else RUNS / args.run
    sections = _load_sections(run_dir)
    if len(sections) < 2:
        print(f"[dedup] only {len(sections)} section(s) -- nothing to compare")
        return
    print(f"[dedup] {run_dir.name}: {len(sections)} non-blocked sections")

    # --- Section-level near-dup ---
    titles = [t for t, _ in sections]
    svecs = embed([b[:1500] for _, b in sections], model="bge-m3:latest")
    sec_dups = []
    if svecs and len(svecs) == len(sections):
        for i in range(len(sections)):
            for j in range(i):
                c = cosine(svecs[i], svecs[j])
                if c >= args.sec_thresh:
                    sec_dups.append((round(c, 3), titles[i], titles[j]))
    sec_dups.sort(reverse=True)

    # --- Paragraph-level near-dup (across different sections) ---
    flat = []  # (sec_idx, para_text)
    for si, (_, body) in enumerate(sections):
        for p in _paras(body):
            flat.append((si, p))
    para_dups = []
    if flat:
        pvecs = embed([p[:600] for _, p in flat], model="bge-m3:latest")
        if pvecs and len(pvecs) == len(flat):
            for a in range(len(flat)):
                for b in range(a):
                    if flat[a][0] == flat[b][0]:
                        continue  # same section
                    c = cosine(pvecs[a], pvecs[b])
                    if c >= args.para_thresh:
                        para_dups.append((round(c, 3), titles[flat[a][0]], titles[flat[b][0]]))
    para_dups.sort(reverse=True)

    print(f"\n=== SECTION near-dups (cosine >= {args.sec_thresh}): {len(sec_dups)} ===")
    for c, a, b in sec_dups[:20]:
        print(f"  {c}  '{a[:45]}'  ~=  '{b[:45]}'")
    print(f"\n=== PARAGRAPH duplicates (cosine >= {args.para_thresh}): {len(para_dups)} ===")
    for c, a, b in para_dups[:20]:
        print(f"  {c}  para in '{a[:40]}'  ~=  para in '{b[:40]}'")

    ok = (not sec_dups) and (not para_dups)
    print(f"\nRESULT: {'PASS (zero duplicate content)' if ok else 'FAIL'} "
          f"-- {len(sec_dups)} dup sections, {len(para_dups)} dup paragraphs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
