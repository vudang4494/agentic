#!/usr/bin/env python3
"""Lightweight progress monitor for the Deep Research Pipeline.

Run: python3 files/monitor.py
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE / "output"

STATE  = OUT_DIR / "state.json"
REPORT = OUT_DIR / "report.json"
WORDS_PER_PAGE = 400


def _total_sections() -> int:
    try:
        sys.path.insert(0, str(HERE))
        import deep_research
        return sum(len(c["passes"]) for c in deep_research.CHAPTERS)
    except Exception:
        return 96


TOTAL = _total_sections()


def get_progress():
    if not STATE.exists():
        return None
    with open(STATE) as f:
        d = json.load(f)
    return {
        "written": len(d.get("passes", {})),
        "words":   d.get("total_words", 0),
        "tokens":  d.get("total_tokens", 0),
        "age":     time.time() - os.path.getmtime(STATE),
    }


def get_report():
    if REPORT.exists():
        with open(REPORT) as f:
            return json.load(f)
    return None


def main():
    print("Deep Research Pipeline -- Progress Monitor")
    print("=" * 55)
    print(f"State:  {STATE}")
    print(f"Report: {REPORT}")
    print(f"Total sections: {TOTAL}")
    print("=" * 55)
    print()

    last = None
    while True:
        p = get_progress()
        if p:
            pct = f"{100 * p['written'] / max(TOTAL, 1):.0f}%"
            pages = p["words"] // WORDS_PER_PAGE
            line = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"{p['written']}/{TOTAL} ({pct}) | "
                f"Words: {p['words']:,} (~{pages}p) | "
                f"Tokens: {p['tokens']:,} | "
                f"Last update: {p['age']:.0f}s ago"
            )
            if line != last:
                print(line, flush=True)
                last = line
            if p["written"] >= TOTAL:
                print("\nDONE! Check files/output/book.pdf")
                r = get_report()
                if r:
                    print(f"  Time:  {r.get('total_time_min', '?')} min")
                    print(f"  Pages: ~{r.get('pages', '?')}")
                sys.exit(0)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for pipeline to start...", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
