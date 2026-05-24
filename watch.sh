#!/bin/bash
# watch.sh -- one-shot snapshot of pipeline progress.
# For continuous tailing, use: ./run.sh watch  (or python3 files/monitor.py)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STATE="files/output/state.json"
STDOUT_LOG="files/output/pipeline.stdout.log"

clear
echo "=================================================="
echo "   Deep Research Pipeline -- Progress Snapshot"
echo "=================================================="
echo ""

if [ ! -f "$STATE" ]; then
    echo "  Status: NO STATE FILE (pipeline not started yet)"
    echo ""
    echo "  Start: ./run.sh"
    exit 0
fi

python3 - "$STATE" <<'PY'
import json, sys, time, os
from pathlib import Path

state_path = Path(sys.argv[1])
with open(state_path) as f:
    d = json.load(f)

# Derive total sections from deep_research.CHAPTERS
sys.path.insert(0, "files")
try:
    import deep_research
    total = sum(len(c["passes"]) for c in deep_research.CHAPTERS)
except Exception:
    total = 96

passes = len(d.get("passes", {}))
words = d.get("total_words", 0)
tokens = d.get("total_tokens", 0)
pages = words // 400
pct = passes * 100 // max(total, 1)

print(f"  Sections: {passes}/{total} ({pct}%)")
print(f"  Words:    {words:,} (~{pages}p)")
print(f"  Tokens:   {tokens:,}")
print()

bar_len = 32
filled = passes * bar_len // max(total, 1)
print(f"  [{'=' * filled}{'-' * (bar_len - filled)}]")
print()

if passes >= 2:
    age = time.time() - os.path.getmtime(state_path)
    print(f"  Last update: {int(age)}s ago")
    remaining = total - passes
    eta_min = remaining * 2.5
    if eta_min > 60:
        print(f"  ETA: ~{int(eta_min//60)}h {int(eta_min%60)}m")
    else:
        print(f"  ETA: ~{int(eta_min)}m")
PY

echo ""
echo "=================================================="
echo "  Recent pipeline log:"
echo "--------------------------------------------------"
if [ -f "$STDOUT_LOG" ]; then
    grep -E "OK:|CHECKPOINT|ERROR|REVIEW" "$STDOUT_LOG" | tail -5 | sed 's/^/  /'
else
    echo "  (no log yet)"
fi
echo ""
echo "=================================================="
echo "  Commands:"
echo "    ./run.sh watch                 (live monitor)"
echo "    tail -f $STDOUT_LOG"
echo "    pkill -f files/deep_research.py   (kill)"
echo "    ./run.sh                       (re-run / resume)"
echo "=================================================="
