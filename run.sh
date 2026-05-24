#!/bin/bash
# ============================================================
# run.sh -- Deep Research Pipeline launcher
# ============================================================
# Stage 1 of the Agentic Deep Research roadmap.
# Default mode wraps the pipeline in an autonomous runner with
# health monitoring, crash recovery, and PDF render on finish.
#
# Usage:
#   ./run.sh             # autonomous runner (recommended)
#   ./run.sh direct      # run pipeline once, no watchdog
#   ./run.sh watch       # tail progress only
#   ./run.sh review      # autonomous runner + LLM-as-judge review pass
#
# Outputs land in files/output/ as book.{md,html,pdf},
# state.json, report.json, pipeline.log, runner.log.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_FILE="$SCRIPT_DIR/files/output/state.json"
RUNNER_LOG="$SCRIPT_DIR/files/output/runner.log"
MODEL="gemma3:4b"

cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Deep Research Pipeline -- Agentic Book Generator"
echo "  Model: $MODEL"
echo "============================================================"
echo ""

echo "[1/3] Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running."
    echo "  Start with: ollama serve"
    exit 1
fi
echo "  Ollama is reachable."

echo ""
echo "[2/3] Checking $MODEL..."
MODEL_OK=$(curl -s http://localhost:11434/api/tags | python3 -c "
import json,sys
data=json.load(sys.stdin)
print('yes' if '$MODEL' in [m['name'] for m in data.get('models',[])] else 'no')
" 2>/dev/null || echo "no")
if [ "$MODEL_OK" != "yes" ]; then
    echo "ERROR: $MODEL not found. Pull with: ollama pull $MODEL"
    exit 1
fi
echo "  $MODEL is available."

echo ""
echo "[3/3] Pipeline status..."
if [ -f "$STATE_FILE" ]; then
    DONE=$(python3 -c "
import json
with open('$STATE_FILE') as f: d=json.load(f)
print(len(d.get('passes', {})))
" 2>/dev/null || echo "0")
    echo "  Checkpoint: $DONE section(s) already done -- resuming."
else
    echo "  No checkpoint -- starting fresh."
fi

echo ""
echo "============================================================"
echo "  Output: files/output/book.pdf"
echo "  Runner log:   $RUNNER_LOG"
echo "  Live stdout:  files/output/pipeline.stdout.log"
echo "============================================================"
echo ""

case "${1:-}" in
    direct)
        echo "Mode: direct (no watchdog)"
        shift
        python3 files/deep_research.py "$@"
        ;;
    review)
        echo "Mode: autonomous + LLM-as-judge review"
        DEEP_RESEARCH_REVIEW=1 python3 files/runner.py
        ;;
    watch)
        echo "Mode: watch (Ctrl+C to stop)"
        python3 files/monitor.py
        ;;
    *)
        echo "Mode: autonomous runner"
        python3 files/runner.py
        ;;
esac
