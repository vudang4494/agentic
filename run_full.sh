#!/bin/bash
# ============================================================
# run_full.sh -- Full end-to-end v3 pipeline
#
# Lam moi thu tu dau den cuoi, khong biet truoc kich ban.
# Prompt thô -> Gemma4 -> Discovery -> Outline -> Investigate
# -> Qwen viet -> Verify -> Assemble -> PDF
#
# Usage:
#   TOPIC="RLHF" OUT_NAME="rlhf_v5" \
#     CANONICAL_IDS="2203.02155,2305.18290,..." \
#     ./run_full.sh
#
# Hoac:
#   export TOPIC="RLHF"
#   ./run_full.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Config ----
TOPIC="${TOPIC:-}"
OUT_NAME="${OUT_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
CANONICAL_IDS="${CANONICAL_IDS:-}"
N_ROUNDS="${N_ROUNDS:-2}"
MODEL_DISCOVERY="${MODEL_DISCOVERY:-gemma4:e4b}"
MODEL_WRITER="${MODEL_WRITER:-batiai/qwen3.6-35b:iq3}"

# ---- Helpers ----
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---- Validate ----
[ -n "$TOPIC" ] || die "TOPIC chua dat. Dung: TOPIC=\"RLHF\" ./run_full.sh"

RUN_DIR="$SCRIPT_DIR/files/output/runs/$OUT_NAME"

# ---- Pre-flight checks ----
log "=== PRE-FLIGHT CHECKS ==="

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    die "Ollama khong chay. Start: ollama serve"
fi
log "Ollama: OK"

DISCO_OK=$(curl -s http://localhost:11434/api/tags | python3 -c "
import json,sys
d=json.load(sys.stdin)
names=[m['name'] for m in d.get('models',[])]
print('yes' if '$MODEL_DISCOVERY' in names else 'no')
" 2>/dev/null) || DISCO_OK="no"
[ "$DISCO_OK" = "yes" ] || die "Model discovery '$MODEL_DISCOVERY' chua co. Pull: ollama pull $MODEL_DISCOVERY"
log "Discovery model ($MODEL_DISCOVERY): OK"

WRIT_OK=$(curl -s http://localhost:11434/api/tags | python3 -c "
import json,sys
d=json.load(sys.stdin)
names=[m['name'] for m in d.get('models',[])]
print('yes' if '$MODEL_WRITER' in names else 'no')
" 2>/dev/null) || WRIT_OK="no"
[ "$WRIT_OK" = "yes" ] || die "Model writer '$MODEL_WRITER' chua co. Pull: ollama pull $MODEL_WRITER"
log "Writer model ($MODEL_WRITER): OK"

# ---- Setup output dir ----
log "=== SETUP ==="
log "Topic: $TOPIC"
log "Out: $RUN_DIR"
log "Canonical IDs: ${CANONICAL_IDS:-none}"
mkdir -p "$RUN_DIR"

# ---- Stage 0+1+2: Discovery + Outline + Investigate ----
log "=== PIPELINE: DISCOVERY -> OUTLINE -> INVESTIGATE -> ASSEMBLE ==="

python3 files/deep_research_v3.py \
    --topic "$TOPIC" \
    --out-name "$OUT_NAME" \
    --max-rounds "$N_ROUNDS" \
    --no-smoke \
    ${CANONICAL_IDS:+"--canonical-arxiv-ids" "$CANONICAL_IDS"}

# ---- Check result ----
if [ ! -f "$RUN_DIR/book.md" ]; then
    die "Pipeline chua tao book.md. Co loi xay ra."
fi

log "=== PIPELINE COMPLETE ==="

# ---- Stats ----
if [ -f "$RUN_DIR/state.json" ]; then
    python3 -c "
import json, sys
try:
    s = json.load(open('$RUN_DIR/state.json'))
    secs = s.get('sections', {})
    total_w = s.get('total_words', 0)
    g_list = [v.get('grounding', 0) for v in secs.values() if v.get('grounding')]
    print(f'  Sections: {len(secs)}')
    print(f'  Total words: {total_w:,}')
    if g_list:
        print(f'  Avg grounding: {sum(g_list)/len(g_list):.3f}')
        print(f'  All g=1.0: {all(g==1.0 for g in g_list)}')
except Exception as e:
    print(f'  Stats error: {e}')
" 2>&1
fi

# ---- PDF render ----
# Use the ROBUST renderer (research/mathfix math normalization + tectonic with -Z continue-on-errors,
# weasyprint fallback). Plain pandoc here would choke on the LaTeX / undefined-macro / raw-% cases the
# pipeline now emits and silently "skip", leaving a successful run with no book.pdf.
if command -v pandoc > /dev/null 2>&1; then
    log "=== RENDER PDF (robust: mathfix + tectonic) ==="
    # NOTE: pipe to tail only truncates the log. Check ${PIPESTATUS[0]} (render_book.py's exit code),
    # NOT the pipeline status (that would be tail's, which is ~always 0 and masks render failures).
    python3 "$SCRIPT_DIR/files/scripts/render_book.py" --run "$OUT_NAME" 2>&1 | tail -3
    if [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "$RUN_DIR/book.pdf" ]; then
        log "PDF: $RUN_DIR/book.pdf"
    else
        log "PDF render FAILED (book.md is intact; re-run: python3 files/scripts/render_book.py --run $OUT_NAME)"
    fi
fi

log "=== DONE ==="
log "Book: $RUN_DIR/book.md"
log "State: $RUN_DIR/state.json"
