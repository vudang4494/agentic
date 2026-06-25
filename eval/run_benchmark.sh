#!/bin/bash
# Multi-run benchmark driver: run the FIXED pipeline on N diverse topics (FULL pipeline, natural
# discovery scale ~96 sections each), BAER-evaluate each, then aggregate into one multi-run
# benchmark (mean +/- std). Sequential (one local Ollama/GPU). Resume-safe: deep_research_v3.py
# continues each topic from its state.json, so re-running this script picks up where it left off.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"
mkdir -p output/runs
LOG="output/runs/benchmark.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# topic | out-name | canonical-arxiv-ids (foundational papers, P0b-injected + protected)
TOPICS=(
  "Reinforcement Learning from Human Feedback (RLHF)|bench_rlhf|2203.02155,2305.18290,1707.06347"
  "Diffusion Models for Generative AI|bench_diffusion|2006.11239,2112.10752"
  "Retrieval-Augmented Generation (RAG)|bench_rag|2005.11401,2004.04906"
  "Mixture-of-Experts (MoE) in Large Language Models|bench_moe|2101.03961,1701.06538"
)

log "##### MULTI-RUN BENCHMARK START (${#TOPICS[@]} topics, full pipeline, natural scale) #####"
RUNS=()
for entry in "${TOPICS[@]}"; do
  IFS='|' read -r topic out ids <<< "$entry"
  RUNS+=("$out")
  log "=== TOPIC: $topic  ->  $out ==="
  python3 pipeline/deep_research_v3.py --topic "$topic" --out-name "$out" \
    --no-smoke --render --canonical-arxiv-ids "$ids" >> "output/runs/$out.log" 2>&1
  rc=$?
  log "  pipeline rc=$rc ; running BAER for $out"
  python3 eval/benchmark_book.py "$out" > "output/runs/$out.baer.log" 2>&1
  log "  BAER done -> output/runs/$out/book_eval_report.md"
done

log "=== AGGREGATE -> output/benchmark.{md,json} ==="
python3 eval/aggregate_benchmark.py --out output/benchmark "${RUNS[@]}" 2>&1 | tee -a "$LOG"
log "##### BENCHMARK COMPLETE #####"
