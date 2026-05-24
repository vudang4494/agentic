#!/bin/bash
# Stage 2++ full-pipeline run -- all fixes active.
# Tavily auto-disables on HTTP 432 (quota exhausted); Brave kicks in if BRAVE_API_KEY is set.
#
# Required env (set externally or in a .env file you source before this script):
#   TAVILY_API_KEY  -- optional; web search ON if set (free 1000 queries/mo at https://tavily.com)
#   BRAVE_API_KEY   -- optional; fallback web search (free 2000/mo at https://brave.com/search/api/)
#
# Without either web-search key, the pipeline falls back to arxiv + Wikipedia + DDG only.
set -e
cd "$(dirname "$0")/.."

if [ -z "${TAVILY_API_KEY:-}" ] && [ -z "${BRAVE_API_KEY:-}" ]; then
    echo "[launch_book2] note: no TAVILY_API_KEY or BRAVE_API_KEY set -- web search disabled, arxiv+wiki+ddg only" >&2
fi

export DEEP_RESEARCH_REVIEW=1
export DEEP_RESEARCH_OUT_NAME=book2
# No --topic -> uses the hardcoded 96-section LLM outline for apples-to-apples comparison
exec python3 files/runner.py
