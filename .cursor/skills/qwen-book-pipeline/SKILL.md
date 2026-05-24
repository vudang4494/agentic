---
name: qwen-book-pipeline
description: Implement end-to-end Qwen3.6-35B-A3B book pipeline for generating bilingual EN-VN technical PDFs. Use when building technical books, generating PDFs from prompts, orchestrating LLM pipelines with vLLM, managing STORM-style research workflows, or when users mention "book pipeline", "Qwen", "technical book", "bilingual PDF", or "generate book".
---

# Qwen Book Pipeline

## Quick Start

```bash
# Install vLLM + deps
python3.11 -m venv .venv && source .venv/bin/activate
uv pip install -U 'vllm>=0.11.0' && pip install -r requirements.txt

# Tavily API key
export TAVILY_API_KEY="tvly-..."

# Start vLLM server (4 hardware presets)
./scripts/start_vllm.sh 24gb   # or 48gb, 96gb, h100x8

# Run pipeline
python -m src.pipeline "Viết handbook về LLM bilingual EN-VN, ~400 pages, depth handbook"
```

## Architecture — 6 Stages

```
STAGE 0: Spec Compiler   (1 LLM, ~5s)  → BookSpec JSON
STAGE 1: STORM Outline   (~30 LLM + 30 searches, ~5 min) → BookOutline tree
STAGE 2: Research        (~500 leaves × 5 queries, ~30 min) → Map[leaf → sources+facts]
STAGE 3: Writer         (~500 LLM, 2-4 hours) → Map[leaf → markdown]
STAGE 4: Citations      (deterministic, ~5 min) → references.bib
STAGE 5: Quality Gates  (~500 LLM judge, ~15 min) → pass/warn/fail
STAGE 6: Render PDF     (Pandoc → XeLaTeX, ~2 min) → book.pdf
```

## Project Structure

```
qwen-pipeline/
├── scripts/start_vllm.sh     # vLLM server, 4 hardware presets
├── src/
│   ├── config.py             # PipelineConfig, LLMConfig, SearchConfig
│   ├── schemas.py            # Pydantic models for 6 stages
│   ├── prompts.py            # All prompts centralized
│   ├── llm_client.py         # Qwen3.6 async client
│   ├── search_client.py      # Tavily client + on-disk cache
│   ├── stage_outline.py      # Stages 0+1 (spec + STORM)
│   ├── stage_research.py     # Stage 2 (per-leaf research)
│   ├── stage_writer.py       # Stage 3 (writer w/ coherence context)
│   ├── stage_citations.py    # Stage 4 (URL → BibEntry)
│   ├── stage_quality.py      # Stage 5 (gates + LLM judge)
│   ├── stage_render.py       # Stage 6 (Pandoc → PDF)
│   └── pipeline.py           # Main orchestrator + CLI
├── bookkit/                  # Render layer (Pandoc/XeLaTeX)
└── output/
    ├── .checkpoints/         # Stage checkpoints (resumable)
    ├── .search_cache/        # Tavily query cache (SHA256)
    └── book.pdf
```

## Key Design

- **Concurrency**: sequential within chapter (coherence via prev_summary), parallel across chapters via vLLM continuous batching
- **Checkpointing**: each stage saves JSON to `output/.checkpoints/`. Resume loads checkpoint, skips completed stages. Force re-run: `--no-resume`
- **Thinking mode**: OFF for stages 0/2/4/5 (deterministic), ON for stages 1/3 (reasoning benefits)
- **JSON decoding**: vLLM guided JSON → zero parse failures
- **Research cache**: SHA256(query) → JSON on disk, deduplicate cross-leaves

## Hardware Presets (`scripts/start_vllm.sh`)

| Preset | GPU | Quant | Context | Model VRAM |
|--------|-----|-------|---------|------------|
| 24gb | RTX 4090 | AWQ Q4 | 128K | ~21 GB |
| 48gb | RTX 6000 Ada / 2×4090 | FP8 | 256K | ~37 GB |
| 96gb | RTX PRO 6000 | BF16 | 256K | ~71 GB |
| h100x8 | 8×H100 80GB | BF16 TP=8 | 1M | ~70 GB |

## Config Tuning

- **Throughput**: `src/config.py > LLMConfig.max_concurrent_requests` — higher = faster but OOM risk
- **Research depth**: `SearchConfig.search_depth="basic"` ($0.005) vs `"advanced"` ($0.01)
- **Writer**: `temperature_creative` (default 0.7), `use_thinking_writer` (default True), `max_tokens_writer` (default 6000)
- **Prompts**: edit `src/prompts.py` only — all prompts centralized

## Stage 3 — Writer (Bottleneck, 80% time)

6 context inputs per writer call:

| Input | Source | Purpose |
|-------|--------|---------|
| Atomic facts | Stage 2 | Citations with URL |
| Prev subsection summary | Same chapter, prior leaf | Smooth transition |
| Next subsection title | Same chapter, next leaf | Avoid preempting |
| Notation table | Stage 1 | Symbol consistency |
| Subsection spec | Outline | id, word_budget, depth_hint |
| Style guide | BookSpec | Bilingual rules, math, code |

Output: `{markdown: str, local_refs: dict[int, str]}` — `local_refs` maps `[N]` to URLs.

## Stage 4 — Citation Resolution

Deterministic (0 LLM calls):
1. Collect all `(leaf_id, local_num, url)` from `local_refs`
2. Resolve URL → metadata: arXiv API (free) for arXiv URLs, Semantic Scholar API for academic, HTML `<title>` scrape fallback
3. Dedupe by BibTeX key
4. Rewrite `[N]` → `[@bibkey]` (Pandoc format)
5. Write `references.bib`

## Stage 5 — Quality Gates

**Deterministic** (regex, fast): citation density, math balance, code fences, env balance, word budget ±15%, no hallucinated refs.

**LLM-judge** (slow): no meta-commentary, coherence with prev, no verbatim repetition.

Failed leaves → log warnings + optional retry with augmented prompt.

## Checkpoint Schema

```
output/.checkpoints/
├── 00-spec.json              # BookSpec
├── 01-outline.json           # BookOutline (3-level tree, ~500 leaves)
├── 02-research.json          # Map[leaf_id → LeafResearch]
├── 03-content.json           # Map[leaf_id → LeafContent] (pre-citation)
├── 04-bib.json               # list[BibEntry]
├── 04-content-cited.json     # Map[leaf_id → LeafContent] (post-citation)
└── 05-quality.json           # Map[leaf_id → QualityReport]
```

## Failure Modes

| Failure | Frequency | Mitigation |
|---------|-----------|------------|
| Hallucinated citations | High | Stage 5 check + retry; Stage 4 verify bibkey exists |
| Notation drift cross-chapter | Medium | Pre-compute notation table in Stage 1 |
| Same content repeated | Medium | Pass prev_summary to writer; LLM-judge catches |
| Math doesn't render | Low | Lint script catches before xelatex |
| Word budget overflow | Medium | Quality gate warns; retry with shorten instruction |
| vLLM OOM mid-pipeline | Low | Checkpoint resume; reduce max_concurrent_requests |
| Tavily rate limit | Low | Semaphore-bounded concurrency + retry + cache |
| Pandoc XeLaTeX fail | Low | Lint catches most |

## Reference

- Full file-by-file breakdown: see [pipeline-reference.md](pipeline-reference.md)
- CLAUDE.md in `files/`: project configs, code standards, hyperparameters
- WORKPLAN.md in `files/`: 8-week execution plan, budget, decision gates
