# agentic

> **Local-first Agentic Deep Research platform.** Hand it a topic; it plans a
> 12-chapter outline, runs ~100 atomic *research → write → verify* rounds, and
> assembles a 300-400+ page, LaTeX-typeset technical book with a single
> deduplicated References page — all on your laptop, no API key required.

<p align="left">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="python"  src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="runtime" src="https://img.shields.io/badge/runtime-Ollama-black.svg">
  <img alt="render"  src="https://img.shields.io/badge/render-tectonic%20%2F%20WeasyPrint-orange.svg">
  <img alt="status"  src="https://img.shields.io/badge/status-stage%203%2B%20shipped-success.svg">
</p>

Default stack is Ollama-served `gemma3:4b` (writer) + `qwen3.5:4b` (query gen
+ judge) + `bge-m3` (embedder), with arXiv, Wikipedia, Tavily, Brave, and
DuckDuckGo as research providers. Provider-agnostic — swap in any
OpenAI-compatible endpoint.

<p align="center">
  <img src="pipeline.jpg" alt="Agentic Deep Research pipeline" width="640">
  <br>
  <em>End-to-end pipeline: planner → per-section research / write / verify loop → assemble & render.</em>
</p>

---

## Table of contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [CLI reference](#cli-reference)
- [Environment variables](#environment-variables)
- [File layout](#file-layout)
- [How it stays grounded](#how-it-stays-grounded)
- [Roadmap](#roadmap)
- [License](#license)

---

## Highlights

| Capability | What ships today |
|---|---|
| **Local-first** | Ollama-served LLMs only (default `gemma3:4b`, override via env) |
| **Multi-provider retrieval** | arXiv + Wikipedia + Tavily + Brave + DDG, all gated by env |
| **Full-text grounding** | top-2 sources per section get a 350-word body extract, not a 80-word snippet |
| **Self-critique** | LLM-as-judge scores per-`[N]` citation grounding and triggers re-search |
| **Iterative loop** | low-grounding sections re-query with reviewer hint, capped at 2 rounds |
| **Concept memory** | every concept introduced is tracked; later sections see an "ALREADY DEFINED" prohibition list |
| **Outline planning** | `--topic "..."` triggers a research-grounded outline generator |
| **Citation hygiene** | strips `[N>max]` orphans, denies zero-citation gaming, prefilters noisy domains |
| **LaTeX-quality PDF** | renders via `tectonic` (paper-quality math) with WeasyPrint fallback |
| **Resume-safe** | per-section state checkpoint, autonomous watchdog with Ollama health checks |

See [WORKPLAN.md](WORKPLAN.md) for the full roadmap (stages 0 → 5) and architecture notes.

---

## Quick start

```bash
# 1. Install Ollama + pull the default model stack
brew install ollama
ollama serve &
ollama pull gemma3:4b      # writer (default)
ollama pull qwen3.5:4b     # query generator + citation judge
ollama pull bge-m3         # embedder for ranking + filtering

# 2. Python deps
pip install -r files/requirements.txt
brew install pandoc tectonic    # tectonic for paper-quality LaTeX PDF render

# 3. (optional) configure web-search keys
cp .env.example .env
# fill in TAVILY_API_KEY and/or BRAVE_API_KEY — pipeline degrades gracefully without

# 4. Run
./run.sh             # autonomous runner (writes book.{md,html,pdf})
./run.sh direct      # single pass, no watchdog
./run.sh review      # autonomous + LLM-as-judge review pass
./run.sh watch       # live monitor

# 5. Spawn a custom-topic book
python3 files/deep_research.py --topic "Diffusion Models for Image Generation" \
  --n-chapters 12 --n-passes 10 --out-name diffusion_book --review
```

Outputs land in `files/output/` as `book.{md,html,pdf}` (plus `state.json`,
`report.json`, and timestamped logs).

---

## Pipeline

```
                 .------------------------------------------------------.
  topic str ---> | Planner agent  (qwen3.5:4b + scoping research)       |  Stage 3
                 |   scoping_search (arxiv + wiki + tavily)             |
                 |   -> 12-chapter outline JSON (self-corrected)        |
                 '----------------------------+-------------------------'
                                              | CHAPTERS[]
                                              v
                 .------------------------------------------------------.
  per section -> | (1) Query generator       (qwen3.5:4b)               |  Stage 2
                 |     prompt -> 3-5 search queries (JSON)              |
                 | (2) Multi-provider search (tavily/brave/arxiv/wiki)  |
                 |     -> raw sources                                   |
                 | (3) Prefilter             (bge-m3 cosine + domain    |
                 |                            noise threshold)          |
                 | (4) Rank top-k            (bge-m3)                   |
                 | (5) Full-text enrich      (trafilatura, top-2 only)  |
                 |     -> EVIDENCE block                                |
                 | (6) Writer                (gemma3:4b -- configurable)|
                 |     + context block (continuity + concepts already   |
                 |       defined cross-chapter, "DO NOT redefine")      |
                 |     -> section markdown with [N] citations           |
                 | (7) Sanitize + math normalize + clean_citations      |
                 |     (strip H1/H2 / refs / orphan [N>max])            |
                 | (8) LLM-as-judge per-citation verify (qwen3.5:4b)    |
                 |     supports / partial / unrelated / contradicts     |
                 |     -> grounding score 0..1                          |
                 |        zero citations + sources -> grounding 0.0     |
                 |        (denies the "writer gaming" pathology)        |
                 | (9) If grounding < 0.55 AND round < 2:               |
                 |       re-query with reviewer hint -> back to (2)     |
                 |     else persist sources + queries + concepts        |
                 '----------------------------+-------------------------'
                                              v
                 .------------------------------------------------------.
  assemble    -> | book.md = sanitized sections + dedup'd References    |  Stage 1
                 | render = pandoc + tectonic (LaTeX) -> book.pdf       |
                 '------------------------------------------------------'
```

---

## CLI reference

```bash
python3 files/deep_research.py [OPTIONS]
```

| Flag | Default | Effect |
|------|---------|--------|
| `--batch N` | 2 | Reserved for parallel section generation (future) |
| `--start-ch N` | 1 | Resume from chapter N |
| `--start-pp N` | 1 | Resume from section N within that chapter |
| `--end-ch N` | none | Stop after chapter N (use `--start-ch 1 --end-ch 1` for a smoke test) |
| `--review` | off | LLM-as-judge prose review on top of grounding verification |
| `--no-render` | render on | Skip PDF rendering at end |
| `--no-research` | research on | Disable Stage 2 (use only the writer's pretrained knowledge) |
| `--topic "..."` | hardcoded LLMs outline | Stage 3 planner generates a fresh outline |
| `--n-chapters N` | 12 | Number of chapters (with `--topic`) |
| `--n-passes N` | 8 | Sections per chapter (use 10-11 to target >400 pages) |
| `--out-name X` | book | Output basename: produces `X.{md,html,pdf,state.json,...}` |

---

## Environment variables

| Variable | Effect |
|---|---|
| `TAVILY_API_KEY` | Enable Tavily web search (free 1000/mo at tavily.com) |
| `BRAVE_API_KEY` | Enable Brave Search fallback (free 2000/mo at brave.com/search/api/) |
| `DEEP_RESEARCH_WRITER_MODEL` | Override writer model (default `gemma3:4b`) |
| `DEEP_RESEARCH_REVIEW=1` | Enable review pass (equivalent to `--review`) |
| `DEEP_RESEARCH_TOPIC` | Topic for planner (equivalent to `--topic`) |
| `DEEP_RESEARCH_N_CHAPTERS` / `DEEP_RESEARCH_N_PASSES` | Outline shape |
| `DEEP_RESEARCH_OUT_NAME` | Output basename |
| `DEEP_RESEARCH_END_CH` | Stop after this chapter |

---

## File layout

```
agentic/
├── run.sh                       # default-config launcher
├── watch.sh                     # one-shot progress snapshot
├── scripts/
│   └── launch_book2.sh          # example: Stage 3+ full run with review
├── README.md  CLAUDE.md  WORKPLAN.md  HANDOFF.md  LICENSE
├── .env.example                 # all env vars documented
├── .mcp.json                    # MCP server config (Claude Code reads at root)
├── pipeline.jpg                 # architecture diagram
└── files/
    ├── deep_research.py         # main pipeline (writer + assemble + render)
    ├── runner.py                # autonomous watchdog
    ├── monitor.py               # progress CLI
    ├── mcp_server.py            # MCP server (configured at root via .mcp.json for Claude Code)
    ├── requirements.txt
    ├── mcp_requirements.txt
    ├── archive/                 # legacy pipelines kept for historical reference
    ├── research/                # Stage 2+/3 agentic layer
    │   ├── __init__.py
    │   ├── types.py             # Source, Query dataclasses
    │   ├── search.py            # tavily / brave / arxiv / wiki / ddg adapters
    │   ├── query_gen.py         # section prompt -> search queries (JSON, qwen3.5:4b)
    │   ├── notes.py             # dedup / prefilter / rank / enrich / format EVIDENCE
    │   ├── embeddings.py        # bge-m3 batched + cosine sim
    │   ├── fetch.py             # disk-cached HTTP fetcher + full-text extraction
    │   ├── verify.py            # per-[N] grounding judge (qwen3.5:4b)
    │   ├── planner.py           # topic -> outline JSON + self-correction
    │   └── cache/               # HTTP fetch cache (gitignored)
    └── output/                  # all generated artifacts (gitignored)
```

---

## How it stays grounded

The fragile parts of "agentic" pipelines are usually:

1. **Citation gaming** — writer LLMs learn that "no citations = no failed verifications = perfect score"
   and drop all `[N]` markers. We deny zero-citation sections with `grounding = 0.0` when evidence was
   provided, forcing the writer to either cite or hedge.
2. **Source noise** — web search APIs occasionally surface tangential domains (YouTube transcripts,
   social-media histories, vendor marketing). A bge-m3 cosine prefilter drops anything below 0.30
   similarity, with a stricter 0.55 threshold for noisy-domain hits.
3. **Off-by-one citations** — writers sometimes emit `[9]` when only 8 sources were provided.
   `clean_citations()` strips these post-write before they reach the verifier.
4. **Binary content leakage** — full-text fetchers can return PDF/zip bytes that look like text.
   `_looks_binary()` checks content-type + magic bytes (`%PDF`, `PK\x03\x04`) + control-char ratio.
5. **Concept repetition across chapters** — without a global memory, agents redefine "attention" in
   every chapter that touches it. We extract every concept introduced (H3/H4 headers + bold terms)
   and emit a chapter-keyed "ALREADY DEFINED" prohibition list into each later section's prompt.

---

## Roadmap

| Stage | What | Status |
|---|---|---|
| 0 | Atomic-call book generator (96 hardcoded sections) | shipped |
| 1 | Continuity context + LLM-as-judge prose review + sanitization | shipped |
| 2 | Researcher layer (search + retrieval + grounded citations) | shipped |
| 2+ | Tavily + full-text + verifier + iterative loop + zero-cite penalty | shipped |
| 3 | Planner agent (topic → outline) + self-correction | shipped |
| 3+ | Cross-section concept tracker + outline dedupe directives | shipped |
| 4 | Multi-agent orchestration (Researcher / Writer / Reviewer split, parallel sections) | in progress |
| 5 | Citation-graph following + second-hop retrieval | planned |

---

## License

[MIT](LICENSE). Generated book content is yours.
