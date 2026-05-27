# agentic -- Agent Context

## Mission

**Agentic Deep Research platform** -- a local-first system that researches a topic
and produces a book-length technical reference, grounded in retrieved sources, with
self-critique, citation verification, and iterative re-search loops. Provider-agnostic;
default stack is Ollama-served `gemma3:4b` + `qwen3.5:4b` + `bge-m3` on Apple Silicon.

Every change must move the codebase toward this mission. Polish work that does not
serve the agentic-research direction should be refused or redirected.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 0 | Atomic-call book generator (96 hardcoded sections) | shipped |
| 1 | Continuity context + LLM-as-judge prose review + sanitization | shipped |
| 2 | Researcher layer (search + retrieval + grounded citations) | shipped |
| 2+ | Tavily provider + full-text enrichment + citation verifier + iterative loop + zero-cite penalty | shipped |
| 3 | Planner agent (topic → outline) + self-correction | shipped |
| 3+ | Cross-section concept tracker + outline dedupe directives | shipped |
| 4 | Multi-agent orchestration (Researcher / Writer / Reviewer split, parallelism) | planned |
| 5 | Citation-graph following + second-hop retrieval | planned |

Full design notes: `WORKPLAN.md`. User-facing docs: `README.md`. Session handoff: `HANDOFF.md`.

## Claude-native pipeline structure

The four pillars Claude reads at startup:

| Pillar | Where | Purpose |
|---|---|---|
| **Plan**   | `WORKPLAN.md` (root) | Roadmap (stages 0-5), design notes per stage, decision log |
| **Memory** | `~/.claude/projects/<encoded-cwd>/memory/` + `MEMORY.md` index | Cross-session memory: project north star, status snapshots, pipeline critique, user feedback |
| **Skill / agent context** | `CLAUDE.md` (this file, auto-loaded by Claude Code) | What the agent must know to operate the project |
| **MCP**    | `.mcp.json` (root) | MCP server config (`agentic-deep-research` server -> `files/mcp_server.py`) |

`HANDOFF.md` at root captures the in-progress state when a session ends -- read it first when resuming work.

## Top-level layout

```
agentic/
├── run.sh / watch.sh           # entry points
├── scripts/                    # production launchers
├── .mcp.json                   # MCP server config (Claude convention)
├── .env.example                # all env vars documented
├── .gitignore
├── README.md                   # user-facing docs
├── CLAUDE.md                   # this file -- agent context
├── WORKPLAN.md                 # roadmap + design notes
├── HANDOFF.md                  # session handoff snapshot
├── LICENSE
└── files/                      # Stage 3+ production pipeline
    ├── deep_research.py        # main pipeline (writer + assemble + render)
    ├── runner.py               # autonomous watchdog
    ├── monitor.py              # progress CLI
    ├── mcp_server.py           # optional MCP server
    ├── research/               # Stage 2+/3 agentic layer (search, verify, planner, ...)
    ├── archive/                # legacy pipelines kept for reference
    └── output/                 # generated artifacts (gitignored)
```

## Output artifacts (`files/output/`)

All gitignored. Named after the `--out-name` flag (default `book`):

| File | Producer | Purpose |
|------|----------|---------|
| `book.md / .html / .pdf` | `deep_research.py` + `render_pdf` | Final artifacts |
| `book.clean.md` | render | Intermediate fed to pandoc |
| `book.state.json` | pipeline | Per-section checkpoint with sources / queries / verify scores |
| `book.report.json` | pipeline | End-of-run statistics |
| `book.pipeline.log` | pipeline | Timestamped log |
| `book.runner.log` | runner | Watchdog log |
| `book.pipeline.stdout.log` | runner | Captured subprocess stdout |

## Common commands

```bash
# Default LLM book run
./run.sh

# Custom topic via planner agent
python3 files/deep_research.py --topic "Diffusion Models" \
  --n-chapters 12 --n-passes 10 --out-name diffusion --review

# Single-chapter smoke test
python3 files/deep_research.py --start-ch 1 --end-ch 1 --out-name smoke --review

# Resume after crash (runner auto-detects but can override)
python3 files/deep_research.py --start-ch 5 --start-pp 3 --out-name book

# Kill a running pipeline
pkill -f files/runner.py && pkill -f files/deep_research.py
```

## Pipeline knobs

In `files/deep_research.py`:

| Knob | Default | Effect |
|------|---------|--------|
| `MODEL` | `gemma3:4b` (env `DEEP_RESEARCH_WRITER_MODEL`) | Writer LLM |
| `WORD_BUDGET` | 1500 | Ceiling on the dynamic per-section target (was 4200, the source of citation-gaming pressure — see W1 in [WORKPLAN.md](WORKPLAN.md)) |
| `WORD_TARGET_PER_SOURCE` | 220 | Per-section target = `n_evidence_sources * 220`, floored / capped |
| `WORD_TARGET_FLOOR` | 400 | Floor when research returns 0 sources |
| `WORD_TARGET_NO_EVIDENCE` | 900 | Fallback when `--no-research` |
| `MIN_REVIEW_SCORE` | 6 | Prose-review threshold (1-10 scale) |
| `CONTINUATION_WORDS` | 120 | Prior-section tail forwarded as context |

In `files/research/__init__.py`:

| Knob | Default |
|---|---|
| `PROVIDERS_DEFAULT` | `("arxiv", "wikipedia", "brave", "ddg", "tavily")` |
| `TOP_K_DEFAULT` | 8 sources per section |
| `FULL_TEXT_TOP_N` | 2 (top-2 get 350w body, rest keep 80w excerpt) |
| `QUERY_GEN_MODEL` / `JUDGE_MODEL` | `qwen3.5:4b` |
| `EMBED_MODEL` | `bge-m3:latest` |
| `MIN_GROUNDING` | 0.55 (below triggers re-search) |
| `MAX_RESEARCH_ROUNDS` | 2 |

## Ollama setup

```bash
ollama serve
ollama pull gemma3:4b qwen3.5:4b bge-m3
```

Reference: Apple M4 Metal runs `gemma3:4b` at ~35 tok/s; one section round (research +
write + verify) takes ~150-200s depending on iteration count.

## Render

`tectonic` (LaTeX) is preferred — paper-quality math. WeasyPrint is the fallback.

```bash
brew install pandoc tectonic
```
