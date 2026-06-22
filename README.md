# agentic — Local-first Agentic Deep Research

> Hand it a topic. It **discovers** the scope, lets the **outline emerge from the evidence**, then runs
> a per-section **research → rank → gate → write → verify** loop and assembles a LaTeX-typeset technical
> book — **100% on local models, no API key, every claim gated and auditable.**

<p align="left">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="python"  src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="runtime" src="https://img.shields.io/badge/runtime-Ollama%20(local)-black.svg">
  <img alt="render"  src="https://img.shields.io/badge/render-tectonic%20%2F%20WeasyPrint-orange.svg">
  <img alt="status"  src="https://img.shields.io/badge/status-full--run%20validated-success.svg">
</p>

It is **not** a text generator. The goal is an on-topic, grounded, **auditable** book: drift, repetition,
and ungrounded claims are blocked at gates, not patched in the writer.

---

## See it in action

A real end-to-end result is committed so you can judge it by the output, not the claims:

**[`files/output/runs/agentic_2025_full/`](files/output/runs/agentic_2025_full/)**
- [`book.pdf`](files/output/runs/agentic_2025_full/book.pdf) — **605 pages**, tectonic-typeset math, per-section references.
- [`book_eval_report.md`](files/output/runs/agentic_2025_full/book_eval_report.md) — the **BAER** transparent quality report (deterministic, reproducible).
- [`README.md`](files/output/runs/agentic_2025_full/README.md) — how it was generated + honest caveats.

Headline numbers (BAER): 288 sections · ~258k words · **0 near-duplicate sections** · **90% of sections carry a formula** · coverage 6/6 must-cover + 10/10 canonical terms · references 78.5% on-topic.

---

## Model stack (all local, via Ollama + transformers)

| Role | Model | Notes |
|------|-------|-------|
| Discovery / Outline / Query-gen / **Judge** | `gemma4:e4b` | topic profile, evidence-driven outline, query gen, topic + citation judges |
| **Writer** | `batiai/qwen3.6-35b:iq3` | prose + formulas + algorithms, grounded in cited evidence |
| Embedding (retrieval + verify) | `bge-m3:latest` | unified dense retrieval / dedup / cosine prefilter |
| Rerank | `BAAI/bge-reranker-v2-m3` | cross-encoder (transformers, not Ollama) |
| Grounding | `vectara/hallucination_evaluation_model` (HHEM v2) | per-claim entailment (advisory signal — see note below) |

> **Invariant — Verifier ≠ Writer:** the writer is **qwen**; grounding is scored by **HHEM** and
> topic/citation by **gemma**. The writer never grades its own prose (no self-preference bias).
> **LOCAL-ONLY:** every model runs on `localhost` — no Claude/OpenAI/external API at runtime.

---

## Pipeline

```
Topic ─▶ Discovery ─▶ Outline (from evidence) ─▶ Deep Investigation (per section) ─▶ Assemble ─▶ Render
```

| # | Stage | Model | What happens |
|---|-------|-------|--------------|
| 0 | **Discovery** | gemma4:e4b | scoping queries → `TopicProfile` (canonical terms, must-cover, out-of-scope) + **P0b** canonical-paper injection |
| 1 | **Outline** | gemma4:e4b | **chunked** drafting — chapter skeleton, then per-chapter sections that *emerge from that chapter's evidence* (kills the archetype "matrix") |
| 2 | **Deep Investigation** (loop, per section) | ↓ | query-gen → search (arXiv/Wikipedia/DDG) → cosine **prefilter** → RRF rank + cross-encoder rerank → **gates** → write → **verify** |
| 3 | **Assemble** | — | `book.md` + math hygiene (single-source [`research/mathfix.py`](files/research/mathfix.py)) + per-section references |
| 4 | **Render** (`--render`) | — | pandoc → **tectonic** (LaTeX, `-Z continue-on-errors`); WeasyPrint fallback |

### The gates — what *actually* enforces (honest; P0 applied 2026-06-22)
- **P0a** domain gate — the **primary live hard gate**: hard-blocks a section whose *evidence* is off-domain (≈0.40, **pre-writer**). ~27% of sections block here.
- **P0b** canonical inject / **P0c** seen-penalty — foundational papers are protected (exempt from prefilter + dedup); over-represented sources penalized. *(P0c cross-section propagation was a single-run no-op — **fixed in P0**.)*
- **Prefilter** — drop sources below cosine **0.48** to the section (grey domains 0.65); canonical exempt.
- **Post-writer verify (P0):** grounding (G3) is now **log-only/advisory** (removed from the gate — HHEM strict-NLI under-scores synthesized prose). **Topic (G4) is enforced** (clean-accept + a topic-drift block). **Citation precision (G2) now runs for real** (it was previously wrapped behind the dead grounding bar and never executed; `cite_precision = 1.0` used to be an un-measured default).
- **⚠️ Known limitation (next: P0-2b):** the G2 citation judge is prompted for *strict direct-match*, so on synthesized prose it scores ~0.3–0.4 — below the 0.45 accept bar — so no section yet clean-accepts (all ship `degraded`). The judge needs softening to accept paraphrase/implication (or the threshold recalibrated) before the faithfulness gate is fully "green". Stated plainly rather than hidden.

> Thresholds live in **code** (`deep_investigate.py`, `notes.py`, `config.py`), not docs — see [RULES.md](RULES.md) for the full table.

---

## Quality eval — BAER (transparent, deterministic)

Judge any finished run by numbers, not adjectives:

```bash
python3 files/eval/benchmark_book.py <run-name>   # -> book_eval_report.md + book_eval.json
```

**B**enchmark (pages/words/citations) · **A**nalyze (cross-section redundancy, anti-matrix, reference
on-topic %, coverage, technical-depth %) · **E**val (gates labelled **real vs inert**) · **R**eport.
Calls no model → same run always yields the same report. Companion probes:
`bench_hhem_discrimination.py` (grounding scorer health), `bench_math_split_bm25.py` (math/BM25 safety).

---

## Quick start

```bash
# 1. Local models (Ollama)
brew install ollama && ollama serve &
ollama pull gemma4:e4b
ollama pull batiai/qwen3.6-35b:iq3
ollama pull bge-m3:latest

# 2. Python deps + render toolchain
pip install -r files/requirements.txt
brew install pandoc tectonic          # tectonic = paper-quality LaTeX PDF

# 3. (optional) extra web search — degrades gracefully without
cp .env.example .env                  # add TAVILY_API_KEY if you have one

# 4. Run (smoke = first 2 chapters; --no-smoke = full book)
python3 files/deep_research_v3.py --topic "RLHF" --out-name rlhf_v4 \
  --canonical-arxiv-ids "2203.02155,2305.18290,1706.03762" --no-smoke --render
#   or: ./run_full.sh

# 5. Evaluate the result
python3 files/eval/benchmark_book.py rlhf_v4
```

Monitor a long run / inspect a finished one:
```bash
python3 files/monitor.py                      # live progress across runs
python3 files/report.py files/output/runs/rlhf_v4   # per-section metrics from state.json
python3 files/eval/smoke_test_p0.py --topic "Transformer" --canonical-ids "1706.03762"  # quick gate check
```

Each run lands in `files/output/runs/<out-name>/` (`book.{md,pdf}`, `book_eval_report.md`,
`state.json`, `topic_profile.json`, `outline_profile.json`). Runs are **resume-safe** — re-running
continues from `state.json` and never rewrites an accepted section.

### CLI (orchestrator v3)

| Flag | Default | Effect |
|------|---------|--------|
| `--topic` | required | Research topic |
| `--out-name` | from topic | Output folder |
| `--n-chapters N` / `--sections-per-chapter N` | from discovery | Outline size hints |
| `--canonical-arxiv-ids "id,id"` | none | Force-inject + protect foundational papers (P0b) |
| `--providers arxiv wikipedia ddg` | all 3 | Retrieval providers |
| `--max-rounds N` | 3 | Max research rounds per section |
| `--no-smoke` | smoke (2 ch) | Run the full book |
| `--render` | off | Render PDF |

---

## Repo layout

```
files/
├── deep_research_v3.py         # orchestrator (LIVE) — run_v3()
├── deep_research.py            # legacy v2 (pre-planned outline; not the live path)
├── monitor.py · report.py      # live progress across runs · per-section metrics from state.json
├── research/
│   ├── discovery.py            # Stage 0 — TopicProfile + P0b canonical inject
│   ├── outline_from_research.py# Stage 1 — chunked, evidence-driven outline
│   ├── deep_investigate.py     # Stage 2 — per-section research/gate/write/verify loop
│   ├── query_gen.py · query_router.py · search.py · rerank.py
│   ├── notes.py                # RRF rank + P0a/P0c gates + prefilter + math-safe BM25
│   ├── faithfulness.py         # G3 grounding (HHEM)
│   ├── verify.py               # G4 topic + G2 citation (gemma), cross-ref
│   ├── mathfix.py              # single-source math/LaTeX normalization (render-safe)
│   ├── embeddings.py · fetch.py · config.py · canonical_seeds.py · types.py
├── eval/
│   ├── benchmark_book.py       # BAER quality report
│   ├── bench_hhem_discrimination.py · bench_math_split_bm25.py
│   ├── test_math_char_safety.py · test_verify_optim.py
│   └── smoke_test_p0.py         # quick P0a/b/c gate check (no full run)
├── scripts/render_book.py      # Stage 4 — pandoc + tectonic / weasyprint
└── output/runs/<name>/         # per-run output (the committed example lives here)
```

Operational doctrine: [CLAUDE.md](CLAUDE.md) (pipeline + invariants) · [RULES.md](RULES.md) (gate/threshold table) · [files/GLOSSARY.md](files/GLOSSARY.md) (terms).

---

## Notable fixes that make it auditable
- **Grounding scorer revived (now advisory)** — HHEM was a no-op under transformers 5.x (T5 `embed_tokens` loaded as zeros → constant ~0.502 for every pair). Fixed by re-tying embeddings in `_get_hhem` + a startup discrimination-assertion (no version pin). It now discriminates clean NLI pairs, but strict-NLI under-scores synthesized prose (~0.05–0.10 even when faithful), so grounding is treated as an **advisory** signal — the enforced faithfulness check is citation-precision (G2) and the live quality signal is topic (G4).
- **Outline emerges from evidence** — chunked drafting replaces the single giant-JSON call that overflowed the model and fell back to a templated topic×archetype matrix.
- **Render never dies on one bad formula** — `mathfix` neutralizes broken LaTeX spans to literal, escapes raw `%`/stray `$`, bounds unbalanced-brace paragraphs, and is idempotent; tectonic renders a 250k-word book directly.
- **No internal logs leak into the book**; raw `[N]` citations resolve to per-section references.

---

## License

[MIT](LICENSE). Generated book content is yours.
