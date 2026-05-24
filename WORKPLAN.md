# Workplan -- Agentic Deep Research

**Mission:** become a local-first Agentic Deep Research platform that researches a
topic and produces a book-length technical reference, grounded in retrieved
sources, with self-critique and re-search loops.

**Status (2026-05-23):** Stages 0 and 1 shipped; Stage 2 is the next concrete piece
of work.

---

## Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | Atomic-call book generator (96 fixed-prompt sections) | shipped |
| 1 | Continuity context + LLM-as-judge reviewer + sanitization | shipped |
| 2 | Researcher layer (search + retrieval + grounded citations) | shipped |
| 2+ | Tavily provider + full-text enrichment + citation verifier + iterative loop | **shipped 2026-05-23** |
| 3 | Planner agent (outline generated from `--topic`) + outline self-correction | **shipped 2026-05-23** |
| 3+ | Cross-section concept tracker + outline dedupe directives | **shipped 2026-05-23** |
| 4 | Multi-agent orchestration (separate Researcher / Writer / Reviewer processes) | planned |
| 5 | Citation graph following + secondary-hop retrieval | planned |

---

## Stages 0 + 1: what shipped

### Stage 0 -- atomic-call generator

- 12 chapters x 8 sections = 96 LLM calls. Each call targets one focused topic.
- Per-section checkpoint to `state.json`; auto-resume from crash.
- Autonomous runner with Ollama health monitoring, stall detection, and
  macOS notifications.
- PDF render at the end via pandoc + WeasyPrint.

Baseline output (gemma3:4b, Apple M4 Metal): 124,394 words / ~310-415 pages /
35 tok/s / 100% success rate. See `files/output/benchmark.md`.

### Stage 1 -- continuity + critique + cleanup

Three problems identified after the Stage 0 benchmark:

1. **Context isolation.** Each of the 96 calls ran independent. Sections re-
   introduced concepts already covered. *Fix:* the previous section's last ~120
   words plus the titles of sections already covered in the chapter are fed
   forward as a context block.
2. **No self-evaluation.** The pipeline only checked word count, not quality.
   *Fix:* optional `--review` flag adds an LLM-as-judge pass scoring depth /
   coherence / format on 1-10. Below threshold (default 6) the section is
   regenerated once with the reviewer's feedback appended.
3. **Format hallucination.** The model emitted its own H1/H2 headings,
   References blocks, and Conclusion sections that broke the assembled
   structure. *Fix:* stricter system prompt + `sanitize()` strips
   model-hallucinated H1/H2 / References / Conclusion / meta-intros at both
   generation and assembly time.

---

## Stage 2+ -- what shipped on 2026-05-23 (after the honest audit)

The first Stage 2 baseline scored 5/10 on "Agentic Deep Research" -- it had real retrieval but
narrow providers, decorative citations, no iteration, no verification. The 2026-05-23 follow-up
shipped the seven concrete fixes from the audit:

| # | Fix | File |
|---|---|---|
| 1 | **Tavily provider** -- AI-friendly web search, gated on `TAVILY_API_KEY` env. Augments arxiv+wiki when set; degrades silently when not. | `research/search.py` |
| 2 | **Full-text enrichment** -- top-2 ranked sources have their excerpts replaced with up to 350 words of real page content (trafilatura optional, fallback HTML strip). Writer no longer reads 80-word search snippets. | `research/fetch.py`, `research/notes.py` |
| 3 | **Citation grounding judge** -- after every section, each `[N]` marker is fed (claim, source-excerpt) to a small LLM judge (qwen3.5:4b) that returns `supports / partial / contradicts / unrelated / no_evidence`. Aggregate grounding score persists in state. | `research/verify.py` |
| 4 | **Iterative research loop** -- when grounding < 0.55 the section is re-researched with a hint derived from the weak-citation reasons, then rewritten. Capped at 2 rounds. | `deep_research.py` `run()` |
| 5 | **Writer model configurable** -- `DEEP_RESEARCH_WRITER_MODEL` env overrides the default `gemma3:4b`. Recommended upgrades: `qwen2.5:7b` or `qwen2.5:14b`. | `deep_research.py` constants |
| 6 | **Stage 3 planner** -- `--topic "..."` triggers a research-grounded outline generator (scope research -> JSON outline -> self-correction). | `research/planner.py` |
| 7 | **Cross-section concept tracker** -- every section's H3/H4 headers + bold terms are extracted post-write and stored as `concepts`. `build_context` shows later sections an "ALREADY DEFINED" prohibition list with chapter origins. Plus a one-shot outline dedupe pass auto-injects `[OUTLINE-DEDUPE]` directives into prompts where high-traffic concepts recur. | `deep_research.py` `extract_concepts`, `build_context`, `dedupe_outline` |

Smoke-test on one section (Self-Attention) after the changes: research 23s, write 115s,
verify 25s, grounding 0.66/0.55 -> done in 1 round. Per-section budget ~165s; full-book
estimate ~4.5h for 96 sections.

Audit of the hardcoded CHAPTERS after the dedupe pass: 63/96 prompts gained at least one
"already covered" directive. Worst recurring concepts: `attention` (21 sections), `fine-tuning`
(16), `embedding` (12), `RAG` (9), `transformer` (9), `quantization` (8), `LoRA` (7), `scaling
laws` (6). The dedupe pass marks each section after the first occurrence so the writer is
explicitly told to skip definitions and jump into the section-specific aspect.

## Stage 2 (initial) -- the researcher layer

The biggest unaddressed gap: the pipeline name promises "deep research" but
content is generated from a hardcoded `CHAPTERS` prompt list with no retrieval,
no tools, no grounding. Stage 2 closes that.

### Flow

```
for each section:
    1. Query generator  (LLM)        section prompt -> 3-5 search queries
    2. Retriever        (no LLM)     arxiv API + Wikipedia REST -> raw hits
    3. Notes compiler   (no LLM)     dedup + rank by relevance -> top-k
    4. Writer           (LLM, existing gen())  evidence + context -> markdown,
                                                with [N] citations
    5. Reviewer         (existing review_section)  if low score, route back to
                                                    step 1 with reviewer's hint
```

Compared to the existing pipeline, only step 1-3 are new. Step 4 reuses `gen()`
unchanged (it already accepts a context block; we add an evidence block beside
it). Step 5 reuses `review_section()` but its retry path becomes a re-search
rather than a same-evidence regeneration.

### Module layout (`files/research/`)

```
files/research/
├── __init__.py
├── search.py       # provider adapters: arxiv_search(), wiki_search(),
│                   # ddg_search() (off by default)
├── fetch.py        # fetch_arxiv_abstract(), fetch_url_text() + disk cache
├── query_gen.py    # LLM call: section prompt -> [{q, intent}, ...]
├── notes.py        # dedup/rank/format snippets into an EVIDENCE block
└── cache/          # JSONL per section, content-addressed by URL
```

### Search providers (v1)

- **arxiv** -- `arxiv` PyPI lib over `export.arxiv.org/api/query`. No key. Use
  for any topic that has academic literature.
- **Wikipedia REST** -- `en.wikipedia.org/api/rest_v1/page/summary/...`. No key.
  Use for definitions and conceptual grounding.
- **DuckDuckGo HTML** -- gated behind `--web` flag in v1. Scrapes HTML results;
  rate-limit aware (1 req/sec). Use for current events, vendor docs.

Caching: every successful fetch is written to
`files/research/cache/<sha1(url)>.json` with `{url, fetched_at, title, text,
metadata}`. Subsequent runs read the cache; force-refresh via `--no-cache`.

### Query generation prompt (gemma3:4b)

```
SYSTEM: You are a research assistant. Given a section prompt, output ONLY a JSON
array of 3-5 search queries, each <= 12 words, optimized for arxiv / Wikipedia.
Return the JSON now, no prose.

USER:
Chapter: {ch_t}
Section: {pp_t}
Section prompt: {prompt}
```

Expected output:
```json
[{"q": "scaled dot-product attention derivation", "intent": "primary source"},
 {"q": "transformer multi-head attention complexity", "intent": "supporting"},
 {"q": "attention is all you need vaswani 2017", "intent": "canonical citation"}]
```

Failure mode: gemma3:4b sometimes wraps JSON in prose. `query_gen.py` extracts
the first `[...]` block; on parse failure, falls back to a deterministic
template (`{section title} survey`, `{section title} 2024`, ...).

### Evidence block format (fed to writer)

```
EVIDENCE (cite as [N] -- do not invent papers or URLs):

[1] Vaswani et al. (2017), "Attention Is All You Need" (arxiv:1706.03762)
    Excerpt: "We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms..."

[2] Wikipedia, "Transformer (deep learning architecture)" (en.wikipedia.org)
    Excerpt: "...self-attention computes a weighted sum of values, where the
    weight assigned to each value is determined by..."

[3] ... up to 8 sources, ranked by relevance, ~80 words excerpt each ...
```

Total evidence block: ~600-800 tokens. System prompt rule added:

> Cite using `[N]` referring to numbered sources in the EVIDENCE block. Do NOT
> invent papers, URLs, dates, or authors. If evidence is insufficient for a
> point, omit it or say so.

`assemble()` learns to read `state["passes"][k]["sources"]` and emit a single
References section at the end of the book, deduped across all sections.

### State / audit schema additions

`state["passes"][key]` gains:
```json
{
  "...existing fields...",
  "sources": [
    {"id": "arxiv:1706.03762", "title": "Attention Is All You Need",
     "authors": ["Vaswani", "..."], "year": 2017, "url": "...",
     "excerpt": "...", "relevance_score": 0.87}
  ],
  "queries": ["scaled dot-product attention derivation", "..."]
}
```

Per-section evidence is also written to `output/evidence.jsonl` (one line per
section) for offline audit.

### Integration point in `deep_research.py`

Single edit in `run()`, around the existing `gen()` call:

```python
context_block = build_context(state, ch_n, pp_n)

# NEW for Stage 2:
queries  = research.query_gen.queries_for(prompt, ch_t, pp_t, client)
sources  = research.search.gather(queries, providers=cfg.providers)
sources  = research.notes.rank(sources, prompt, top_k=8)
evidence = research.notes.format_for_prompt(sources)

content, stats, w = gen(client, ch_n, ch_t, pp_n, pp_t, prompt, budget,
                        context_block, evidence_block=evidence)

state["passes"][key]["sources"]  = sources
state["passes"][key]["queries"]  = [q["q"] for q in queries]
```

Reviewer retry path becomes: if score < threshold, re-run `query_gen` with
`reviewer_hint=issues`, gather fresh evidence, then re-call `gen()`. The
existing "regenerate-once" budget cap stays.

### Decision questions (open until user confirms)

1. Phase 1 first, or jump to Phase 2 (planner) so the topic is no longer
   hardcoded?
2. Should query_gen / reviewer use a larger model (`qwen2.5:7b`?) while the
   writer keeps `gemma3:4b`? Two-model setup is straightforward via separate
   `OllamaClient` instances.
3. Enable DuckDuckGo from day one, or gate behind `--web` for v1?
4. Should the pipeline accept a `--topic` CLI argument in v1 (and use a default
   "Large Language Models" outline), or stay topic-fixed until Stage 3?

---

## Stages 3-5 (sketch only)

### Stage 3 -- planner agent

Replace the hardcoded `CHAPTERS` list with an LLM-generated outline. Planner
takes `--topic`, emits an outline JSON, user can edit, then pipeline runs as
today. Outline itself can be researched first ("what are the major subareas of
X?").

### Stage 4 -- critique-driven re-search

Today the reviewer can only force a same-evidence regenerate. With the
researcher layer in place, low scores re-enter the loop at step 1 with a
modified query (e.g. reviewer says "missing formal definitions" -> query_gen
biases toward textbook / canonical sources).

### Stage 5 -- multi-agent orchestration

Split Researcher / Writer / Reviewer into separate processes communicating
through `state.json` + a queue. Parallelism across sections (independent
research subtrees). Possibly use LangGraph or a small custom state machine.

---

## What was archived during normalization (2026-05-23)

To stop the codebase carrying multiple competing standards, the following moved
to `archive/` directories:

- `files/archive/multipass_pipeline.py` -- 48-pass legacy generator
- `files/archive/run_pipeline.py` -- its autonomous runner
- `files/archive/deep_agent_pipeline.py` -- 4-stage reference design
- `files/archive/watch.py` -- legacy monitor
- `files/output/archive/BENCHMARK.legacy.md` -- 48-pass benchmark report
- `files/output/archive/autonomous.legacy.log` -- legacy runner log
- `files/output/archive/benchmark.legacy.json` -- legacy benchmark data

All output filenames now use the neutral form (`book.md`, `state.json`,
`pipeline.log`, ...). Page count no longer appears in any filename.
