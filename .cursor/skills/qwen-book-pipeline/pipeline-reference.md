# Qwen Book Pipeline — Detailed Reference

## Stage 0 — Spec Compiler

Input: 1 user prompt (EN or VN)
Output: `BookSpec` Pydantic model

```python
class BookSpec(BaseModel):
    title: str
    target_pages: int           # 50-2000
    target_words: int           # = target_pages * 400
    languages: list[str]       # ["vi", "en"]
    style_guide: StyleGuide
    audience: Literal["beginner", "intermediate", "advanced", "expert"]
    depth: Literal["overview", "textbook", "handbook", "reference"]
    domain: str
    chapter_count_target: int   # 8-15
    constraints: list[str]
```

LLM: 1 call, thinking=OFF, temperature=0.0, guided JSON.

## Stage 1 — STORM Outline

3 sub-stages:
1. **Perspective discovery**: 1 LLM call → 5-8 "lenses" (theoretical, architectural, training, alignment, application, historical, engineering)
2. **Simulated conversations**: 5-8 perspectives × 4 turns × 2 calls = ~50 calls + 30 Tavily searches. Writer↔Expert dialog loops, knowledge accumulates each turn.
3. **Outline synthesis**: 1 LLM call on all accumulated knowledge → 3-level tree (~500 leaves), notation table, word budgets summing to target.

Output: `BookOutline` with `all_leaves()` helper.

## Stage 2 — Research

Per leaf (parallel with semaphore-bounded concurrency):
1. Generate 3-5 search queries (1 LLM call, JSON guided)
2. Parallel Tavily search (5 results × 5 queries = 25 raw)
3. Dedupe by URL, keep top 8
4. Extract atomic facts with citations (1 LLM call)
5. Filter facts so every `source_url` exists in collected sources (anti-hallucination)

Cache: SHA256(query) → JSON on disk in `output/.search_cache/`.

## Stage 3 — Writer

Sequential within chapter (coherence via prev_summary) → parallel across chapters (vLLM continuous batching).

Context assembly per writer call:
- Atomic facts (Stage 2) with URLs
- Prev subsection summary (first 200 chars of prior leaf)
- Next subsection title
- Notation table (cross-book macros)
- Subsection spec (id, word_budget, depth_hint)
- Style guide (bilingual rules, math, code conventions)

Output: `{markdown: str, local_refs: dict[int, str]}` — `local_refs` maps `[N]` inline markers to URLs.

## Stage 4 — Citation Resolution

Deterministic (0 LLM calls):
1. Collect all `(leaf_id, local_num, url)` tuples
2. Resolve: arXiv API → arXiv URLs, Semantic Scholar → academic, HTML `<title>` scrape → fallback
3. Dedupe by BibTeX key
4. Rewrite `[N]` → `[@bibkey]` (Pandoc citeproc format)
5. Write `bookkit/references.bib`

## Stage 5 — Quality Gates

Deterministic checks (regex, fast):
- Citation density ≥5/1000 words → warn
- Math balance: all `$...$` and `$$...$$` paired → fail
- Code fences: all triple-backtick pairs → fail
- Env balance: `\begin{X}` matches `\end{X}` → fail
- Word budget within ±15% → warn
- Every `[@bibkey]` exists in references.bib → fail

LLM-judge checks:
- No meta-commentary ("In this section we will...")
- Smooth coherence with previous subsection
- No verbatim repetition from prior subsection

Failed leaves → log warnings + optional retry with augmented prompt.

## Stage 6 — Render

1. Assemble chapter markdown from subsection contents
2. Write to `bookkit/chapters/NN-name.md`
3. Write `bookkit/references.bib`
4. Run `bookkit/build.sh`: lint → pandoc → xelatex 2-pass (for ToC + cross-refs)
5. Copy `book.pdf` → `output/book.pdf`

## Pydantic Schemas (src/schemas.py)

```python
# Stage 0
class BookSpec(BaseModel): ...

# Stage 1
class BookOutline(BaseModel):
    title: str
    perspectives: list[Perspective]
    chapters: list[Chapter]     # ~10
    notation_table: list[dict]  # cross-book consistency
    def all_leaves(self) -> list[LeafNode]: ...

# Stage 2
class LeafResearch(BaseModel):
    leaf_id: str
    queries_used: list[str]
    sources: list[SearchResult]
    facts: list[AtomicFact]
    summary: str

class AtomicFact(BaseModel):
    content: str
    source_url: str

# Stage 3
class LeafContent(BaseModel):
    leaf_id: str
    markdown: str
    word_count: int
    local_refs: dict[int, str]  # {1: url1, 2: url2, ...}

# Stage 4
class BibEntry(BaseModel):
    key: str                    # "vaswani2017attention"
    bibtex_type: Literal["article", "inproceedings", "book", "misc", ...]
    url: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    arxiv_id: str | None

# Stage 5
class QualityIssue(BaseModel):
    severity: Literal["warn", "fail"]
    check: str
    message: str

class QualityReport(BaseModel):
    leaf_id: str
    overall: Literal["pass", "warn", "fail"]
    issues: list[QualityIssue]
```

## LLM Client Details (src/llm_client.py)

```python
class LLMResponse:
    content: str       # Raw output
    think_block: str   # Everything between <think>...</think>
    text_only: str     # content minus think block

class QwenClient:
    async def chat(self, messages, thinking: bool, ...) -> LLMResponse
    async def chat_json[T](self, messages, schema: type[T], ...) -> T
```

Qwen3-specific: `"chat_template_kwargs": {"enable_thinking": thinking}` for thinking mode toggle.

## Search Client Details (src/search_client.py)

```python
class SearchResult(BaseModel):
    url: str
    title: str
    content: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

# On-disk cache: SHA256(query) → JSON file in output/.search_cache/
# Dedupe by URL, keep top 8 ranked by score
```

## Cost & Time Estimates

| Stage | LLM | Search | Cost | Time |
|-------|-----|--------|------|------|
| 0. Spec | 1 | 0 | $0 (local) | 5s |
| 1. Outline | ~30 | ~30 | $0 + $0.15 | 5 min |
| 2. Research | ~3000 | ~2500 | $0 + $12.50 | 30 min |
| 3. Writing | ~500 | 0 | $0 | 2-4 hours |
| 4. Citations | 0 | ~50 | $0 + $0.25 | 5 min |
| 5. Quality | ~500 | 0 | $0 | 15 min |
| 6. Render | 0 | 0 | $0 | 2 min |
| **Total** | **~4000** | **~2580** | **~$13** | **~3-5 hours** |

## Dependencies (pyproject.toml pins)

```
torch>=2.4.0
vllm>=0.11.0
pydantic>=2.0
httpx
tenacity
rich
tavily-python
arxiv
requests
beautifulsoup4
```

## Limitations

1. **No fact verification**: model trusts Tavily sources. Spot-check important claims.
2. **No domain experts in loop**: hallucinated theorems possible.
3. **Math correctness**: writer may produce wrong derivations.
4. **Image/figure generation**: pipeline does not generate figures.
5. **Cross-chapter consistency**: writers only see prev subsection summary, not full prev chapter.
6. **Notation drift**: no hard enforcement of notation table.
