# Danh Gia -- Cau Truc Pipeline Agentic Deep Research

> Document current pipeline architecture, model assignments, and quality gates.
> Last updated: 2026-06-13 | Version: v3.5-p0fix | Status: FIXES APPLIED

---

## 0. Quality Assessment Reports

> **Last Updated:** 2026-06-15 | Grading: Academic Research Standards (DREAM, DR3-Eval)

---

### 2026-06-15: llm_book_v36 (FINAL ASSESSMENT)

**Source:** `output/runs/llm_book_v36/book.pdf` (~850 pages)

| Metric | Score | Grade | Status | Weight |
|--------|-------|-------|--------|--------|
| Part N Pattern | 1.00 | A | ✅ PASS (0) | 25% |
| Cross-References | 0.85 | B+ | ✅ PASS (180, 0.64/s) | 20% |
| Matrix Pattern | 0.10 | F | ❌ FAIL (15 themes) | 15% |
| Semantic Overlap | 0.60 | C | ⚠️ MODERATE | 20% |
| Citation Diversity | 0.30 | F | ❌ LOW (8 unique) | 10% |
| Filler Content | 0.60 | C | ⚠️ 213 phrases | 10% |
| **OVERALL** | **0.62** | **C+** | ⚠️ NEEDS WORK | 100% |

**Stats:**
- Words: **224,446**
- Sections: **283** (H2)
- Subsections: **940** (H3)
- Chapters: **42**
- Cross-refs: **180** (0.64/section)
- Citations: **3,806** | Unique: **8**
- Part N: **0** ✅
- References: **666 arXiv** + 71 Wiki

**Key Findings:**

**✅ Strengths:**
- Zero Part N pattern
- 180 cross-references (good narrative flow)
- 940 H3 subsections (detailed structure)
- Comprehensive arXiv coverage (666 papers)
- Long-form technical depth (224K words)

**❌ Weaknesses:**

1. **Matrix Pattern (CRITICAL)** - 15 themes lặp 5-38x:
```
Evaluation: 38x    Fine-tuning: 22x    Alignment: 22x
Historical Origins: 21x    Design Principles: 21x
Objective Functions: 21x    Training: 20x
```

2. **Chapter Title Issues** - Self-referential chapters:
```
# 4. Chain-of-thought (cot): Training Paradigms and Chain-of-thought (cot)
# 12. Architecture: Multimodal Extensions and Architecture
# 39. Attention head: Model Compression and Attention head
```

3. **Citation Diversity (LOW)** - Chỉ 8 unique IDs, dù 3,806 citations
   → Một paper được cite ~475 lần = claim diversity kém

4. **Semantic Overlap** - Cùng bucket suffix lặp:
```
"Historical Origins and Motivating Problems" xuất hiện 21x
"Core Definitions and Formalism" xuất hiện 14x
"Design Principles and Architectural Choices" xuất hiện 21x
```

**Verdict:** Sách dài + comprehensive, nhưng structure chưa đạt Grade S do matrix pattern còn nguyên.

---

### 2026-06-14: llm_book_v36 (Initial)

**Source:** `output/runs/llm_book_v36/book.pdf`

| Metric | Score | Grade | Status | Weight |
|--------|-------|-------|--------|--------|
| Part N Pattern | 1.00 | A | ✅ PASS (0 instances) | 25% |
| Cross-References | 1.00 | A | ✅ PASS (161 total) | 20% |
| Matrix Pattern | 0.00 | F | ❌ FAIL (8x theme) | 15% |
| Semantic Overlap | 1.00 | A | ✅ PASS | 20% |
| Citation Diversity | 0.40 | F | ⚠️ LOW (8 unique) | 10% |
| Filler Content | 1.00 | A | ✅ PASS | 10% |
| **OVERALL** | **0.795** | **B+** | ⚠️ MODERATE | 100% |

**Stats:**
- Words: 199,011 (+149% vs v35)
- Sections (H2): 280 (+87% vs v35)
- Chapters (H1): 40
- Cross-refs: 161 (0.57/section avg)
- Part N: 0 ✅
- Citations: 3,491

**Matrix Pattern Themes (Still Present):**
| Theme | Repetitions |
|-------|-------------|
| Evaluation | 38x ❌ |
| Training | 20x ❌ |
| Application | 18x ❌ |
| Transformer block | 8x |
| Attention | 8x |
| Reasoning | 8x |

**Issues:**
1. **Matrix Pattern (FAIL):** 8+ themes repeat across chapters
2. **Citation Diversity (LOW):** 8 unique sources, max ~21%
3. **Section Title Repetition:** "Design Principles", "Objective Functions" lặp 7 lần

**Grade S Progress:**
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Part N | 0 ✅ | 0 | ✅ PASS |
| Cross-refs | 161 (0.57/s) | ≥560 (2/s) | ⚠️ Needs 4x more |
| Matrix themes | 8+ ⚠️ | 0 | FAIL |
| Overall | 0.795 | ≥0.95 | ⚠️ Progress |

---

### 2026-06-12: llm_book_v35 (Full Assessment)

**Source:** `eval/reports/llm_book_v35_quality_assessment.md`

|| Metric | Score | Grade | Status | Weight |
||--------|-------|-------|--------|--------|
|| Part N Pattern | 0.00 | F | FAIL (56 instances) | 25% |
|| Title Uniqueness | 0.00 | F | FAIL (49 dup, 1643 overlap) | 15% |
|| Semantic Content | 1.00 | A | PASS (100% unique) | 20% |
|| Cross-References | 0.00 | F | FAIL (0.0/section) | 20% |
|| Heading Hygiene | 1.00 | A | PASS (0 issues) | 10% |
|| Book Coherence | 0.70 | C | MODERATE | 10% |
|| **OVERALL** | **0.625** | **B-** | FAIL | 100% |

**Stats:** Pages: 157 | Words: 79,443 | Citations: 1,856 (~12/page)

---

### Root Cause Diagnosis

```
Writer quality:   Good    (100% semantic uniqueness)
Outline quality:  Bad     (Part N + Matrix patterns)
Cross-refs:      Bad     (0/section -- writer prompt weak)
```

| Layer | Cong viec | Chat luong | Van de | Fix |
|-------|----------|-----------|--------|------|
| `outline_from_research.py` | Tao cau truc chuong/muc | Rat kem | Part N (56), Matrix (49 dup titles) | Can sua |
| `deep_investigate.py` | Viet noi dung | Tot | Cross-refs = 0 (prompt chua du manh) | Can them prompt |
| `verify.py` | Kiem tra chat luong | Tot | Da co nhung khong retry/block duoc | Khop voi writer |
| `product_quality_verifiers.py` | Danh gia cuoi | Tot | Danh gia dung | OK |

**Ket luan:** Tat ca 3 loi deu la **config/prompt**, khong phai bug code.

---

### Chi Tiet Root Cause

#### 1. Part N Pattern + Matrix Pattern -- Loi o Outline

**Vi tri:** `outline_from_research.py` - `draft_outline_from_buckets()` va `_semantic_fallback_outline()`

**Van de trong `draft_outline_from_buckets()` (dong 109-159):**

Prompt LLM chi noi chung chung ve Part N, nhung khong huong dan cach xu ly khi so chuong lon hon so bucket:

```
- Chapter titles must be SPECIFIC and UNIQUE -- do NOT use generic prefixes
- NO "(Part N)" pattern anywhere
```

LLM van tao Part N vi: no doc `evidence_map` voi 6 bucket co dinh (foundations/math/architectures/training/evaluation/applications), va khi `n_chapters > 6`, LLM lap lai bucket + Part N de phan biet.

**Van de trong `_semantic_fallback_outline()` (dong 432-632):**

a) **Matrix Pattern chua xoa:** Section title van theo format `bucket: term -- template_label`

```python
# Dong 632: van tao ra "Foundations: Scaling -- Core Definitions"
sec_title = f"{bucket}: {sec_term_cap} -- {template_short_label}"
```

Ket qua: 160 section voi 49 trung ten, chi khac prefix bucket.

b) **Fallback khong topic-specific:** Khi LLM that bai, fallback su dung 6 bucket co dinh thay vi term pool tu topic.

#### 2. Cross-References = 0 -- Loi o Writer Prompt

**Vi tri:** `deep_investigate.py` - `_build_writer_prompt()` dong 201

```python
# Dong 201 -- chi yeu cau, khong enforce
"- Cross-reference at least 2 prior sections by title"
```

**Van de:**
- Prompt chi la "yeu cau" khong phai "bat buoc"
- Khong co retry neu writer bo qua
- `verify_cross_references_v2()` kiem tra nhung khong retry
- `cross_ref_count` duoc tra ve nhung khong block

#### 3. Title Uniqueness = 0 -- Loi o Outline (tiep)

**Vi tri:** `audit_outline()` dong 303-306

```python
# semantic_overlap duoc phat hien nhung KHONG block
if semantic_overlap_issues:
    issues.append("semantic_overlap_high")
# Khong co "BLOCK" hay "retry" nhu Part N
```

1643 high-overlap pairs duoc phat hien nhung pipeline van tiep tuc.

---

### Fix Plan Chi Tiet

| # | File | Hanh dong | Muc do | Trang thai |
|---|------|----------|--------|--------|
| F1 | `outline_from_research.py` | Prompt LLM: them rule tranh Part N khi n_ch > n_buckets | P0 | Can lam |
| F2 | `outline_from_research.py` | Fix fallback: bo "bucket:" prefix trong section title | P0 | Can lam |
| F3 | `outline_from_research.py` | Block neu semantic_overlap > 10 pairs | P0 | Can lam |
| F4 | `deep_investigate.py` | Manh hon writer prompt: cross-refs >= 2 (bat buoc) | P0 | Can lam |
| F5 | `deep_investigate.py` | Retry + block neu cross-refs < 2 | P0 | Can lam |
| F6 | `outline_from_research.py` | Fallback: dung topic-specific terms thay vi 6 bucket co dinh | P1 | Nen lam |
| F7 | `product_quality_verifiers.py` | Jaccard threshold 0.30 -> 0.20 | P1 | Nen lam |

**Fix code cu the:**

```python
# F2: fix section title trong fallback (dong 632)
# TRUOC:
sec_title = f"{bucket}: {sec_term_cap} -- {template_short_label}"
# SAU:
sec_title = f"{sec_term_cap}: {template_short_label}"  # Khong co bucket

# F4: manh hon writer prompt (dong 201)
# TRUOC:
"- Cross-reference at least 2 prior sections by title"
# SAU:
"- CRITICAL: You MUST include at least 2 cross-references to prior sections.
  Use exact titles like 'As discussed in Section X.X: [exact title]'.
  If you fail, the section will be REJECTED. Count your cross-refs before finishing: >= 2."

# F5: retry neu cross-refs < 2 (sau dong 568)
if cross_refs_found < 2:
    current_hint = (
        f"Section lacks cross-references ({cross_refs_found}/2 found). "
        "Add sentences like: 'As discussed in Section 1.2: [Title]', etc."
    )
    # Retry writing
```

---

### Grade S Progress

|| Metric | llm_book_v35 | Target | Status |
||--------|--------------|--------|--------|
|| Part N Pattern | 56 FAIL | 0 PASS | FAIL |
|| Cross-refs | 0/section FAIL | >=2/section PASS | FAIL |
|| Matrix Pattern | 49 dup titles FAIL | 0 PASS | FAIL |
|| Semantic Content | 100% PASS | 100% PASS | PASS |
|| Word Count | 79K PASS | >=50K PASS | PASS |
|| Heading Hygiene | 0 issues PASS | 0 PASS | PASS |
|| **Grade** | **B- (0.625)** | **A (>=0.95)** | FAIL |

---

### 2026-06-12: llm_book_v35 (Content Eval)

**Source:** `eval/reports/book_quality_eval_llm_book_v35.md`

|| Category | Score | Notes |
||----------|-------|-------|
|| Structure | 4/10 | Matrix + Part repetition |
|| Content Quality | 6/10 | Technical correct, repetitive |
|| Evidence/Grounding | 8/10 | Diverse citations |
|| Writing | 6/10 | Academic tone, filler present |
|| Coherence | 5/10 | Weak, lacks progression |
|| **Overall** | **5.8/10** | Trung binh |

---


## 1. Muc Tieu San Pham

**Agentic Deep Research** la he thong local-first, khong can API key, tim kiem su that ve mot chu de va tao ra mot cuon sach ky thuat 300-400+ trang voi nguon trich dan ro rang.

**Input:** Mot chu de (vd: "Diffusion Models")
**Output:** `book.md` + `book.pdf` + `state.json` + `report.json`

---

## 2. Kien Truc Tong Quan

```
Topic
  │
  v
.------------------------------------------------------.
|  STAGE 0: DISCOVER  (gemma-4-12b, fast)             |
|    3 broad scoping queries -> ~20 sources             |
|    LLM synthesizes: TopicProfile                     |
|      - name, description, subtitle                   |
|      - key_concepts[], initial_queries[]            |
'------------------------------------------------------'
  │
  v
.------------------------------------------------------.
|  STAGE 1: OUTLINE FROM RESEARCH  (qwen3.6-35b)      |
|    LLM reads gathered sources                        |
|    -> hierarchical chapter/section outline            |
|    (outline is OUTPUT, not INPUT)                    |
'------------------------------------------------------'
  │
  v
.------------------------------------------------------.
|  STAGE 2: DEEP INVESTIGATE  (per section)           |
|    Loop: QGN -> RSR -> rank -> enrich -> WRT -> VFY|
|    max 2 rounds per section                          |
'------------------------------------------------------'
  │
  v
.------------------------------------------------------.
|  STAGE 3: ASSEMBLE  -> book.md -> PDF               |
'------------------------------------------------------'
```

---

## 3. Chi Tiet Tung Stage

### 3.1. Stage 0: DISCOVER

**Model:** `gemma4:e4b` (6.4 GB, fast)

**Input:** Topic string
**Output:** `TopicProfile`

**Xu ly:**
1. Generate 3 broad scoping queries (arxiv, wiki, web)
2. Gather ~20 sources from providers
3. LLM reads sources and synthesizes:
   - `name`: book title
   - `description`: scope summary
   - `subtitle`: hook line
   - `key_concepts[]`: 5-10 core concepts
   - `initial_queries[]`: 3-6 queries for outline stage

**File:** `research/discovery.py`

---

### 3.2. Stage 1: OUTLINE FROM RESEARCH

**Model:** `batiai/qwen3.6-35b:iq3` (MoE 35B/3B, best prose)

**Input:** `TopicProfile` + sources from discovery
**Output:** `OutlineProfile`

**Xu ly:**
1. LLM reads all gathered sources
2. LLM designs hierarchical structure:
   - `N` chapters (configurable, default 12)
   - `M` sections per chapter (configurable, default 8)
   - Each section has `title` + `pr` (writing directive)
3. Post-process: fill missing `pr` from `title`
4. Output includes `coverage_gaps[]` from evidence

**JSON output schema:**
```json
{
  "title": "...",
  "subtitle": "...",
  "chapters": [
    {
      "n": 1,
      "t": "Chapter title",
      "coverage_note": "...",
      "sections": [
        {"n": 1, "t": "Section title", "pr": "writing directive"}
      ]
    }
  ],
  "coverage_gaps": []
}
```

**File:** `research/outline_from_research.py`

---

### 3.3. Stage 2: DEEP INVESTIGATE (per section loop)

#### Loop hien tai

```
For each (chapter, section):
  Round 1:
    (1) QGN -> 3-5 search queries
    (2) RSR gather (arxiv/wiki/ddg)
    (3) RRF fusion + top-20 candidates
    (4) RRK cross-encoder rerank -> top-8
    (5) Full-text enrich (top-2, 350 words each)
    (6) WRT (qwen3.6) -> markdown with [N] citations
    (7) VFY (gemma-4) -> grounding score
    (8) if grounding >= 0.55: DONE
        else: retry with reviewer hint

  Round 2 (if Round 1 failed):
    -> refined queries based on reviewer feedback
    -> repeat steps (1)-(7)
    -> if still < 0.55: accept with warning
```

#### (1) QGN -- Query Generator

**Model:** `gemma4:e4b`
**File:** `research/query_gen.py` + `query_router.py`

- Semantic routing via archetype templates (0 LLM cost when matched)
- LLM fallback for novel topics
- Output: 3-5 `Query` objects with intent tags

#### (2) RSR -- Research Gather

**Model:** API/parsing (no LLM)
**File:** `research/search.py`

Providers (configurable):
| Provider | Weight | Coverage |
|----------|--------|----------|
| arxiv | 3x boost | academic papers |
| wikipedia | 3x boost | encyclopedic |
| ddg | 1x | web fallback |

#### (3) RRF -- Reciprocal Rank Fusion

**File:** `research/notes.py`

- Fuse results from multiple providers
- arxiv/wikipedia: 3x priority boost
- top-20 candidates passed to reranker

#### (4) RRK -- Cross-Encoder Rerank

**File:** `research/rerank.py`

- Cross-encoder relevance scoring
- `BAAI/bge-reranker-v2-m3` model
- top-8 final candidates

#### (5) Enrich -- Full-Text Extraction

**File:** `research/fetch.py`

- trafilatura for web pages
- arxiv DOI fetch for papers
- top-2 sources: 350-word body extract
- remaining 6: 80-word excerpts

#### (6) WRT -- Writer

**Model:** `batiai/qwen3.6-35b:iq3`
**File:** `research/deep_investigate.py`

Prompt includes:
- EVIDENCE block (full text from top-2 + excerpts from rest)
- CONTINUATION context (prior section tail, 120 words)
- CONCEPTS context (already-defined terms across chapters)
- SECTION directive (from `pr` field)

Output: markdown body with `[N]` citation markers

#### (7) VFY -- Verifier

**Model:** `gemma4:e4b` + HHEM v2
**File:** `research/verify.py` + `faithfulness.py`

Two-tier verification:
1. **HHEM v2** (flan-t5-base, 0.1B): claim-level grounding
2. **LLM judge** (gemma-4): remaining borderline citations

CRAG thresholds:
| Grounding | Action |
|-----------|--------|
| >= 0.80 | ACCEPT |
| 0.40 - 0.80 | AMBIGUOUS -> LLM judge |
| < 0.40 | INCORRECT -> discard + re-search |

---

### 3.4. Stage 3: ASSEMBLE

**File:** `pipeline/deep_research_v3.py` + `scripts/postprocess_book.py`

1. Concatenate all sanitized sections
2. Deduplicate References page
3. Render PDF via `tectonic` (LaTeX) or WeasyPrint fallback

---

## 4. Model Stack

| Tier | Model | Role | Size | Speed |
|------|-------|------|------|-------|
| **Research** | `gemma4:e4b` | QGN, VFY | 6.4 GB | ~20 tok/s |
| **Writing** | `batiai/qwen3.6-35b:iq3` | WRT, PLN, RVW | MoE 35B/3B | ~15 tok/s |
| **Embedding** | `bge-m3:latest` | cosine rank | - | ~50 tok/s |

**Rationale:**
- Gemma 4 12B: fast, cheap, good enough for research-layer tasks (QGN, VFY)
- Qwen 3.6 35B MoE: best prose quality, used only where quality matters most (WRT, PLN, RVW)
- Split reduces cost while maximizing output quality

---

## 5. Key Knobs

| Knob | Default | Effect |
|------|---------|--------|
| `WORD_BUDGET` | 1500 | Ceiling per section |
| `WORD_TARGET_PER_SOURCE` | 220 | Target = sources * 220 |
| `WORD_TARGET_FLOOR` | 400 | Fallback when 0 sources |
| `MIN_GROUNDING` | 0.55 | Below -> re-search |
| `MAX_RESEARCH_ROUNDS` | 2 | Max retries per section |
| `TOP_K_RETRIEVE` | 20 | Candidates before rerank |
| `TOP_K_FINAL` | 8 | After RRK rerank |
| `FULL_TEXT_TOP_N` | 2 | Sources with 350w body |
| `GROUND_UPPER` | 0.80 | ACCEPT threshold |
| `GROUND_LOWER` | 0.40 | INCORRECT threshold |

---

## 6. Quality Gates

| Gate | Metric | Threshold | Action if Fail |
|------|--------|-----------|----------------|
| Citation grounding | mean supported/total claims | >= 0.55 | re-search |
| Zero-citation penalty | grounding | 0.0 | forces citation |
| Off-by-one citation | max([N]) vs len(sources) | N <= len(sources) | clean_citations() |
| Source noise | cosine similarity | >= 0.30 | prefilter |
| Noisy domain | cosine similarity | >= 0.55 | strict gate |

---

## 7. Output Artifacts

```
output/runs/<name>/
├── book.md          # assembled markdown
├── book.html        # HTML render
├── book.pdf         # PDF (tectonic/WeasyPrint)
├── book.clean.md    # sanitized (no H1/H2/refs)
├── state.json       # per-section checkpoint
├── topic_profile.json
├── outline_profile.json
├── report.json      # end-of-run statistics
├── pipeline.log     # timestamped log
└── runner.log      # watchdog log
```

---

## 8. File Structure

```
files/
├── deep_research_v3.py         # v3 orchestrator (True Deep Research)
├── deep_research.py            # v2 orchestrator (pre-planned outline)
├── runner.py                   # watchdog + auto-restart
├── monitor.py                  # progress CLI
├── report.py                   # post-run analysis
├── research/
│   ├── discovery.py            # Stage 0: topic scoping
│   ├── outline_from_research.py # Stage 1: outline generation
│   ├── deep_investigate.py     # Stage 2: per-section loop
│   ├── query_gen.py            # QGN: generate search queries
│   ├── query_router.py         # QGN: semantic archetype routing
│   ├── search.py               # RSR: multi-provider gather
│   ├── notes.py                # RRF: rank + format evidence
│   ├── embeddings.py            # bge-m3 cosine similarity
│   ├── rerank.py               # RRK: cross-encoder rerank
│   ├── fetch.py                # full-text extraction
│   ├── verify.py               # VFY: LLM judge (cosine prefilter)
│   ├── faithfulness.py         # VFY: HHEM v2 claim grounding
│   ├── types.py                # Source, Query dataclasses
│   └── planner.py             # PLN: v2 outline generator
├── eval/
│   ├── paper_eval.py          # paper-quality evaluation
│   └── reports/               # eval outputs
└── output/
    └── runs/<name>/           # per-run artifacts
```

---

## 9. Dua Ra San Pham

### Pull models
```bash
ollama pull gemma4:e4b
ollama pull batiai/qwen3.6-35b:iq3
ollama pull bge-m3:latest
```

### Chay smoke test (2 chapters)
```bash
python3 pipeline/deep_research_v3.py --topic "Diffusion Models" --out-name diffusion_v3
```

### Chay day du (tat ca chapters)
```bash
python3 pipeline/deep_research_v3.py --topic "Diffusion Models" \
  --out-name diffusion_v3 --no-smoke
```

### Chay benchmark voi P0 fixes
```bash
python3 pipeline/deep_research_v3.py \
  --topic "RLHF: Reinforcement Learning from Human Feedback" \
  --out-name rlhf_p0benchmark \
  --canonical-arxiv-ids "2203.02155,2305.18290,2009.14165,2204.05862" \
  --n-chapters 4 --sections-per-chapter 4 --max-rounds 3
```

### Theo doi tien do
```bash
python3 tools/monitor.py
```

### Kill pipeline
```bash
pkill -f legacy/runner.py && pkill -f legacy/deep_research.py
```

---

## 10. Root Cause Analysis & Fix Priorities

### Quality Score Breakdown

```
                    Impact on Grade S
                    Low         High
              +-----------+-----------+
      High    |  Filler   |  PART N   | <- FIX FIRST
   Priority   |  Content  |  CROSS-REF|
              +-----------+-----------+
      Low     | Citation  |  MATRIX   |
   Priority   |  Diversity|  Pattern  |
              +-----------+-----------+
```

### Priority Fix Matrix

| Priority | Issue | Weight | Impact | File | Fix |
|----------|-------|--------|--------|------|-----|
| **P0** | Part N Pattern | 25% | 40%->100% | `outline_from_research.py` | Remove `(Part N)` suffix |
| **P0** | Cross-References | 20% | 0%->100% | `deep_investigate.py` | Add prompt requirement |
| **P0** | Matrix Pattern | 15% | 0%->100% | `outline_from_research.py` | Use unique titles, not buckets |
| **P1** | Filler Content | 10% | 47%->100% | `deep_investigate.py` | Writer quality prompt |
| **P2** | Citation Diversity | 10% | 40%->100% | `notes.py` | Enforce max % per source |

### Grade S Checklist

```
□ Part N Pattern:     0 instances
□ Cross-References:  >=20 total (>=2/section avg)
□ Matrix Pattern:     0 themes repeated >5x
□ Semantic Overlap:   0 high-overlap pairs
□ Citation Diversity: >=15 unique sources
□ Filler Content:    <=10 template phrases
□ Word Count:        >=50,000 words
□ Chapters:          8-20 chapters
□ Sections/Chapter:  3-5 sections
□ Progression Logic: Foundations -> Frontiers
□ Book Coherence:    Clear narrative flow
```

### Ideal Chapter Structure (Grade S)

```markdown
# ❌ BAD (Matrix Pattern)
Foundations: Theory
Foundations: Methods
Foundations: Applications
Math: Theory
Math: Methods
... (44x Foundations, 39x Math)

# ✅ GOOD (Grade S)
1. Origins and Motivating Problems of LLMs
2. Transformer Architecture: Self-Attention Deep Dive
3. Pre-training Objectives: Masked Language Modeling
4. Scaling Laws and Compute-Optimal Training
5. Instruction Tuning and Alignment Techniques
6. RLHF: Reward Modeling and Policy Optimization
7. Evaluation Benchmarks: HELM, MT-Bench, AlpacaFarm
8. Chain-of-Thought Reasoning in LLMs
9. Tool Use and Retrieval-Augmented Generation
10. Multimodal Large Language Models
11. Efficient Fine-tuning: LoRA, QLoRA, Adapter Methods
12. Knowledge Distillation and Quantization
13. Safety, Bias, and Ethical Considerations
14. Open Problems and Future Directions
```

### Ideal Section Structure (Grade S)

```markdown
# ❌ BAD (Part N Pattern)
"Foundations: Historical Origins (Part 2)"
"Foundations: Theory (Part 2)"
...

# ✅ GOOD (Grade S)
"Chapter 1: Origins"
  ├── 1.1 The Turing Test and Early Dreams
  ├── 1.2 Neural Network Winter (1980s-1990s)
  └── 1.3 The Transformer Revolution (2017)

"Chapter 2: Architecture"
  ├── 2.1 Self-Attention Mechanism
  ├── 2.2 Multi-Head Attention
  └── 2.3 Positional Encoding

"Chapter 3: Pre-training"
  ├── 3.1 Masked Language Modeling (BERT-style)
  ├── 3.2 Causal Language Modeling (GPT-style)
  └── 3.3 Mixed-Objectives Training
```

### Cross-References Example (Grade S)

```markdown
# ❌ BAD (0 cross-refs)
This section covers the Transformer architecture.

# ✅ GOOD (>=2 cross-refs)
Building on the self-attention mechanism introduced in Chapter 2,
the feed-forward layers in this section provide non-linear
transformation capacity [1]. As discussed in Section 1.3, the
Transformer revolution (Vaswani et al., 2017) introduced this
architecture [2]. We also see connections to pre-training
objectives from Chapter 3...
```
