# Reference example — `agentic_2025_full`

A concrete, end-to-end output of the Agentic Deep Research pipeline, committed so readers can
inspect a real result transparently. **Everything here was produced by local models only**
(gemma4:e4b, qwen3.6-35b:iq3, bge-m3, bge-reranker-v2-m3, HHEM) — no external API at runtime.

## What to look at
| File | What it is |
|------|-----------|
| **`book.pdf`** | The rendered book — **605 pages**, typeset math (tectonic/xdvipdfmx). |
| **`book_eval_report.md`** | **BAER** transparent quality report (deterministic, reproducible). |
| `book_eval.json` | Machine-readable per-section metrics behind the report. |
| `book.md` | The assembled Markdown source. |
| `outline_profile.json` | The 24-chapter × 12-section outline (structure). |
| `topic_profile.json` | Discovery output: canonical terms, must-cover, out-of-scope. |

## How it was generated
```bash
python3 files/deep_research_v3.py \
  --topic "Xu hướng Agentic từ Agent từ 2025 đến hiện tại" \
  --out-name agentic_2025_full --no-smoke \
  --n-chapters 24 --sections-per-chapter 12 --max-rounds 2 --render \
  --canonical-arxiv-ids "2210.03629,2303.11366,2302.04761,2305.16291,2304.03442,2308.08155"
```
Run time ~23h on local hardware (288 sections, qwen 35B writer). Re-evaluate any run with:
`python3 files/eval/benchmark_book.py agentic_2025_full`.

## Transparent quality (BAER — numbers, not adjectives)
- **605 pages · 288 sections** (269 accepted, 19 P0a-blocked & omitted) · ~258k words · 4652 citations.
- **Redundancy:** mean 0.001, max 0.253, **0 near-duplicate section pairs** (8-gram Jaccard).
- **Technical depth:** **90% of sections carry a formula** (3403 inline + 669 display + 56 code blocks + 352 algorithm markers).
- **Coverage:** must-cover 6/6 · canonical terms 10/10 · canonical recall 1.0.
- **References on-topic:** 78.5% (reranker cosine ≥ 0.50) · providers arxiv 1667 / wiki 357 / ddg 118.
- **G4 topic (real signal):** 93% pass ≥0.50, mean 0.817, 17 distinct values.
- **G2 cite-precision (real signal):** mean 0.794.

## Honest caveats (this is one snapshot, mid-improvement)
- The **outline is templated** ("X: Foundations and Motivation" / "Inside X: Formal Definitions" …) —
  it fell back to the archetype generator because a 288-section JSON overflowed the small model.
  Fixed afterward (chunked evidence-driven outline); a fresh run no longer shows this matrix.
- The **19 P0a-blocked sections were omitted** from the book (the domain gate correctly refused
  sections with no on-topic evidence rather than letting the writer drift).
- At generation time **HHEM grounding was inert** (a degenerate scorer); it has since been fixed
  (embedding re-tie), so grounding is a real signal for new runs. cite-precision (G2) and topic
  (G4) were the live discriminating signals for this book.
- `book.pdf` was re-assembled + re-rendered with the latest fixes (no internal-log leakage,
  per-section references, render-robust math). The section *content* is from the original run.
