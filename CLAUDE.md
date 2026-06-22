# CLAUDE.md — Agentic Deep Research Platform

> **Nguồn sự thật vận hành cho agent.** Ngưỡng & hành vi THẬT nằm trong CODE (`files/research/*.py`, `config.py`), không phải trong doc. Khi nghi ngờ một con số → đọc code, đừng trích doc làm fact. Bảng ngưỡng đầy đủ: `RULES.md`.

## 0. Quy tắc phản hồi
- Trả lời = **tiếng Việt có dấu** + English term khi cần.

## 1. Thứ tự đọc mỗi phiên
1. `files/GLOSSARY.md` — thuật ngữ chuẩn (ĐỌC TRƯỚC).
2. `files/memory/short-memory.md` — snapshot trạng thái + run mới nhất.
3. File này — doctrine, pipeline, model stack, guardrails.
4. `RULES.md` — bảng ngưỡng & gate (khi đụng tới quality gate / debug drift).

## 2. Doctrine (CỐ ĐỊNH — không đổi)
**Hệ thống KHÔNG được biết trước kịch bản.** Outline phải **emerge từ evidence**, tuyệt đối không pre-template `chapters × concepts` (đó chính là lỗi *matrix pattern* gây trùng lặp — xem Guardrail 3).

```
Prompt thô → Discovery (TopicProfile) → Outline (từ evidence)
   → Deep Investigation mỗi Section: QGN → Search → Rerank → Gate(P0a) → Write → Verify
   → Assemble → (Render PDF)
```

## 3. Pipeline THẬT (orchestrator + 12 stage)
- **Orchestrator LIVE:** `files/deep_research_v3.py :: run_v3()`. Launcher: `run_full.sh`.
- **Legacy v2 (ĐỪNG sửa như đang live):** `files/deep_research.py` (140KB, outline pre-fixed) + `runner.py` + `run.sh` + `watch.sh`. Vẫn được import bởi `monitor.py` / `eval/run_eval.py` → còn sống nhưng KHÔNG phải đường chính.
- Resume qua run-dir `files/output/runs/<name>/{topic_profile,outline_profile,state}.json` — **không bao giờ viết lại Section đã có trong `state.json`.**

| # | Stage | File | Vai trò |
|---|-------|------|---------|
| 0 | Discovery (DSC) | `research/discovery.py` | TopicProfile (canonical papers/terms, out_of_scope) + **P0b** canonical injection |
| 1 | Outline (OUT) | `research/outline_from_research.py` | chapters/sections TỪ evidence; `outline_audit` (advisory) |
| 2 | Deep Investigation | `research/deep_investigate.py` | vòng lặp per-section `max_rounds` (CLI=3, run_v3 nội bộ=2) |
| 2a | Query Gen (QGN) | `research/query_gen.py` + `query_router.py` | LLM khi có `domain_context`, else archetype |
| 2b | Search (RSR) | `research/search.py` | providers: arxiv, wikipedia, ddg (default); tavily chỉ khi `TAVILY_API_KEY` set |
| 2c | Rerank (RRK) | `research/rerank.py` | `bge-reranker-v2-m3` (transformers, KHÔNG Ollama) |
| 2d | Rank + Gate | `research/notes.py` | RRF(BM25+cosine) + **P0a** domain gate + **P0c** seen-penalty + prefilter + **evidence-pool rescue** (post-prefilter on-topic <5 → mượn sibling sources, P0c-EXEMPT, `deep_investigate.py:416,433`) |
| 2e | Writer (WRT) | `deep_investigate.py` (inline) | `qwen3.6-35b:iq3` |
| 2f | Grounding (VFY) | `research/faithfulness.py` | HHEM v2 — **ADVISORY/log-only**, không hard-block một mình |
| 2g | Topic / Cross-ref | `research/verify.py` | topic = **G4** blend term-heuristic + `answer_relevance` gemma LOCAL (`verify.py:401-411`) + StageE floor; cross-ref = regex string match |
| 3 | Assemble | `deep_research_v3.py` | book.md + math/heading hygiene (Stage F) |
| 4 | Render `--render` | `files/scripts/render_book.py` | book.pdf / book.html |

Module phụ trợ (load-bearing): `config.py` (hằng số), `canonical_seeds.py` (P0b seeds), `embeddings.py`, `fetch.py`, `planner.py`, `types.py`.

**Verify layer — LIVE vs LEGACY (đừng sửa nhầm):**
- **LIVE (gọi mỗi round, post-P0 2026-06-22):** `faithfulness.grounding_score` (HHEM, **G3 log-only/advisory** — đã bỏ khỏi gate) + `verify.topic_relevance_check` (**G4** blend gemma LOCAL — **ENFORCED**: điều kiện `gate_ok` + StageE best-topic) + `verify.verify_section` (**G2** — **GIỜ CHẠY**, cite_precision đo thật; clean-accept cần ≥0.45 — ⚠️ judge strict-match floor ~0.3-0.4 → clean-accept=0, cần **P0-2b**) + `verify.verify_cross_references_v2` (regex đếm). **Gate cứng SỐNG = P0a (pre-writer ~0.40) + StageD word-count.**
- **LEGACY-only** (chỉ `deep_research.py`/scripts/eval gọi — ĐỪNG sửa như live): `verify_section_v2`, `crag_decision`, `strip_refine`, `scrub_unsupported_citations`. (`verify_section` + `answer_relevance` GIỜ dùng LIVE bởi G2/G4.)
- Đổi accept/grounding → sửa `deep_investigate.py:606-740`, **KHÔNG** sửa `crag_decision`/`verify_section_v2` (no-op với run thật).
- **Verifier ≠ Writer (bất biến — chống self-preference):** grounding = **HHEM**, topic/citation = **gemma** — model verify TÁCH khỏi writer (**Qwen**). ĐỪNG để Qwen tự chấm prose của chính nó (model tự duyệt văn mình → bias). Quyết định: 2026-06-16.

## 4. Model stack (THẬT)
| Vai trò | Model |
|---------|-------|
| Discovery / Outline / Query-Gen / Judge | `gemma4:e4b` |
| Writer | `batiai/qwen3.6-35b:iq3` |
| Embed | **`bge-m3:latest` THỐNG NHẤT (#3)**: retrieval (notes.rank/prefilter RRF) + query_router + verify-side đều bge-m3 — `config.py:34` EMBED_MODEL, `query_router.py:210` _EMBED_MODEL, `embeddings.py:8` DEFAULT_MODEL, `verify.py:35`. (Trước là split nomic; unify vì nomic cần prefix `search_query:`/`search_document:` mà code KHÔNG truyền → asymmetric.) **0 ref nomic sống** trong pipeline. |
| Rerank | `BAAI/bge-reranker-v2-m3` (transformers) |
| Grounding | `vectara/hallucination_evaluation_model` (HHEM v2) |

> **LOCAL-ONLY (bất biến):** mọi model trong pipeline chạy cục bộ (Ollama `localhost:11434` + transformers). Mọi judge verify — P0a domain, topic G4 (`answer_relevance`), citation-integrity G2 (`verify_section`) — đều dùng **gemma4:e4b LOCAL**; grounding = **HHEM local**; embed = **bge-m3:latest local** (thống nhất #3). **TUYỆT ĐỐI KHÔNG gọi Claude/OpenAI/external API lúc runtime.** Claude (tôi) chỉ để review + chuẩn hóa docs + thiết kế cấu trúc verify, KHÔNG phải một model trong pipeline.

## 5. Ngưỡng gate THẬT (code = chuẩn; chi tiết → `RULES.md`)
| Gate | Giá trị thật (code) | File |
|------|---------------------|------|
| P0a domain gate | `ev_threshold = min(0.40, max(0.30, min_topic_relevance−0.10)) ≈ 0.40` — HARD BLOCK round cuối | `deep_investigate.py:524` |
| **Accept Section** | clean-accept (P0): topic≥0.50 (G4) AND n_cites>0 AND cross-ref AND **cite_precision≥0.45 (G2, đo thật)**; grounding log-only. ⚠️ cite-judge strict → cite_precision floor ~0.3-0.4 → clean-accept=0, cần P0-2b | `deep_investigate.py:752,808` |
| StageE HARD BLOCK | **KHÔNG BAO GIỜ fire** (cần grounding≥0.70) | `deep_investigate.py:751` |
| P0c penalty | `max(0.05, (1 − seen/max_seen)²)`; canonical + pool-rescued **EXEMPT** ⚠️ bug aliasing → **no-op trong 1 run** (xem §Upgrade P0) | `notes.py:324` |
| Prefilter cosine | **0.48** (grey-domain 0.65) — Rank7: was 0.45 | `notes.py:109` |
| Min words / Cross-ref | 120 từ / 2·1·0 theo số prior sections | `deep_investigate.py` |

> ✅ **P0 ĐÃ APPLY (2026-06-22):** grounding bỏ khỏi gate (log-only); G2 `verify_section` **GIỜ CHẠY** (cite_precision đo thật, ≠1.0); P0c aliasing fixed (run_seen_counts 0→23); StageE chặn best-topic sau-loop; best-round topic-first. ⚠️ **CÒN LẠI (P0-2b):** cite-judge strict-match → cite_precision ~0.3-0.4 < 0.45 → clean-accept vẫn = 0. Gate cứng SỐNG vẫn = **P0a (pre-writer ~0.40)** + word-count. → `plan.md` §Upgrade.

⚠️ Các số `0.80 grounding`, `0.80 topic_purity`, `jaccard 0.30/0.70` trong tài liệu cũ là **ASPIRATIONAL (target), KHÔNG được enforce**. Đừng trích chúng làm hành vi thật.

## 6. 8 Guardrails — tránh đi sai hướng product/process
1. **Output goal > volume.** Đây là technical book đúng-topic, grounded, auditable — KHÔNG phải máy sinh chữ. Run 700 trang mà drift/lặp = **FAIL**. Đừng tối ưu section/word/completion trước topic purity & non-redundancy.
2. **Grounding KHÔNG phải chất lượng (P0: đã bỏ khỏi gate → log-only/advisory).** HHEM strict-NLI ~0.05–0.10 trên prose. Sau P0: **G4 topic** ENFORCED (gate_ok + StageE), **G2 cite_precision** GIỜ CHẠY (đo thật) — nhưng cite-judge strict-match floor ~0.3-0.4 < 0.45 → clean-accept=0 tới khi **P0-2b** (soften judge). **Gate cứng SỐNG = P0a pre-writer.** Đừng green-light run vì grounding; faithfulness gate chưa "xanh" tới P0-2b.
3. **Outline EMERGE từ evidence — GIẾT matrix pattern.** Không pre-template chapters×concepts. Nếu `outline_audit` trả `ok=false` (matrix/coherence/overlap) → **sửa OUTLINE trước Stage 2**, không vá ở writer.
4. **Fix ở GATE, không ở writer.** Drift / off-topic evidence / canonical thiếu / nguồn dominate phải chặn ở P0a/P0b/P0c/prefilter (`notes.py`, `deep_investigate.py`). Không cho writer "cứ viết rồi tính".
5. **Canonical papers được PROTECT, không penalize.** `protected_source_ids` bypass cosine prefilter + EXEMPT khỏi P0c. Mọi thay đổi retrieval/dedup phải giữ exemption này (nếu mất → canonical recall sụp về 0).
6. **Enforce reference relevance theo SECTION.** Canonical recall cao che giấu sourcing kém per-section (~45% ref off-topic ở v36). Siết prefilter/domain gate; không accept section chỉ vì có ≥6 citation.
7. **Ngưỡng sống trong CODE, doc là advisory.** Đừng quote số trong doc làm fact — đọc `config.py` / `deep_investigate.py` / `notes.py` / `verify.py`. *Lưu ý:* `product_quality_verifiers.py` là eval-time-only, KHÔNG chạy trong pipeline; đừng coi GATE-0..6 trong đó là đang bảo vệ run.
8. **Một nguồn sự thật pipeline.** Orchestrator = `deep_research_v3.py`; mọi stage logic = `files/research/*.py`. `files/deep_research.py` là legacy v2 — đừng sửa nó như đang live. Memory gọn (short ≤50 dòng, long <200).

## 7. Lệnh thường dùng
```bash
# Full run (orchestrator v3)
python3 files/deep_research_v3.py --topic "RLHF" --out-name rlhf_v4 \
  --canonical-arxiv-ids "2203.02155,2305.18290,1706.03762" --no-smoke
# hoặc: ./run_full.sh

# Smoke test P0a/b/c + RULES
python3 files/eval/smoke_test_p0.py --topic "Transformer" --canonical-ids "1706.03762,1607.06450"

python3 files/monitor.py                 # theo dõi tiến độ
python3 files/report.py <run_dir>        # phân tích state.json sau run
pkill -f files/deep_research_v3.py       # dừng
```

## 8. Trạng thái hiện tại
Xem `files/memory/short-memory.md`. **Base** = orchestrator `deep_research_v3.py` + research layer, gồm: LOCAL-only, **Verifier≠Writer** (model tách thật), outline anti-matrix (#1), embed `bge-m3` thống nhất (#3), anchoring an-toàn KHÔNG-mất-nguồn (#5), evidence-pool rescue (completeness), render tectonic robust. **P0 (2026-06-22) ĐÃ APPLY:** grounding bỏ khỏi gate (advisory), G2 `verify_section` chạy thật (cite_precision đo, ≠ default 1.0), P0c aliasing fixed; G4 topic ENFORCED. ⚠️ Faithfulness gate **chưa "xanh"**: cite-judge strict-match → cite_precision floor ~0.3-0.4 < 0.45 → clean-accept=0; **P0-2b** (soften judge) là bước kế. Roadmap → `plan.md` §Upgrade. `llm_book_v36` là book CŨ — tham khảo lịch sử.
