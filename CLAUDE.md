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
| 2d | Rank + Gate | `research/notes.py` | RRF(BM25+cosine) + **P0a** domain gate + **P0c** seen-penalty + prefilter |
| 2e | Writer (WRT) | `deep_investigate.py` (inline) | `qwen3.6-35b:iq3` |
| 2f | Grounding (VFY) | `research/faithfulness.py` | HHEM v2 |
| 2g | Topic / Cross-ref | `research/verify.py` | topic = HEURISTIC term-overlap (KHÔNG LLM) → quantized {0.5,0.75,1.0}; cross-ref = string match |
| 3 | Assemble | `deep_research_v3.py` | book.md + math/heading hygiene (Stage F) |
| 4 | Render `--render` | `files/scripts/render_book.py` | book.pdf / book.html |

Module phụ trợ (load-bearing): `config.py` (hằng số), `canonical_seeds.py` (P0b seeds), `embeddings.py`, `fetch.py`, `planner.py`, `types.py`.

**Verify layer — LIVE vs LEGACY (đừng sửa nhầm):**
- **LIVE** (chạy trong `investigate_section`, `deep_investigate.py:~606-740`): `faithfulness.grounding_score` (HHEM) + `verify.topic_relevance_check` (HEURISTIC, KHÔNG LLM) + `verify.verify_cross_references_v2` (regex đếm).
- **LEGACY-only** (chỉ `deep_research.py`/scripts/eval gọi — ĐỪNG sửa như live): `verify_section`, `verify_section_v2`, `crag_decision`, `answer_relevance`, `strip_refine`, `scrub_unsupported_citations`.
- Đổi accept/grounding → sửa `deep_investigate.py:606-740`, **KHÔNG** sửa `crag_decision`/`verify_section_v2` (no-op với run thật).

## 4. Model stack (THẬT)
| Vai trò | Model |
|---------|-------|
| Discovery / Outline / Query-Gen / Judge | `gemma4:e4b` |
| Writer | `batiai/qwen3.6-35b:iq3` |
| Embed | **SPLIT**: retrieval (notes.rank/prefilter RRF) + query_router chạy `nomic-embed-text` runtime (`investigate_section` default `deep_investigate.py:214`, orchestrator KHÔNG override). `bge-m3:latest` CHỈ ở verify-side (`verify.py:35`) + default `embeddings.py`. config nomic KHỚP path retrieval (≠ pure drift) |
| Rerank | `BAAI/bge-reranker-v2-m3` (transformers) |
| Grounding | `vectara/hallucination_evaluation_model` (HHEM v2) |

> **LOCAL-ONLY (bất biến):** mọi model trong pipeline chạy cục bộ (Ollama `localhost:11434` + transformers). Mọi judge verify — P0a domain, topic G4 (`answer_relevance`), citation-integrity G2 (`verify_section`) — đều dùng **gemma4:e4b LOCAL**; grounding = **HHEM local**; embed = **bge-m3/nomic local**. **TUYỆT ĐỐI KHÔNG gọi Claude/OpenAI/external API lúc runtime.** Claude (tôi) chỉ để review + chuẩn hóa docs + thiết kế cấu trúc verify, KHÔNG phải một model trong pipeline.

## 5. Ngưỡng gate THẬT (code = chuẩn; chi tiết → `RULES.md`)
| Gate | Giá trị thật (code) | File |
|------|---------------------|------|
| P0a domain gate | `ev_threshold = min(0.40, max(0.30, min_topic_relevance−0.10)) ≈ 0.40` — HARD BLOCK round cuối | `deep_investigate.py:479` |
| Accept Section | grounding ≥ **0.70** AND topic_relevance ≥ **0.50** AND n_cites > 0 AND cross_refs đủ | `deep_investigate.py:671` |
| StageE HARD BLOCK | grounding pass NHƯNG topic < 0.50 → block (grounding KHÔNG đủ một mình) | `deep_investigate.py:702` |
| P0c penalty | `max(0.05, (1 − seen/max_seen)²)`; canonical **EXEMPT** | `notes.py:311` |
| Prefilter cosine | 0.45 (grey-domain 0.65) | `notes.py:101` |
| Min words / Cross-ref | 120 từ / 2·1·0 theo số prior sections | `deep_investigate.py` |

⚠️ Các số `0.80 grounding`, `0.80 topic_purity`, `jaccard 0.30/0.70` trong tài liệu cũ là **ASPIRATIONAL (target), KHÔNG được enforce**. Đừng trích chúng làm hành vi thật.

## 6. 8 Guardrails — tránh đi sai hướng product/process
1. **Output goal > volume.** Đây là technical book đúng-topic, grounded, auditable — KHÔNG phải máy sinh chữ. Run 700 trang mà drift/lặp = **FAIL**. Đừng tối ưu section/word/completion trước topic purity & non-redundancy.
2. **Grounding KHÔNG phải chất lượng.** Run v36: `g=1.0` toàn bộ 280 section → HHEM bão hòa (gộp mega-premise), vô nghĩa làm tín hiệu. Tín hiệu thật = `topic_relevance` — NHƯNG nó là heuristic thô lượng tử hóa {0.5,0.75,1.0} (`verify.py:342`, KHÔNG LLM), chỉ là proxy. Đừng green-light run chỉ vì grounding.
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
Xem `files/memory/short-memory.md`. Tóm tắt: run mới nhất = **`llm_book_v36`** (40 chương / 280 section / ~196K từ state.json / 712 trang PDF; `g=1.0` toàn bộ; `topic_relevance` mean 0.78, 23 section ở sàn 0.50; 12/12 canonical injected). Blocker mở: matrix pattern + reference off-topic (Guardrail 3 & 6).
