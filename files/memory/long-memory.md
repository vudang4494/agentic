# Long Memory — Nhật ký nén theo phiên

> **Mục đích:** Lưu lịch sử quyết định/phát hiện/kết quả, mới nhất trước. Không lặp `short-memory.md`. Mỗi entry 1–3 dòng/mục; archive khi milestone.

```text
### [YYYY-MM-DD] <tiêu đề>
- Bối cảnh: / Phát hiện: / Thay đổi: / Bằng chứng: / Còn lại:
```

---

## Session log (mới nhất trước)

### [2026-06-16] #3+#4+#5 SAFE (no info loss) + validate_v39
- **#3 embed unify** → `bge-m3:latest` toàn bộ (config/query_router/deep_investigate; bỏ nomic dùng-thiếu-prefix). **#5 anchoring AN TOÀN:** anchor (must_cover_terms[0]) CHỈ vào rank_rrf + rerank (ordering/chọn top-8); **prefilter — chỗ hard-drop duy nhất — giữ section_prompt gốc** → KHÔNG bao giờ thu nhỏ pool. **#4 citation-aware grounding** (parse [N], premise = nguồn được cite, strip marker) = **warn-first** (log grounding_cited vs per-source; gate KHÔNG đổi) + **G2 fail-CLOSED** (lỗi verify_section → cite_prec=0.0, retry/best-effort, không hard-block, không mất content).
- **Validation `validate_v39`** (safe version, exit 0): KHÔNG mất thông tin — prefilter kept 7-16, chỉ drop off-topic/grey, 0 empty-pool/HARD-BLOCK, 4 section ship (3219 từ > v37). #4: grounding(cited)=1.0=per-source trên section sạch (bật gate sẽ không mass-block). **G2 bắt đúng**: 2.1 cite_prec=0.0 trên section evidence borderline (rel=0.425) mà g/topic bỏ sót → vẫn ship best-effort. Đã dừng v38 (bản prefilter-anchored kém an toàn) trước khi sửa.
- Unit test 16/16 (thêm test #4 citation-aware phân biệt). Còn lại: re-tune floor/ngưỡng nếu cần (data đã có); cân nhắc bật #4 làm gate (thấp vì G2 đã lấp).

---

### [2026-06-16] Ship #1+G6, validation run validate_v37, fix R7 + chapter-title
- **Đã ship + push** (branch chore/normalize-docs-cleanup): #1 outline anti-matrix (section title term-centric theo global index, pool 15 coprime spp=7); G6 bge-m3 cosine dedup section-vs-prior warn-first (≥0.85 log, KHÔNG block).
- **Validation `validate_v37`** (2ch×2sec, max_rounds 2, topic "LLMs"): pipeline OK exit 0, không vỡ. **G2 cite_prec phân biệt 0.75-1.0; G4 topic liên tục 0.77-0.90** (hết quantize {0.5,0.75,1.0}). grounding vẫn 1.0 trên section sạch (per-source-max chưa đủ de-sat; #4 citation-aware mới làm thật, NHƯNG G2 đã lấp chỗ trống). G6 chạy sạch (không false-warn). #1 confirmed: title đa dạng, hết matrix.
- **2 bug phụ (từ validate_v37) đã sửa:** (a) R7 `_validate_canonical_urls` loại MỌI canonical thiếu URL + gán nhãn sai "redirect" → giờ chỉ loại DDG-redirect thật, giữ title-only paper; (b) chapter subtitle kết "and {term}" → "Transformer: ... and Transformer" → bỏ term khỏi subtitle + guard "X: X".
- **Còn lại (cần run thêm):** #3 embed unify (re-tune floor 0.45), #4 citation-aware grounding (warn-first), #5 query anchoring (re-measure trên outline non-templated).

---

### [2026-06-16] Audit + tối ưu tầng Verify; đính chính embed claim
- **Bối cảnh:** Re-check công việc chuẩn hóa + duyệt cấu trúc Verify (nhiễu/lỗi thời); chuẩn hóa cả Claude-agent docs lẫn product.
- **Phát hiện:**
  - **Đính chính embed:** docs trước claim "code dùng bge-m3, config nomic=drift" là SAI → SPLIT: retrieval+query_router=`nomic-embed-text` (runtime), verify-side=`bge-m3` (`verify.py:35`); config nomic khớp retrieval.
  - **Nhiễu verify:** message P0a in "< 0.50" nhưng enforce ≈0.40; `topic_relevance_check` nhận `model=` nhưng KHÔNG gọi LLM → heuristic quantized {0.5,0.75,1.0}; grounding bão hòa (gộp mega-premise); `check_evidence_domain(min_relevance=0.50)` = param chết.
  - **Dead/legacy:** 6/8 hàm `verify.py` chỉ `deep_research.py`/scripts/eval gọi, KHÔNG live v3.
  - **Bug tiềm ẩn:** `faithfulness.py:183` except-fallback NameError; cross-ref 3 chỗ gate chồng + bug ngưỡng `:583` (flat min thay vì rule động).
- **Thay đổi (APPLY-NOW, 0 đổi hành vi):** code: message `:499`→in `ev_threshold`, comment `:450/:476`, xoá param chết `notes.py:461`, docstring `verify.py:242`. docs: GLOSSARY (0.60→0.40, grounding saturation, topic heuristic, embed split), CLAUDE/RULES (embed split, providers, topic heuristic, bản đồ Verify LIVE/LEGACY, target G0-G6), short-memory.
- **Đã implement (LOCAL-only, unit-test 13/13 pass):** G3 grounding de-saturated (per-source max HHEM, `faithfulness.py`), G4 topic judge gemma-local (blend `answer_relevance` + StageE floor), G2 citation-integrity (`verify_section` cite_precision fail-open), G5a fix cross-ref bug `:583` + gộp gate, G5b numeric-ref hint, fix NameError `faithfulness.py`. Mọi judge = model local; KHÔNG Claude/external trong pipeline. Test: `files/eval/test_verify_optim.py`.
- **Còn lại:** validation run đầy đủ (Ollama) để tinh chỉnh ngưỡng; G5b resolve "Section N.M" vs outline thật; G6 redundancy; unify embed model.

---

### [2026-06-15] Đánh giá llm_book_v36 + chuẩn hóa docs/cleanup
- **Bối cảnh:** Hoàn thành book v36 (712 trang); audit chất lượng + chuẩn hóa CLAUDE/RULES/memory cho khớp code thật; dọn file lạc mục tiêu.
- **Phát hiện chính:**
  - **g=1.0 toàn bộ 280 section → HHEM bão hòa, vô nghĩa làm tín hiệu.** Tín hiệu thật = `topic_relevance` (mean 0.782; 23 section ở sàn 0.50).
  - **Matrix pattern còn sống:** outline templated (35 anchor × 8 section); RAG/CoT bị tách 2 anchor; jaccard ~0.80 giữa section gần-trùng. `outline_audit` `ok=false` nhưng chỉ advisory → redundancy lọt. (FIX_SUMMARY/DEVELOPMENT_SUMMARY claim "fixed=0" là OVER-CLAIM.)
  - **~45% reference off-topic** (laser/MHD/QFT...) do prefilter 0.45 + domain gate ~0.40 quá lỏng cho chapter theme-generic.
  - **Doc drift nặng:** CLAUDE/short-memory ghi v3.4/v3.5, "32 section/19K từ/4 canonical", module ở `files/*.py`, P0a=0.60/0.80 — ĐỀU SAI. Thật: deep_research_v3.py + `files/research/*.py`, 280 section, 12 canonical, P0a≈0.40, grounding accept 0.70, topic 0.50.
  - **Embed SPLIT (đính chính 06-16):** retrieval+query_router chạy `nomic-embed-text`, verify-side `bge-m3`. config nomic KHỚP path retrieval — KHÔNG phải pure drift.
- **Thay đổi đã làm:**
  - Viết lại `CLAUDE.md` (pipeline 12-stage thật + 8 guardrails + ngưỡng code), `RULES.md` (bảng OPERATIVE vs TARGET), `short-memory.md` (snapshot v36), `long-memory.md` (de-dup entry 06-08).
  - Xoá cruft (~4MB): `cv_*_run.log`, `.DS_Store`, `danhgia/` rỗng, `*.bak`/`*.prepolish` (v36), tất cả `__pycache__`.
- **Bằng chứng:** ngưỡng verify trực tiếp tại `deep_investigate.py:218-220,479,671,702`, `notes.py:101,311`, `config.py`.
- **Còn lại:** (1) sửa outline generator khử matrix → audit `ok=true`; (2) siết per-section reference relevance; (3) **commit file core đang untracked** (deep_research_v3.py + research layer + GLOSSARY + memory + run_full.sh); (4) consolidate doc sprawl (eval/*SUMMARY*, WORKPLAN→plan, benchmark.md → archive).

---

### [2026-06-08] 7-run comprehensive eval + scoring rubric
- **Bối cảnh:** Sau 7 topics v3, cần phân loại run nào benchmark-được.
- **Phát hiện:** g=1.0 + topic=1.0 mọi run; arxiv recall **0%** (foundational papers không retrieve); paper `2510.22344` (FAIR-RAG) dominate 50-75% mọi run; rlhf_v3 §1.1=143w toàn RAG (evidence gate fail); diffusion_v3 3 zero-cite section; boilerplate tăng theo size.
- **Thay đổi:** `eval_v3_runs.py` + rubric 5 tiêu chí. Scoring: A=rag/longctx/agentic; B=transformer/llm_agentic_2026; C=rlhf/diffusion.
- **Còn lại:** fix P0a/P0b/P0c rồi rerun rlhf+diffusion. Báo cáo: `eval/reports/eval_v3_runs_20260608.md`.

### [2026-06-08] Shipped P0a/b/c fixes
- **P0a** (`notes.py`+`deep_investigate.py`): `check_evidence_domain()` LLM-gate trước writer, block/retry khi topic thấp. *(Lưu ý 06-15: ngưỡng thật về sau ≈0.40, không phải 0.60 như ghi lúc này.)*
- **P0b** (`discovery.py`+`deep_research_v3.py`): canonical injection, `--canonical-arxiv-ids`, force-fetch + protect.
- **P0c** (`notes.py`): seen-count penalty trong RRF; protected papers exempt. *(Formula sau cùng: `max(0.05,(1−f)²)`.)*
- **Bằng chứng:** syntax check pass. **Còn lại:** rerun verify.

### [2026-06-07] P3 verified + đồng bộ memory/product flow
- Fresh eval `attention_v3_p3` xác nhận P3a/b/c SHIPPED+VERIFIED: dup sections 32→0, R0 1→0, chapter specificity 0.15→1.0, primary coverage 83→100%, generic rate 100→0%.
- Tái cấu trúc 3 file: CLAUDE=luật, short=snapshot, long=journal.

### [2026-06-07] P3 outline repair + R7 canonical URL hygiene
- Lỗi ở post-processing, không phải model. Regex generic title, `_postprocess_outline` mọi return path, dedup section xuyên chương, loại DDG redirect khỏi canonical, R0 detection trong `discovery_eval.py`.

### [2026-06-04] True Deep Research v3 redesign
- Outline phải emerge từ evidence (không pre-fix). Tạo `discovery.py` / `outline_from_research.py` / `deep_investigate.py` / `deep_research_v3.py`. Smoke: 16 section, g=1.0, 365 cites, 31.6 min.

### [2026-06-04] Paper eval sprint + postprocess
- Tạo `paper_eval.py` + `postprocess_book.py` + arXiv boost. grounding 0.979, pass 98%, arXiv chỉ 7.6%.

### [2026-06-03] Migration writer → `batiai/qwen3.6-35b:iq3`
- MoE IQ3 tốt hơn cho quality/cost local; grounding tăng.

### [2026-05-29] bookv7 fix-batch
- 12 fix cấu trúc; WeasyPrint OK, tectonic còn lỗi `symbb`.

### [2026-05-27] Honest audit
- Xác định 4 structural gaps; W1 đã sửa; tạo baseline trung thực. Còn W2-W4.
