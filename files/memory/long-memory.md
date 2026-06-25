# Long Memory — Nhật ký nén theo phiên

> **Mục đích:** Lưu lịch sử quyết định/phát hiện/kết quả, mới nhất trước. Không lặp `short-memory.md`. Mỗi entry 1–3 dòng/mục; archive khi milestone.

```text
### [YYYY-MM-DD] <tiêu đề>
- Bối cảnh: / Phát hiện: / Thay đổi: / Bằng chứng: / Còn lại:
```

---

## Session log (mới nhất trước)

### [2026-06-25] DOCTRINE chốt: cải thiện AGENTIC (orchestration/inference), KHÔNG train — + P0.5–0.8 + clean-run
- **DOCTRINE (user affirmed, BẤT BIẾN):** Agentic Deep Research lên chất lượng ở tầng **orchestration/inference** (retrieval/verify/revise-loop/prompt/evidence-select), **KHÔNG train model & KHÔNG build dataset** (giữ topic-agnostic, prompt-robust, auditable). Codify: `CLAUDE.md mục 2/mục 6.9`, `RULES #8 + Product-FAIL #7`, `plan.md P1.5`.
- **Lever cuối = P1.5 verify-revise loop:** feed G2 per-`[N]` verdict (đã có `cite_res["verdicts"]`) ngược writer làm retry-hint surgical → revise đúng citation hỏng. KHÔNG weight.
- **Correctness/measurement fixes session này (PR #13-17):** P0.5 bestround-ships-failing-body + smoke-hollow-book (`quality="ok"` + completeness tin được); P0.6 G2 batch-judge parser (gemma emit 1-array/dòng → fail-closed pad floor giả; fix → discrimination GOOD 0.72→1.00 vs BAD→0.06/0.00); P0.7 claim-aware excerpt (`_best_passage` argmax cosine); P0.8 `.env` auto-load (tavily im lặng OFF → giờ ON, effective providers gồm tavily).
- **Clean-accept THẬT (đo qua nhiều run):** 7% (false floor) → 25-32% (arxiv-down) → **40% written / 0-6% block** (smoke arxiv+tavily+canonical+all-fix). Retrieval lever (tavily+arxiv) cắt block 21%→6%. Trần còn lại = writer-grounding → P1.5 (agentic, không train).

### [2026-06-23] P0-2b thực thi: soften cite-judge → faithfulness gate "xanh"
- **Làm:** `verify.py:47-75` (JUDGE_SYS + JUDGE_BATCH_SYS) — thay dòng "**Be strict: direct match only, not topical overlap**" bằng "judge by MEANING: `supports` = evidence states/implies/**faithfully paraphrases** claim; topical-overlap-no-support KHÔNG phải supports; contradicts/unrelated giữ strict". `_VERDICT_SCORE` (no_evidence 0.3) + `min_cite_precision=0.45` **GIỮ NGUYÊN** (change tối thiểu — không hạ mù).
- **Guard mới:** `files/eval/bench_cite_discrimination.py` gọi THẬT `verify.verify_section` trên labeled sections (GOOD paraphrase / BAD_unrelated / BAD_contradict) → assert GOOD≥0.45 ∧ gap≥0.30. Kết quả: **GOOD 0.72 (PASS) · BAD_unrelated 0.18 · BAD_contradict 0.20 · gap +0.54/+0.52**. Judge discriminate thật, KHÔNG rubber-stamp (chống "nới-để-qua" thành 1.0-giả lần 2).
- **Validation `p0_validate3` (RLHF smoke, prose THẬT):** cite_prec spread **0.275/0.321/0.411/0.481/0.487** (≠ 1.0, ≠ floored-constant); faithful section ACCEPT `quality="ok"` cite_prec **0.481/0.487**; round dưới gate (0.275-0.411) bị **retry** (loop re-research đẩy 0.321→0.487 mới qua). Gate 0.45 cắt sạch accept/retry → **gate SỐNG, discriminate**. (arxiv timeout suốt run — degrade graceful qua wiki/ddg.)
- **Kết:** P0-2 chuyển PARTIAL→DONE (quality "ok" giờ đạt); faithfulness gate hết INERT/fake-1.0 → SỐNG. **Bước kế = P1** (matrix HARD gate, paragraph-dedup, math-validation, near-miss rescue, held-out judge). *Follow-up nhỏ:* persist cite_precision accept-round vào state.json (field=None dù accept; BAER đọc log nên không vỡ).
- **Docs:** RULES/CLAUDE/GLOSSARY/README/short+long-memory/plan sync post-P0-2b. Commit + PR.

### [2026-06-22] P0 thực thi: decouple G2 + grounding log-only + fix P0c → lộ P0-2b
- **Làm:** `deep_investigate.py` — bỏ grounding khỏi gate (G3 log-only); `gate_ok = n_cites>0 AND topic≥0.50 AND cross-ref`; `verify_section` (G2) chạy khi `n_cites>0 AND topic_ok` (bất kể grounding) → cite_precision đo thật; `cite_precision=None` khi không đo (log hết phát default 1.0); best-round **topic-first**; StageE chuyển **sau-loop** gate theo best-topic; P0c `if run_seen_counts is None` (`:304`). Reviewer độc lập: **GO, 0 blocking**.
- **Validation `p0_validate2` (RLHF 1ch×4sec):** ✅ P0-1 G2 chạy thật cite_precision **0.30/0.367/0.374/0.410** (≠1.0); ✅ P0-3 run_seen_counts **0→23**; ❌ P0-2 quality "ok" = **0** — vì cite_precision (0.3-0.4) < min_cite_precision 0.45.
- **Phát hiện P0-2b (NEW):** gemma `verify_section` judge prompt "**Be strict: direct match only, not topical overlap**" + `_VERDICT_SCORE` {supports 1.0, partial 0.5, no_evidence 0.3, unrelated/contra 0} → trên prose synthesized đa số citation chấm no_evidence/unrelated → cite_precision **floor ~0.3-0.4** (CÙNG bệnh strict-NLI như HHEM). Citations IN-range (không phải out-of-range). → P0 mới làm lỗi HIỆN ra (hết fake 1.0), chưa làm gate dùng được. **P0-2b = soften judge (paraphrase/implication) / dùng cosine liên tục / recalibrate + discrimination test.**
- **Docs:** RULES/CLAUDE/short-memory/plan cập nhật post-P0 (G2 LIVE nhưng floor, P0-2b). Commit + PR.

### [2026-06-22] Đánh giá grounded (22-agent) → phát hiện verify post-writer INERT
- **Bối cảnh:** đánh giá product có grounding thật (đọc code + benchmark + nội dung sách); mọi verification holds:true.
- **Phát hiện then chốt (verified, tự re-check bằng số):** per-source-max grounding **không bao giờ chạm 0.70** (max 0.458; quality field: 0 "ok", mọi section "degraded" cả 4 run). Vì `base_ok` cần grounding≥0.70 → **base_ok LUÔN false** → (a) clean-accept không fire; (b) `verify_section` (G2) trong `if base_ok` **KHÔNG BAO GIỜ chạy** → `cite_precision=1.0` là DEFAULT init (BAER parse 93 dòng `cite_prec=1.000` từ retry-hint, gắn nhãn nhầm "G2 REAL"); (c) StageE topic-block (cần g≥0.70) **không fire**. → **Gate cứng SỐNG duy nhất = P0a domain-evidence (~0.40 pre-writer)**; mọi verify post-writer chỉ LOG. **SUPERSEDE "faithfulness thật = G2 cite_precision" ở entry 06-21** — G2 không chạy.
- **Điểm (harsh, evidence-based):** tổng **C+/B−**. Faithfulness C− · Eval C+ (vòng tròn: topic≡accept, 0 ground-truth) · Architecture B− (render/resume tốt; bug P0c aliasing no-op) · sách-RLHF B− (toán DPO đúng nhưng eqn malformed + LaTeX leak) · sách-605pg C+ (matrix 269/269 forced scale) · novelty B−.
- **Thay đổi:** clean toàn bộ docs (RULES/CLAUDE/GLOSSARY/README + memory) về đúng "verify post-writer INERT, P0a là gate sống" + viết mục Upgrade roadmap vào `plan.md`.
- **Bằng chứng:** grounding max 0.458, `bench_rlhf.log` 93×`cite_prec=1.000`, `benchmark_book.py:248`, `deep_investigate.py:729` base_ok, `:301` `run_seen_counts = x or {}` aliasing.

---

### [2026-06-21] Chuẩn hóa docs + sync GitHub; làm rõ grounding = ADVISORY
- **Bối cảnh:** verify toàn bộ docs/memory vs code (audit 13-agent read-only) để hết nhiễu sau chuỗi HHEM-fix/evidence-pool/benchmark; đồng bộ GitHub.
- **Làm rõ grounding (supersede mọi note "G3 de-saturated" / "grounding ≥0.70 = gate chất lượng" ở entry cũ bên dưới):** có HAI số — `grounding`=per-source-MAX (gate dùng; là soft conjunct của `base_ok` ở 0.70 NHƯNG không block một mình: thiếu → ship `quality='degraded'`) và `grounding_cited`=strict cited (~0.06 trên prose, BAER post-hoc, **ADVISORY**). ~~Faithfulness thật = G2 cite_prec; tín hiệu phân biệt LIVE = G4 topic (G2 saturate 1.0)~~ **(⚠️ SAI — supersede 06-22: G2 KHÔNG BAO GIỜ chạy, cite_precision=1.0 là default; gate sống = P0a).**
- **Embed:** xác nhận **UNIFIED** `bge-m3:latest` mọi path (`config.py:34`/`notes.py:111`/`query_router.py:210`/`verify.py:35`); 0 ref nomic sống — supersede note "Embed SPLIT" ở entry 06-15/06-16.
- **Thay đổi:** chuẩn hóa RULES/CLAUDE/GLOSSARY/README (grounding advisory, embed unify, **G2 fail-CLOSED**, evidence-pool, line-refs) + refresh short-memory snapshot. **PR #9 merged → main**.
- **Bằng chứng:** `deep_investigate.py:227,729,745,850`, `faithfulness.py` grounding_score (per-source-max + `grounding_cited`), `notes.py:109/110/324`. Docs-only, 0 đổi hành vi pipeline.
- **Còn lại:** semantic/LLM-judge eval (gap lớn nhất — BAER chỉ cơ học); deeper retriever topic ngách.

---

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
- **Phát hiện:** g=1.0 + topic=1.0 mọi run; arxiv recall **0%** (foundational papers không retrieve); paper `2510.22344` (FAIR-RAG) dominate 50-75% mọi run; rlhf_v3 mục 1.1=143w toàn RAG (evidence gate fail); diffusion_v3 3 zero-cite section; boilerplate tăng theo size.
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
