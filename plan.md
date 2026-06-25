# Plan -- Discovery Deep Research Improvement

**Purpose:** Evaluate and evolve a truly prompt-emergent Deep Research pipeline where structure, chapters, subchapters, and section content arise from raw prompt + discovered evidence, not from pre-scripted topic templates.

**Status:** P3a/b/c SHIPPED. **P0 + P0-2b ✅ DONE (2026-06-23)** — faithfulness gate (G2) từ INERT/fake-1.0 → SỐNG: decouple G2 + grounding log-only + P0c aliasing fixed + **cite-judge soften** (faithful prose ACCEPT, discriminate GOOD 0.72 vs BAD 0.18/0.20). **Ưu tiên kế = P1** (mục Upgrade dưới: matrix HARD gate, paragraph-dedup, math-validation, near-miss rescue, held-out judge).

---

# Upgrade Roadmap (2026-06-22, eval-driven) — ƯU TIÊN CAO NHẤT

> Nguồn: đánh giá grounded 22-agent (đọc code + benchmark + nội dung sách thật), mọi finding adversarial-verify `holds:true`. Điểm tổng product hiện tại **C+/B−**. Roadmap này sửa đúng các điểm yếu đã verify. **Bất biến giữ nguyên:** LOCAL-only · Verifier≠Writer · fix-ở-GATE-không-ở-writer · outline emerge-from-evidence.

**Phát hiện nền (lý do có roadmap này):** mọi verify **post-writer** (G2 citation / G3 grounding / G4 topic / StageE) hiện **INERT**. `base_ok` yêu cầu per-source-max grounding ≥ 0.70, nhưng HHEM strict-NLI trên prose synthesized chỉ ~0.05–0.10 (max thực **0.458**) → `base_ok` LUÔN false → clean-accept không fire, mọi section ship `quality='degraded'`, `verify_section` (G2) **không bao giờ chạy** → `cite_precision=1.0` là **default** (BAER parse nhầm từ log). **Gate cứng SỐNG duy nhất = P0a domain-evidence (~0.40, pre-writer).**

Mỗi item: **Vấn đề (bằng chứng)** → **Fix (file:dòng)** → **Acceptance (đo bằng gì)**.

## P0 — Faithfulness sống lại (verify post-writer đang chết)

### P0-1. Decouple G2 khỏi thanh grounding chết
- **Vấn đề:** `verify_section` (G2 citation-vs-source) nằm trong `if base_ok:` (`deep_investigate.py:737`); `base_ok` cần grounding≥0.70 (`:729`) không bao giờ pass → G2 **không bao giờ chạy**, `cite_precision=1.0` là default (`:732`).
- **Fix:** tách `gate_ok = (n_cites>0 AND topic≥min_topic AND has_min_cross_refs)` (bỏ grounding khỏi điều kiện cứng); chạy `verify_section` khi `gate_ok` (bất kể grounding); accept khi `gate_ok AND cite_precision≥min_cite_precision`. Giữ grounding log advisory.
- **Acceptance:** re-run benchmark → `cite_precision_mean < 1.0` CÓ phân bố (không phải toàn 1.0); một số section fail G2 → retry/block; log có dòng "Citation integrity (G2)" thật.
- **✅ DONE (2026-06-22, validation `p0_validate2` 4 sec):** G2 chạy thật, cite_precision đo được **0.30/0.367/0.374/0.410** (≠ default 1.0); citations in-range (max[N]≤n_sources); reviewer độc lập GO. `deep_investigate.py:737-748` (chạy khi `n_cites>0 AND topic_ok`).

### P0-2. Re-baseline / bỏ grounding khỏi `base_ok`
- **Vấn đề:** per-source-max grounding max 0.458 < min_grounding 0.70 → 0/390 section "ok", 100% "degraded"; StageE (`:751`) cần g≥0.70 nên không bao giờ fire (topic-drift không bị chặn).
- **Fix:** bỏ `grounding >= min_grounding` khỏi `base_ok`; chuyển grounding sang log thuần (cả `grounding` per-source-max lẫn `grounding_cited`). StageE đổi điều kiện chặn topic-drift độc lập grounding (topic<min_topic + n_cites>0 → block/retry).
- **Acceptance:** quality field có lại "ok"; StageE fire trên topic-fail thật; không section nào ship "degraded" chỉ vì grounding.
- **✅ DONE (2026-06-23, sau P0-2b):** grounding log-only + bỏ khỏi `base_ok` ✅; best-round topic-first ✅; StageE sau-loop theo best-topic ✅; **quality "ok" GIỜ ĐẠT** (sau khi P0-2b soften judge → cite_precision faithful prose ≥0.45). `deep_investigate.py:712-716,804-813,867`.

### P0-2b. ⚠️ NEW (từ P0 validation): G2 cite-judge strict-match → cite_precision floor ~0.3-0.4
- **Vấn đề (verified):** `verify.py` judge prompt ghi "Be strict: 'supports' requires a **direct match**, not just topical overlap" + thang `_VERDICT_SCORE` {supports 1.0, partial 0.5, no_evidence 0.3, unrelated/contradicts 0}. Trên prose **synthesized/paraphrase**, gemma hiếm chấm "supports" → đa số no_evidence/unrelated → cite_precision floor **~0.30-0.41** (CÙNG bệnh strict-NLI như HHEM/G3). Citations in-range (không phải out-of-range artifact). Ngưỡng `min_cite_precision=0.45` → **0 clean-accept** → P0 mới chỉ làm lỗi HIỆN ra, chưa làm gate dùng được.
- **Fix (3 lựa chọn — design fork về độ-strict-faithfulness):** (a) **soften judge** — "supports = evidence states OR clearly implies/paraphrases the claim" (bỏ "direct match only"), phù hợp prose synthesized; (b) **dùng cosine liên tục** (mean cos) thay verdict-bucket; (c) **recalibrate** `min_cite_precision` về dải discriminate thật. Khuyến nghị (a) + giữ discrimination test.
- **Acceptance:** sau fix → cite_precision có spread rõ + clean-accept >0 ("ok" xuất hiện) trên section tốt, ĐỒNG THỜI vẫn fail section citation kém (discrimination test good-vs-injected-bad), không floor/saturate.
- **✅ DONE (2026-06-23, chọn (a) soften judge + giữ discrimination test):** sửa `verify.py:47-75` — `supports` = "evidence states/implies/**faithfully paraphrases** the claim" (bỏ "direct match only, not topical overlap"); contradicts/unrelated giữ strict; `_VERDICT_SCORE` + `min_cite_precision=0.45` GIỮ NGUYÊN (không hạ mù). **Discrimination test mới** `files/eval/bench_cite_discrimination.py` (gọi thật `verify.verify_section`): GOOD=**0.72** (PASS gate) vs BAD_unrelated=**0.18** / BAD_contradict=**0.20** (gap +0.54/+0.52 ≫ 0.30). **Smoke RLHF thật** (`p0_validate3`): faithful section ACCEPT `quality="ok"` cite_prec **0.481**; section yếu floor (R1 0.321→R3 0.487 mới qua, hoặc <0.45 → degraded) → gate SỐNG, discriminate, KHÔNG rubber-stamp. *Follow-up nhỏ (không thuộc P0-2b):* persist `cite_precision` accept-round vào state.json (hiện field=None dù accept; BAER đọc từ log nên không vỡ).

### P0-3. Fix bug aliasing P0c (seen-penalty no-op trong 1 run)
- **Vấn đề:** `deep_investigate.py:301` `run_seen_counts = run_seen_counts or {}` — dict rỗng `{}` là falsy → rebind sang local mới; propagate-back (`:832`) ghi vào bản copy bị vứt. Hệ quả: P0c seen-penalty (`notes.py:322`) **không bao giờ fire cross-section** trong 1 run (state.json `run_seen_counts` len=0 dù 454 source). Một paper có thể dominate >50% mà không bị phạt.
- **Fix:** `if run_seen_counts is None: run_seen_counts = {}` (giữ object identity của caller).
- **Acceptance:** sau fresh run, `run_seen_counts` non-empty; grep log thấy P0c penalty fire; không source nào >50% sections.
- **✅ DONE (2026-06-22):** `if run_seen_counts is None` (`deep_investigate.py:304`); validation: `run_seen_counts` len **0→23**, propagation về orchestrator OK.

## P0.5 — Correctness fixes (phát hiện bởi verify đa-tầng 2026-06-23) ✅ DONE

> Verify (10 finder + adversarial refute từng finding, 89 finding 0 bị bác) lộ **2 bug correctness** ngoài roadmap — đã fix.

### P0.5-1. `bestround-ships-failing-body` — accept ship body đã TRƯỢT G2
- **Vấn đề (reproduce):** best-round chọn **topic-first** (`:712-716`) nhưng accept `break` ở round có cite≥0.45; hàm return `best_content`. Khi best-topic-round ≠ accept-round → ship body round topic cao (đã trượt cite-gate) với `quality="ok"`. Bằng chứng persist: `p0_validate3` sec 1.2 (`ok`) content 19 markers vs `n_citations=15` → **4/14 section** content≠metadata.
- **Fix (`deep_investigate.py`):** pin `best_*` (content+sources+topic+**n_cites+cite_markers+cross_refs**) vào ĐÚNG round (best cho degraded, accept cho accepted) — init `:297-300`, selection `:724-726`, accept-override `:760-766`, return dùng `best_n_cites/best_cite_markers/best_cross_refs`. → content & metadata luôn cùng 1 round; "ok" body luôn là round qua G2.
- **Acceptance:** fresh run → markers(content) == n_citations mọi section (0 mismatch); "ok" body = round accept. (validation `p0_validate4` đang chạy.)

### P0.5-2. `smoke-hollow-book` — assemble emit 70/84 heading rỗng
- **Vấn đề:** smoke investigate 2 chapter nhưng `assemble_book` duyệt **full outline** → section vắng state.json (raw="") không match skip BLOCKED → emit `## heading` body rỗng. `p0_validate3/book.md`: 84 heading, **14 có body, 70 stub rỗng**.
- **Fix (`deep_research_v3.py:131-154`):** `if key not in sections: continue`; gom `sec_lines` per-chapter, bỏ chapter rỗng (`if not sec_lines: continue`).
- **Acceptance:** re-assemble `p0_validate3` → **14 heading, 14 body, 0 hollow** ✅ (đã verify).

## P0.6 — G2 measurement fix (chẩn đoán "vì sao clean-accept thấp", 2026-06-24) ✅ DONE

> Chẩn đoán đa-agent (5 giả thuyết, reproduce G2 thật trên `p0_validate4`): clean-accept thấp (7%) **chủ yếu là BUG ĐO, không phải quality**. `_judge_batch`: gemma4:e4b emit **1 mảng `[{...}]` mỗi DÒNG** chứ không phải 1 mảng lớn; `_JSON_ARRAY_RE.search()` chỉ bắt mảng đầu → parse 1 verdict → **fail-closed pad** phần còn lại `no_evidence=0.3` → floor giả (sec 1.3: giữ 1/15 verdict, pad 31). Cộng thêm gemma quit ~15 verdict trong 1 call lớn (done_reason=stop) + timeout 60s → wholesale no_evidence.

- **Fix (`verify.py`, gate-side, LOCAL-only):** (1) parser thu **TẤT CẢ** object (`_JSON_ARRAY_RE.finditer` đa-mảng + fallback `re.finditer(r'\{...\}')` bare-object) `:199-220`; (2) **chunk queued ~6 + 1 retry/chunk** `:330-345`; (3) `DEFAULT_TIMEOUT 60→150` `:25`; (4) `_extract_claim` giữ câu **CUỐI** trước marker `:97-110`; (5) **`AUTO_SUPPORT_COS 0.75→0.90`** `:33` (paraphrase 0.75-0.90 phải qua judge — bỏ false auto-support inflate, +0.0 với GOOD nhưng −với BAD).
- **Acceptance ✅:** Discrimination `bench_cite_discrimination.py` **GOOD 0.72→1.00 vs BAD 0.18/0.20→0.06/0.00 (gap +0.94/+1.00)** — bất đối xứng, BAD đi XUỐNG → KHÔNG rubber-stamp. Reproduce prose thật `p0_validate4`: faithful bị floor giả (1.1→0.500, 2.7→0.468) **ACCEPT thật**; weak (no_evidence-dominant) vẫn <0.45.
- **DON'T (verified harmful):** hạ `min_cite_precision`, bump `no_evidence` 0.3, đụng `AUTO_UNRELATED_COS`, soften judge thêm, yếu fail-closed pad, để writer tự chấm.
- **Residual #1 = retrieval/excerpt quality** (verdict no_evidence-dominant: excerpt là head-slice/LaTeX-dump không chứa fact của `[N]`).

## P0.7 — Claim-aware excerpt selection (2026-06-24) ✅ DONE
> Gốc no_evidence: `enrich_top_sources` thay excerpt bằng **550 từ ĐẦU** của full-text (`fetch_full_text` head-slice / math-window), không phải passage gần claim → judge thấy "generic" → no_evidence.
- **Fix (`notes.py`):** thêm `_best_passage(body, query, max_words, embed_model)` — fetch body DÀI hơn (≥1600 từ) → chia window chồng lấp 50% → trả **window cosine cao nhất với `section_prompt`** (argmax, bge-m3). `enrich_top_sources` nhận thêm `section_prompt`/`embed_model` (optional → legacy head-slice giữ nguyên, backward-compat cho `deep_research.py`/`regen_broken.py`). Call site live `deep_investigate.py:489` truyền `section_prompt`+`embed_model`.
- **Acceptance:** unit test `_best_passage` PASS (chọn đúng window chứa fact, head-slice trượt); excerpt-query cosine tăng by-construction (argmax). End-to-end cite_prec lift → đo ở fresh run. Guard: discrimination KHÔNG đổi (#3 không đụng judge).
- **Lưu ý:** excerpt tốt hơn giúp CẢ writer (ground chặt hơn) LẪN judge (thấy passage đúng) → double win. arxiv down → benefit chủ yếu trên wiki/ddg (re-fetchable).

## P1.5 — Verify-revise loop: fix writer-grounding AGENTIC (KHÔNG train) — **[NEXT, lever đúng-bản-chất]**

> Lever đúng-bản-chất-agentic (xem doctrine `CLAUDE.md mục 2`/`mục 6.9`). Residual cuối = **writer grounding** (smoke arxiv+tavily = 40% accept / 0-6% block, NHƯNG ~60% vẫn `degraded`, cite_prec 0.25–0.44 dù retrieval+excerpt đã tốt). Fix ở LOOP, không weight.
- **Vấn đề:** pipeline đã có 3-round retry + G2 chấm **per-`[N]`** (`verify.verify_section` trả `verdicts:[{n,verdict,reason}]`), nhưng degraded section đi hết 3 round vẫn floor → **feedback chưa dùng sắc**: retry-hint hiện gom chung (`weak_summary`), không chỉ writer ĐÚNG citation nào hỏng + vì sao.
- **Fix (`deep_investigate.py` retry-hint, gate-side):** lấy `cite_res["verdicts"]` (đã có sẵn) → dựng hint **per-citation**: *"[5] no_evidence — excerpt không nêu X → đổi nguồn HOẶC bỏ claim; [8] unrelated — citation sai → thay"*. Round sau writer revise ĐÚNG chỗ trượt (surgical), không viết lại mù.
- **Acceptance:** % degraded → ok tăng (near-miss 0.40-0.44 vượt 0.45 sau revise có-hướng); discrimination giữ (không nới gate).

## P1 — Cấu trúc sách & độ tin eval

> ✅ **RE-AUDIT GROUNDED 2026-06-23 (5-agent, verify trực tiếp code + run thật).** Cả 5 item **CÒN THẬT** (không cái nào bị fix đã-ship hóa giải). Thứ tự thực thi đã xác nhận: **P1-3 → P1-1 → P1-4 → P1-2 → P1-5** (leverage = impact × readiness ÷ invariant-risk). Mỗi item dưới kèm STILL-REAL + bằng chứng + fix-site + impact/readiness.

### P1-3. Math validation gate (chống eqn hỏng + LaTeX leak) — **[RANK 1] impact MED · readiness HIGH · risk 0**
- **STILL-REAL (2 bug reproduce ngay trong phiên):** (a) **FALSE-NEGATIVE** — mẫu Bradley-Terry thiếu ngoặc `\frac{\exp(..)}{\exp(.. + \exp(..)}` (7 `(` vs 6 `)`) → `_math_span_valid`==True (brace balance OK, macro allowlisted, **KHÔNG có paren check**) → **ship math SAI** (`bench_rlhf/book.md:98`; tái diễn pool2×3, diffusion×1, agentic×4). (b) **FALSE-POSITIVE** — `mathfix.py:229` đếm substring `\left`/`\right` đụng `\leftarrow`/`\rightarrow` → math hợp lệ bị đẩy thành literal `$$..$$` trong backtick (rlhf 7 spans, agentic 21). **"$$-in-backticks leak" chính là output của neutralizer, KHÔNG phải writer leak.** `\coloneqq` thiếu trong `_MACRO_ALLOWLIST`.
- **Fix-site:** `mathfix.py:222-234` (`_math_span_valid`): (1) thay substring `\left`/`\right` bằng whole-word regex `\\left(?![A-Za-z])`; (2) thêm paren-balance SAU khi strip `\(`,`\)` + token `\[A-Za-z]+`; (3) thêm `\coloneqq` + siblings vào allowlist `:182-219`. Regression: `test_math_char_safety.py:87-88`.
- **Acceptance:** `_math_span_valid('r_i \leftarrow ...')`==True, `('a \rightarrow b')`==True, BT-thiếu-ngoặc==False; test suite cũ KHÔNG regress; book.md re-scan: BT neutralize, `\leftarrow` typeset lại. **DON'T:** đụng writer/verifier; nới brace-balance `:227` (đang đúng, unbalanced-brace=0 mọi nơi).

### P1-1. Matrix thành HARD gate (chống template ở scale) — **[RANK 2] impact HIGH · readiness HIGH · 1 tripwire**
- **STILL-REAL (chunked outline KHÔNG chữa suffix-matrix):** `audit_outline` chỉ **prefix-only** `startswith(f"{bucket}:")` trên 6-item `_STRUCTURAL_BUCKETS` (`outline_from_research.py:427-433`), không có suffix check; block chỉ fire >50 pattern (`:456`). Bằng chứng run: **`p0_validate3` (tôi vừa chạy)** = 84 sec, 12 ch × **cùng 7 suffix** ('Core Mechanisms'×4, 'Design and Trade-offs'×4, 'Practical Methods'×4, 'Evaluation'×4...), audit chỉ flag `['coherence_low']`, **0 matrix flag**. `agentic_2025_full` 288 sec cũng ship suffix-matrix. **Gốc:** `_evidence_sections_for_chapter` fallback phát `f"{theme}: {angles[j]}"` từ list `angles` cố định 12 mục (`:245-254`), lặp y hệt mọi chapter khi per-chapter LLM trả None (`:237-238`).
- **Fix-site:** `outline_from_research.py:417-457` — port suffix-detector `benchmark_book.py:145-146` (`Counter(t.split(':')[-1])`, ngưỡng ≥3/suffix qua nhiều chapter) vào `audit_outline` + condition `MATRIX_PATTERN_BLOCK` theo tỷ lệ; phụ: list `angles` `:245-254`.
- **Acceptance:** outline suffix-matrix → reject/retry TRƯỚC Stage 2. **TRIPWIRE (outline-emerges):** retry phải regen qua **chunked LLM evidence-path**, KHÔNG snap về `_semantic_fallback_outline` (`:810-828`, bản thân là suffix-matrix generator). Tune ngưỡng để domain-term hợp lệ ('Evaluation' là topic thật) không false-positive.

### P1-4. Near-miss rescue (0.35–0.40) thay vì drop cứng — **[RANK 3] impact HIGH · readiness MED (cần validation run)**
- **STILL-REAL:** `ev_topic_rel` tính 1 lần (`deep_investigate.py:514`), nhánh duy nhất `if ev_topic_rel < ev_threshold` (≈0.40, `:528`), round cuối raise RuntimeError vô điều kiện (`:546-552`); **0 nhánh re-query**. Evidence-pool rescue (`:420`) trigger trên `len(filtered)<5` (thin-pool, score-agnostic, CHẠY TRƯỚC gate) → KHÔNG cứu near-miss-score. Run data (7 sách, 159 block): **73% (116) ở dải 0.30-0.40**, avg 0.357, lệch trung bình chỉ **0.043**; ~16.4 section/sách mất oan.
- **Fix-site:** `deep_investigate.py:529-552` — chèn 1 round re-query nhắm `must_cover` + re-gate TRƯỚC khi raise (tái dùng query_gen+search+check_evidence_domain).
- **Acceptance:** một phần near-miss được rescue, block-rate giảm mà faithfulness (cite_prec) không tụt. **TRIPWIRE (Guardrail 6):** re-retrieve/re-gate THẬT, **KHÔNG hạ `ev_threshold`** (27% true-off-domain <0.30 vẫn phải drop); giữ hard-block làm fallback cuối. Phải có **validation run** chứng minh re-query nâng pool 0.357 → >0.40 (lý do readiness MED).

### P1-2. Paragraph/sentence dedup lúc assemble — **[RANK 4] impact MED · readiness HIGH · polish (tension fix-at-gate)**
- **STILL-REAL:** assemble (`deep_research_v3.py:121-192`) chỉ dedup heading-title (`:164-167`), **0 dedup câu/đoạn**. Empirical: `bench_rlhf_pool2/book.md` (80k từ) 1 câu lặp **17× xuyên 4 chapter** (9-12), 1 câu khác 12×, 17 câu lặp ≥2× (kể cả sau normalize số). Section-Jaccard mù (`near_dup_pairs=0`, max 0.06). Numeric-ref bịa "Section 2.1: ..." CHỈ ở `agentic_2025_full` (lines 13,75; 0 heading khớp) — hẹp, không phải mọi sách.
- **Fix-site:** `deep_research_v3.py:155` (dedup pass sau join, trước normalize_math) + scrub numeric-ref ở `_sanitize_section_content` `:55-69`.
- **Acceptance:** đếm câu boilerplate trùng giảm; 0 numeric-ref bịa. **TRIPWIRE (Verifier≠Writer):** **deletion-only** câu trùng exact/near-exact, **KHÔNG đụng occurrence đầu**, KHÔNG orphan `[N]` cite, KHÔNG rewrite prose (đây là presentation-scrub như log-strip sẵn có); bge-m3 cosine phải LOCAL.

### P1-5. Held-out judge độc lập (phá vòng tròn eval) — **[RANK 5] impact HIGH · readiness MED · BLOCKED (cần model)**
- **STILL-REAL (tautological):** BAER topic (`benchmark_book.py:108` đọc `topic_relevance` pipeline ghi từ G4) + ref-on-topic (`:176-179` đọc `src.relevance` = `notes.rank()` cosine, **CÙNG scorer** với prefilter/P0a). `topic_pass ≡ accept_rate` về cấu trúc (cùng ngưỡng 0.50). **0 judge độc lập** trong `eval/` (`benchmark_book.py:7` "does NOT call any model"). `paper_eval.py` dùng gemma+qwen (cùng họ) và KHÔNG wire vào BAER.
- **Fix-site:** `benchmark_book.py:241` (pass judge độc lập trên sample accepted sections) + `aggregate_benchmark.py:25-36` SIGNALS.
- **Acceptance:** 1 số chất lượng decorrelated với accept_rate (kappa/agreement vs G4). **BLOCKER:** phải **chọn+pull (SWAP, KHÔNG train)** 1 model LOCAL **khác họ** (KHÔNG gemma-G4, KHÔNG qwen-writer) trong Ollama trước khi code; cô lập thành pass optional để giữ determinism "no-model" của BAER. Eval-side only — held-out judge là **eval artifact** (đo), KHÔNG phải training data.

## P2 — Logic agentic sâu hơn (xây năng lực, không chỉ chứng minh)

### P2-1. Citation-graph 2nd-hop retrieval cho topic ngách
- **Vấn đề:** pool thưa cho sub-topic ngách → near-miss block; retrieval hiện chỉ 1-hop (search provider).
- **Fix:** với section pool mỏng, follow references của top paper (arxiv refs/semantic-scholar) để fetch nguồn 2nd-hop on-topic → nạp qua cùng prefilter (faithful) + P0c-exempt như evidence-pool.
- **Acceptance:** pool-depth tăng cho topic ngách; near-miss block giảm; rescue-fire count đo được.

### P2-2. Primary-source routing cho citation định nghĩa/phương trình
- **Vấn đề:** marker `[N]` ở định nghĩa/equation đôi khi trỏ secondary aggregator (emergentmind/DDG-redirect) thay vì paper gốc (vd τ/Ω(τ) trỏ explainer thay vì Yao 2022; Voyager trỏ survey).
- **Fix:** rule ưu tiên primary-source khi cite block định nghĩa/equation (match về canonical arxiv ID nếu có trong pool).
- **Acceptance:** citation ở dòng định nghĩa/equation trỏ primary arxiv ID (đo % primary-cite trên equation lines).

## Thứ tự đề xuất
**P0 ✅ DONE** (faithfulness ảo→thật + P0c). **P1 (re-audit grounded 2026-06-23) thứ tự = P1-3 → P1-1 → P1-4 → P1-2 → P1-5:**
1. **P1-3 math** trước — zero-risk, 2 bug reproduce được trong phiên, pure-Python + test harness có sẵn, không đụng writer/verifier.
2. **P1-1 matrix** — impact ngang P1-3 (giết suffix-matrix Guardrail 3) nhưng có 1 tripwire outline-invariant (retry phải qua LLM-path, không fallback template) → làm sau fix mechanical an toàn.
3. **P1-4 near-miss** — impact cao (~16 sec/sách mất ở mép gate) nhưng readiness MED: cần validation run chứng minh re-query nâng pool >0.40 (không ship mù).
4. **P1-2 dedup** — thật & visible nhưng là post-writer cosmetic scrub (tension fix-at-gate) → polish, deletion-only.
5. **P1-5 held-out judge** — chiến lược nhất (chỉ nó bắt được G4 false-positive) nhưng BLOCKED trên việc chọn model LOCAL khác-họ; eval-side only → sequence cuối.
Mỗi P1 item phải có validation run đo Acceptance trước khi tin. Sau P1 → **P2** (citation-graph 2nd-hop, primary-source routing).

---

## 0. Product Doctrine -- Emergent-from-Prompt Only

### Non-negotiable principle
The pipeline **must not know the scenario in advance**.

That means the system must **not**:
- hard-code topic-specific outlines
- pre-decide chapter lists for a known benchmark topic
- use hidden domain templates as "correct answers"
- benchmark by special-casing per-topic logic

The system **must**:
- start from raw prompt only
- let Gemma4 analyze the prompt and discovered evidence
- derive TopicProfile from discovery, not assumptions
- derive chapter/subchapter structure from evidence clusters
- research each section after structure quality is accepted
- write with Qwen only after evidence passes quality gates
- verify and deduplicate at both section-level and book-level

### Canonical flow
```
raw prompt
-> prompt analysis (Gemma4)
-> discovery deep research
-> TopicProfile
-> outline-from-evidence
-> structure quality review
-> section research
-> evidence quality gates
-> Qwen writing
-> verify / dedupe / coherence review
-> assemble book
```

### Role split
- **Gemma4:** prompt analyzer, evidence judge, topic gate, drift detector, overlap detector
- **Discovery:** infer scope, canonical concepts, boundaries, chapter candidates from evidence
- **Qwen3.6 30B active 3B:** technical writer only after evidence is accepted
- **Verify layer:** grounding, topic relevance, coverage adequacy, duplication check
- **Assembler:** final logical book, not just concatenated sections

### Hard rule
Do **not** hard-code answers. Do hard-code **quality contracts**.

---

## 1. Overview

### Problem
v2 has structural weaknesses: outline decided too early, retrieval justifies pre-existing structure, topic scope drifts, primary-source coverage stays low even when grounding looks high.

v3 improves this by moving to retrieval-first discovery, but current product reality shows that structure quality alone is not enough. A book can look structurally sound while still failing topic purity, canonical recall, or cross-section coherence.

### New pattern
```
Old: topic -> outline first -> retrieve later -> write
New: topic -> discover evidence -> TopicProfile -> outline-from-evidence -> investigate -> write
Target: raw prompt -> discover evidence -> TopicProfile -> outline-from-evidence -> quality gates -> investigate -> verify -> dedupe -> assemble
```

### Scope (in-scope modules)
`discovery.py` `outline_from_research.py` `deep_investigate.py` `query_router.py` `search.py` `notes.py` `fetch.py` `verify.py`

### Non-goals
No benchmark cheating, no hidden topic templates, no pre-scripted chapter answers, no replacement of paper-eval framework.

---

## 2. Product Requirements for a High-Quality Book

### R1 -- Prompt-emergent structure
All chapters and subchapters must emerge from the prompt and discovered evidence, not from a known scenario or fixed domain script.

### R2 -- Structure quality before writing
Do not start section writing until chapter/subchapter structure passes specificity, uniqueness, and coverage review.

### R3 -- Evidence quality before writing
Do not let the writer see evidence that fails section-domain relevance, canonical sufficiency, or diversity checks.

### R4 -- Writer as executor, not planner
Qwen writes only after research and gates pass. It does not invent the plan, invent missing evidence, or re-scope the topic on its own.

### R5 -- Verify more than grounding
Grounding is necessary but insufficient. The pipeline must also verify topic relevance, section adequacy, coverage, and duplication.

### R6 -- No repeated content
No duplicate titles, no repeated concept explanations across sections unless explicitly progressive, and no discourse boilerplate dominating the book.

### R7 -- Book-level logic
A finished book must read as a coherent technical reference, not as a bag of independently acceptable sections.

---

## 3. Hypotheses

| # | Hypothesis | Evidence needed |
|---|-----------|---------------|
| H1 | Outline specificity improves | chapter/section names more topic-specific than v2 |
| H2 | Section prompts improve | fewer empty/generic prompts |
| H3 | Topic drift decreases | fewer off-topic sections |
| H4 | Canonical discovery improves | better paper/term/subtopic surfacing |
| H5 | Retrieval quality improves indirectly | better downstream query and evidence quality |
| H6 | Grounding alone is not enough | outline quality + specificity must also improve |

**Key principle:** Do not compare only grounding scores. Compare the whole planning chain.

---

## 3. Metrics & Pass/Fail

### Discovery metrics

| Metric | Definition | Target | 6/6 avg | verdict |
|--------|------------|--------|---------|---------|
| TopicProfile completeness | 11 fields present | >= 90% | 90.91% | PASS |
| Canonical term precision | % canonical terms judged relevant | >= 0.80 | n/a | unknown |
| Seed query usefulness | % seed queries judged helpful | >= 0.75 | n/a | unknown |
| Fallback rate | % runs using semantic fallback | <= 20% | true | FAIL |

### Outline metrics

| Metric | Definition | Target | 6/6 avg | verdict |
|--------|------------|--------|---------|---------|
| Chapter specificity | specificity score 0-1 | >= 0.80 | 1.00 | PASS |
| Section specificity | specificity score 0-1 | >= 0.80 | 1.00 | PASS |
| Generic title rate | "Chapter 1", "Part 1" patterns | <= 10% | 0% | PASS |
| Empty prompt rate | sections with missing/weak prompt | 0% | 0% | PASS |
| Duplicate section titles | repeated section names | 0 | 0 | PASS |
| Prompt quality | specificity score 0-1 | >= 0.75 | 1.00 | PASS |

> Note: all specificity/prompt scores use 0-1 scale (computed by `discovery_eval.py`). Plan targets translated: LLM 4/5 = 0.80.

### Downstream metrics

| Metric | Definition | Target | 6/6 avg | verdict |
|--------|------------|--------|---------|---------|
| Topic relevance pass rate | sections passing relevance gate | >= 95% | n/a | unknown |
| Grounding average | mean supported/total claims | >= baseline | 1.000 | PASS |
| arxiv coverage | % citations from arxiv | > 7.6% baseline | 100% | PASS |
| Gold paper recall | must-cite / should-cite recovery | > baseline | n/a | unknown |

### Product realism

| Metric | Target |
|--------|--------|
| Discovery runtime | acceptable for local usage |
| Failure recoverability | high |
| Output usefulness | clearly better than baseline |

### Pass / Fail Criteria

**Minimum pass (all required):**
- [x] outline specificity improves -- **PASS** (0.15 -> 1.00 on attention_v3_p3)
- [x] empty/generic prompts reduced -- **PASS** (prompt quality 0.70 -> 1.00)
- [ ] topic drift reduced -- **unknown** (not measured yet)
- [x] primary-source coverage improves -- **PASS** (83% -> 100%)
- [x] no collapse in grounding -- **PASS** (g=1.000)

**Fail conditions (dien xau khong xay ra):**
- outlines remain generic -- avoided (0% generic chapters)
- primary-source coverage does not improve -- avoided (100%)
- fallback path triggered too often -- avoided (low rate)

---

## 5. Architecture Quality Gates

### Gate Layer A -- Structure quality
Before section research begins, the outline must pass:
- chapter specificity
- section specificity
- duplicate-title = 0
- generic-title rate near zero
- coverage-note preservation
- chapter/subchapter logical flow review

### Gate Layer B -- Evidence quality
Before writer is called, each section must pass:
- section topic relevance gate
- evidence adequacy gate
- canonical sufficiency check when relevant
- evidence diversity guard
- seen-count penalty to avoid one-paper domination
- zero-evidence / zero-cite prevention

### Gate Layer C -- Writing quality
Writer must operate under a strict contract:
- follow section goal
- cover must-cover terms
- avoid drift into adjacent domains
- avoid redefining already-covered concepts unless needed
- avoid filler / discourse boilerplate
- preserve technical clarity and explicit logic

### Gate Layer D -- Verification quality
After writing, each section must be checked for:
- grounding
- topic relevance
- missing must-cover terms
- drift terms / off-topic content
- citation validity
- adequacy for its intended section role

### Gate Layer E -- Book-level coherence
Before final assembly, the book should be audited for:
- cross-section concept overlap
- repeated explanations
- weak chapter roles
- chapter-to-chapter logical progression
- markdown heading hygiene
- final usefulness as a book, not just a set of sections

---

## 6. Experiment Matrix

### Experiment A -- Discovery-only evaluation
Run Discovery + outline for all 6 benchmark topics. Score TopicProfile completeness, outline specificity, chapter naming quality, prompt usefulness, JSON/fallback failure rate.

**Benchmark topics:** `Attention Mechanisms` | `Diffusion Models` | `Agentic AI Systems` | `Retrieval-Augmented Generation` | `RLHF and DPO` | `Long Context Language Models`

**Status:** 6/6 COMPLETE. All 6 runs verified: spec=1.00, dups=0, R0=0, primary>=98.7%, TP completeness=90.91%. Fallback rate=100% (all runs used semantic fallback -- target <=20%).

### Experiment B -- 2-chapter smoke comparison (v2 vs v3)
Run 2 chapters per topic on both v2 and v3. Measure: topic drift, section relevance, grounding, primary-source coverage, generic section rate.

**Status:** pending -- waiting on P1-P4.

### Experiment C -- Full-run comparison
Run full book for 1-2 topics on both v2 and v3. Measure: paper-eval outputs, human outline review, section usefulness, truncation rate, factual utility.

**Status:** pending -- waiting on P1-P4.

### Experiment D -- Ablation
| # | Variant | Comparison |
|---|---------|-----------|
| D1 | Retrieval-first vs outline-first | v2 vs v3 |
| D2 | Discovery model | gemma4:e4b vs alternatives |
| D3 | Provider mix | arxiv only vs +wiki vs +tavily+ddg |
| D4 | Query routing | router on vs off |
| D5 | Canonical seed | seed injection off vs on |

**Status:** pending -- waiting on P1-P4.

---

## 7. Prerequisite Fixes (P1-P4 + P0 fidelity gates)

> These fixes must be completed before Experiment B/C/D can produce trustworthy results. A benchmark run while these are broken will be misleading.

### P0 -- Fidelity gates required for trustworthy benchmarking

**Modules:** `notes.py` `deep_investigate.py` `discovery.py` `deep_research_v3.py`

These are not topic templates. They are quality contracts that preserve the emergent-from-prompt doctrine while preventing wrong-domain writing and evidence collapse.

#### P0a -- Section Topic Relevance Gate
- Run `check_evidence_domain()` before writer
- If evidence pool topic relevance < 0.60, retry QGN with refined hint
- Goal: no RLHF section written from RAG evidence, no diffusion section written from agentic/RAG evidence

#### P0b -- Canonical Arxiv Injection
- Accept optional `--canonical-arxiv-ids`
- Force-fetch canonical papers via `arxiv_by_id()`
- Preserve with `protected_source_ids` so they survive cosine gate / quota logic
- Goal: no more 0% canonical recall caused only by recency-biased retrieval

#### P0c -- Seen-Count Penalty
- Penalize sources that appear too often across the same run
- Example schedule: `max(0.1, 1 - seen_count/50 * 0.8)`
- Protected canonical papers are exempt
- Goal: reduce single-paper domination (e.g. FAIR-RAG in 50-75% of sections)

### Why these matter
- Round 2 may repeat Round 1 with nearly identical queries/sources
- Best-round selection may ship the wrong round due to broken scoring
- Planner failures collapse to a generic outline path
- Partial runs fail breadth-sensitive eval checks that only apply to full books
- Even a structurally good book may still fail topic purity or canonical fidelity

### P1 -- Prevent wasted research loops
**Modules:** `deep_investigate.py` `query_gen.py` `query_router.py`

- Round 2 must differ from Round 1 in query intent or source set
- Track query signatures and source IDs across rounds
- If source overlap > threshold, stop early or force diversification
- Validate: fewer Round 2 entries without added value; lower repeated-source reuse

### P2 -- Fix best-round selection
**Module:** `deep_investigate.py`

- Replace brittle additive comparison with consistent tuple ranking
- Rank rounds by (grounding, topic_relevance, citation_presence)
- Ensure later rounds win only when actually better
- Validate: shipped section from strongest verified round; no silent downgrade

### P3 -- Outline repair + canonical URL hygiene (shipped v3.2)
**Modules:** `outline_from_research.py` `discovery.py`

The v3.2 "P3" bundles two fixes that address the R0/R7 risk patterns:

**P3a -- Non-destructive outline repair** (`outline_from_research.py`)
- Only normalize truly generic labels: `^(Part|Chapter|Section)\s+\d+\s*$`
- Never overwrite a semantically-specific chapter title
- Duplicate-title check after every normalization step
- Global cross-chapter section title dedup with `(Part N)` suffixes
- `_postprocess_outline` applied to ALL return paths (model output + semantic fallback)

**P3b -- Canonical URL hygiene** (`discovery.py`)
- Reject DDG redirect URLs in `canonical_papers`
- Prefer direct arxiv/wikipedia links
- Part of P3 hygiene, not optional cleanup

**P3c -- Coverage note preservation** (`outline_from_research.py`)
- Extract `coverage_note` from raw output when post-processed field is empty

**Scope note:** the original plan's P3 ("fix planner fallback path") is a separate concern. The planner module is NOT involved in the v3.2 P3 fixes.

### P4 -- Partial-run eval fairness
**Modules:** `eval/metrics.py` `eval/run_eval.py`

- Support explicit or auto-detected partial-run mode
- Relax breadth-sensitive: `should_cite_recall`, `subtopic_coverage`
- Keep intrinsic unchanged: grounding, loops, forbidden_domains, zero_cite, word_count
- Report must state when partial-run logic is active
- Validate: smoke runs no longer fail misleadingly; full-run strictness unchanged

### Implementation Tracker

| Fix | Evidence | Status |
|-----|----------|--------|
| P3a: outline non-destructive repair | `attention_v3_p3` (fresh): chapter spec 1.00, generic rate 0%, 0 R0 triggers | SHIPPED + VERIFIED |
| P3a: cross-chapter section dedup | `attention_v3_p3` (fresh): 0 duplicate sections | SHIPPED + VERIFIED |
| P3b: canonical URL hygiene | `attention_v3_p3` (fresh): 100% primary sources (54/54) | SHIPPED + VERIFIED |
| P3c: coverage_note preservation | `attention_v3_p3` (fresh): 0 R0 triggers (all_coverage_notes_empty not fired) | SHIPPED + VERIFIED |
| P3: planner client/fallback | NOT part of v3.2 P3; separate concern | pending |
| P1: retry-diversity + source-overlap guard | Overlap guard present but no source-overlap gating yet | partial |
| P2: best-round scoring stability | Tuple ranking code present; stability unverified | partial |
| P4: partial-run eval mode | No implementation yet | pending |

> Status legend: **SHIPPED** = code deployed, evidence measured | **partial** = code present, evidence weak | **pending** = not yet implemented

### Expected outcome
Round 2 only when it adds value. Repeated query/source processing decreases. Outline evaluation becomes trustworthy. Partial-run reports become interpretable without diluting full-run standards.

---

## 8. Risks and Failure Modes

### R0 -- Outline post-processing corruption **[CONFIRMED in practice]**
Raw model output may already be specific and sound, but post-processing makes it generic or duplicated.

**Concrete trigger conditions (CONFIRMED pipeline corruption):**
- final chapter titles identical to section titles
- duplicate section titles in final but not in raw
- raw specificity high, final drops sharply after normalization
- coverage notes empty for all chapters after post-processing

**Typical causes:** weak `startswith("Chapter")` heuristics; replacing chapter title with first section title without validation; missing guards on already-meaningful titles.

**Rule:** When `_raw` output is good but final artifact is poor, classify as **pipeline-corruption first**, not model-quality.

### R1 -- Generic outline despite evidence -- **CONFIRMED**
Generic chapter names emitted even with retrieval-first design.

### R2 -- Short evidence context
Discovery evidence too shallow for good TopicProfile quality.

### R3 -- JSON / schema instability
Outline generation fails parsing and falls back too often.

### R4 -- Grounding metric illusion -- **CONFIRMED**
High grounding coexists with poor topic specificity and factual usefulness.

### R5 -- Primary-source undercoverage
Better discovery does not automatically solve weak arxiv recall.

### R6 -- Writer bottleneck remains
Even if Discovery improves, writer can still truncate or produce mediocre prose.

### R7 -- Canonical paper contamination -- **CONFIRMED**
Canonical slots populated by DDG redirect URLs instead of real papers.

**Rule:** treat canonical URL validation as part of P3 hygiene, not optional cleanup.

### R8 -- Scenario leakage / hidden template dependence -- **MUST AVOID**
Pipeline appears emergent but actually relies on topic-specific hidden assumptions, pre-scripted outlines, or benchmark-special casing.

**Rule:** any topic-specific answer logic is a product failure, even if the resulting book looks good.

### R9 -- Section-level redundancy
Two sections explain the same concept at the same depth with only wording changes.

**Rule:** cross-section concept overlap must be measured and suppressed.

### R10 -- Book-level coherence gap
Individual sections pass locally, but the assembled book lacks progression, chapter role clarity, or technical learning flow.

**Rule:** final book quality must be judged above section-level metrics.

---

## 9. Current Product Reality (2026-06-07)

> This section summarizes confirmed findings from the v3.2 smoke test. For operational status, see `short-memory.md`.

### Confirmed findings

| Metric | Before P3 | After P3 (6/6 topics avg) |
|--------|-----------|----------------------------------|
| Chapter specificity | 0.15 | **1.00** |
| Generic chapter rate | 100% | **0%** |
| Prompt quality | 0.70 | **1.00** |
| Primary-source coverage | 83% | **100%** |
| Duplicate section titles | 32 | **0** |
| R0 pipeline triggers | 1 | **0** |
| TopicProfile completeness | 64% | **91%** |

- `discovery_eval.md` exists for all 6 benchmark topics (`attention_v3_p3`, `diffusion_v3`, `agentic_v3`, `rag_v3`, `rlhf_v3`, `longctx_v3`)

### Confirmed gaps

- Experiment A complete: 6/6 verified. All runs: spec=1.00, dups=0, R0=0, primary>=98.7%, TP=90.91%
- **New finding:** fallback_rate=100% on all 6 runs (target<=20%) -- semantic fallback always triggered; target unmet
- P1 retry diversity: pending
- P2 best-round scoring: pending
- P4 partial-run eval: pending
- topic drift: not measured yet

---

## 8. Evaluation Procedure

### Evaluation Tracker

| Exp | Topics | Evidence | Status | Blocker |
|-----|--------|----------|--------|---------|
| A: Discovery-only | 6/6 | spec=1.00, dups=0, R0=0, primary>=98.7%, TP=90.91% | COMPLETE | none |
| B: Smoke comparison | TBD | none | pending | P1 |
| C: Full-run | TBD | none | pending | P1, P2, P4 |
| D: Ablation | TBD | none | pending | P1, P2, P4 |

### Phase 1 -- Discovery artifact inspection
For each benchmark topic: run Discovery, save `TopicProfile.json`, save `outline_profile.json`, run `discovery_eval.py` to score specificity and completeness. All 6 topics must have `discovery_eval.md` before Phase 2.

### Phase 2 -- 2-chapter smoke comparison
Run v2 and v3 for each topic (2 chapters each). Compare outline quality, section relevance, grounding, primary-source coverage, generic rate.

### Phase 3 -- Full-run validation
Choose 1-2 representative topics. Run full v2 baseline + full v3 candidate. Run paper eval. Compare metrics and human judgment.

### Phase 4 -- Ablation
Vary: discovery model, provider mix, router on/off, seed wiring on/off. Identify whether gains come from architecture, provider breadth, router quality, or model choice.

### Deliverables (per evaluation run)
`TopicProfile.json` | outline artifact | specificity scorecard | topic relevance summary | grounding summary | primary-source coverage summary | fallback/failure notes | recommendation (keep/revise/reject Discovery)

---

## 9. Short-term Recommendation

Treat v3 Discovery as a **promising experimental redesign**, not yet a proven replacement for v2.

**Validate first (in order):**
1. outline specificity (fix P3 first -- R0 is already confirmed)
2. prompt completeness
3. topic drift reduction
4. primary-source coverage improvement (already strong)
5. robustness of fallback behavior (fix P3)

**Do not overclaim:**
- no multi-agent behavior
- no citation-graph retrieval
- no production-readiness at full-book scale
- do not treat high grounding alone as success (R4 confirmed)

---

## 10. Executive Summary

Discovery redesign is valuable **only if** it changes the pipeline from:

> `retrieve to support a pre-existing outline`

to:

> `retrieve first, then let evidence define scope, outline, and downstream section research`

**Current verdict (2026-06-07):**

Experiment A COMPLETE (6/6 topics verified). All runs: spec=1.00, dups=0, R0=0, primary>=98.7%, TP completeness=90.91%. Discovery + Outline pipeline is structurally sound across diverse AI/ML topics.

**Key new finding:** Fallback rate = 100% on all 6 runs (target <= 20%). Semantic fallback always triggers. Root cause needs investigation -- likely model JSON parsing instability or outline schema mismatch. This is a P3-adjacent concern.

**Scope of evidence:** 6 benchmark topics across AI/ML domain. P1, P2, P4 pending.

**Remaining:** Experiment A on 5 more topics, then P1 before B/C/D.

---

**What changed in v3.2:**
- P3a non-destructive repair: specific chapter titles (0% generic), informative prompts
- P3b canonical URL hygiene: 100% primary sources
- P3c cross-chapter dedup + coverage_note: SHIPPED + VERIFIED (fresh eval: 0 dups, 0 R0 triggers)
- Plan + eval: metrics aligned to 0-1 scale; P3 scope clarified; evidence-based status
