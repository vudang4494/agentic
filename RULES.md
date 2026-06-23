# RULES.md — Product Guardrails + Bảng Ngưỡng + Agent Efficiency

> **Vai trò:** Đây là **nguồn ngưỡng chuẩn duy nhất**. Mọi doc khác defer về đây. Phân biệt rõ **OPERATIVE** (code enforce thật) vs **TARGET** (aspirational, chưa enforce). Mục tiêu academic chi tiết → `files/eval/PRODUCT_QUALITY_CRITERIA.md` (không lặp ở đây).

## Mục tiêu tối thượng
Product này **KHÔNG** phải hệ thống sinh chữ dài. Đây là hệ thống tạo **technical book / research artifact đúng topic, đúng evidence, có logic học thuật, auditable**.
**Run dài nhưng sai topic / drift / lặp = FAIL.**

## 7 câu hỏi bắt buộc trước khi chấp nhận run
1. Book có đúng topic không?
2. Canonical papers của topic có thực sự xuất hiện (evidence + prose) không?
3. Có adjacent-domain contamination không?
4. Có paper nào dominate bất thường (>50% sections) không?
5. Các chapter có thực sự nói điều khác nhau không?
6. Sách đọc như một cuốn coherent book chưa?
7. Nếu bỏ chỉ số `g`, còn dám gọi đây là output tốt không?

**Bất kỳ câu nào = "không chắc" → FAIL.**

## Thứ tự ưu tiên quyết định
1. Đúng topic → 2. Đúng evidence/canonical → 3. Không lặp/drift → 4. Logic học thuật toàn sách → 5. Grounding/citation correctness → 6. Độ dài/số section → 7. Văn phong.

## Product-level FAIL (bất kỳ điều nào)
1. `must_cite_recall = 0%` cho topic có canonical rõ ràng.
2. Adjacent domain chi phối narrative nhiều sections.
3. Một paper dominate >50% sections.
4. Lặp semantic nghiêm trọng giữa chapter/section.
5. Heading inflation / assembly corruption.
6. Section viết ra dù evidence topic gate fail.

---

## ⭐ BẢNG NGƯỠNG: OPERATIVE (code) vs TARGET (aspirational)

> **Quy tắc vàng:** khi viết/sửa code hay đánh giá run, dùng cột OPERATIVE. Cột TARGET là đích cần đạt, KHÔNG phản ánh hành vi hiện tại.

> ✅ **P0 + P0-2b ĐÃ APPLY (2026-06-22 → 2026-06-23) — trạng thái GATE hiện tại:**
> - Grounding **bỏ khỏi gate** → log-only/advisory. `base_ok` cũ thay bằng `gate_ok = n_cites>0 AND topic≥0.50 AND cross-ref` (`deep_investigate.py:753`).
> - G2 `verify_section` **CHẠY** khi `n_cites>0 AND topic_ok` (bất kể grounding) → `cite_precision` **đo thật** (`:740-742`).
> - StageE chuyển **sau-loop** (`:804-813`), chặn topic-drift theo **best-topic**; best-round chọn **topic-first** (`:712-716`).
> - P0c aliasing **fixed** (`:304`) → `run_seen_counts` populate (0→23).
> - **P0-2b (NEW):** cite-judge prompt **soften** (`verify.py`): `supports` = "evidence states/implies/**paraphrases faithfully**" (bỏ "direct match only"); `contradicts`/`unrelated` giữ strict. → trên prose THẬT, section faithful đo **cite_prec 0.481 > 0.45 → ACCEPT (`quality="ok"` > 0)**; section yếu vẫn floor (0.411 < 0.45 → `degraded`). **Discrimination test** `bench_cite_discrimination.py`: GOOD=0.72 (PASS) vs BAD_unrelated=0.18 / BAD_contradict=0.20 (gap +0.5) — judge phân biệt thật, KHÔNG rubber-stamp 1.0. `min_cite_precision=0.45` + `no_evidence=0.3` GIỮ NGUYÊN (GOOD vượt gate có headroom).
> → **Gate cứng: P0a (~0.40, PRE-writer) + StageD word-count + G2 cite_prec≥0.45 (giờ là gate SỐNG, cho qua section faithful).** Chi tiết: `plan.md` §Upgrade.

| Check | OPERATIVE (enforce thật) | File:dòng | TARGET (doc/đích) |
|-------|--------------------------|-----------|-------------------|
| **P0a domain gate** (gate cứng SỐNG duy nhất) | `ev_threshold = min(0.40, max(0.30, min_topic_rel−0.10)) ≈ 0.40` → HARD BLOCK round cuối, PRE-writer | `deep_investigate.py:524` | ≥ 0.80 topic purity |
| Topic relevance (G4) | **ENFORCED** (P0): điều kiện của `gate_ok` (clean-accept) + StageE chặn best-topic<0.50 | `deep_investigate.py:752,808` | enforce thật ✅ |
| Grounding (HHEM/G3) | **LOG-only / advisory** (P0: đã bỏ khỏi gate) — strict-NLI ~0.05–0.10 trên prose, KHÔNG phải metric | `deep_investigate.py:692,860` | — |
| StageE hard block | best-topic < 0.50 (sau-loop, không cần grounding) → BLOCK | `deep_investigate.py:808` | (giữ) |
| Citation | n_cites > 0 (điều kiện `gate_ok`) | `deep_investigate.py:752` | diversity ≥ 10 unique |
| Citation precision (G2) | **GATE SỐNG** (P0+P0-2b): ≥ **0.45** để clean-accept; đo thật per-[N]. Judge soften (paraphrase=supports) → faithful prose 0.481 ACCEPT, weak 0.411 degraded. Discrimination GOOD 0.72 vs BAD 0.18/0.20 | `deep_investigate.py:740-755`, `verify.py:47-75` | discriminate, không floor ✅ |
| Cross-refs | 2 nếu ≥2 prior; 1 nếu 1 prior; 0 nếu đầu | `deep_investigate.py:630` | ≥ 2/section |
| Min word count | **120** từ → retry, round cuối HARD BLOCK | `deep_investigate.py:781` | — |
| P0c seen-penalty | `max(0.05, (1 − seen/max_seen)²)`; canonical + pool-rescued EXEMPT | `notes.py:324` | < 50%/paper |
| Prefilter cosine | drop < **0.48** (Rank7: was 0.45 — trim weakest off-topic tail ~9%); grey-domain < **0.65** | `notes.py:109,110` | — |
| Semantic overlap (outline) | flag jaccard ≥ **0.50** (advisory, không block) | `outline_from_research.py:342` | jaccard < 0.30 BLOCK |
| Heading hygiene (assemble) | dup/orphan → **WARN only** (không block) | `deep_research_v3.py:220` | FAIL nếu dup |
| max_rounds | CLI default **3**, `run_v3()` nội bộ **2** | `deep_investigate.py:226` | — |
| Embed model | **UNIFIED** `bge-m3:latest` mọi path (retrieval notes.rank/prefilter RRF, query_router, verify-side) | `config.py:34`; `notes.py:111`; `query_router.py:210`; `verify.py:35` | — (đã unify #3) |

⚠️ `config.py MIN_GROUNDING=0.55` KHÔNG phải giá trị vận hành (runtime `min_grounding=0.70`). Embed đã **unify về `bge-m3:latest`** trên mọi path (`config.py:34`, `notes.py:111`, `query_router.py:210`, `verify.py:35`) — không còn split nomic; 0 ref nomic sống (chỉ comment lịch sử "was nomic").

---

## Gate A–F (map về code thật)

| Gate | Tên | Enforcement thật | File |
|------|-----|------------------|------|
| **A** Structure | Outline audit: no-generic-title, no-dup-section, no-matrix-pattern, semantic-overlap, canonical coverage | matrix giờ **chặn TẠI NGUỒN**: `draft_outline_chunked` (chapter skeleton + per-chapter sections, mỗi call JSON nhỏ hợp lệ, section emerge từ evidence chương) thay single-shot 288-section (vỡ JSON → archetype fallback). Audit vẫn advisory nhưng generator KHÔNG còn sinh matrix; chunked-outline GIỮ dù audit soft-fail | `outline_from_research.py` |
| **B** Evidence | P0a domain gate / P0b canonical inject / P0c seen-penalty / prefilter | **HARD BLOCK** (P0a round cuối) — IMPLEMENTED · **evidence-pool rescue**: post-prefilter on-topic < 5 → `investigate_section` mượn sibling sources on-topic (qua cùng prefilter), **P0c-EXEMPT** (`notes.rank_rrf p0c_exempt_ids`) — `deep_investigate.py:242,416,433`, `notes.py:216,322` | `deep_investigate.py`, `notes.py`, `discovery.py` |
| **C** Writing | min 120 từ, citation cleanup | **HARD BLOCK** word-count; dedup chưa wire vào writer → PARTIAL | `deep_investigate.py` |
| **D** Verify | clean-accept = topic≥0.50 (G4) + n_cites>0 + cross-ref + cite_precision≥0.45 (G2); grounding log-only | **P0+P0-2b applied:** G2 CHẠY + judge soften → trên prose thật faithful section ACCEPT (`quality="ok"`, cite_prec 0.481), weak section degraded (0.411) → gate SỐNG, discriminate (không rubber-stamp, không floor). grounding log-only; StageE chặn best-topic<0.50 sau-loop. | `deep_investigate.py:740-813`, `verify.py:47-75` |
| **E** Coherence/Assembly | dup/orphan heading | **WARN only** → PARTIAL | `deep_research_v3.py` |

> GATE-0..6 / DR3-Eval / DREAM trong `PRODUCT_QUALITY_CRITERIA.md` + `product_quality_verifiers.py`: **chưa được import vào runtime** (`deep_research_v3.py` không gọi) → **DOC_ONLY, không chạy lúc generate**. Đừng giả định chúng đang bảo vệ run.

---

## Verify layer — hiện trạng + target chuẩn hóa

**LIVE (v3, post-P0+P0-2b 2026-06-23):** `grounding_score` (HHEM, **G3 log-only/advisory** — đã bỏ khỏi gate) + `topic_relevance_check` (G4 — **ENFORCED**: điều kiện `gate_ok` + StageE best-topic) + `verify_section` (G2 cite_precision — **GATE SỐNG**, đo thật, judge soften P0-2b → faithful prose ≥0.45 ACCEPT) + `verify_cross_references_v2` (regex đếm). **Gate cứng SỐNG = P0a (pre-writer ~0.40) + StageD word-count + G2 cite_prec≥0.45.** **LEGACY-only** (`deep_research.py`/scripts/eval — đừng sửa như live): `verify_section_v2`, `crag_decision`, `strip_refine`, `scrub_unsupported_citations`.

**Lưu ý:** grounding "bão hòa 1.0" của v36 thực chất là **HHEM degenerate** (mất weight-tying dưới transformers 5.x → `embed_tokens`=0 → hằng số ~0.502 cho mọi cặp). **ĐÃ FIX** (re-tie `embed_tokens←shared` trong `_get_hhem`): benchmark `bench_hhem_discrimination.py` giờ 100% phân biệt (ENTAILED 0.79 vs CONTRA 0.04 vs UNREL 0.005), support 0/4 claim sai. `topic_relevance` quantized {0.5,0.75,1.0} là chữ ký **pre-G4 (v36)** — nay blend judge thật (17 distinct trên run agentic). `product_quality_verifiers.py` = eval-only.

**Gate order (✅ = đã implement LOCAL, ⏳ = roadmap; PHẢI validation run đầy đủ để tinh chỉnh ngưỡng):**
- G1 **P0a** domain ≈0.40 HARD ✅ (pre-writer, gate cứng SỐNG) · G3 **grounding = log-only/advisory** (P0: bỏ khỏi gate) · G4 **topic** ✅ ENFORCED (gate_ok + StageE best-topic) · G2 **citation integrity** ✅ GATE SỐNG (P0+P0-2b; cite_precision đo thật, judge soften → faithful prose ≥0.45 clean-accept, weak floor) · G5a cross-ref `:630` · G5b numeric-ref hint (soft). **→ P0+P0-2b done: G2 discriminate thật (GOOD 0.72 vs BAD 0.18/0.20), faithful section `quality="ok"`. Roadmap kế = P1 (`plan.md`).**
- ⏳ Roadmap: G5b resolve "Section N.M" vs outline thật (HARD); G6 redundancy cosine vs prior (soft); unify embed model; tinh chỉnh ngưỡng sau validation run.
- **Invariant:** mọi HARD message in ngưỡng thật (biến, không literal); 1 grounding + 1 topic definition; 1 embed model; **mọi judge = model LOCAL (gemma/HHEM/bge), KHÔNG Claude/external lúc runtime.** Unit test: `python3 files/eval/test_verify_optim.py`.

---

## Hardcoded Agent Rules (đồng bộ Guardrails ở CLAUDE.md)
1. Không tối ưu completion rate / word count trước quality.
2. Grounding pass ≠ success (P0: grounding bỏ khỏi gate, log-only). StageE giờ **fire theo best-topic<0.50 sau-loop** (`:804-813`), độc lập grounding — chặn topic-drift thật.
3. Outline sai logic → sửa outline trước, không đắp prompt writer.
4. Drift phát hiện → chặn ở gate, không để writer "viết xong rồi tính".
5. Canonical = protected + P0c-exempt — không phá exemption.
6. Root-cause fix > patch bề mặt.
7. Sau mỗi stage: trim context + archive intermediates + update short-memory.

## Agent Efficiency — Token & Memory
- **State files thay vì inline context:** pipeline state → `state.json`; outline/section body ghi file, truyền path. Không re-send full conversation giữa stage.
- **Trim sau mỗi stage:** giữ TopicProfile/outline/verdict, bỏ scratch + evidence không dùng.
- **Compact prompts:** query gen template-based; verify fixed rubric; writer chỉ nhận section goal + evidence path + summary prior sections.
- **short-memory.md ≤ 50 dòng** (xoá resolved sau 48h; dùng `P0a/b/c`, `WRT[g=…]`). **long-memory.md < 200 dòng**, mỗi entry 1–3 dòng, archive milestone.
- **Cache:** LLM output → file (không giữ RAM); embeddings → vector store; search results → disk, chỉ giữ top-K.

## Compact Notation
```
DSC Discovery · OUT Outline · RSR Research[q,src] · QGN QueryGen · RRK Rerank · RRF RankFusion
EVG[rel=0.41] P0a gate · P0b(canon=N) · P0c(pen=0.20x) · WRT[g=0.85,wc=760] · VFY[topic=0.75]
FAIL hard block · PASS all gates green
```

## Claim Policy
Không claim "book-quality / research-grade / benchmark-ready" khi chưa có nhiều run liên tiếp chứng minh: canonical recall, topic purity, evidence-domination control, book coherence.
