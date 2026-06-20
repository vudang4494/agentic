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

| Check | OPERATIVE (enforce thật) | File:dòng | TARGET (doc/đích) |
|-------|--------------------------|-----------|-------------------|
| P0a domain gate | `ev_threshold = min(0.40, max(0.30, min_topic_rel−0.10)) ≈ 0.40` → HARD BLOCK round cuối | `deep_investigate.py:524` | ≥ 0.80 topic purity |
| Topic relevance accept | ≥ **0.50** (G4 blend score) | `deep_investigate.py:228,745` | ≥ 0.80 |
| Grounding (HHEM/G3) | **ADVISORY** — soft conjunct của `base_ok` (default `min_grounding=0.70`) NHƯNG không block một mình: round cuối thiếu grounding → ship `quality='degraded'`, không hard-block. Strict-NLI ~0.05–0.10 trên prose faithful → KHÔNG phải metric chất lượng; faithfulness thật = G2 cite_precision | `deep_investigate.py:227,729,850` | — |
| StageE hard block | g≥0.70 ∧ topic<0.50 ∧ cites>0 → BLOCK (block do **topic**, không do grounding) | `deep_investigate.py:751` | (giữ — đúng tinh thần) |
| Citation | n_cites > 0 (bắt buộc) | `deep_investigate.py:729` | diversity ≥ 10 unique |
| Citation precision (G2) | ≥ **0.45** (clean-accept conjunct; **fail-CLOSED → 0.0** khi verify lỗi; `verify_section` per-[N], gemma local) | `deep_investigate.py:230,738,745` | supported/total |
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
| **D** Verify | topic 0.50 + n_cites>0 + cross-ref + **cite_precision ≥0.45** (G2) [+ grounding, advisory] | **HARD BLOCK** (accept + StageE) trên **topic(G4) + cite_precision(G2) + cross-ref**. ⚠️ **grounding (HHEM/G3) KHÔNG phải metric chất lượng**: HHEM phân biệt được cặp NLI sạch (sau re-tie `embed_tokens`, `bench_hhem_discrimination.py` 100%) NHƯNG là **strict-NLI** → chấm ~0.05–0.10 trên prose synthesized/paraphrase dù content faithful (verified trên bench_rlhf: 0/63 accepted ≥0.70 mà content đúng). Faithfulness THẬT = **G2 cite_precision** (gemma per-`[N]`, mềm hơn). grounding chỉ log advisory, đã cắt premise+claim vừa 512-token HHEM. | `deep_investigate.py`, `verify.py`, `faithfulness.py` |
| **E** Coherence/Assembly | dup/orphan heading | **WARN only** → PARTIAL | `deep_research_v3.py` |

> GATE-0..6 / DR3-Eval / DREAM trong `PRODUCT_QUALITY_CRITERIA.md` + `product_quality_verifiers.py`: **chưa được import vào runtime** (`deep_research_v3.py` không gọi) → **DOC_ONLY, không chạy lúc generate**. Đừng giả định chúng đang bảo vệ run.

---

## Verify layer — hiện trạng + target chuẩn hóa

**LIVE (v3, trong `investigate_section`):** `grounding_score` (HHEM, **G3 ADVISORY** — re-tied nên phân biệt cặp NLI sạch, nhưng strict-NLI không hợp prose synthesized → log-only, không gate chất lượng; G2 là faithfulness thật) + `topic_relevance_check` (G4: blend heuristic + `answer_relevance` gemma local, StageE floor) + `verify_section` (G2 cite_precision ≥0.45, **fail-CLOSED → 0.0** khi verify lỗi) + `verify_cross_references_v2` (regex). **LEGACY-only** (`deep_research.py`/scripts/eval — đừng sửa như live): `verify_section_v2`, `crag_decision`, `strip_refine`, `scrub_unsupported_citations`.

**Lưu ý:** grounding "bão hòa 1.0" của v36 thực chất là **HHEM degenerate** (mất weight-tying dưới transformers 5.x → `embed_tokens`=0 → hằng số ~0.502 cho mọi cặp). **ĐÃ FIX** (re-tie `embed_tokens←shared` trong `_get_hhem`): benchmark `bench_hhem_discrimination.py` giờ 100% phân biệt (ENTAILED 0.79 vs CONTRA 0.04 vs UNREL 0.005), support 0/4 claim sai. `topic_relevance` quantized {0.5,0.75,1.0} là chữ ký **pre-G4 (v36)** — nay blend judge thật (17 distinct trên run agentic). `product_quality_verifiers.py` = eval-only.

**Gate order (✅ = đã implement LOCAL, ⏳ = roadmap; PHẢI validation run đầy đủ để tinh chỉnh ngưỡng):**
- G1 **P0a** domain ≈0.40 HARD ✅ · G3 **grounding = ADVISORY** (HHEM re-tied nên phân biệt cặp NLI sạch 100%, NHƯNG strict-NLI chấm ~0.05–0.10 trên prose synthesized dù faithful → KHÔNG dùng làm metric/gate chất lượng; faithfulness thật = G2. premise+claim đã cắt vừa 512 token tránh truncate) · G4 **topic judge thật** ✅ (`topic_relevance_check` blend heuristic + `answer_relevance` **gemma local**; floor bảo vệ StageE) · G2 **citation integrity** ✅ (`verify_section` per-`[N]`, cite_precision ≥ 0.45, **fail-CLOSED → 0.0** khi lỗi, **gemma local**) · G5a cross-ref 1 rule động ✅ (dynamic `min_refs_needed` gate `:630` + gộp gate) · G5b numeric-ref hint ✅ (chống fabrication, soft).
- ⏳ Roadmap: G5b resolve "Section N.M" vs outline thật (HARD); G6 redundancy cosine vs prior (soft); unify embed model; tinh chỉnh ngưỡng sau validation run.
- **Invariant:** mọi HARD message in ngưỡng thật (biến, không literal); 1 grounding + 1 topic definition; 1 embed model; **mọi judge = model LOCAL (gemma/HHEM/bge), KHÔNG Claude/external lúc runtime.** Unit test: `python3 files/eval/test_verify_optim.py`.

---

## Hardcoded Agent Rules (đồng bộ Guardrails ở CLAUDE.md)
1. Không tối ưu completion rate / word count trước quality.
2. Grounding pass ≠ success (StageE đã block g-pass + topic-fail — giữ nguyên).
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
