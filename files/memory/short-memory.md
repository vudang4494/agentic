# Short Memory — Trạng thái hiện tại

> Snapshot ngắn (≤50 dòng). Thứ tự đọc: 1.`GLOSSARY.md` 2.file này 3.`long-memory.md`. Ngưỡng đầy đủ → `RULES.md`. Chi tiết kiến trúc → `CLAUDE.md`.

## Hiện trạng (2026-06-16)
- **Version:** v3 (run label `llm_book_v36`). KHÔNG có version string trong code.
- **Orchestrator LIVE:** `files/deep_research_v3.py`. Legacy v2 = `files/deep_research.py` (đừng sửa như live).
- **Stage logic:** `files/research/*.py`. Resume qua `files/output/runs/<name>/state.json`.

## Run mới nhất — `llm_book_v36` (LỚN NHẤT, đã đánh giá)
- 40 chương / **280 section** / 195,951 từ (state.json) · book assembled 224,446 từ / **712 trang PDF**.
- Grounding = **1.0 toàn bộ 280** ở run v36 (HHEM bão hòa lúc đó) → **G3 đã de-saturate** (xem Blocker 4); cần validation run để re-baseline.
- topic_relevance mean **0.782**; phân bố {0.5: 23 section, 0.75: 198, 1.0: 59} → 23 section ở sàn = yếu nhất.
- 0 blocked · 0 zero-citation · n_cites mean 13.6 (min 6) · **12/12 canonical** injected+protected.

## Blocker đang mở
1. **Matrix pattern (outline):** `outline_audit` v36 `ok=false` (matrix + coherence_low); section gần-trùng jaccard ~0.80; RAG/CoT bị tách 2 anchor. Outline audit chỉ advisory → redundancy lọt. → Sửa Stage A (Guardrail 3).
2. **~45% reference off-topic:** prefilter 0.45 + domain gate ~0.40 quá lỏng; per-section sourcing kém dù canonical recall cao. → Siết gate (Guardrail 6).
3. **Embed SPLIT:** retrieval+query_router chạy `nomic-embed-text` (runtime), verify-side chạy `bge-m3:latest`. config nomic khớp retrieval; unify về 1 model trước khi tuning.
4. **Verify layer (tối ưu + VALIDATED, LOCAL-only):** validation `validate_v37` (06-16): pipeline OK không vỡ; **G2 cite_prec 0.75-1.0 + G4 topic 0.77-0.90 PHÂN BIỆT được** (hết mù); grounding vẫn 1.0 trên section sạch (#4 citation-aware nếu cần, nhưng G2 đã lấp); #1 outline hết matrix; G6 sạch. Test: `python3 files/eval/test_verify_optim.py`. Pipeline TUYỆT ĐỐI không gọi Claude/external.

## Ngưỡng vận hành (nhắc nhanh — chuẩn ở RULES.md)
- P0a domain gate ≈ 0.40 · accept: grounding ≥0.70 ∧ topic ≥0.50 ∧ cites>0 ∧ cross-ref đủ.
- StageE block: g-pass + topic<0.50. P0c floor 0.05, canonical exempt. Prefilter 0.45/0.65. Min 120 từ.

## Tiếp theo
1. Sửa outline generator (khử matrix/template) → re-run audit phải `ok=true`.
2. Siết per-section reference relevance; nâng sàn topic_relevance.
3. Commit các file core đang untracked (deep_research_v3.py + research layer + GLOSSARY + memory + run_full.sh).

## Lệnh nhanh
```bash
python3 files/deep_research_v3.py --topic "<T>" --out-name <name> --canonical-arxiv-ids "<ids>" --no-smoke
python3 files/eval/smoke_test_p0.py --topic "Transformer" --canonical-ids "1706.03762,1607.06450"
python3 files/monitor.py
```
