# Short Memory — Product Base (snapshot)

> Snapshot ngắn (≤50 dòng) của **TRẠNG THÁI BASE hiện tại**. Đọc: 1.`GLOSSARY.md` 2.file này 3.`long-memory.md`. Ngưỡng đầy đủ → `RULES.md`. Kiến trúc → `CLAUDE.md`. Lịch sử run/version → `long-memory.md` (KHÔNG để ở đây).

## Base (2026-06-21)
- **Orchestrator DUY NHẤT:** `files/deep_research_v3.py` + stage logic `files/research/*.py`. Launcher `./run_full.sh`. Resume qua `files/output/runs/<name>/state.json`.
- **Legacy (KHÔNG phải base — đừng sửa như live):** `files/deep_research.py` (v2, còn bị monitor/run_eval/runner import) + `files/archive/*` (v1).
- **100% model LOCAL:** gemma4:e4b (discovery/outline/QGN/judge) · qwen3.6-35b:iq3 (writer) · **bge-m3:latest (embed UNIFIED mọi path #3)** · bge-reranker-v2-m3 · HHEM. **TUYỆT ĐỐI không gọi Claude/external lúc runtime.**

## Verify layer — ✅ P0 + P0-2b ĐÃ APPLY (2026-06-23)
- **Gate cứng SỐNG = P0a domain-evidence (~0.40, PRE-writer)** + StageD word-count 120 + empty-pool + **G2 cite_prec≥0.45 (giờ gate sống)**.
- **P0:** grounding **bỏ khỏi gate** (G3 log-only/advisory); `gate_ok = n_cites>0 AND topic≥0.50 AND cross-ref` (`deep_investigate.py:753`); **G2 `verify_section` CHẠY** (`:740-742`) → cite_precision **đo thật**; StageE chặn best-topic<0.50 sau-loop (`:804-813`); best-round topic-first (`:712-716`); **P0c aliasing fixed** (`:304`, run_seen_counts 0→23).
- **P0-2b (NEW):** cite-judge prompt **soften** (`verify.py:47-75`): `supports` = states/implies/**paraphrases faithfully** (bỏ "direct match only"); contradicts/unrelated giữ strict. → trên prose THẬT faithful section đo **cite_prec 0.481 > 0.45 → ACCEPT (`quality="ok"` > 0)**; weak floor (R1 0.321→R3 0.487 mới qua, hoặc <0.45 → degraded). **Discrimination test** `files/eval/bench_cite_discrimination.py`: GOOD 0.72 (PASS) vs BAD_unrelated 0.18 / BAD_contradict 0.20 (gap +0.5) → judge phân biệt thật, KHÔNG rubber-stamp. `min_cite_precision=0.45` + `no_evidence=0.3` GIỮ NGUYÊN. **Faithfulness gate giờ "xanh".**
- **Verifier ≠ Writer** (bất biến): grounding=HHEM, topic/cite=gemma, writer=Qwen.

## Ngưỡng vận hành (chuẩn → RULES.md)
- **P0a ≈0.40 (gate cứng SỐNG, pre-writer).** Min word 120 (HARD). Prefilter **0.48/0.65** (bge-m3). max_rounds CLI 3 / run_v3 nội bộ 2.
- clean-accept (P0+P0-2b) = topic≥0.50 (G4, ENFORCED) ∧ n_cites>0 ∧ cross-ref ∧ **cite_prec≥0.45 (G2, đo thật, judge soften)**; grounding log-only. ✅ faithful prose ACCEPT (cite_prec ~0.48), weak floor. P0c cross-section giờ fire (aliasing fixed).

## Base này đã gồm (đã merge main + push)
- #1 outline anti-matrix (chunked) · G6 bge-m3 dedup warn · **#3 embed unify bge-m3** · #5 anchoring SAFE (không mất nguồn) · #4 citation-aware grounding warn-first · **G2 fail-CLOSED→0.0**.
- **HHEM re-tie** (hết degenerate 0.502; nay advisory) · **agentic evidence-pool rescue** (post-prefilter on-topic<5 → mượn sibling, P0c-exempt; block −21%, faithfulness giữ) · **mathfix single-source** + render tectonic robust · **4-topic benchmark** (accept 0.724±0.058; cite_prec/canonical/near-dup std=0) + **HF dataset** `vudang449/agentic-deep-research-eval`.
- Unit test verify: `python3 files/eval/test_verify_optim.py`. Docs + HF card khớp code; **eval 2026-06-22 phát hiện verify post-writer INERT** (đã clean docs về đúng sự thật).

## Open → ROADMAP UPGRADE (`plan.md` §Upgrade)
- **P0 + P0-2b ✅ DONE (2026-06-23):** decouple G2 (chạy thật) · grounding log-only · P0c aliasing fixed (0→23) · **P0-2b: cite-judge soften → faithful prose ACCEPT (cite_prec 0.48, quality=ok), discrimination GOOD 0.72 vs BAD 0.18/0.20.** Faithfulness gate "xanh".
- **→ P1 (NEXT, cấu trúc):** matrix thành HARD gate (suffix/skeleton detector) · paragraph-dedup lúc assemble · math-validation gate (balance + chống LaTeX leak) · near-miss rescue (0.35–0.40).
- **P1 (eval):** held-out judge từ model khác họ / gold set (chống self-eval vòng tròn).
- **P2 (agentic sâu hơn):** citation-graph 2nd-hop cho topic ngách · primary-source routing cho citation định nghĩa/phương trình.

## Lệnh nhanh
```bash
./run_full.sh                              # hoặc: python3 files/deep_research_v3.py --topic "<T>" --out-name <n> --no-smoke
python3 files/eval/test_verify_optim.py    # unit test verify
python3 files/monitor.py
```
