# Short Memory — Product Base (snapshot)

> Snapshot ngắn (≤50 dòng) của **TRẠNG THÁI BASE hiện tại**. Đọc: 1.`GLOSSARY.md` 2.file này 3.`long-memory.md`. Ngưỡng đầy đủ → `RULES.md`. Kiến trúc → `CLAUDE.md`. Lịch sử run/version → `long-memory.md` (KHÔNG để ở đây).

## Base (2026-06-21)
- **Orchestrator DUY NHẤT:** `files/deep_research_v3.py` + stage logic `files/research/*.py`. Launcher `./run_full.sh`. Resume qua `files/output/runs/<name>/state.json`.
- **Legacy (KHÔNG phải base — đừng sửa như live):** `files/deep_research.py` (v2, còn bị monitor/run_eval/runner import) + `files/archive/*` (v1).
- **100% model LOCAL:** gemma4:e4b (discovery/outline/QGN/judge) · qwen3.6-35b:iq3 (writer) · **bge-m3:latest (embed UNIFIED mọi path #3)** · bge-reranker-v2-m3 · HHEM. **TUYỆT ĐỐI không gọi Claude/external lúc runtime.**

## Verify layer — ✅ P0 ĐÃ APPLY (2026-06-22)
- **Gate cứng SỐNG = P0a domain-evidence (~0.40, PRE-writer)** + StageD word-count 120 + empty-pool.
- **P0 đã sửa:** grounding **bỏ khỏi gate** (G3 log-only/advisory); `gate_ok = n_cites>0 AND topic≥0.50 AND cross-ref` (`deep_investigate.py:752`); **G2 `verify_section` GIỜ CHẠY** khi n_cites>0 AND topic_ok → cite_precision **đo thật** (RLHF 0.30-0.41, ≠ default 1.0); StageE chặn best-topic<0.50 sau-loop; best-round topic-first; **P0c aliasing fixed** (`:304`, run_seen_counts 0→23).
- ⚠️ **CÒN LẠI (P0-2b):** gemma cite-judge **strict-match** ("direct match only") → cite_precision floor ~0.3-0.4 < `min_cite_precision=0.45` → **clean-accept vẫn = 0** (mọi section `degraded`). G4 topic ENFORCED ✅. Faithfulness gate chưa "xanh" tới khi soften judge / recalibrate.
- **Verifier ≠ Writer** (bất biến): grounding=HHEM, topic/cite=gemma, writer=Qwen.

## Ngưỡng vận hành (chuẩn → RULES.md)
- **P0a ≈0.40 (gate cứng SỐNG, pre-writer).** Min word 120 (HARD). Prefilter **0.48/0.65** (bge-m3). max_rounds CLI 3 / run_v3 nội bộ 2.
- clean-accept (P0) = topic≥0.50 (G4, ENFORCED) ∧ n_cites>0 ∧ cross-ref ∧ **cite_prec≥0.45 (G2, đo thật)**; grounding log-only. ⚠️ cite_prec floor ~0.3-0.4 → 0 clean-accept (P0-2b). P0c cross-section giờ fire (aliasing fixed).

## Base này đã gồm (đã merge main + push)
- #1 outline anti-matrix (chunked) · G6 bge-m3 dedup warn · **#3 embed unify bge-m3** · #5 anchoring SAFE (không mất nguồn) · #4 citation-aware grounding warn-first · **G2 fail-CLOSED→0.0**.
- **HHEM re-tie** (hết degenerate 0.502; nay advisory) · **agentic evidence-pool rescue** (post-prefilter on-topic<5 → mượn sibling, P0c-exempt; block −21%, faithfulness giữ) · **mathfix single-source** + render tectonic robust · **4-topic benchmark** (accept 0.724±0.058; cite_prec/canonical/near-dup std=0) + **HF dataset** `vudang449/agentic-deep-research-eval`.
- Unit test verify: `python3 files/eval/test_verify_optim.py`. Docs + HF card khớp code; **eval 2026-06-22 phát hiện verify post-writer INERT** (đã clean docs về đúng sự thật).

## Open → ROADMAP UPGRADE (`plan.md` §Upgrade)
- **P0 ✅ DONE (mechanism, 2026-06-22):** decouple G2 (chạy thật, cite_precision đo) · grounding bỏ khỏi gate (log-only) · P0c aliasing fixed (0→23). → **P0-2b (NEW, NEXT):** cite-judge strict-match floor ~0.3-0.4 < 0.45 → clean-accept=0; soften judge (paraphrase/implication) hoặc recalibrate + discrimination test.
- **P1 (cấu trúc):** matrix thành HARD gate (suffix/skeleton detector) · paragraph-dedup lúc assemble · math-validation gate (balance + chống LaTeX leak) · near-miss rescue (0.35–0.40).
- **P1 (eval):** held-out judge từ model khác họ / gold set (chống self-eval vòng tròn).
- **P2 (agentic sâu hơn):** citation-graph 2nd-hop cho topic ngách · primary-source routing cho citation định nghĩa/phương trình.

## Lệnh nhanh
```bash
./run_full.sh                              # hoặc: python3 files/deep_research_v3.py --topic "<T>" --out-name <n> --no-smoke
python3 files/eval/test_verify_optim.py    # unit test verify
python3 files/monitor.py
```
