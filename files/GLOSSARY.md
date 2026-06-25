# Glossary -- Thuật ngữ hệ thống

> **Mục đích:** Định nghĩa chuẩn tất cả thuật ngữ. Dùng cho mọi file trong project.

---

## A. Kiến trúc Pipeline

| Term | Viết tắt | Định nghĩa |
|------|-----------|-------------|
| **Deep Investigation** | DI | Giai đoạn 2: với mỗi section, thực hiện research (tìm nguồn) -> write (viết) -> verify (kiểm tra) |
| **Discovery** | | Giai đoạn 0: phân tích topic, tạo TopicProfile |
| **Outline Generation** | | Giai đoạn 1: tạo cấu trúc chương/phần từ evidence |
| **Topic Profile** | TP | Data structure chứa: title, description, canonical papers, key concepts, search queries |
| **Outline Profile** | OP | Data structure chứa: title, chapters[], sections[] |
| **Chapter** | CH | Một chương trong sách |
| **Section** | SEC | Một phần trong chương, ví dụ "1.1 Introduction" |
| **State File** | | `state.json` -- lưu trạng thái run: sections đã viết, tổng words, seen_counts |

## B. Kiểm soát chất lượng

| Term | Định nghĩa |
|------|-------------|
| **Hard Block** | Khi kiểm tra fail, DỪNG và báo lỗi. Không viết section. Đối lập với Soft Block. |
| **Soft Block** | Khi kiểm tra fail, vẫn tiếp tục nhưng đánh dấu chất lượng giảm. |
| **Orchestration-layer improvement (doctrine)** | Cải thiện chất lượng ở tầng **orchestration/inference** (retrieval/verify/revise-loop/prompt/evidence-selection) — **KHÔNG fine-tune model, KHÔNG build dataset** (giữ topic-agnostic, prompt-robust, auditable). Bottleneck writer-grounding → verify-revise loop. Xem `CLAUDE.md mục 2`. |
| **Domain Relevance Gate (P0a)** | Kiểm tra evidence pool đúng domain trước khi viết, qua `notes.check_evidence_domain()` = keyword-overlap + optional gemma judge (KHÔNG phải LLM-judge thuần). Threshold THẬT ≈ 0.40 (`deep_investigate.py:524`), KHÔNG phải 0.60. Accept-topic (writer) = 0.50. Ngưỡng chuẩn: RULES.md |
| **Evidence Gate (P0a/B)** | Trước writer: (1) pool không rỗng (else HARD BLOCK); (2) domain-relevance ≥ ev_threshold≈0.40. KHÔNG có gate "đủ terms" riêng. |
| **Grounding Score** | Điểm HHEM v2 NLI (0-1). **G3 = log-only/advisory** (P0 2026-06-22: đã bỏ khỏi gate): strict-NLI ~0.05–0.10 trên prose synthesized → KHÔNG phải metric, không hard-block. (g=1.0 v36 là HHEM degenerate cũ, đã fix.) |
| **Topic Relevance Score** | Điểm content đúng chủ đề (0-1). **G4 blend** 0.6·`answer_relevance` (gemma LOCAL) + 0.4·term-overlap (`verify.py:401-411`). **P0: ENFORCED** — điều kiện `gate_ok` (clean-accept) + StageE chặn best-topic<0.50. |
| **Citation Count** | Số lần nguồn được trích dẫn trong text. Zero citation = section không có evidence |
| **Verify signals (G2/G3/G4) — POST-P0+P0-2b (2026-06-23)** | Gate cứng SỐNG = **P0a pre-writer + G2 cite_prec≥0.45**. **G2 cite_precision** = `verify_section` per-`[N]` (gemma) **GATE SỐNG** → cite_precision **đo thật** (KHÔNG default 1.0). **P0-2b:** judge prompt **soften** (paraphrase=supports, contradicts/unrelated giữ strict) → prose faithful đo ~0.48 ≥0.45 → **ACCEPT (`quality="ok"`)**; weak floor → degraded. Discrimination `bench_cite_discrimination.py`: GOOD 0.72 vs BAD 0.18/0.20. **G3 grounding** = log-only/advisory. **G4 topic** = ENFORCED. → `plan.md` mục Upgrade (kế = P1). |

## C. Retrieval (Tìm kiếm nguồn)

| Term | Viết tắt | Định nghĩa |
|------|-----------|-------------|
| **Source** | | Một nguồn tìm được: có id, title, url, excerpt |
| **Canonical Paper** | | Paper nền tảng bắt buộc phải có (ví dụ "Attention Is All You Need") |
| **BM25** | | Sparse retrieval: tìm theo keyword matching. Không cần embedding |
| **Cosine Similarity** | | Dense retrieval: tìm theo embedding vector similarity |
| **Reciprocal Rank Fusion** | RRF | Gộp nhiều rankers (BM25 + Cosine) bằng công thức 1/(k+rank) |
| **Reranker / Cross-Encoder** | RRK | Mô hình re-rank kết quả retrieval. Model: BAAI/bge-reranker-v2-m3 |
| **Embedding Model** | | **UNIFIED** `bge-m3:latest` (#3): retrieval (notes.rank/prefilter RRF), query_router, và verify-side đều dùng cùng 1 model — KHÔNG còn nomic runtime path. `config.py:34`, `notes.py:111`, `query_router.py:210`, `embeddings.py:8`, `verify.py:35`. (Trước split vì nomic cần prefix search_query/document mà code không truyền → asymmetric; 0 ref nomic sống, chỉ comment "was nomic".) |
| **Primary Source** | | arxiv.org hoặc wikipedia -- nguồn đáng tin cậy |
| **Secondary Source** | | Blog, medium, substack -- nguồn phụ |
| **Grey Domain** | | Domain có thể kém tin cậy (đã được whitelist) |

## D. Mô hình AI

| Term | Model | Vai trò |
|------|-------|---------|
| **Writer** | `batiai/qwen3.6-35b:iq3` | Viết nội dung section |
| **Query Generator** | QGN | Sinh search queries từ section title |
| **Judge / Verifier** | | Grounding = HHEM (model). Topic relevance = G4 blend heuristic + `answer_relevance` (gemma LOCAL). Citation integrity = `verify_section` G2 (gemma LOCAL). P0a domain = check_evidence_domain (gemma). |
| **Discovery Model** | | Phân tích topic, tạo TopicProfile |
| **Outline Model** | | Tạo outline từ evidence |

## E. Các lỗi đã biết

| Term | Định nghĩa | Root cause |
|------|-------------|------------|
| **Domain Mismatch** | Evidence tìm từ domain sai (ví dụ: RAG paper cho RLHF section) | Query generator gửi nhầm archetype |
| **Evidence Domination** | Một paper xuất hiện trong >50% sections | Không có penalty cho nguồn trùng lặp |
| **Canonical Miss** | Paper nền tảng không được retrieve | Search API ưu tiên paper mới |
| **Zero Citation** | Section không trích dẫn nguồn nào | Evidence rỗng hoặc writer không cite |

## F. Ký hiệu Logging

| Ký hiệu | Viết đầy | Ý nghĩa |
|---------|-----------|----------|
| `RSR` | Research | Thu thập nguồn |
| `QGN` | Query Gen | Sinh truy vấn |
| `WRT` | Write | Viết section |
| `VFY` | Verify | Kiểm tra grounding |
| `RVW` | Review | Đánh giá chất lượng |
| `RRK` | Rerank | Re-rank kết quả retrieval |
| `RRF` | Rank Fusion | Gộp BM25 + Cosine |
| `DI` | Deep Investigation | Toàn bộ stage 2 |

## G. Experiment Labels

| Label | Ý nghĩa |
|-------|----------|
| **Experiment A** | Benchmark 7 topics trước khi có fixes |
| **Experiment B** | Benchmark 7 topics sau khi có P0a/b/c fixes |
| **Smoke Test** | Test nhanh 1 section để verify fixes hoạt động |
| **Run** | Một lần chạy pipeline trên một topic |
| **Eval artifact** (BAER / `product_quality_verifiers.py`) | Công cụ **ĐO** chất lượng run đã xong (`files/eval/*`) — eval-only, dùng để chấm, KHÔNG phải training data. |

## H. Acronyms thường gặp

| Acronym | Giải nghĩa |
|---------|-------------|
| RRF | Reciprocal Rank Fusion |
| RRK | Reranker / Cross-encoder rerank |
| LLM | Large Language Model |
| QGN | Query Generator |
| VFY | Verifier |
| WRT | Writer |
| DI | Deep Investigation |
| TP | Topic Profile |
| OP | Outline Profile |
| arxiv | arxiv.org -- kho paper học thuật |
| wiki | Wikipedia |
| ddg | DuckDuckGo web search |
| CoT | Chain-of-Thought |
| DPO | Direct Preference Optimization |
| RLHF | Reinforcement Learning from Human Feedback |
| RoPE | Rotary Position Embedding |
| HHEM | Hallucination Evaluation Model |
