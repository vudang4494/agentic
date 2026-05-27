# Eval report -- Transformer Architecture and Attention Mechanism

- Run ID: `20260527_132048`
- Difficulty: `easy`
- Outline: 3 chapters x 4 passes = 12 sections actually generated
- Elapsed: 41.3 min

## Overall: FAIL (6/10 checks passed)

| Check | Target | Actual | Pass |
|---|---|---|---|
| `must_cite_recall` | >= 0.8 | 0.0 | FAIL |
| `should_cite_recall` | >= 0.4 | 0.0 | FAIL |
| `grounding_mean` | >= 0.6 | 0.509 | FAIL |
| `zero_cite_sections` | <= 0 | 0 | OK |
| `loop_section_pct` | <= 0.1 | 0.0 | OK |
| `research_round_2_rate` | <= 0.4 | 0.667 | FAIL |
| `forbidden_domain_hits` | <= 0 | 0 | OK |
| `subtopic_coverage` | >= 0.7 | 0.9 | OK |
| `median_words_min` | >= 600 | 783.0 | OK |
| `median_words_max` | <= 1600 | 783.0 | OK |

## Aggregate metrics

| Metric | Value |
|---|---|
| `must_cite_recall` | 0.0 |
| `should_cite_recall` | 0.0 |
| `grounding_mean` | 0.509 |
| `zero_cite_section_count` | 0 |
| `loop_section_count` | 0 |
| `loop_section_pct` | 0.0 |
| `research_round_2_rate` | 0.667 |
| `forbidden_domain_hits` | 0 |
| `subtopic_coverage` | 0.9 |
| `median_words` | 783.0 |
| `mean_citations_per_1000w` | 12.28 |
| `total_tokens` | 12201 |

**Missed must-cite arxiv IDs:** `1607.06450`, `1706.03762`

**Missing expected subtopics:** `scaled dot-product attention`

## Per-section metrics

| Sec | Title | Words | Cites | Grounding | Sources | Round | Zero? | Loop? |
|---|---|---|---|---|---|---|---|---|
| 1.1 | Core Attention Mechanism: Self-A | 894 | 10 | 0.74 | 8 | 2 | - | - |
| 1.2 | The Transformer Decoder-Encoder: | 850 | 7 | 0.26 | 8 | 2 | - | - |
| 1.3 | Scaling Laws and Training Dynami | 755 | 10 | 0.39 | 8 | 2 | - | - |
| 1.4 | Attention Variants and Architect | 669 | 8 | 0.60 | 8 | 1 | - | - |
| 2.1 | Vision Transformers (ViT): Patch | 811 | 7 | 0.66 | 8 | 1 | - | - |
| 2.2 | Language Models and BERT: Bidire | 728 | 9 | 0.39 | 8 | 2 | - | - |
| 2.3 | Multimodal Transformers: Integra | 941 | 8 | 0.66 | 8 | 1 | - | - |
| 2.4 | Temporal Modeling: Time Series A | 751 | 13 | 0.76 | 8 | 1 | - | - |
| 3.1 | Generative Transformers: Diffusi | 840 | 8 | 0.49 | 8 | 2 | - | - |
| 3.2 | Large Language Models (LLMs): Co | 603 | 15 | 0.69 | 8 | 2 | - | - |
| 3.3 | Personalized and Interpretable A | 829 | 9 | 0.41 | 8 | 2 | - | - |
| 3.4 | Emerging Trends: Hybrid Architec | 755 | 8 | 0.06 | 8 | 2 | - | - |
