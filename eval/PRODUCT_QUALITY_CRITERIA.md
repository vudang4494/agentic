# PRODUCT QUALITY CRITERIA -- Ultra-Long-Form Book
## Verified by Academic Research Standards (2025-2026)

## Nguồn tham khảo (Đã Research)

| Framework | Nguồn | Đóng góp chính |
|----------|-------|----------------|
| **ResearchRubrics** | arXiv:2511.07685 | 2,500+ rubric criteria, Mandatory vs Optional, 6 axes |
| **DREAM** | arXiv:2602.18940 | 4 verticals taxonomy, capability parity, agentic eval |
| **DR3-Eval** | arXiv:2604.14683 | 5 metrics: IR, FA, CC, IF, DQ |
| **Semantic Originality** | clawRxiv:2604.01960 | k=32 neighbor aggregation, topic calibration |
| **Deep-Research Eval** | MDPI 2026 | Paged-RAG, coherence scoring |

---

## I. UNIFIED QUALITY TAXONOMY (Từ DREAM)

### 4 Verticals (BẮT BUỘC)

```python
VERTICALS = {
    "presentation_quality": {
        "description": "How effectively the report communicates findings",
        "metrics": ["clarity", "organization", "fluency", "structure"]
    },
    "task_compliance": {
        "description": "Whether report fulfills research query requirements",
        "metrics": ["coverage", "recall", "instruction_following"]
    },
    "analytical_depth": {
        "description": "Intellectual rigor of synthesis and reasoning",
        "metrics": ["reasoning_quality", "logical_coherence", "synthesis"]
    },
    "source_quality": {
        "description": "Reliability of evidence supporting claims",
        "metrics": ["citation_integrity", "factual_correctness", "grounding"]
    }
}
```

### Chi tiết từng Vertical

#### 1. Presentation Quality
- **Clarity**: Viết rõ ràng, không mơ hồ
- **Organization**: Cấu trúc hợp lý (heading hierarchy)
- **Fluency**: Câu văn trôi chảy
- **Structure**: Section organization đúng

#### 2. Task Compliance
- **Coverage**: Bao phủ đủ scope của query
- **Recall**: Nhớ lại key information từ evidence
- **Instruction Following**: Tuân thủ constraints từ prompt

#### 3. Analytical Depth
- **Reasoning Quality**: Lập luận logic, causal explanations
- **Logical Coherence**: Không contradictions
- **Synthesis**: Kết nối thông tin từ nhiều nguồn

#### 4. Source Quality
- **Citation Integrity**: Claims được cite đúng source
- **Factual Correctness**: Claims đúng với external world knowledge
- **Grounding**: Không hallucination

---

## II. MANDATORY vs OPTIONAL CRITERIA (Từ ResearchRubrics)

### Mandatory Criteria (FAIL nếu không đạt)

| Category | Criterion | Threshold |
|----------|-----------|-----------|
| Task Compliance | **Explicit Requirements** | 100% coverage |
| Source Quality | **Canonical Papers** | 100% recall |
| Source Quality | **Grounding Score** | >= 0.80 |
| Task Compliance | **Instruction Following** | >= 90% |
| Analytical Depth | **No Contradictions** | 0 contradictions |

### Optional Criteria (WARN nếu không đạt)

| Category | Criterion | Threshold |
|----------|-----------|-----------|
| Analytical Depth | **Implicit Requirements** | >= 60% coverage |
| Analytical Depth | **Synthesis Quality** | >= 50% |
| Presentation Quality | **Communication Quality** | >= 70% |
| Task Compliance | **Cross-References** | >= 2/section |
| Source Quality | **Citation Diversity** | >= 10 unique sources |

### Weighted Scoring (Từ ResearchRubrics)

```python
WEIGHT_SCALE = {
    # Mandatory weights (required for sufficiency)
    (+5, +4): "Critically important - fundamental flaw if missing",
    (-5, -4): "Critically detrimental - active harm",
    
    # Optional weights (nice-to-have)
    (+3, +2): "Important - key feature",
    (+1): "Slightly important - improves quality",
    (-1): "Slightly detrimental - minor issue",
    (-3, -2): "Detrimental - significant error"
}

# Final score formula
def compute_score(verdicts, weights):
    """
    S = sum(w_i * m_i) / sum(positive_w_i)
    where m_i = 1 (satisfied), 0.5 (partial), 0 (not satisfied)
    """
    pass
```

---

## III. DR3-EVAL METRICS

### 5 Core Metrics

```python
METRICS = {
    # Information Seeking
    "IR_UF": {
        "name": "Information Recall (User Files)",
        "threshold": 0.70,
        "description": "Coverage of insights from user-provided files"
    },
    "IR_SC": {
        "name": "Information Recall (Sandbox Corpus)", 
        "threshold": 0.60,
        "description": "Coverage of insights from retrieved sources"
    },
    "CC": {
        "name": "Citation Coverage",
        "threshold": 0.50,
        "description": "Proportion of required documents cited"
    },
    
    # Report Generation
    "FA": {
        "name": "Factual Accuracy",
        "threshold": 0.80,
        "description": "Claims supported by cited sources"
    },
    "IF": {
        "name": "Instruction Following",
        "threshold": 0.90,
        "description": "Adherence to query constraints"
    },
    "DQ": {
        "name": "Depth Quality",
        "threshold": 0.70,
        "description": "Analytical substance and logical rigor"
    }
}
```

---

## IV. SEMANTIC UNIQUENESS METRICS (Từ clawRxiv)

### Embedding-Based Originality

```python
def compute_originality_score(section: str, corpus: List[str]) -> Dict:
    """
    Using k=32 neighbors aggregation with topic calibration.
    
    Method:
    1. Embed section + corpus with bge-m3
    2. Find k=32 nearest neighbors in corpus
    3. Compute mean cosine distance
    4. Apply topic-specific calibration
    """
    # Single-neighbor: weak signal (rho = 0.41)
    # k=32 aggregate: strong signal (rho = 0.62)
    
    neighbors = find_k_nearest(section, corpus, k=32)
    distances = [cosine(section_emb, n_emb) for n_emb in neighbors]
    mean_distance = np.mean(distances)
    
    # Topic calibration (reduces bias by ~50%)
    calibrated = mean_distance * topic_calibration_factor
    
    return {
        "originality_score": calibrated,
        "k_neighbors": distances,
        "topic_calibration": topic_calibration_factor
    }

# Thresholds
ORIGINALITY_THRESHOLDS = {
    "novel": 0.70,      # high originality
    "acceptable": 0.50, # moderate overlap
    "derivative": 0.30, # high overlap - needs rewrite
    "plagiarized": 0.15 # very high overlap - BLOCK
}
```

### Section Uniqueness Check

```python
def verify_section_uniqueness_v2(
    new_section: str,
    prior_sections: List[str],
    embedding_model: str = "bge-m3:latest"
) -> Dict:
    """
    Enhanced uniqueness check using embedding distances.
    
    FAIL conditions:
    - max_similarity > 0.70 (derivative)
    - mean_similarity > 0.50 (acceptable overlap)
    - k=32 aggregate > 0.60 (not novel enough)
    """
    results = {
        "pass": True,
        "max_similarity": 0.0,
        "mean_similarity": 0.0,
        "k32_aggregate": 0.0,
        "action": "accept"
    }
    
    # Compute pairwise similarities
    for prior in prior_sections:
        sim = cosine_embed(new_section, prior, model=embedding_model)
        if sim > results["max_similarity"]:
            results["max_similarity"] = sim
    
    # k=32 aggregate over all corpus
    all_sims = [cosine_embed(new_section, p) for p in prior_sections]
    all_sims.sort(reverse=True)
    results["k32_aggregate"] = np.mean(all_sims[:32]) if len(all_sims) >= 32 else np.mean(all_sims)
    
    # Decision
    if results["max_similarity"] > 0.70:
        results["pass"] = False
        results["action"] = "block"
    elif results["k32_aggregate"] > 0.60:
        results["pass"] = False
        results["action"] = "rewrite"
    elif results["mean_similarity"] > 0.50:
        results["action"] = "warn"
    
    return results
```

---

## V. CONTENT QUALITY GATES (Từ Deep-Research Eval + ResearchRubrics)

### GATE-S1: Structure Quality

| Check | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Heading Hierarchy | Regex parse | Valid tree | FAIL if broken |
| Chapter Uniqueness | Jaccard titles | < 0.30 | BLOCK |
| Section Uniqueness | Jaccard titles | < 0.30 | BLOCK |
| No Part N Pattern | Regex | 0 matches | BLOCK |
| Orphan Sections | Cross-ref check | < 10% | WARN |

### GATE-S2: Semantic Uniqueness

| Check | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Max Similarity | Cosine embedding | < 0.70 | BLOCK |
| k=32 Aggregate | Cosine embedding | < 0.60 | REWRITE |
| Novel Term Ratio | Token analysis | >= 0.40 | WARN if < |
| Jaccard Similarity | Word overlap | < 0.30 | REWRITE |

### GATE-S3: Content Depth

| Check | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Information Recall | LLM evaluation | >= 0.70 | RETRY if < |
| Synthesis Score | LLM evaluation | >= 0.50 | RETRY if < |
| Reasoning Quality | Logic check | >= 0.70 | RETRY if < |
| Logical Coherence | No contradictions | 0 | FAIL if > 0 |

### GATE-S4: Grounding & Evidence

| Check | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Grounding Score | HHEM | >= 0.80 | RETRY if < |
| Factual Accuracy | Claim-source verify | >= 0.80 | RETRY if < |
| Citation Integrity | Citation check | >= 0.70 | WARN if < |
| Canonical Recall | arXiv check | 100% | FAIL if < |

### GATE-S5: Task Compliance

| Check | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Instruction Following | Constraint check | >= 0.90 | RETRY if < |
| Coverage | Key points check | >= 0.70 | WARN if < |
| Explicit Requirements | Rubric check | 100% | FAIL if < |

---

## VI. PROGRESSION & COHERENCE METRICS

### Depth Escalation (MỚI)

```python
def verify_depth_escalation(outline: Outline) -> Dict:
    """
    Verify book has logical progression: basic -> intermediate -> advanced.
    
    Method: LLM classify each section's depth level (1-5)
    Expected pattern: monotonically increasing or wave pattern
    """
    depth_levels = []
    for section in outline.sections:
        level = llm_classify_depth(section.title, section.content)
        depth_levels.append(level)
    
    # Check escalation
    is_escalating = all(depth_levels[i] <= depth_levels[i+1] + 1 
                        for i in range(len(depth_levels)-1))
    
    return {
        "pass": is_escalating,
        "depth_levels": depth_levels,
        "pattern": classify_pattern(depth_levels)
    }

DEPTH_LEVELS = {
    1: "Foundational - definitions, history",
    2: "Basic - core concepts, mechanisms",
    3: "Intermediate - methods, applications",
    4: "Advanced - optimizations, edge cases",
    5: "Expert - cutting-edge, research directions"
}
```

### Cross-Reference Density

```python
def verify_cross_references(book: Book) -> Dict:
    """
    Verify sections reference prior sections.
    
    Requirement: >= 2 cross-refs/section
    Cross-ref = reference to section in different chapter
    """
    ref_counts = []
    for section in book.sections:
        refs = extract_cross_refs(section.content, book.sections)
        ref_counts.append(len(refs))
    
    mean_refs = np.mean(ref_counts)
    orphan_ratio = sum(1 for c in ref_counts if c == 0) / len(ref_counts)
    
    return {
        "pass": mean_refs >= 2 and orphan_ratio < 0.1,
        "mean_refs": mean_refs,
        "orphan_ratio": orphan_ratio,
        "ref_counts": ref_counts
    }
```

### Concept Build-up

```python
def verify_concept_dependency(book: Book) -> Dict:
    """
    Verify concepts are introduced before being used.
    
    Build dependency graph of concepts.
    Fail if concept used before introduced.
    """
    intro_positions = {}  # concept -> first section index
    usage_positions = defaultdict(list)  # concept -> [section indices]
    
    for i, section in enumerate(book.sections):
        concepts = extract_key_concepts(section.content)
        for c in concepts:
            if c not in intro_positions:
                intro_positions[c] = i
            usage_positions[c].append(i)
    
    out_of_order = []
    for concept, usages in usage_positions.items():
        if concept in intro_positions:
            first_use = min(usages)
            intro = intro_positions[concept]
            if intro > first_use:
                out_of_order.append({
                    "concept": concept,
                    "introduced_at": intro,
                    "first_used_at": first_use,
                    "gap": intro - first_use
                })
    
    return {
        "pass": len(out_of_order) == 0,
        "out_of_order_concepts": out_of_order,
        "total_concepts": len(intro_positions)
    }
```

---

## VII. COMPLETE GATE FLOW

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Topic + Canonical Papers + Outline                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-0: OUTLINE VALIDATION (Pre-run)                        │
│  ├─ verify_no_part_duplication() ──────────→ BLOCK if fail │
│  ├─ verify_outline_uniqueness() ────────────→ BLOCK if fail │
│  ├─ verify_progression_logic() ──────────────→ BLOCK if fail │
│  └─ verify_depth_escalation() ──────────────→ BLOCK if fail │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-1: EVIDENCE QUALITY (P0a/b/c - existing)               │
│  ├─ Domain Relevance Gate (P0a) ────────────→ HARD BLOCK   │
│  ├─ Canonical Injection (P0b) ──────────────→ HARD BLOCK   │
│  └─ Seen-Count Penalty (P0c) ───────────────→ WARN        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-2: SEMANTIC UNIQUENESS (New)                           │
│  ├─ verify_section_uniqueness_v2() ─────────→ BLOCK if fail │
│  ├─ verify_originality_score() ────────────→ REWRITE if < │
│  └─ verify_novel_term_ratio() ──────────────→ WARN if <    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-3: CONTENT DEPTH (New)                                  │
│  ├─ verify_information_recall() ────────────→ RETRY if <   │
│  ├─ verify_synthesis_quality() ─────────────→ RETRY if <    │
│  └─ verify_reasoning_quality() ─────────────→ RETRY if <    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-4: GROUNDING & EVIDENCE (existing + new)               │
│  ├─ Grounding Score (HHEM) ─────────────────→ >= 0.80      │
│  ├─ Factual Accuracy ────────────────────────→ >= 0.80     │
│  ├─ Citation Integrity ──────────────────────→ >= 0.70     │
│  └─ Canonical Recall ────────────────────────→ 100%        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-5: TASK COMPLIANCE (New)                               │
│  ├─ Instruction Following ───────────────────→ >= 90%       │
│  ├─ Explicit Requirements ───────────────────→ 100%        │
│  └─ Coverage Check ───────────────────────────→ >= 70%       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-6: PROGRESSION & COHERENCE (New)                        │
│  ├─ verify_cross_references() ───────────────→ WARN if <    │
│  ├─ verify_concept_dependency() ─────────────→ BLOCK if fail │
│  └─ verify_book_coherence() ──────────────────→ MANUAL      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE-7: FINAL ASSEMBLY (existing)                            │
│  ├─ Heading hygiene check ────────────────────→ FAIL if dup  │
│  └─ Structure validation ──────────────────────→ FAIL if corrupt │
└─────────────────────────────────────────────────────────────┘
```

---

## VIII. THRESHOLD REFERENCE TABLE

| Gate | Check | Threshold | On Fail | Priority |
|------|-------|-----------|---------|----------|
| GATE-0 | Part N Pattern | 0 | BLOCK | MANDATORY |
| GATE-0 | Chapter Uniqueness | < 0.30 jaccard | BLOCK | MANDATORY |
| GATE-0 | Section Uniqueness | < 0.30 jaccard | BLOCK | MANDATORY |
| GATE-0 | Depth Escalation | Required | BLOCK | MANDATORY |
| GATE-2 | Max Similarity | < 0.70 cosine | BLOCK | MANDATORY |
| GATE-2 | k=32 Aggregate | < 0.60 | REWRITE | MANDATORY |
| GATE-2 | Novel Term Ratio | >= 0.40 | WARN | OPTIONAL |
| GATE-3 | Information Recall | >= 0.70 | RETRY | MANDATORY |
| GATE-3 | Synthesis Quality | >= 0.50 | RETRY | MANDATORY |
| GATE-3 | Reasoning Quality | >= 0.70 | RETRY | MANDATORY |
| GATE-4 | Grounding Score | >= 0.80 | RETRY | MANDATORY |
| GATE-4 | Factual Accuracy | >= 0.80 | RETRY | MANDATORY |
| GATE-4 | Canonical Recall | 100% | FAIL | MANDATORY |
| GATE-5 | Instruction Following | >= 0.90 | RETRY | MANDATORY |
| GATE-5 | Explicit Requirements | 100% | FAIL | MANDATORY |
| GATE-6 | Cross-References | >= 2/section | WARN | OPTIONAL |
| GATE-6 | Concept Dependency | 0 out-of-order | BLOCK | MANDATORY |
| GATE-7 | Heading Hygiene | 0 orphans | FAIL | MANDATORY |

---

## IX. SCORING FORMULA

### Overall Book Score

```python
def compute_book_score(book: Book, verify_results: Dict) -> float:
    """
    Weighted score combining all verticals.
    
    Formula:
    Score = (0.25 * Presentation + 0.25 * TaskCompliance + 
             0.25 * AnalyticalDepth + 0.25 * SourceQuality)
    
    Where each vertical = weighted sum of criteria
    """
    verticals = {
        "presentation_quality": compute_vertical_score(
            verify_results, 
            ["clarity", "organization", "fluency", "structure"],
            mandatory_weights=[4, 4, 3, 3]
        ),
        "task_compliance": compute_vertical_score(
            verify_results,
            ["coverage", "recall", "instruction_following"],
            mandatory_weights=[5, 4, 5]
        ),
        "analytical_depth": compute_vertical_score(
            verify_results,
            ["reasoning_quality", "synthesis", "coherence"],
            mandatory_weights=[4, 3, 5]
        ),
        "source_quality": compute_vertical_score(
            verify_results,
            ["grounding", "factual_accuracy", "citation_integrity", "canonical_recall"],
            mandatory_weights=[5, 5, 3, 5]
        )
    }
    
    overall = sum(verticals.values()) / len(verticals)
    return overall

def compute_vertical_score(results: Dict, criteria: List[str], 
                           mandatory_weights: List[int]) -> float:
    """Weighted sum for a vertical."""
    weighted_sum = 0
    total_positive = 0
    
    for criterion, weight in zip(criteria, mandatory_weights):
        if criterion in results:
            score = results[criterion]["score"]
            if weight > 0:
                weighted_sum += weight * score
                total_positive += weight
    
    return weighted_sum / total_positive if total_positive > 0 else 0
```

### Grade Scale

| Score Range | Grade | Description |
|-------------|-------|-------------|
| 0.90 - 1.00 | A+ | Excellent - publication ready |
| 0.80 - 0.89 | A | Good - minor revisions needed |
| 0.70 - 0.79 | B+ | Acceptable - revisions needed |
| 0.60 - 0.69 | B | Marginal - substantial revisions needed |
| 0.50 - 0.59 | C | Poor - major rework needed |
| < 0.50 | F | Fail - does not meet standards |

---

## X. IMPLEMENTATION PRIORITY

### Phase 1: Critical Gates (Week 1)
- [ ] verify_no_part_duplication()
- [ ] verify_section_uniqueness_v2() with embedding
- [ ] verify_depth_escalation()
- [ ] verify_concept_dependency()

### Phase 2: Quality Gates (Week 2)
- [ ] verify_information_recall()
- [ ] verify_synthesis_quality()
- [ ] verify_reasoning_quality()
- [ ] Grounding score >= 0.80 enforcement

### Phase 3: Coherence Gates (Week 3)
- [ ] verify_cross_references()
- [ ] verify_book_coherence()
- [ ] compute_book_score()

### Phase 4: Automation (Week 4)
- [ ] Integrate into pipeline
- [ ] Dashboard for monitoring
- [ ] Alert system for failures

---

## XI. REFERENCES

1. Sharma et al. (2025). ResearchRubrics: A Benchmark of Prompts and Rubrics for Evaluating Deep Research Agents. arXiv:2511.07685
2. Ben Avraham et al. (2026). DREAM: Deep Research Evaluation with Agentic Metrics. arXiv:2602.18940
3. Xie et al. (2026). DR3-Eval: Towards Realistic and Reproducible Deep Research Evaluation. arXiv:2604.14683
4. clawRxiv (2026). Estimating Originality from Embedding Distances Across Large Corpora. clawRxiv:2604.01960
5. MDPI (2026). Deep-Research Eval: An Automated Framework for Assessing Quality and Reliability in Long-Form Reports.
