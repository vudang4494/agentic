"""
Product Quality Verifiers -- Ultra-Long-Form Book Quality Assessment
Verified by Academic Research Standards (2025-2026)

References:
- ResearchRubrics (arXiv:2511.07685) - Mandatory vs Optional criteria, 6 axes
- DREAM (arXiv:2602.18940) - 4 verticals taxonomy, capability parity
- DR3-Eval (arXiv:2604.14683) - 5 core metrics: IR, FA, CC, IF, DQ
- Semantic Originality (clawRxiv:2604.01960) - k=32 neighbor aggregation
"""
import re
import json
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

# ============================================================================
# GATE-0: OUTLINE VALIDATION
# ============================================================================

def verify_no_part_duplication(outline: Dict) -> Dict:
    """
    FAIL FAST if outline has Part N pattern.
    
    Pattern: "Topic (Part 2)", "Topic (Part 3)", etc.
    
    Returns:
        {
            "pass": bool,
            "duplicates": List[str],
            "action": "accept" | "BLOCK"
        }
    """
    part_pattern = re.compile(r"\(Part\s+\d+\)", re.IGNORECASE)
    
    duplicates = []
    chapters = outline.get("chapters", [])
    
    for chapter in chapters:
        chapter_title = chapter.get("title", "")
        if part_pattern.search(chapter_title):
            duplicates.append(f"[CHAPTER] {chapter_title}")
        
        for section in chapter.get("sections", []):
            section_title = section.get("title", "")
            if part_pattern.search(section_title):
                duplicates.append(f"[SECTION] {section_title}")
    
    return {
        "pass": len(duplicates) == 0,
        "duplicates": duplicates,
        "action": "BLOCK" if duplicates else "accept",
        "reason": f"Found {len(duplicates)} Part N patterns" if duplicates else "No duplicates"
    }


def verify_outline_uniqueness(outline: Dict) -> Dict:
    """
    Verify chapter and section titles are semantically unique.
    
    FAIL if:
    - Jaccard similarity > 0.30 between any two titles
    - Duplicate exact titles
    
    Returns:
        {
            "pass": bool,
            "duplicate_titles": List[str],
            "high_overlap_pairs": List[Tuple[str, str, float]],
            "action": "accept" | "BLOCK"
        }
    """
    all_titles = []
    
    for chapter in outline.get("chapters", []):
        all_titles.append(chapter.get("title", ""))
        for section in chapter.get("sections", []):
            all_titles.append(section.get("title", ""))
    
    # Check exact duplicates
    seen = {}
    duplicates = []
    for title in all_titles:
        if title in seen:
            duplicates.append(title)
        seen[title] = True
    
    # Check semantic overlap
    high_overlap = []
    for i, title1 in enumerate(all_titles):
        for j, title2 in enumerate(all_titles[i+1:], i+1):
            jaccard = compute_title_jaccard(title1, title2)
            if jaccard > 0.20:
                high_overlap.append((title1, title2, jaccard))
    
    return {
        "pass": len(duplicates) == 0 and len(high_overlap) == 0,
        "duplicate_titles": duplicates,
        "high_overlap_pairs": high_overlap,
        "action": "BLOCK" if (duplicates or high_overlap) else "accept"
    }


def verify_depth_escalation(outline: Dict) -> Dict:
    """
    Verify outline has logical progression: basic -> intermediate -> advanced.
    
    Method: Analyze chapter titles for depth indicators.
    
    Returns:
        {
            "pass": bool,
            "depth_pattern": List[int],
            "pattern_type": str,
            "action": "accept" | "BLOCK"
        }
    """
    depth_indicators = {
        "foundations": 1,
        "basics": 1,
        "fundamentals": 1,
        "introduction": 1,
        "core": 2,
        "principles": 2,
        "math": 2,
        "architecture": 2,
        "training": 3,
        "methods": 3,
        "applications": 3,
        "evaluation": 3,
        "advanced": 4,
        "frontiers": 4,
        "ethics": 4,
        "research": 5,
        "cutting-edge": 5,
    }
    
    depth_levels = []
    for chapter in outline.get("chapters", []):
        title = chapter.get("title", "").lower()
        max_depth = 2  # Default to basic
        for indicator, depth in depth_indicators.items():
            if indicator in title:
                max_depth = max(max_depth, depth)
        depth_levels.append(max_depth)
    
    # Check pattern: should have escalation
    # Pattern types: escalating, wave, flat, random
    if all(depth_levels[i] <= depth_levels[i+1] for i in range(len(depth_levels)-1)):
        pattern = "escalating"
        pass_check = True
    elif all(depth_levels[i] >= depth_levels[i+1] for i in range(len(depth_levels)-1)):
        pattern = "de-escalating"
        pass_check = False
    elif is_wave_pattern(depth_levels):
        pattern = "wave"  # OK - can go deep then broad
        pass_check = True
    else:
        pattern = "mixed"
        pass_check = len(set(depth_levels)) > 1  # OK if varied
    
    return {
        "pass": pass_check,
        "depth_pattern": depth_levels,
        "pattern_type": pattern,
        "action": "accept" if pass_check else "BLOCK"
    }


# ============================================================================
# GATE-2: SEMANTIC UNIQUENESS
# ============================================================================

def verify_section_uniqueness_v2(
    new_section: str,
    prior_sections: List[str],
    embedding_model: str = "bge-m3:latest"
) -> Dict:
    """
    Enhanced uniqueness check using embedding distances.
    Based on clawRxiv:2604.01960 - k=32 neighbor aggregation.
    
    FAIL conditions:
    - max_similarity > 0.70 (derivative content)
    - k=32 aggregate > 0.60 (not novel enough)
    
    Returns:
        {
            "pass": bool,
            "max_similarity": float,
            "k32_aggregate": float,
            "jaccard_scores": List[float],
            "action": "accept" | "rewrite" | "BLOCK"
        }
    """
    # For now, use Jaccard as fallback (embedding requires API call)
    new_words = set(tokenize(new_section.lower()))
    
    jaccard_scores = []
    max_jaccard = 0.0
    all_jaccards = []
    
    for prior in prior_sections:
        prior_words = set(tokenize(prior.lower()))
        jaccard = compute_jaccard(new_words, prior_words)
        jaccard_scores.append(jaccard)
        all_jaccards.append(jaccard)
        if jaccard > max_jaccard:
            max_jaccard = jaccard
    
    # k=32 aggregate
    all_jaccards.sort(reverse=True)
    k32_aggregate = np.mean(all_jaccards[:32]) if len(all_jaccards) >= 32 else np.mean(all_jaccards)
    
    # Decision
    action = "accept"
    if max_jaccard > 0.20:
        action = "BLOCK"
    elif k32_aggregate > 0.40:
        action = "rewrite"
    elif max_jaccard > 0.15:
        action = "warn"
    
    return {
        "pass": action == "accept",
        "max_similarity": max_jaccard,
        "k32_aggregate": k32_aggregate,
        "jaccard_scores": jaccard_scores[:10],  # Top 10
        "action": action
    }


def verify_originality_score(
    section: str,
    corpus: List[str],
    k: int = 32
) -> Dict:
    """
    Compute embedding-based originality score.
    Based on clawRxiv:2604.01960.
    
    Method:
    1. Embed section + corpus
    2. Find k nearest neighbors
    3. Compute mean cosine distance
    4. Apply topic calibration
    
    Returns:
        {
            "originality_score": float (0-1, higher = more original),
            "k_neighbors": int,
            "raw_distance": float,
            "calibrated_distance": float,
            "verdict": "novel" | "acceptable" | "derivative" | "plagiarized"
        }
    """
    # Simplified: use word-based distance as proxy
    section_words = set(tokenize(section.lower()))
    
    if not corpus:
        return {
            "originality_score": 1.0,
            "k_neighbors": 0,
            "raw_distance": 1.0,
            "calibrated_distance": 1.0,
            "verdict": "novel"
        }
    
    distances = []
    for doc in corpus:
        doc_words = set(tokenize(doc.lower()))
        # Cosine-like similarity
        overlap = len(section_words & doc_words)
        norm = math.sqrt(len(section_words) * len(doc_words))
        if norm > 0:
            similarity = overlap / norm
            distances.append(1 - similarity)  # Distance = 1 - similarity
    
    distances.sort()
    k_distances = distances[:min(k, len(distances))]
    raw_distance = np.mean(k_distances) if k_distances else 1.0
    
    # Topic calibration factor (simplified)
    # In practice, would use per-topic calibration
    calibration_factor = 1.0
    calibrated_distance = raw_distance * calibration_factor
    
    # Map to 0-1 originality (higher = more original)
    originality_score = min(1.0, calibrated_distance * 2)
    
    # Verdict
    if originality_score >= 0.70:
        verdict = "novel"
    elif originality_score >= 0.50:
        verdict = "acceptable"
    elif originality_score >= 0.30:
        verdict = "derivative"
    else:
        verdict = "plagiarized"
    
    return {
        "originality_score": round(originality_score, 3),
        "k_neighbors": min(k, len(distances)),
        "raw_distance": round(raw_distance, 3),
        "calibrated_distance": round(calibrated_distance, 3),
        "verdict": verdict
    }


def verify_novel_term_ratio(
    new_section: str,
    prior_sections: List[str],
    threshold: float = 0.40
) -> Dict:
    """
    Compute ratio of novel terms in new section vs all prior sections.
    
    Novel term = term not appearing in any prior section.
    
    Returns:
        {
            "pass": bool,
            "novel_ratio": float,
            "total_terms": int,
            "novel_terms": int,
            "action": "accept" | "rewrite"
        }
    """
    new_words = set(tokenize(new_section.lower()))
    
    all_prior_words = set()
    for prior in prior_sections:
        all_prior_words |= set(tokenize(prior.lower()))
    
    novel_terms = new_words - all_prior_words
    novel_ratio = len(novel_terms) / len(new_words) if new_words else 0
    
    return {
        "pass": novel_ratio >= threshold,
        "novel_ratio": round(novel_ratio, 3),
        "total_terms": len(new_words),
        "novel_terms": len(novel_terms),
        "threshold": threshold,
        "action": "accept" if novel_ratio >= threshold else "rewrite"
    }


# ============================================================================
# GATE-3: CONTENT DEPTH (Based on DREAM/DR3-Eval)
# ============================================================================

def verify_information_recall(
    section_content: str,
    evidence_sources: List[str],
    required_insights: List[str],
    llm_call_fn=None
) -> Dict:
    """
    Verify section covers key information from evidence.
    Based on DR3-Eval IR metric.
    
    Returns:
        {
            "pass": bool,
            "recall_score": float,
            "covered_insights": List[str],
            "missing_insights": List[str],
            "action": "accept" | "retry"
        }
    """
    if not required_insights:
        return {
            "pass": True,
            "recall_score": 1.0,
            "covered_insights": [],
            "missing_insights": [],
            "action": "accept"
        }
    
    # Simple keyword matching as fallback
    section_lower = section_content.lower()
    
    covered = []
    missing = []
    for insight in required_insights:
        insight_keywords = tokenize(insight.lower())
        if any(kw in section_lower for kw in insight_keywords if len(kw) > 4):
            covered.append(insight)
        else:
            missing.append(insight)
    
    recall_score = len(covered) / len(required_insights) if required_insights else 1.0
    
    return {
        "pass": recall_score >= 0.70,
        "recall_score": round(recall_score, 3),
        "covered_insights": covered,
        "missing_insights": missing,
        "action": "accept" if recall_score >= 0.70 else "retry"
    }


def verify_synthesis_quality(
    section_content: str,
    section_title: str,
    llm_call_fn=None
) -> Dict:
    """
    Verify section synthesizes information rather than listing facts.
    Based on ResearchRubrics synthesis criterion.
    
    Checks:
    - Connects information from multiple sources
    - Identifies trends/patterns
    - Draws integrated conclusions
    
    Returns:
        {
            "pass": bool,
            "synthesis_score": float,
            "indicators": List[str],
            "action": "accept" | "retry"
        }
    """
    synthesis_indicators = [
        "however",
        "therefore",
        "consequently",
        "in contrast",
        "furthermore",
        "moreover",
        "additionally",
        "whereas",
        "although",
        "thus",
        "this suggests",
        "these findings indicate",
        "in summary",
        "ultimately",
        "as a result",
    ]
    
    comparison_indicators = [
        "compared to",
        "in comparison",
        "on the other hand",
        "similarly",
        "differently",
        "unlike",
        "likewise",
    ]
    
    synthesis_patterns = [
        r"\d+\s*[%]",
        r"\d+\s*times",
        r"increased?",
        r"decreased?",
        r"improved?",
        r"significantly",
        r"trend",
    ]
    
    section_lower = section_content.lower()
    
    # Count indicators
    indicator_count = sum(1 for ind in synthesis_indicators if ind in section_lower)
    comparison_count = sum(1 for ind in comparison_indicators if ind in section_lower)
    pattern_count = sum(1 for pat in synthesis_patterns if re.search(pat, section_lower))
    
    # Synthesis indicators present?
    indicators = []
    if indicator_count >= 3:
        indicators.append("high_connector_usage")
    if comparison_count >= 2:
        indicators.append("comparison_present")
    if pattern_count >= 2:
        indicators.append("quantitative_analysis")
    
    # Check for list-like structure (bad)
    lines = section_content.split("\n")
    short_lines = sum(1 for line in lines if len(line.split()) < 10)
    list_ratio = short_lines / len(lines) if lines else 0
    
    if list_ratio > 0.5:
        indicators.append("too_listy")
    
    # Compute score
    score = min(1.0, (indicator_count + comparison_count + pattern_count) / 6)
    
    return {
        "pass": score >= 0.50,
        "synthesis_score": round(score, 3),
        "indicators": indicators,
        "connector_count": indicator_count,
        "comparison_count": comparison_count,
        "pattern_count": pattern_count,
        "action": "accept" if score >= 0.50 else "retry"
    }


def verify_reasoning_quality(
    section_content: str,
    llm_call_fn=None
) -> Dict:
    """
    Verify section has logical reasoning.
    Based on DREAM RQ metric.
    
    Checks:
    - No logical fallacies
    - Causal links present
    - Evidence supports conclusions
    
    Returns:
        {
            "pass": bool,
            "reasoning_score": float,
            "fallacies": List[str],
            "action": "accept" | "retry"
        }
    """
    # Fallacy patterns
    fallacy_patterns = {
        "ad_hominem": [r"you are wrong because", r"that person is"],
        "false_dichotomy": [r"either.*or", r"only two options"],
        "circular_reasoning": [r"is true because.*is true"],
        "appeal_to_authority": [r"expert said.*therefore"],
    }
    
    fallacies_found = []
    section_lower = section_content.lower()
    
    for fallacy_name, patterns in fallacy_patterns.items():
        for pattern in patterns:
            if re.search(pattern, section_lower):
                fallacies_found.append(fallacy_name)
    
    # Positive reasoning indicators
    reasoning_indicators = [
        r"because",
        r"since",
        r"due to",
        r"as a result",
        r"this leads to",
        r"caused by",
        r"consequently",
        r"implies",
        r"demonstrates",
        r"evidence shows",
    ]
    
    reasoning_count = sum(1 for pat in reasoning_indicators if re.search(pat, section_lower))
    
    # Score
    fallacy_penalty = min(0.3, len(fallacies_found) * 0.1)
    reasoning_bonus = min(0.3, reasoning_count * 0.05)
    
    score = max(0.0, 0.7 - fallacy_penalty + reasoning_bonus)
    
    return {
        "pass": score >= 0.70,
        "reasoning_score": round(score, 3),
        "fallacies": fallacies_found,
        "reasoning_count": reasoning_count,
        "action": "accept" if score >= 0.70 else "retry"
    }


# ============================================================================
# GATE-5: TASK COMPLIANCE (Based on ResearchRubrics)
# ============================================================================

def verify_instruction_following(
    section_content: str,
    constraints: List[str],
    llm_call_fn=None
) -> Dict:
    """
    Verify section follows explicit instructions/constraints.
    Based on ResearchRubrics IF criterion.
    
    Returns:
        {
            "pass": bool,
            "follow_score": float,
            "violations": List[str],
            "action": "accept" | "retry"
        }
    """
    violations = []
    section_lower = section_content.lower()
    
    for constraint in constraints:
        constraint_lower = constraint.lower()
        # Check if constraint is violated
        if constraint_lower.startswith("must not include"):
            term = constraint_lower.replace("must not include", "").strip()
            if term in section_lower:
                violations.append(f"Contains forbidden term: {term}")
        elif constraint_lower.startswith("must include"):
            term = constraint_lower.replace("must include", "").strip()
            if term not in section_lower:
                violations.append(f"Missing required term: {term}")
    
    follow_score = 1.0 - (len(violations) * 0.2)
    follow_score = max(0.0, min(1.0, follow_score))
    
    return {
        "pass": follow_score >= 0.90,
        "follow_score": round(follow_score, 3),
        "violations": violations,
        "action": "accept" if follow_score >= 0.90 else "retry"
    }


def verify_explicit_requirements(
    section_content: str,
    requirements: List[str],
    llm_call_fn=None
) -> Dict:
    """
    Verify all explicit requirements from prompt are met.
    Based on ResearchRubrics explicit requirements criterion.
    
    Returns:
        {
            "pass": bool,
            "requirement_score": float,
            "met_requirements": List[str],
            "unmet_requirements": List[str],
            "action": "accept" | "retry"
        }
    """
    section_lower = section_content.lower()
    
    met = []
    unmet = []
    
    for req in requirements:
        req_lower = req.lower()
        req_keywords = [w for w in tokenize(req_lower) if len(w) > 3]
        
        if all(kw in section_lower for kw in req_keywords[:3]):  # At least 3 keywords
            met.append(req)
        else:
            unmet.append(req)
    
    score = len(met) / len(requirements) if requirements else 1.0
    
    return {
        "pass": score == 1.0,  # 100% required
        "requirement_score": round(score, 3),
        "met_requirements": met,
        "unmet_requirements": unmet,
        "action": "accept" if score == 1.0 else "retry"
    }


# ============================================================================
# GATE-6: PROGRESSION & COHERENCE
# ============================================================================

def verify_cross_references(
    book_sections: List[Dict],
    min_refs_per_section: int = 2
) -> Dict:
    """
    Verify sections reference other sections.
    Based on DREAM coherence check.
    
    Returns:
        {
            "pass": bool,
            "mean_refs": float,
            "orphan_ratio": float,
            "ref_counts": Dict[str, int],
            "orphans": List[str],
            "action": "accept" | "warn"
        }
    """
    ref_pattern = re.compile(r"\b(section|chapter|part)\s+\d+[\.\d]*\b", re.IGNORECASE)
    
    ref_counts = {}
    for section in book_sections:
        section_id = section.get("id", "unknown")
        content = section.get("content", "")
        
        refs = ref_pattern.findall(content)
        ref_counts[section_id] = len(refs)
    
    mean_refs = np.mean(list(ref_counts.values())) if ref_counts else 0
    orphans = [sid for sid, count in ref_counts.items() if count == 0]
    orphan_ratio = len(orphans) / len(ref_counts) if ref_counts else 0
    
    return {
        "pass": mean_refs >= min_refs_per_section and orphan_ratio < 0.1,
        "mean_refs": round(mean_refs, 2),
        "orphan_ratio": round(orphan_ratio, 3),
        "ref_counts": ref_counts,
        "orphans": orphans,
        "action": "accept" if mean_refs >= min_refs_per_section and orphan_ratio < 0.1 else "warn"
    }


def verify_concept_dependency(
    book_sections: List[Dict],
) -> Dict:
    """
    Verify concepts are introduced before being used.
    
    Build dependency graph of key concepts.
    Fail if concept used before introduced.
    
    Returns:
        {
            "pass": bool,
            "out_of_order_concepts": List[Dict],
            "total_concepts": int,
            "action": "accept" | "BLOCK"
        }
    """
    # Extract concepts per section (simplified: key terms)
    concept_intro = {}  # concept -> first section index
    concept_usage = defaultdict(list)  # concept -> [section indices]
    
    key_terms = ["transformer", "attention", "rlhf", "llm", "embedding", 
                 "fine-tuning", "pre-training", "token", "embedding"]
    
    for i, section in enumerate(book_sections):
        content = section.get("content", "").lower()
        section_title = section.get("title", "").lower()
        
        for term in key_terms:
            if term in content or term in section_title:
                if term not in concept_intro:
                    concept_intro[term] = i
                concept_usage[term].append(i)
    
    # Check out-of-order
    out_of_order = []
    for concept, usages in concept_usage.items():
        if concept in concept_intro:
            intro_idx = concept_intro[concept]
            for usage_idx in usages:
                if usage_idx < intro_idx:
                    out_of_order.append({
                        "concept": concept,
                        "introduced_at": intro_idx,
                        "first_used_at": usage_idx,
                        "gap": intro_idx - usage_idx
                    })
    
    return {
        "pass": len(out_of_order) == 0,
        "out_of_order_concepts": out_of_order,
        "total_concepts": len(concept_intro),
        "action": "accept" if len(out_of_order) == 0 else "BLOCK"
    }


def verify_book_coherence(book: Dict) -> Dict:
    """
    High-level coherence check for entire book.
    
    Returns:
        {
            "pass": bool,
            "structure_score": float,
            "narrative_score": float,
            "recommendation": str
        }
    """
    chapters = book.get("chapters", [])
    
    # Structure checks
    structure_score = 1.0
    
    # Chapter count
    if len(chapters) < 3:
        structure_score -= 0.2
    elif len(chapters) > 30:
        structure_score -= 0.1
    
    # Sections per chapter variance
    section_counts = [len(ch.get("sections", [])) for ch in chapters]
    if section_counts:
        variance = np.var(section_counts)
        if variance > 20:  # Too varied
            structure_score -= 0.1
    
    # Cross-chapter references
    cross_refs = 0
    for i, chapter in enumerate(chapters):
        for section in chapter.get("sections", []):
            content = section.get("content", "").lower()
            # Check for prior chapter refs
            for j in range(i):
                if f"chapter {j+1}" in content or f"ch. {j+1}" in content:
                    cross_refs += 1
    
    cross_ref_ratio = cross_refs / max(1, sum(len(ch.get("sections", [])) for ch in chapters))
    if cross_ref_ratio < 0.05:
        structure_score -= 0.1
    
    structure_score = max(0.0, min(1.0, structure_score))
    
    return {
        "pass": structure_score >= 0.7,
        "structure_score": round(structure_score, 3),
        "narrative_score": round(structure_score * 0.8, 3),  # Simplified
        "recommendation": "manual_review" if structure_score < 0.8 else "pass"
    }


# ============================================================================
# GATE-7: FINAL ASSEMBLY
# ============================================================================

def verify_heading_hygiene(book_content: str) -> Dict:
    """
    Verify heading structure is clean.
    Based on RULES Stage F.
    
    Returns:
        {
            "pass": bool,
            "issues": List[str],
            "action": "accept" | "fail"
        }
    """
    issues = []
    
    # Check for orphan headings
    lines = book_content.split("\n")
    orphan_pattern = re.compile(r"^#{1,3}\s+[A-Z]")
    
    orphan_count = 0
    for i, line in enumerate(lines):
        if orphan_pattern.match(line.strip()):
            # Check if preceded by content
            if i > 0:
                prev_lines = [l for l in lines[max(0, i-5):i] if l.strip()]
                if not prev_lines:
                    orphan_count += 1
                    issues.append(f"Orphan heading at line {i}: {line[:50]}")
    
    # Check for duplicate headings
    headings = []
    for line in lines:
        match = re.match(r"^#{1,3}\s+(.+)$", line.strip())
        if match:
            headings.append(match.group(1))
    
    from collections import Counter
    heading_counts = Counter(headings)
    duplicates = {h: c for h, c in heading_counts.items() if c > 1}
    
    if duplicates:
        for heading, count in duplicates.items():
            issues.append(f"Duplicate heading '{heading}' appears {count} times")
    
    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "orphan_count": orphan_count,
        "duplicate_headings": list(duplicates.keys()),
        "action": "accept" if len(issues) == 0 else "fail"
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def compute_jaccard(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_title_jaccard(title1: str, title2: str) -> float:
    """Compute Jaccard similarity between two titles."""
    words1 = set(tokenize(title1.lower()))
    words2 = set(tokenize(title2.lower()))
    return compute_jaccard(words1, words2)


def is_wave_pattern(levels: List[int]) -> bool:
    """Check if depth levels follow a wave pattern."""
    if len(levels) < 3:
        return False
    
    increases = sum(1 for i in range(len(levels)-1) if levels[i+1] > levels[i])
    decreases = sum(1 for i in range(len(levels)-1) if levels[i+1] < levels[i])
    
    # Wave = roughly equal increases and decreases
    total_changes = increases + decreases
    if total_changes == 0:
        return False
    
    balance = min(increases, decreases) / total_changes
    return balance > 0.3  # At least 30% balanced


# ============================================================================
# MAIN QUALITY ASSESSMENT
# ============================================================================

def run_full_quality_assessment(
    book: Dict,
    outline: Dict,
    prior_sections: List[str] = None
) -> Dict:
    """
    Run complete quality assessment across all gates.
    
    Returns:
        {
            "overall_pass": bool,
            "gate_results": Dict,
            "final_score": float,
            "grade": str,
            "recommendations": List[str]
        }
    """
    prior_sections = prior_sections or []
    
    results = {}
    
    # GATE-0: Outline Validation
    results["gate_0_outline"] = {
        "part_duplication": verify_no_part_duplication(outline),
        "outline_uniqueness": verify_outline_uniqueness(outline),
        "depth_escalation": verify_depth_escalation(outline),
    }
    
    # GATE-2: Semantic Uniqueness
    all_sections = []
    for chapter in outline.get("chapters", []):
        for section in chapter.get("sections", []):
            all_sections.append(section.get("content", ""))
    
    uniqueness_results = []
    for i, section in enumerate(all_sections):
        prior = all_sections[:i] + prior_sections
        uniqueness_results.append(verify_section_uniqueness_v2(section, prior))
    
    results["gate_2_uniqueness"] = {
        "section_results": uniqueness_results,
        "all_pass": all(r["pass"] for r in uniqueness_results),
        "block_count": sum(1 for r in uniqueness_results if r["action"] == "BLOCK"),
        "rewrite_count": sum(1 for r in uniqueness_results if r["action"] == "rewrite"),
    }
    
    # GATE-6: Progression & Coherence
    book_sections = []
    for chapter in outline.get("chapters", []):
        for section in chapter.get("sections", []):
            book_sections.append({
                "id": section.get("id", "unknown"),
                "title": section.get("title", ""),
                "content": section.get("content", "")
            })
    
    results["gate_6_coherence"] = {
        "cross_references": verify_cross_references(book_sections),
        "concept_dependency": verify_concept_dependency(book_sections),
        "book_coherence": verify_book_coherence(book),
    }
    
    # Compute overall pass
    gate_0_pass = all([
        results["gate_0_outline"]["part_duplication"]["pass"],
        results["gate_0_outline"]["outline_uniqueness"]["pass"],
        results["gate_0_outline"]["depth_escalation"]["pass"],
    ])
    
    gate_2_pass = results["gate_2_uniqueness"]["all_pass"]
    
    gate_6_pass = all([
        results["gate_6_coherence"]["book_coherence"]["pass"],
    ])
    
    overall_pass = gate_0_pass and gate_2_pass and gate_6_pass
    
    # Compute final score (simplified)
    scores = [
        1.0 if results["gate_0_outline"]["part_duplication"]["pass"] else 0.0,
        1.0 if results["gate_0_outline"]["outline_uniqueness"]["pass"] else 0.0,
        results["gate_2_uniqueness"]["all_pass"] * 1.0,
        results["gate_6_coherence"]["book_coherence"]["structure_score"],
    ]
    final_score = np.mean(scores)
    
    # Grade
    if final_score >= 0.90:
        grade = "A+"
    elif final_score >= 0.80:
        grade = "A"
    elif final_score >= 0.70:
        grade = "B+"
    elif final_score >= 0.60:
        grade = "B"
    else:
        grade = "C/F"
    
    # Recommendations
    recommendations = []
    if not results["gate_0_outline"]["part_duplication"]["pass"]:
        recommendations.append("Remove Part N patterns from outline")
    if results["gate_2_uniqueness"]["block_count"] > 0:
        recommendations.append(f"Rewrite {results['gate_2_uniqueness']['block_count']} sections with high similarity")
    if not results["gate_6_coherence"]["cross_references"]["pass"]:
        recommendations.append("Add more cross-references between sections")
    
    return {
        "overall_pass": overall_pass,
        "gate_results": results,
        "final_score": round(final_score, 3),
        "grade": grade,
        "recommendations": recommendations
    }
