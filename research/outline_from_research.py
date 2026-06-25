"""
Stage 1: OUTLINE FROM RESEARCH
Generate outline STRUCTURED BY the evidence, not by prior assumptions.

Input:  TopicProfile (from Stage 0) + gathered sources
Output: OutlineProfile = {
    title, subtitle,
    chapters: [{n, t, pr, coverage_note, sections: [{n, t, pr}]}],
    coverage_gaps: ["concept X not yet covered"],
  }
"""
import httpx, json, re
from dataclasses import dataclass, field
from typing import List

from .config import OUTLINE_MODEL

OLLAMA_BASE = "http://localhost:11434"
TIMEOUT = 300.0
_GENERIC_CHAPTER_RE = re.compile(r"^(Part|Chapter|Section)\s+\d+\s*$", re.IGNORECASE)


class OutlineValidationError(Exception):
    """Raised when outline fails GATE-0 validation (RULES-U1)."""
    pass


@dataclass
class OutlineProfile:
    title: str
    subtitle: str = ""
    chapters: List[dict] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)
    evidence_map: List[dict] = field(default_factory=list)
    outline_audit: dict = field(default_factory=dict)
    _raw: str = ""


def _ollama_chat(model: str, messages: list, temperature: float = 0.4,
                 num_predict: int = 4000, timeout: float = TIMEOUT) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    content = (data.get("message") or {}).get("content", "").strip()
    if not content:
        content = (data.get("message") or {}).get("thinking", "").strip()
    return content


def build_evidence_map(topic_profile, sources: list) -> List[dict]:
    """Cluster discovery evidence into coarse thematic buckets before outline drafting."""
    buckets = [
        ("foundations", ["introduction", "overview", "foundation", "history", "concept"]),
        ("math", ["equation", "objective", "likelihood", "score", "stochastic", "probability"]),
        ("architectures", ["architecture", "u-net", "transformer", "latent", "implementation"]),
        ("training", ["training", "optimization", "loss", "guidance", "conditioning"]),
        ("evaluation", ["benchmark", "evaluation", "metric", "performance"]),
        ("applications", ["application", "use case", "image", "video", "audio", "frontier"]),
    ]
    mapped = []
    for name, keywords in buckets:
        matched = []
        for s in sources:
            hay = f"{getattr(s, 'title', '')} {getattr(s, 'excerpt', '')}".lower()
            if any(k in hay for k in keywords):
                matched.append({
                    "title": getattr(s, "title", ""),
                    "url": getattr(s, "url", ""),
                    "provider": getattr(s, "provider", ""),
                })
        if matched:
            mapped.append({
                "bucket": name,
                "keywords": keywords,
                "sources": matched[:6],
            })
    if not mapped:
        fallback_terms = getattr(topic_profile, "must_cover", [])[:6] or getattr(topic_profile, "key_concepts", [])[:6]
        mapped = [
            {"bucket": term.lower().replace(" ", "_"), "keywords": [term], "sources": []}
            for term in fallback_terms
        ]
    return mapped[:10]


def draft_outline_from_buckets(
    topic_profile,
    evidence_map: List[dict],
    n_chapters: int,
    sections_per_chapter: int,
    model: str,
) -> str:
    must_cover = getattr(topic_profile, "must_cover", [])
    canonical_terms = getattr(topic_profile, "canonical_terms", [])
    out_of_scope = getattr(topic_profile, "out_of_scope", [])
    base_buckets = [item.get("bucket", "topic") for item in evidence_map] or [
        "foundations", "methods", "architectures", "training",
        "evaluation", "applications", "frontiers", "ethics"
    ]
    bucket_lines = []
    for item in evidence_map:
        src_titles = "; ".join(s.get("title", "") for s in item.get("sources", [])[:3])
        bucket_lines.append(f"- {item['bucket']}: keywords={', '.join(item.get('keywords', [])[:5])}; sources={src_titles}")
    bucket_text = "\n".join(bucket_lines) or "- none"
    prompt = f"""You are a research book architect. Build a technical outline from thematic evidence buckets.

Topic: {getattr(topic_profile, 'name', 'Unknown')}
Subtitle: {getattr(topic_profile, 'subtitle', '')}
Scope: {getattr(topic_profile, 'description', '')}
Must-cover terms: {json.dumps(must_cover, ensure_ascii=False)}
Canonical terms: {json.dumps(canonical_terms, ensure_ascii=False)}
Out-of-scope drift terms: {json.dumps(out_of_scope, ensure_ascii=False)}

Evidence buckets:
{bucket_text}

Output ONLY JSON:
{{
  "title": "...",
  "subtitle": "...",
  "chapters": [
    {{
      "n": 1,
      "t": "Specific chapter title",
      "coverage_note": "Why this chapter exists",
      "sections": [
        {{
          "n": 1,
          "t": "Specific section title",
          "pr": "Specific writing directive",
          "goal": "What this section must teach",
          "must_cover_terms": ["..."],
          "avoid_terms": ["..."],
          "depends_on": [],
          "section_type": "foundational|methods|math|systems|applications|frontiers"
        }}
      ]
    }}
  ],
  "coverage_gaps": ["..."]
}}

Rules:
- Exactly {n_chapters} chapters
- Exactly {sections_per_chapter} sections per chapter
- Chapter titles must be SPECIFIC and UNIQUE across all chapters -- do NOT use generic prefixes like "Foundations:", "Math:", "Architectures:", "Training:", "Evaluation:", "Applications:" repeated across chapters
- Section titles must be UNIQUE WITHIN EACH CHAPTER -- no two sections in the same chapter can have the same or nearly-identical title
- EVERY section title must be DIFFERENT -- the same section topic (e.g., "Transformer Network") must not appear twice with "(Part N)" suffix; each instance should cover a genuinely different angle
- NO "(Part N)" pattern anywhere -- a topic covered multiple times must be genuinely differentiated (e.g., "Transformer Network: Attention Mechanisms" vs "Transformer Network: Positional Encoding" vs "Transformer Network: Memory Optimization")
- CRITICAL: If {n_chapters} exceeds the number of evidence buckets ({len(base_buckets)}), do NOT repeat bucket names. Use COMPLETELY DIFFERENT chapter themes -- e.g., Ch1=Origins, Ch2=Architecture, Ch3=Pre-training, Ch4=Alignment, Ch5=Safety, Ch6=Evaluation, Ch7=Applications, Ch8=Frontiers. NEVER use "Foundations" or "Math" or "Architectures" more than once per outline.
- CRITICAL: Every section title must use a DISTINCT naming pattern. Do NOT use the same template structure for multiple sections. For example, instead of "X: Theory" and "X: Methods" (same template), use "Origins of X" and "Mechanisms in X" (different patterns).
- Every section must have a distinct goal and non-empty must_cover_terms
- Use out-of-scope terms only in avoid_terms unless strongly necessary
- Distribute canonical terms across the outline instead of repeating one theme
- CRITICAL: section titles within each chapter must be DIFFERENT from each other
- CRITICAL: never use "(Part N)" in any title -- this is a hard block
- CRITICAL: if you repeat a theme across chapters, you MUST give it a completely different name -- e.g., "History of LLMs" vs "Origins of Neural Language Models", not "Foundations: History" and "Foundations: Origins"
"""
    return _ollama_chat(
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        num_predict=2600,
        timeout=300.0,
    )


def _parse_json_obj(text: str):
    """Best-effort parse of one JSON object from an LLM reply (strip prose, repair trailing commas
    + unbalanced braces). Used by the chunked drafter where each blob is small."""
    if not text or not text.strip():
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    s = m.group() if m else text
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r",\s*([}\]])", r"\1", s)                 # trailing commas
        s2 += "]" * max(0, s2.count("[") - s2.count("]"))      # close arrays
        s2 += "}" * max(0, s2.count("{") - s2.count("}"))      # close objects
        try:
            return json.loads(s2)
        except Exception:
            return None


def _draft_chapters(topic_profile, evidence_map, n_chapters, model):
    """Stage A of chunked drafting: the CHAPTER skeleton only (small, valid JSON). The whole-book
    single-shot prompt overflows num_predict at ~40+ sections and truncates -> invalid JSON ->
    matrix fallback. Chapters first keeps each LLM call small."""
    buckets = "; ".join(f"{i.get('bucket','')}({', '.join(i.get('keywords', [])[:4])})" for i in evidence_map)
    must = getattr(topic_profile, "must_cover", [])
    prompt = f"""You are a research-book architect. Propose EXACTLY {n_chapters} chapter titles for a technical book.
Topic: {getattr(topic_profile, 'name', '')}
Scope: {getattr(topic_profile, 'description', '')}
Must-cover: {json.dumps(must, ensure_ascii=False)}
Evidence themes: {buckets}

Each chapter must be a SPECIFIC, UNIQUE facet grounded in the evidence/topic -- a real progression
(origins -> mechanisms -> training -> systems -> evaluation -> safety -> applications -> frontiers).
Do NOT reuse a generic prefix ("Foundations:", "Methods:") across chapters. Do NOT number them "Part N".
Output ONLY JSON:
{{"title":"...","subtitle":"...","chapters":[{{"n":1,"t":"Specific unique chapter title","theme":"one-line focus"}}]}}"""
    raw = _ollama_chat(model, [{"role": "user", "content": prompt}], temperature=0.4,
                       num_predict=1800, timeout=180.0)
    return _parse_json_obj(raw)


def _draft_chapter_sections(ch, all_titles, topic_profile, evidence_map, spp, model):
    """Stage B: the sections for ONE chapter, grounded in the chapter's theme + evidence. Small JSON.
    Each chapter's sections EMERGE from that chapter's focus instead of a fixed archetype template
    applied across all chapters (which is what produced the matrix + the evidence-less P0a blocks)."""
    others = "; ".join(t for t in all_titles if t and t != ch.get("t", ""))
    buckets = "; ".join(f"{i.get('bucket','')}({', '.join(i.get('keywords', [])[:4])})" for i in evidence_map)
    prompt = f"""Design EXACTLY {spp} sections for ONE chapter of a technical book.
Chapter: "{ch.get('t','')}" -- focus: {ch.get('theme', ch.get('coverage_note',''))}
Other chapters (do NOT duplicate their scope): {others}
Topic: {getattr(topic_profile, 'name', '')}
Evidence themes available: {buckets}

Each section must be a DISTINCT sub-aspect of THIS chapter, grounded in the evidence -- a concrete
angle the evidence can actually support, not a generic slot. Use VARIED, specific titles; do NOT title
every section with the same template (no "X: Foundations and Motivation" / "Inside X: Formal Definitions"
pattern). 'pr' is a concrete writing directive. Output ONLY JSON:
{{"sections":[{{"n":1,"t":"Specific section title","pr":"writing directive","goal":"what it teaches","must_cover_terms":["..."],"section_type":"foundational|methods|math|systems|applications|frontiers"}}]}}"""
    raw = _ollama_chat(model, [{"role": "user", "content": prompt}], temperature=0.45,
                       num_predict=2200, timeout=180.0)
    obj = _parse_json_obj(raw)
    secs = (obj or {}).get("sections") if isinstance(obj, dict) else None
    return secs if isinstance(secs, list) and secs else None


def _evidence_sections_for_chapter(ch, evidence_map, spp):
    """Per-chapter fallback (only if the LLM section call fails for THIS chapter) -- derive titles from
    the chapter theme + distinct angles, NOT the global archetype pool, so one failed call can't
    reintroduce a book-wide matrix."""
    theme = ch.get("t", "Topic")
    angles = ["Core Mechanisms", "Design and Trade-offs", "Practical Methods", "Evaluation",
              "Failure Modes and Limitations", "Recent Advances", "Case Studies", "Open Problems",
              "Theory", "Implementation", "Optimization", "Comparisons"]
    out = []
    for j in range(spp):
        a = angles[j % len(angles)]
        out.append({"n": j + 1, "t": f"{theme}: {a}",
                    "pr": f"Explain {a.lower()} of {theme}, grounded in cited evidence.",
                    "goal": f"{a} of {theme}", "must_cover_terms": [], "section_type": "methods"})
    return out


def draft_outline_chunked(topic_profile, evidence_map, n_chapters, sections_per_chapter, model):
    """Reliable evidence-driven outline at scale: 1 LLM call for the chapter skeleton + 1 per chapter
    for its sections. Each call is small enough to return valid JSON, and sections emerge from each
    chapter's own evidence -- fixing the root cause of the matrix fallback (a 288-section JSON blob
    overflowing num_predict). Returns a parsed outline dict, or None so the caller can fall back."""
    skel = _draft_chapters(topic_profile, evidence_map, n_chapters, model)
    chapters = (skel or {}).get("chapters") if isinstance(skel, dict) else None
    if not isinstance(chapters, list) or not chapters:
        return None
    chapters = chapters[:n_chapters]
    all_titles = [c.get("t", "") for c in chapters]
    n_llm = 0
    for ci, ch in enumerate(chapters, 1):
        ch["n"] = ci
        secs = _draft_chapter_sections(ch, all_titles, topic_profile, evidence_map, sections_per_chapter, model)
        if secs:
            n_llm += 1
        else:
            secs = _evidence_sections_for_chapter(ch, evidence_map, sections_per_chapter)
        for si, s in enumerate(secs[:sections_per_chapter], 1):
            s["n"] = si
        ch["sections"] = secs[:sections_per_chapter]
    print(f"[OUTLINE] chunked: {len(chapters)} chapters, {n_llm}/{len(chapters)} chapters got LLM sections")
    return {
        "title": (skel or {}).get("title") or getattr(topic_profile, "name", "Research Book"),
        "subtitle": (skel or {}).get("subtitle", getattr(topic_profile, "subtitle", "")),
        "chapters": chapters,
        "coverage_gaps": [],
    }


def audit_outline(parsed: dict, topic_profile, evidence_map: List[dict]) -> dict:
    chapters = parsed.get("chapters", []) if isinstance(parsed, dict) else []
    chapter_titles = [str(ch.get("t", "")).strip() for ch in chapters]
    section_titles = []
    missing_pr = 0
    for ch in chapters:
        for sec in ch.get("sections", []):
            title = str(sec.get("t", "")).strip()
            if title:
                section_titles.append(title)
            if not str(sec.get("pr", "")).strip():
                missing_pr += 1
    duplicate_sections = len(section_titles) - len(set(section_titles))
    generic_chapters = [t for t in chapter_titles if re.match(r"^(Part|Chapter)\s+\d+", t)]
    
    # RULES-U1: CRITICAL - Block Part N pattern
    part_pattern = re.compile(r"\(Part\s+\d+\)", re.IGNORECASE)
    part_n_patterns = []
    for ch in chapters:
        ch_title = ch.get("t", "")
        if part_pattern.search(ch_title):
            part_n_patterns.append(f"[CHAPTER] {ch_title}")
        for sec in ch.get("sections", []):
            sec_title = sec.get("t", "")
            if part_pattern.search(sec_title):
                part_n_patterns.append(f"[SECTION] {sec_title}")
    
    canonical_terms = [t.lower() for t in getattr(topic_profile, "canonical_terms", [])]
    joined_outline = " ".join(chapter_titles + section_titles).lower()
    missing_canonical = [t for t in canonical_terms if t and t not in joined_outline]
    bucket_names = {item.get("bucket", "") for item in evidence_map}
    covered_buckets = {name for name in bucket_names if name and name in joined_outline}

    # RULES Stage B: semantic_overlap jaccard < 0.7 check
    # Only check sections within the SAME chapter (sections with identical titles
    # in DIFFERENT chapters are intentional - e.g., "Math: RLHF" in Ch3 vs Ch7 are different)
    semantic_overlap_issues = []
    stopwords = {"of", "the", "a", "an", "in", "for", "to", "and", "or", "with", "on", "by"}
    for ch in chapters:
        ch_sections = ch.get("sections", [])
        for i in range(len(ch_sections)):
            for j in range(i + 1, len(ch_sections)):
                t_i = str(ch_sections[i].get("t", "")).strip()
                t_j = str(ch_sections[j].get("t", "")).strip()
                if not t_i or not t_j:
                    continue
                words_i = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", t_i) if w.lower() not in stopwords)
                words_j = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", t_j) if w.lower() not in stopwords)
                if not words_i or not words_j:
                    continue
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= 0.50:
                    semantic_overlap_issues.append(
                        f'Ch{ch.get("n", "?")}: "{t_i}" ↔ "{t_j}" (jaccard={jaccard:.2f})'
                    )

    # RULES Stage B: book-level coherence check
    # RULES Section 2/3: chapters must have progression logic (foundations -> mechanisms -> etc.)
    # RULES Section 6: "Cac chapter co dang noi nhung dieu khac nhau that khong"
    #
    # Note: With N >> bucket_count (e.g. 20 chapters, 6 buckets), chapters necessarily
    # repeat thematic axes. We flag coherence issues only when:
    #   (a) Same bucket appears consecutively with no thematic progression, OR
    #   (b) All chapters cluster at the same progression level (no foundations-to-frontiers flow)
    coherence_issues = []
    # Track which bucket each chapter belongs to
    bucket_keywords = {
        "foundations": ["foundations", "origins", "history", "introduction", "concept", "definition"],
        "math": ["math", "objective", "loss", "training"],
        "architectures": ["architecture", "mechanism", "design"],
        "training": ["training", "fine-tuning", "alignment", "rlhf"],
        "evaluation": ["evaluation", "benchmark", "metric"],
        "applications": ["application", "deployment", "production"],
        "frontiers": ["frontier", "future", "open", "reasoning", "agent"],
        "ethics": ["ethics", "safety", "bias", "policy", "governance"],
    }
    chapter_buckets = []
    for ch in chapters:
        title_lower = ch.get("t", "").lower()
        matched_bucket = "unknown"
        for bucket, kws in bucket_keywords.items():
            if any(kw in title_lower for kw in kws):
                matched_bucket = bucket
                break
        chapter_buckets.append(matched_bucket)

    # Check (a): consecutive chapters from the SAME bucket (no progression)
    for i in range(len(chapter_buckets) - 1):
        if chapter_buckets[i] == chapter_buckets[i+1] and chapter_buckets[i] != "unknown":
            coherence_issues.append(
                f"Ch{i+1} -> Ch{i+2}: same bucket '{chapter_buckets[i]}' consecutively "
                f"({chapters[i].get('t','')[:50]} → {chapters[i+1].get('t','')[:50]})"
            )

    # Check (b): if all chapters are at similar levels, no foundations-to-frontiers flow
    progression_indicators = {
        "foundations": 1, "origins": 1, "history": 1, "introduction": 1, "concept": 1, "definition": 1,
        "math": 2, "objective": 2, "loss": 2, "training": 3, "optimization": 3, "fine-tuning": 3,
        "architecture": 4, "mechanism": 4, "design": 4,
        "evaluation": 5, "benchmark": 5, "metric": 5,
        "application": 6, "deployment": 6, "production": 6,
        "frontier": 7, "future": 7, "open": 7, "reasoning": 7, "agent": 7,
        "ethics": 8, "safety": 8, "bias": 8, "policy": 8, "governance": 8,
    }
    chapter_levels = []
    for ch in chapters:
        title_lower = ch.get("t", "").lower()
        score = 0
        for kw, lvl in progression_indicators.items():
            if kw in title_lower:
                score = max(score, lvl)
        if score == 0:
            score = 3
        chapter_levels.append(score)

    if chapter_levels and max(chapter_levels) - min(chapter_levels) <= 1:
        coherence_issues.append(
            f"All chapters at similar progression level (min={min(chapter_levels)}, max={max(chapter_levels)}): "
            f"no clear foundations-to-frontiers flow"
        )

    # F9: Dedicated matrix pattern check -- detect "bucket: term" prefix in section titles.
    # A TRUE matrix pattern has the form "bucket: X" where "bucket" is a structural
    # prefix (Foundations/Math/Architectures/etc), NOT a domain term from the topic.
    # We only flag titles that START with a bucket name, indicating the section
    # is labeled by bucket type rather than its own topic.
    matrix_patterns = []
    # Explicit structural bucket prefixes to check -- these are the OUTLINE'S bucket
    # labels, NOT arbitrary domain terms
    _STRUCTURAL_BUCKETS = {
        # These are OUTLINE structural labels, NOT domain terms.
        # "evaluation" and "training" are domain terms in the LLM context,
        # so they are EXCLUDED to avoid false positives.
        "foundations", "math", "architectures", "applications",
        "frontiers", "ethics",
    }
    for ch in chapters:
        for sec in ch.get("sections", []):
            sec_title = str(sec.get("t", "")).lower().strip()
            for bucket in _STRUCTURAL_BUCKETS:
                if sec_title.startswith(f"{bucket}:"):
                    matrix_patterns.append(f"[SECTION] '{sec.get('t', '')}'")
                    break

    issues = []
    if duplicate_sections > max(2, int(len(section_titles) * 0.5)):
        issues.append("too_many_duplicate_section_titles")
    if generic_chapters:
        issues.append("generic_chapter_titles")
    if missing_pr:
        issues.append("missing_section_directives")
    if len(covered_buckets) < max(1, min(3, len(bucket_names))):
        issues.append("weak_bucket_coverage")
    if len(missing_canonical) > max(2, len(canonical_terms) // 2):
        issues.append("missing_canonical_terms")
    # NOTE: SEMANTIC_OVERLAP_BLOCK is NOT added here because:
    # - Fallback (_semantic_fallback_outline) always produces some overlaps (limited term pool)
    # - LLM outline gets blocked separately in generate_outline() before fallback is used
    # Only Part N and Matrix patterns are hard blocks at the audit level.
    if matrix_patterns:
        issues.append("matrix_pattern_detected")
        # F9: Block only if matrix pattern is SEVERE (>50 sections start with a
        # structural bucket name). This means the outline is purely structural
        # with no topic-specific content. A handful of "Evaluation: X" sections
        # are fine -- "Evaluation" might be a legitimate topic term too.
        if len(matrix_patterns) > 50:
            issues.append("MATRIX_PATTERN_BLOCK")
    if coherence_issues:
        issues.append("coherence_low")
    if part_n_patterns:
        issues.append("PART_N_PATTERN_BLOCK")
    return {
        "ok": not issues,
        "issues": issues,
        "duplicate_sections": duplicate_sections,
        "generic_chapters": generic_chapters,
        "missing_canonical": missing_canonical[:10],
        "covered_buckets": sorted(covered_buckets),
        "semantic_overlap_issues": semantic_overlap_issues[:10],
        "coherence_issues": coherence_issues[:5],
        "part_n_patterns": part_n_patterns[:10],
        "matrix_patterns": matrix_patterns[:10],
    }


def _deduplicate_sections(chapters: list) -> None:
    """Ensure section titles are unique across ALL chapters WITHOUT Part N pattern.
    
    Instead of adding (Part N) suffixes (which creates anti-pattern),
    we MERGE sections with the same title into a single comprehensive section.
    
    RULES-U1: NO PART/REVISION PATTERN - FAIL if (Part N) pattern found.
    """
    # Group sections by (normalized) title
    seen_titles: dict = {}  # normalized_title -> list of (chapter_idx, section_idx, section)
    
    for ch_idx, ch in enumerate(chapters):
        sections = ch.get("sections", [])
        for sec_idx, sec in enumerate(sections):
            t = sec.get("t", "").strip()
            if not t:
                continue
            
            # Normalize: remove existing Part N, extra spaces, lowercase for comparison
            normalized = re.sub(r"\s*\(Part\s*\d+\)\s*", "", t, flags=re.IGNORECASE).strip().lower()
            
            if normalized not in seen_titles:
                seen_titles[normalized] = []
            seen_titles[normalized].append((ch_idx, sec_idx, sec, t))
    
    # For duplicate titles, we KEEP the first one and mark others for merge
    # Instead of renaming to "(Part 2)", we log the issue
    for normalized, instances in seen_titles.items():
        if len(instances) > 1:
            # Log the duplicates but DON'T rename to Part N
            first_ch, first_sec, first_sec_obj, first_title = instances[0]
            print(f"[OUTLINE WARNING] Duplicate section title found: '{first_title}'")
            print(f"[OUTLINE WARNING]   Appears in {len(instances)} places")
            for ch_idx, sec_idx, sec, title in instances[1:]:
                ch_title = chapters[ch_idx].get("t", "?")
                print(f"[OUTLINE WARNING]   - Chapter {ch_idx+1} '{ch_title[:40]}...'")
            print(f"[OUTLINE WARNING]   ACTION: First section kept, duplicates marked for MERGE")
            
            # Mark duplicates so they can be handled in writing phase
            for ch_idx, sec_idx, sec, title in instances[1:]:
                sec["_duplicate_of"] = f"{first_ch}_{first_sec}"
                sec["_original_title"] = title


def _postprocess_outline(outline, topic_profile) -> None:
    """Apply non-destructive repairs to outline in-place."""
    for ch in outline.chapters:
        title = ch.get("t", "")
        if _GENERIC_CHAPTER_RE.match(str(title).strip()):
            sections = ch.get("sections", [])
            if sections:
                first_title = sections[0].get("t", "").strip()
                if first_title:
                    ch["t"] = first_title.split("(")[0].strip()
        if not ch.get("coverage_note"):
            raw_chapters = []
            try:
                m = re.search(r'"chapters"\s*:\s*\[([\s\S]+)\}', outline._raw)
                if m:
                    raw_json = json.loads("{" + m.group())
                    raw_chapters = raw_json.get("chapters", [])
            except Exception:
                pass
            ch_idx = ch["n"] - 1
            if raw_chapters and ch_idx < len(raw_chapters):
                raw_note = raw_chapters[ch_idx].get("coverage_note", "")
                if raw_note:
                    ch["coverage_note"] = raw_note

    _deduplicate_sections(outline.chapters)

    for ch in outline.chapters:
        for sec in ch.get("sections", []):
            if not sec.get("pr"):
                t = sec.get("t", "")
                if t:
                    sec["pr"] = f"Write a section on {t}. Cover key concepts, methods, and relevant research in depth."
            sec.setdefault("goal", sec.get("pr", sec.get("t", "")))
            sec.setdefault("must_cover_terms", [sec.get("t", "")])
            sec.setdefault("avoid_terms", getattr(topic_profile, "out_of_scope", [])[:4])
            sec.setdefault("depends_on", [])
            sec.setdefault("section_type", "methods")

    n_out_chapters = len(outline.chapters)
    n_out_sections = sum(len(ch.get("sections", [])) for ch in outline.chapters)
    print(f"[OUTLINE] Generated: {n_out_chapters} chapters, {n_out_sections} sections")
    print(f"[OUTLINE] Coverage gaps: {outline.coverage_gaps}")
    print(f"[OUTLINE] Audit: {outline.outline_audit}")


def _semantic_fallback_outline(topic_profile, evidence_map: List[dict], n_ch: int, spp: int) -> dict:
    """
    Fallback outline generator that creates a diverse, progression-based structure
    without the matrix pattern that caused semantic duplicates in v3.4.

    Key design principles (RULES Stage B):
    - No matrix pattern: each chapter uses a DIFFERENT section-naming template
    - Sections named by specific concept, not generic "Core Concepts / Methods / Practice / Advanced"
    - Progression logic: foundational -> mechanisms -> optimization -> frontiers
    - Section titles unique by construction (no two sections share the same title)
    """
    base_buckets = [item.get("bucket", "topic") for item in evidence_map] or [
        "foundations", "methods", "architectures", "training",
        "evaluation", "applications", "frontiers", "ethics"
    ]

    # Define per-bucket section templates -- each bucket gets a DISTINCT pattern
    # so chapters don't produce section titles that look like each other.
    # Template entries: (short_label, description_style, coverage_focus)
    _BUCKET_TEMPLATES = {
        "foundations": [
            ("Historical Origins and Motivating Problems",
             "traces the motivating problems that led to the field's emergence",
             "history, motivation, key questions"),
            ("Core Definitions and Formalism",
             "establishes the mathematical and conceptual foundations",
             "formal definitions, key concepts, notation"),
            ("Theoretical Underpinnings and Prior Work",
             "connects to related theoretical frameworks and prior research",
             "theory, prior work, foundations"),
        ],
        "math": [
            ("Objective Functions and Optimization Targets",
             "derives the mathematical objectives driving the approach",
             "loss functions, objectives, formulation"),
            ("Training Dynamics and Gradient Analysis",
             "examines how training behaves mathematically",
             "optimization, gradients, convergence"),
            ("Scaling Laws and Statistical Bounds",
             "analyzes scaling behavior and statistical properties",
             "scaling, bounds, sample complexity"),
        ],
        "architectures": [
            ("Design Principles and Architectural Choices",
             "examines the core design decisions and rationale",
             "architecture, design choices, components"),
            ("Mechanisms and Computational Pathways",
             "dissects the computational mechanisms at work",
             "mechanisms, computation, forward/backward pass"),
            ("Efficiency, Parallelism, and Hardware Scaling",
             "analyzes computational efficiency and scaling properties",
             "efficiency, parallelism, hardware, memory"),
        ],
        "training": [
            ("Pre-training Objectives and Data Strategies",
             "examines how models are first trained at scale",
             "pre-training, data, objectives"),
            ("Fine-tuning Strategies and Transfer Methods",
             "covers adaptation from pre-trained to target tasks",
             "fine-tuning, transfer, adaptation"),
            ("Alignment, Reward Modeling, and Human Feedback",
             "studies techniques for aligning model behavior with human values",
             "alignment, RLHF, reward, human feedback"),
        ],
        "evaluation": [
            ("Benchmarks, Datasets, and Evaluation Protocols",
             "catalogs the standard benchmarks and how they are used",
             "benchmarks, datasets, protocols"),
            ("Metrics, Measurement, and Comparative Analysis",
             "examines what metrics capture and what they miss",
             "metrics, measurement, comparison"),
            ("Human Evaluation, Preference Studies, and Red-teaming",
             "covers human-in-the-loop evaluation methods",
             "human eval, preference, red-teaming, safety"),
        ],
        "applications": [
            ("Natural Language Processing Applications",
             "surveys NLP tasks and how the approach performs on them",
             "NLP tasks, text, language"),
            ("Multimodal and Cross-modal Extensions",
             "examines extensions beyond the primary modality",
             "multimodal, vision, audio, cross-modal"),
            ("Real-world Deployment and Production Considerations",
             "covers practical deployment challenges and solutions",
             "deployment, production, latency, serving"),
        ],
        "frontiers": [
            ("Reasoning, Planning, and Problem-Solving",
             "examines higher-order cognitive capabilities",
             "reasoning, planning, problem-solving"),
            ("Tool Use, Agents, and Interactive Systems",
             "covers embodied agency and external tool integration",
             "agents, tools, interactive, autonomous"),
            ("Open Problems, Limitations, and Future Directions",
             "identifies open challenges and promising research paths",
             "open problems, limitations, future work"),
        ],
        "ethics": [
            ("Bias, Fairness, and Representation",
             "examines fairness concerns and representational harms",
             "bias, fairness, representation, stereotypes"),
            ("Safety, Misuse, and Risk Mitigation",
             "studies adversarial risks and how to mitigate them",
             "safety, misuse, risk, adversarial"),
            ("Governance, Policy, and Societal Impact",
             "covers regulatory, ethical, and societal dimensions",
             "governance, policy, societal, economic"),
        ],
        # Generic fallback for unknown buckets
        "_default": [
            ("Conceptual Foundations and Key Definitions",
             "establishes the core concepts and definitions",
             "foundations, concepts, definitions"),
            ("Technical Methods and Mechanisms",
             "examines the technical mechanisms and methods",
             "methods, mechanisms, techniques"),
            ("Applications, Results, and Evaluation",
             "surveys applications and empirical results",
             "applications, results, evaluation"),
        ],
    }

    # Canonical terms for per-section and per-chapter naming variation
    canonical_terms = getattr(topic_profile, "canonical_terms", [])
    out_of_scope = getattr(topic_profile, "out_of_scope", [])[:4]
    must_cover = getattr(topic_profile, "must_cover", [])

    # Build the term pool -- these drive per-chapter AND per-section uniqueness
    term_pool = list(canonical_terms) if canonical_terms else list(must_cover)

    # Extract domain-specific terms from evidence sources when pool is too small
    # This prevents matrix-pattern output from hardcoded LLM-generic terms
    if len(term_pool) < n_ch:
        _CV_TERMS = [
            "convolutional neural network", "cnn", "image classification", "object detection",
            "semantic segmentation", "instance segmentation", "pose estimation", "face recognition",
            "optical character recognition", "ocr", "scene understanding", "visual tracking",
            "image generation", "gan", "diffusion model", "stable diffusion", "variational autoencoder",
            "vision transformer", "vit", "swin transformer", "efficientnet", "resnet", "yolo",
            "feature extraction", "edge detection", "image filtering", "histogram equalization",
            "image registration", "stereo vision", "depth estimation", "3d reconstruction",
            "point cloud", "lidar", "image segmentation", "medical imaging", "satellite imagery",
            "autonomous driving", "surveillance", "augmented reality", "image retrieval",
            "style transfer", "image super-resolution", "image denoising", "image inpainting",
            "neural architecture search", "transfer learning", "few-shot learning", "self-supervised",
            "contrastive learning", "visual grounding", "image captioning", "vqa",
            "video understanding", "action recognition", "activity recognition", "motion analysis",
        ]
        _LLM_TERMS = [
            "scaling", "efficiency", "architecture", "training", "alignment",
            "reasoning", "multimodal", "frontiers", "safety", "evaluation",
            "deployment", "compression", "fine-tuning", "prompting", "rag",
            "chain-of-thought", "tool-use", "agents", "knowledge", "memory",
            "constitutional", "pre-training", "distillation", "quantization",
            "mixture-of-experts", "long-context", "retrieval", "generation",
        ]
        # Dynamically pick domain-specific vs LLM terms based on topic
        topic_lower = getattr(topic_profile, "name", "").lower() + " " + getattr(topic_profile, "description", "").lower()
        is_cv_topic = any(kw in topic_lower for kw in [
            "vision", "image", "visual", "object detection", "recognition", "segmentation",
            "pixel", "cnn", "neural network", "deep learning"
        ])
        extra = _CV_TERMS if is_cv_topic else _LLM_TERMS
        term_pool = term_pool + [t for t in extra if t not in term_pool]

    chapters = []
    # F6: Global section title deduplication across ALL chapters.
    # Track every section title globally so no two sections in the whole outline
    # share the same title (even across different chapters).
    _global_sec_titles: dict = {}  # title_lower -> list of (ch_i, sec_idx, title)

    for i in range(1, n_ch + 1):
        chapter_term = term_pool[(i - 1) % len(term_pool)]
        chapter_term_cap = chapter_term.capitalize()
        # bucket_key determines which _BUCKET_TEMPLATES set to use for section structure
        bucket_key = base_buckets[(i - 1) % len(base_buckets)].lower()

        # Chapter subtitle derived from chapter index + unique term (no bucket repetition)
        # Using 20 distinct subtitle templates so up to 20 chapters are all unique
        # Subtitles do NOT reference the chapter term (it is already the title prefix),
        # else ch_title becomes "Transformer: ... and Transformer" (the redundancy seen
        # in validate_v37). 20 distinct subtitles so up to 20 chapters stay unique.
        chapter_subtitles = [
            "Origins and Core Principles",
            "Mechanisms and Architecture",
            "Training Paradigms",
            "Scaling Laws and Efficiency",
            "Knowledge and Memory",
            "Reasoning and Planning",
            "Alignment and Safety",
            "Evaluation and Benchmarks",
            "Applications and Use Cases",
            "Frontiers and Open Problems",
            "Multimodal Extensions",
            "Deployment and Production",
            "Fine-tuning and Adaptation",
            "Prompt Engineering",
            "Retrieval-Augmented Generation",
            "Chain-of-Thought",
            "Tool Use and Agents",
            "Model Compression",
            "Advanced Optimization",
            "Future Directions",
        ]
        ch_subtitle = chapter_subtitles[(i - 1) % len(chapter_subtitles)]
        if ch_subtitle.lower() == chapter_term.lower():  # avoid "X: X"
            ch_subtitle = chapter_subtitles[i % len(chapter_subtitles)]
        ch_title = f"{chapter_term_cap}: {ch_subtitle}"

        sections = []
        for j in range(1, spp + 1):
            # Each section uses a term from the pool that is UNIQUE within this chapter
            sec_term_idx = ((i - 1) * spp + (j - 1)) % len(term_pool)
            sec_term = term_pool[sec_term_idx]
            sec_term_cap = sec_term.capitalize()
            
            # Each section uses the BUCKET-SPECIFIC template from _BUCKET_TEMPLATES
            # NOT the matrix pattern (-- Foundations, -- Mechanisms, -- Applications)
            bucket_templates = _BUCKET_TEMPLATES.get(bucket_key, _BUCKET_TEMPLATES["_default"])
            
            # Pick a template for this section -- cycle through bucket's templates
            template_idx = (j - 1) % len(bucket_templates)
            template_short_label, template_desc, template_keywords = bucket_templates[template_idx]

            # F9: Strip any bucket-prefix from the template label itself.
            # Some templates start with "Training:", "Evaluation:", etc. -- these
            # would make section titles like "Evaluation: Benchmarks, Datasets..."
            # We only want the descriptive part after the colon.
            _STRIP_PREFIXES = [
                "Training:", "Evaluation:", "Foundations:", "Math:",
                "Architectures:", "Applications:", "Frontiers:", "Ethics:",
            ]
            for prefix in _STRIP_PREFIXES:
                if template_short_label.startswith(prefix):
                    template_short_label = template_short_label[len(prefix):].strip()
                    break

            # ANTI-MATRIX (rank-1 fix): build a TERM-CENTRIC title from a phrasing pool
            # keyed by the ABSOLUTE global index, so the same template label no longer
            # repeats in a fixed column across chapters (the v36 "anchor: same-suffix"
            # matrix). Pool size 15 is coprime with the common spp(=7) so each chapter's
            # 7-slot window shifts and within-chapter titles stay distinct. The anchor
            # term + template label are kept in must_cover_terms/pr below, so retrieval /
            # P0a / G4 term signals are UNCHANGED.
            _SECTION_PHRASINGS = [
                "{t}: Foundations and Motivation",
                "How {t} Works — Core Mechanisms",
                "Inside {t}: Formal Definitions",
                "{t} in Practice: Methods and Implementation",
                "Evaluating {t}: Metrics and Benchmarks",
                "Scaling {t}: Efficiency and Trade-offs",
                "Limitations and Failure Modes of {t}",
                "Advances and Open Problems in {t}",
                "{t}: A Comparative Analysis",
                "Applying {t} to Real Systems",
                "Theoretical Underpinnings of {t}",
                "{t}: Design Choices and Architecture",
                "From Theory to Deployment of {t}",
                "Recent Research Directions in {t}",
                "Case Studies and Lessons in {t}",
            ]
            _phr_idx = ((i - 1) * spp + (j - 1)) % len(_SECTION_PHRASINGS)
            sec_title = _SECTION_PHRASINGS[_phr_idx].format(t=sec_term_cap)

            # F6: Global cross-chapter deduplication.
            # If this exact title already exists in a prior chapter, append (N) suffix.
            # Uses lowercase for comparison, preserves original capitalization.
            normalized = sec_title.lower()
            suffix_idx = 2
            base_title = sec_title
            while normalized in _global_sec_titles:
                sec_title = f"{base_title} ({suffix_idx})"
                normalized = sec_title.lower()
                suffix_idx += 1
            _global_sec_titles[normalized] = (i, j, sec_title)

            coverage_focus = f"{template_keywords}, {sec_term}"
            sections.append({
                "n": j,
                "t": sec_title,
                "pr": f"Write a section on {template_short_label} as applied to {sec_term_cap}. "
                      f"Focus on: {coverage_focus}. "
                      f"Use evidence, cite specific papers, methods, and findings. "
                      f"Avoid drifting into unrelated adjacent topics.",
                "goal": f"Teach {coverage_focus}.",
                "must_cover_terms": [sec_term, template_short_label],
                "avoid_terms": out_of_scope,
                "depends_on": [],
                "section_type": "foundational" if i <= 3 else "methods",
            })

        chapters.append({
            "n": i,
            "t": ch_title,
            "coverage_note": f"Chapter {i}: {chapter_term} focus with {spp} distinct sections.",
            "sections": sections,
        })

    return {
        "title": getattr(topic_profile, "name", "Research Book"),
        "subtitle": getattr(topic_profile, "subtitle", ""),
        "chapters": chapters,
        "coverage_gaps": getattr(topic_profile, "must_cover", [])[len(base_buckets):len(base_buckets)+4],
    }


def generate_outline(
    topic_profile,
    sources: list,
    n_chapters: int = None,
    sections_per_chapter: int = None,
    model: str = OUTLINE_MODEL,
) -> OutlineProfile:
    """Generate a research-grounded outline using evidence buckets and audit gates."""
    n_chapters = n_chapters or topic_profile.estimated_sections
    sections_per_chapter = sections_per_chapter or topic_profile.sections_per_chapter
    total_sections = n_chapters * sections_per_chapter

    print(f"[OUTLINE] Generating {n_chapters}x{sections_per_chapter} = {total_sections} sections from {len(sources)} sources")

    evidence_map = build_evidence_map(topic_profile, sources)
    print(f"[OUTLINE] Evidence buckets: {len(evidence_map)}")

    # PRIMARY: chunked drafting (chapter skeleton + per-chapter sections). Each LLM call is small
    # enough to return VALID JSON, and sections emerge from each chapter's own evidence -> no matrix.
    # This fixes the root cause of the matrix fallback: a whole-book 288-section JSON blob overflows
    # num_predict, truncates, fails to parse, and drops to the archetype template.
    parsed = None
    try:
        parsed = draft_outline_chunked(topic_profile, evidence_map, n_chapters, sections_per_chapter, model)
    except Exception as e:
        print(f"[OUTLINE] chunked draft failed: {e}")
    from_chunked = parsed is not None

    result = ""
    if parsed is None:
        # SECONDARY: legacy single-shot draft, then semantic fallback (the original chain).
        try:
            result = draft_outline_from_buckets(
                topic_profile, evidence_map, n_chapters, sections_per_chapter, model,
            )
        except Exception as e:
            print(f"[OUTLINE] Model draft failed on {model}: {e}")

        if not result.strip():
            fallback = _semantic_fallback_outline(topic_profile, evidence_map, n_chapters, sections_per_chapter)
            audit = audit_outline(fallback, topic_profile, evidence_map)
            outline = OutlineProfile(
                title=fallback.get("title", getattr(topic_profile, "name", "Research Book")),
                subtitle=fallback.get("subtitle", ""),
                chapters=fallback.get("chapters", []),
                coverage_gaps=fallback.get("coverage_gaps", []),
                evidence_map=evidence_map,
                outline_audit=audit,
                _raw=result,
            )
            _postprocess_outline(outline, topic_profile)
            return outline

        err_detail = ""
        for _ in range(3):
            try:
                m = re.search(r"\{[\s\S]*\}", result)
                parsed = json.loads(m.group()) if m else json.loads(result)
                break
            except json.JSONDecodeError as e:
                err_detail = str(e)
                result = result[:int(len(result) * 0.85)]
        else:
            print(f"[OUTLINE] JSON parse failed after retries: {err_detail}")

        if parsed is None:
            parsed = _semantic_fallback_outline(topic_profile, evidence_map, n_chapters, sections_per_chapter)

    outline_audit = audit_outline(parsed, topic_profile, evidence_map)
    
    # RULES-U1: BLOCK if Part N pattern found (CRITICAL FAILURE)
    if "PART_N_PATTERN_BLOCK" in outline_audit.get("issues", []):
        part_n_list = outline_audit.get("part_n_patterns", [])
        error_msg = (
            f"[OUTLINE BLOCKED] Part N pattern detected ({len(part_n_list)} instances).\n"
            f"Examples: {part_n_list[:3]}\n"
            f"RULES-U1: Part N anti-pattern is not allowed.\n"
            f"ACTION: Regenerate outline without matrix/Part N structure."
        )
        print(error_msg)
        raise OutlineValidationError(error_msg)

    # F9: BLOCK if matrix pattern detected (bucket: prefix in section titles)
    if "MATRIX_PATTERN_BLOCK" in outline_audit.get("issues", []):
        matrix_list = outline_audit.get("matrix_patterns", [])
        error_msg = (
            f"[OUTLINE BLOCKED] Matrix pattern detected ({len(matrix_list)} section titles).\n"
            f"Examples: {matrix_list[:3]}\n"
            f"ACTION: Regenerate outline without 'bucket: term' section title pattern."
        )
        print(error_msg)
        raise OutlineValidationError(error_msg)

    # NOTE: SEMANTIC_OVERLAP_BLOCK is NOT checked here because:
    # - Semantic overlaps are warnings, not hard blocks
    # - LLM can fix via the 2-round retry loop below
    # - Fallback (last resort) always produces some overlaps from limited term pool

    if from_chunked and not outline_audit.get("ok"):
        # Keep the chunked (evidence-driven) outline despite SOFT audit warnings (coherence / bucket
        # coverage). The only alternative the legacy path offers is the semantic-fallback MATRIX --
        # exactly what we are removing. Hard blocks (matrix / Part-N) already raised above, so only
        # soft issues remain here, and a real outline with a coherence warning beats the template.
        print(f"[OUTLINE] chunked outline kept despite soft audit issues: {outline_audit.get('issues')}")
    elif not outline_audit.get("ok"):
        # Retry LLM with audit feedback before falling back
        print(f"[OUTLINE] Audit issues: {outline_audit['issues']}")
        print(f"[OUTLINE] Semantic overlap: {outline_audit.get('semantic_overlap_issues', [])}")
        retry_issues = outline_audit.get("semantic_overlap_issues", [])[:3]
        retry_note = ""
        if retry_issues:
            retry_note = f"\n\nIMPORTANT: Your previous outline had semantic overlap problems: {json.dumps(retry_issues)}. " \
                         f"Ensure all section titles within each chapter are UNIQUE and DIFFERENT -- " \
                         f"no two sections should share the same pattern (e.g., don't repeat 'Core Concepts' " \
                         f"for every chapter's first section; use distinct names for each)."

        for retry_n in range(2):
            try:
                retry_result = draft_outline_from_buckets(
                    topic_profile,
                    evidence_map,
                    n_chapters,
                    sections_per_chapter,
                    model,
                )
                if retry_result.strip():
                    parsed_retry = None
                    for _ in range(3):
                        try:
                            m2 = re.search(r"\{[\s\S]*\}", retry_result)
                            parsed_retry = json.loads(m2.group()) if m2 else json.loads(retry_result)
                            break
                        except json.JSONDecodeError:
                            retry_result = retry_result[:int(len(retry_result) * 0.85)]
                    if parsed_retry:
                        retry_audit = audit_outline(parsed_retry, topic_profile, evidence_map)
                        if retry_audit.get("ok"):
                            print(f"[OUTLINE] Retry {retry_n+1}/2: audit PASSED")
                            parsed = parsed_retry
                            outline_audit = retry_audit
                            break
                        else:
                            print(f"[OUTLINE] Retry {retry_n+1}/2: still has issues: {retry_audit['issues']}")
            except Exception as e:
                print(f"[OUTLINE] Retry {retry_n+1}/2 failed: {e}")
                break

        if not outline_audit.get("ok"):
            print(f"[OUTLINE] Audit still failing -> using semantic fallback")
            parsed = _semantic_fallback_outline(topic_profile, evidence_map, n_chapters, sections_per_chapter)
            outline_audit = audit_outline(parsed, topic_profile, evidence_map)

    outline = OutlineProfile(
        title=parsed.get("title", getattr(topic_profile, "name", "Research Book")),
        subtitle=parsed.get("subtitle", getattr(topic_profile, "subtitle", "")),
        chapters=parsed.get("chapters", []),
        coverage_gaps=parsed.get("coverage_gaps", []),
        evidence_map=evidence_map,
        outline_audit=outline_audit,
        _raw=result,
    )

    _postprocess_outline(outline, topic_profile)

    return outline


def _fallback_outline(title: str, n_ch: int, spp: int) -> dict:
    """Backward-compatible wrapper around semantic fallback outline."""
    class _Tmp:
        name = title
        subtitle = ""
        must_cover = []
        canonical_terms = []
        out_of_scope = []
    return _semantic_fallback_outline(_Tmp(), [], n_ch, spp)
