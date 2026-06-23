"""
Stage 2: DEEP INVESTIGATION
Per-section research that discovers and iterates.

Input:  section spec + topic context + all prior sections (for cross-ref)
Output: SectionResult = {
    content, sources[], grounding_score,
    topic_relevance_score,
    new_concepts: [],
    citation_markers: [],
  }

Loop: query_gen → gather → adequacy gate → rank → write → verify → topic gate
       If grounding/topic relevance < threshold: refine_queries → repeat

Key differences from v2:
  - Each section is driven by a structured spec, not only a loose prompt
  - New concepts are flagged for outline expansion
  - Cross-section overlap is checked before accepting a section
  - Topic drift is rejected even when citations exist
"""
import httpx, json, re, time
from dataclasses import dataclass, field
from typing import List, Optional, Callable

OLLAMA_BASE = "http://localhost:11434"
TIMEOUT = 300.0

# Import research layer components
import sys as _sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))


@dataclass
class SectionSpec:
    title: str
    prompt: str = ""
    goal: str = ""
    must_cover_terms: List[str] = field(default_factory=list)
    must_cite: List[str] = field(default_factory=list)
    avoid_terms: List[str] = field(default_factory=list)
    prior_dependencies: List[str] = field(default_factory=list)
    success_checks: List[str] = field(default_factory=list)
    section_type: str = "methods"


@dataclass
class SectionResult:
    content: str = ""
    sources: List = field(default_factory=list)
    grounding_score: float = 0.0
    topic_relevance_score: float = 0.0
    n_citations: int = 0
    new_concepts: List[str] = field(default_factory=list)
    research_rounds: int = 0
    citation_markers: List[str] = field(default_factory=list)
    quality: str = "ok"
    cross_ref_count: int = 0  # GATE-6: Number of cross-references to prior sections


def _ollama_chat(model: str, messages: list, temperature: float = 0.7,
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


def _concept_decomposition(text: str) -> List[str]:
    """Extract named concepts from a text using simple heuristics.

    Better than nothing; full extraction would need an LLM call.
    """
    # Named entities: capitalized multi-word terms
    patterns = [
        r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+\b",  # Multi-word proper nouns
        r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase
        r"\b(?:Transformer|Attention|GPT|LLM|BERT|RLHF|LoRA|RAG|DPO)\b",
        r"\b\w+(?:-\w+){{1,3}}\b",  # Hyphenated terms
    ]
    concepts = []
    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            c = m.group()
            if c.lower() not in seen and len(c) > 3:
                seen.add(c.lower())
                concepts.append(c)
    return list(set(concepts))[:20]


def _build_section_spec(section_prompt: str, section_title: str, section_meta: Optional[dict] = None,
                       topic_context: str = "") -> SectionSpec:
    meta = section_meta or {}
    must_cover = [t for t in meta.get("must_cover_terms", []) if t]
    if not must_cover:
        # P0a fix: if outline didn't provide must_cover terms, extract them from topic_context.
        # This ensures P0a's evidence gate checks against the right domain even when
        # archetype queries are wrong (e.g. "Self-Attention" routed to "reasoning").
        # Uses module-level `re` -- do NOT shadow with a local `import re`.
        if topic_context:
            # Pull out canonical terms and key phrases from topic context
            # Format: "Book: X -- description | Canonical terms: a, b, c | Must cover: d, e"
            lines = topic_context.split("\n")
            for line in lines:
                if "canonical terms:" in line.lower():
                    terms = re.findall(r'\b[A-Z][a-z][^\s,]+(?:\s+[A-Z][a-z][^\s,]+){0,2}\b', line)
                    must_cover.extend([t for t in terms if t.lower() not in {c.lower() for c in must_cover}])
                elif "must cover:" in line.lower():
                    parts = line.split(":", 1)[1]
                    for part in re.split(r'[,;|]', parts):
                        part = part.strip()
                        if part and len(part) > 2:
                            must_cover.append(part)
        if not must_cover and section_title:
            must_cover = [section_title]
    return SectionSpec(
        title=section_title,
        prompt=section_prompt,
        goal=meta.get("goal", section_prompt or section_title),
        must_cover_terms=must_cover[:8],
        must_cite=[t for t in meta.get("must_cite", []) if t][:8],
        avoid_terms=[t for t in meta.get("avoid_terms", []) if t][:8],
        prior_dependencies=[t for t in meta.get("depends_on", []) if t][:8],
        success_checks=[t for t in meta.get("success_checks", []) if t][:8],
        section_type=meta.get("section_type", "methods"),
    )


def _evidence_adequate(spec: SectionSpec, ranked_sources: List) -> tuple:
    if not ranked_sources:
        return False, "no ranked sources"
    source_text = " ".join(
        (getattr(s, "title", "") + " " + (getattr(s, "excerpt", "") or ""))
        for s in ranked_sources[:8]
    ).lower()
    covered_terms = [term for term in spec.must_cover_terms if term and term.lower() in source_text]
    primary_like = sum(1 for s in ranked_sources[:5] if (getattr(s, "provider", "") in ("arxiv", "wikipedia")))
    if primary_like < 2 and len(ranked_sources) >= 2:
        return False, "not enough primary sources for section"
    if spec.must_cover_terms and not covered_terms:
        return False, "must_cover_terms missing from evidence"
    return True, "ok"


def _build_writer_prompt(spec: SectionSpec, chapter_title: str, topic_context: str,
                         context_block: str, evidence_block: str,
                         prior_sections: List[dict] = None) -> str:
    prior_sections = prior_sections or []
    
    # Build cross-reference hint from prior sections
    cross_ref_hint = ""
    if prior_sections:
        prior_titles = [s.get("title", "") for s in prior_sections[-5:]]  # Last 5 sections
        if prior_titles:
            cross_ref_hint = f"\n\nEarlier sections for cross-referencing: {', '.join(prior_titles[-5:])}"
    
    must_cover = ", ".join(spec.must_cover_terms) if spec.must_cover_terms else "(none)"
    avoid_terms = ", ".join(spec.avoid_terms) if spec.avoid_terms else "(none)"
    deps = ", ".join(spec.prior_dependencies) if spec.prior_dependencies else "(none)"
    
    return f"""Chapter: {chapter_title}
Section: {spec.title}
Topic context: {topic_context}
Section type: {spec.section_type}
Goal: {spec.goal}
Must cover: {must_cover}
Avoid or minimize: {avoid_terms}
Depends on earlier material: {deps}{cross_ref_hint}

{context_block}{evidence_block}

Section directive: {spec.prompt or spec.goal}

WRITE RULES:
- Do NOT output any H1 (`#`) or H2 (`##`) heading. The section heading is added by the assembler. You may use H3 (`###`) and H4 (`####`) for sub-topics inside the section.
- Do NOT start with phrases like 'In this section', 'This chapter covers', 'We will discuss',
  or any meta-introduction. Start directly with substantive content. Also AVOID the
  formulaic openings this book overuses: do NOT begin with 'The {{abstract noun}} of', do NOT
  open by restating 'Large Language Models (LLMs)', and do NOT use the word 'necessitates'.
  Open in medias res with a concrete fact, formula, named method, or benchmark result from the
  EVIDENCE -- e.g. lead with a definition, a number, or a specific paper's finding.
- Do NOT write a 'Conclusion', 'Summary', 'Wrap-up', or 'In summary' section at the end.
  End with the last technical point.
- Do NOT write a 'References', 'Bibliography', or 'Further Reading' section. A single
  ## References section is added by the assembler.
- Cross-reference at least 2 prior sections by title (e.g., "As discussed in 'Section X.X: [exact prior section title]'..."). These are MANDATORY -- the section will be REJECTED if fewer than 2 cross-references are included. Count your cross-references before finishing and ensure you have >= 2.
- TECHNICAL DEPTH (this book must TEACH the mechanism, not just summarize it): WHENEVER the
  EVIDENCE provides them, you MUST (a) explain the core algorithm/method STEP BY STEP (a short
  numbered walkthrough of what it computes and why), (b) reproduce the key mathematical
  formula(s) as LaTeX display math on their own line, written as $$ ... $$, and define every
  symbol you use, and (c) give short pseudocode in a fenced code block when the method is
  algorithmic. GROUND every formula/step in a cited source [N] -- do NOT invent equations or
  steps the evidence does not contain (if the math is absent, explain the mechanism in words
  and say so). Prefer ONE correct, fully-derived mechanism over a broad shallow survey.
- If a required term is missing from the evidence, say the literature is limited rather than inventing details.
"""


def investigate_section(
    section_prompt: str,
    chapter_title: str,
    section_title: str,
    topic_context: str = "",       # Topic description + key concepts
    prior_sections: List[dict] = None,  # [{title, content}] for cross-ref
    prior_concepts: List[str] = None,     # concepts already covered elsewhere
    providers: tuple = ("arxiv", "wikipedia", "ddg"),
    embed_model: str = "bge-m3:latest",  # #3 unify retrieval embed with verify-side
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    writer_model: str = "batiai/qwen3.6-35b:iq3",  # WRT: stable writer on this machine
    judge_model: str = "gemma4:e4b",  # Research/Topic gate: stable fast model on 24 GB
    max_rounds: int = 3,
    min_grounding: float = 0.70,
    min_topic_relevance: float = 0.50,
    min_cross_refs: int = 2,  # 1.2-style retry: if only 1 prior section, can accept 1 ref
    min_cite_precision: float = 0.45,  # G2: avg per-[N] citation support (local gemma verify_section); fail-open
    concept_callback: Optional[Callable] = None,
    section_meta: Optional[dict] = None,
    # P0b: canonical source IDs injected at discovery stage -- these survive cosine gate
    protected_source_ids: set = None,
    # P0c: run-level seen-count map -- penalizes over-represented sources across sections
    run_seen_counts: dict = None,
    # Rank6: primary_floor reserves N of top-8 slots for arxiv/wikipedia primary sources
    primary_floor: int = 0,
    # AGENTIC evidence-pool memory: on-topic Source objects already gathered by sibling sections this
    # run. A niche section whose fresh queries return little is RESCUED by reusing these (the cosine
    # prefilter still drops off-topic ones) -> completeness up WITHOUT sacrificing faithfulness.
    evidence_pool: list = None,
) -> SectionResult:
    # P0c: run-level seen-count map -- penalizes over-represented sources across sections
    """
    Deep investigation for one section.

    Key loop:
      1. Generate queries from section prompt + topic context
      2. Gather + rank sources (RRF + RRK)
      3. Write section from evidence
      4. Verify grounding
      5. If low grounding → refine queries and retry
      6. Extract new concepts discovered during research
      7. Return result with all metadata
    """
    from . import search as _search
    from . import query_gen as _qgen
    from . import notes as _notes
    from . import embeddings as _embed
    from . import rerank as _rerank
    from .types import Query, Source

    prior_sections = prior_sections or []
    prior_concepts = prior_concepts or []
    protected_source_ids = protected_source_ids or set()
    spec = _build_section_spec(section_prompt, section_title, section_meta, topic_context)

    # P0b: force-fetch canonical papers directly so they survive in ranked results
    # even when search queries miss them. These bypass the normal search pipeline.
    canonical_forced: List[Source] = []
    if protected_source_ids:
        raw_ids = list(protected_source_ids)
        try:
            canonical_forced = _search.arxiv_by_id(raw_ids)
            print(f"  [P0b] arxiv_by_id returned {len(canonical_forced)} papers for IDs: {raw_ids}")
            if canonical_forced:
                print(f"  [P0b] Force-injected {len(canonical_forced)} canonical papers: "
                      f"{[s.id for s in canonical_forced]}")
        except Exception as e:
            import traceback as _tb
            print(f"  [P0b] arxiv_by_id failed: {e}")
            print(f"  [P0b] traceback: {_tb.format_exc()[:300]}")

    print(f"[INVESTIGATE] {chapter_title} > {section_title}")
    print(f"  protected_source_ids: {len(protected_source_ids)} ({list(protected_source_ids)[:3]}...)")
    print(f"  Topic context: {len(topic_context)} chars")
    print(f"  Prior sections: {len(prior_sections)}")
    print(f"  Prior concepts: {len(prior_concepts)}")

    best_content = ""
    best_score = 0.0
    best_sources = []
    best_evidence = ""
    best_topic_relevance = 0.0
    # Metadata pinned to the SAME round as best_content (was returned from last-round locals,
    # which mismatched best_content whenever best!=last round -- see accept-coupling bugfix).
    best_n_cites = 0
    best_cite_markers: list = []
    best_cross_refs = 0
    _best_round_tuple: tuple = (0.0, 0.0, 0)  # (grounding, topic_relevance, has_cites)
    all_concepts = []
    research_rounds = 0
    accepted = False  # P0: True iff a round cleanly passed the live gates (quality='ok')

    # P0c: seen-count penalty -- sources that appeared in many prior sections get
    # P0-3 fix: `x or {}` rebinds when caller passes an empty dict (falsy) -> propagation
    # back to the orchestrator was lost -> cross-section seen-penalty was a no-op in one run.
    if run_seen_counts is None:
        run_seen_counts = {}
    seen_counts = dict(run_seen_counts)  # copy so per-section mutations don't bleed

    current_hint = ""
    # Track prior queries/sources across rounds to detect wasted re-search
    prior_query_sigs: set = set()
    prior_source_ids: set = set()

    for round_n in range(1, max_rounds + 1):
        research_rounds += 1
        t_r = time.time()

        # --- Query generation ---
        # P0a fix: when topic_context is available, skip archetype routing and use LLM
        # fallback directly. Archetypes can't disambiguate "Self-Attention (transformer)"
        # from "Self-Attention (reasoning/CoT)" without domain context.
        # Pass topic_context as domain_context to query_router so it knows the broader domain.
        hint_text = f"\n\nHint: {current_hint}" if current_hint else ""
        prompt_for_qgen = (spec.prompt or spec.goal) + hint_text

        queries = _qgen.queries_for(
            prompt_for_qgen,
            chapter_title,
            spec.title,
            prior_query_sigs=prior_query_sigs if round_n > 1 else None,
            domain_context=topic_context if topic_context else None,
        )
        # Signature each query for overlap detection
        q_sigs = set()
        for q in queries:
            sig = (q.q.strip().lower()[:60], getattr(q, "intent", "unknown"))
            q_sigs.add(sig)
        print(f"  [R{round_n}] QGN: {len(queries)} queries")
        if current_hint:
            print(f"  [R{round_n}] Hint: {current_hint[:100]}...")

        # --- Gather ---
        raw_sources = _search.gather(
            queries, providers=providers, per_provider_k=3,
        )
        print(f"  [R{round_n}] Gathered: {len(raw_sources)} raw")

        # P0b fix: force-inject canonical (protected) papers into the raw pool
        # so they are available for ranking. Without this, protected IDs only survive
        # the cosine gate but are never fetched in the first place.
        if protected_source_ids:
            def _norm(s):
                s = (s or "").strip()
                s = re.sub(r"[vV]\d+$", "", s)
                if s.startswith("arxiv:"):
                    s = s[6:]
                return s
            existing_norm = {_norm(getattr(s, "id", "") or "") for s in raw_sources}
            existing_urls_norm = {_norm(getattr(s, "url", "") or "") for s in raw_sources}
            protected_sources = []
            for pid in protected_source_ids:
                pid_norm = _norm(pid)
                if pid_norm and pid_norm not in existing_norm and pid_norm not in existing_urls_norm:
                    try:
                        canon = _search.arxiv_by_id([pid])
                        if canon:
                            protected_sources.extend(canon)
                    except Exception:
                        pass
            if protected_sources:
                print(f"  [R{round_n}] P0b: injected {len(protected_sources)} canonical papers")
                raw_sources = protected_sources + raw_sources

        _pool_rescue_ids = set()   # populated by the post-prefilter evidence-pool rescue below

        if not raw_sources and not evidence_pool:
            print(f"  [R{round_n}] No sources gathered")
            if round_n < max_rounds:
                current_hint = "No sources found. Try different query terms."
                continue
            break

        # Guard: detect high source overlap to skip wasted re-search
        if round_n > 1 and raw_sources:
            seen_ids = prior_source_ids
            overlap_count = sum(
                1 for s in raw_sources
                if (getattr(s, "id", "") or "") in seen_ids
                or any((getattr(s, "url", "") or "") == prior_url
                        for prior_url in seen_ids)
            )
            overlap_frac = overlap_count / len(raw_sources)
            print(f"  [OVERLAP r{round_n}] {overlap_count}/{len(raw_sources)} "
                  f"({overlap_frac:.0%}) sources already seen in round 1")
            if overlap_frac > 0.6 and best_content:
                print(f"  [OVERLAP r{round_n}] HIGH overlap — skipping round {round_n}, "
                      "shipping best round so far")
                break

        # --- Rank (RRF) ---
        # #5 ANCHORING (SAFE / NO information loss): the section's domain term anchors ONLY
        # ranking + selection (rank_rrf + rerank below) so on-topic sources rise into the
        # top-8 and off-domain ones fall out -- WITHOUT dropping anything. The prefilter
        # (the only HARD-DROP) keeps the UNANCHORED section_prompt, so anchoring can never
        # shrink the candidate pool. Anchor = short noun phrase (must_cover_terms[0] / title).
        _anchor = (spec.must_cover_terms[0] if spec.must_cover_terms else "") or spec.title
        retrieval_query = f"{_anchor}. {section_prompt}".strip()
        # P0b: prefilter with protected IDs so canonical papers survive the cosine gate
        filtered = _notes.prefilter(
            raw_sources, section_prompt,
            embed_model=embed_model,
            protected_ids=protected_source_ids,
        )
        # AGENTIC evidence-pool rescue (post-prefilter): the REAL block trigger is thin ON-TOPIC
        # evidence -- few/no sources survive the cosine gate, whether because retrieval returned
        # nothing OR returned only off-domain hits. When that happens, reuse on-topic sources gathered
        # by sibling sections this run: run them through the SAME cosine prefilter (off-topic dropped
        # -> faithful) and mark the survivors P0c-exempt so the reuse can actually reach the writer's
        # top-k. Lifts completeness for niche sub-topics WITHOUT weakening faithfulness; only starved
        # sections pay the extra prefilter, well-covered ones are untouched.
        if evidence_pool and len(filtered) < 5:
            _seen = {(getattr(s, "id", "") or getattr(s, "url", "") or "") for s in filtered}
            _pool_kept = _notes.prefilter(
                [s for s in evidence_pool
                 if (getattr(s, "id", "") or getattr(s, "url", "") or "") not in _seen][-80:],
                section_prompt, embed_model=embed_model, protected_ids=protected_source_ids,
            )
            if _pool_kept:
                filtered = filtered + _pool_kept
                _pool_rescue_ids = {(getattr(s, "id", "") or getattr(s, "url", "") or "") for s in _pool_kept}
                print(f"  [R{round_n}] evidence-pool RESCUE: +{len(_pool_kept)} on-topic sibling source(s) (own evidence thin)", flush=True)
        ranked = _notes.rank_rrf(
            filtered if filtered else raw_sources,
            retrieval_query,
            top_k=20, embed_model=embed_model,
            protected_ids=protected_source_ids,
            seen_counts=seen_counts,   # P0c: penalize over-represented sources
            p0c_exempt_ids=_pool_rescue_ids,   # agentic: don't P0c-penalize pool-rescued siblings
            primary_floor=primary_floor,  # Rank6: reserve arxiv/wiki slots
        )

        # P0b: prepend force-fetched canonical papers so they appear in the evidence pool.
        # RRF+RRK may miss canonical papers (low relevance score), so we force-inject them.
        # Deduplicate by normalized ID so we don't double-count.
        if canonical_forced and protected_source_ids:
            def _norm(x):
                x = str(x or "").strip()
                x = re.sub(r"^arxiv:", "", x)
                x = re.sub(r"v\d+$", "", x)
                return x

            ranked_ids_normed = {_norm(getattr(s, "id", "") or getattr(s, "url", "")) for s in ranked}
            to_add = []
            for s in canonical_forced:
                norm = _norm(getattr(s, "id", "") or getattr(s, "url", ""))
                if norm and norm not in ranked_ids_normed:
                    to_add.append(s)
                    ranked_ids_normed.add(norm)
            if to_add:
                ranked = to_add + ranked
                print(f"  [P0b] Prepended {len(to_add)} canonical papers (RRF missed them)")
            else:
                print(f"  [P0b] Canonical papers already in RRF results, skipped prepend")

            # Note: canonical_coverage check is informational only (P0b prepend handles injection)
            # If canonical papers are missing from RRF results, P0b force-prepends them
            protected_normed = {_norm(pid) for pid in protected_source_ids}
            canonical_covered = protected_normed & ranked_ids_normed
            if len(canonical_covered) == 0:
                print(f"  [R{round_n}] WARNING: canonical_coverage=0/{len(protected_source_ids)} "
                      f"(P0b will prepend to evidence)")

        # --- RRK rerank ---
        try:
            ranked = _rerank.rerank(retrieval_query, ranked, top_k=8)
            print(f"  [R{round_n}] RRK: top-{len(ranked)}")
        except Exception as e:
            print(f"  [R{round_n}] RRK failed: {e}, using RRF top-8")
            ranked = ranked[:8]

        # --- Full-text enrichment ---
        # #2 EVIDENCE DEPTH: pull more full-text from MORE sources so the writer actually
        # sees the papers' methods/equations to reproduce (was top-2 @ 350w -> too thin for
        # formulas/algorithms). Adds info, never removes.
        ranked = _notes.enrich_top_sources(ranked, top_n=4, max_words_per=550)
        evidence_ok, evidence_reason = _evidence_adequate(spec, ranked)

        if not evidence_ok:
            print(f"  [R{round_n}] Evidence gate failed: {evidence_reason}")
            if round_n < max_rounds:
                current_hint = f"Evidence gap: {evidence_reason}. Retrieve sources that explicitly cover {', '.join(spec.must_cover_terms[:4])}."
                continue
        evidence_block = _notes.format_for_prompt(ranked)

        # --- P0a: Section Topic Relevance Gate ---
        # Check that the evidence pool actually matches the section's domain before writing.
        # This is the primary defense against rlhf_v3-style failures where evidence returns
        # from an entirely wrong domain (e.g., RAG papers for an RLHF section).
        # Gate uses notes.check_evidence_domain() (keyword overlap + optional gemma judge over
        # evidence titles/excerpts) -- distinct from the post-writer verify.topic_relevance_check on prose.
        if not ranked:
            if round_n < max_rounds:
                current_hint = (
                    "Evidence pool is empty -- no sources retrieved. "
                    "Retry with broader queries."
                )
                print(f"  [R{round_n}] Empty evidence pool -> retry with broader queries")
                continue
            else:
                raise RuntimeError(
                    f"[P0a HARD BLOCK] Section '{spec.title}' blocked: "
                    f"no sources retrieved after {max_rounds} rounds. "
                    f"Evidence pool empty. DO NOT write."
                )
        ev_check = _notes.check_evidence_domain(
            ranked,
            spec.title,
            spec.goal,
            spec.must_cover_terms,
            spec.avoid_terms,
            model=judge_model,
        )
        ev_topic_rel = float(ev_check.get("topic_relevance", 0.0))
        ev_reason = ev_check.get("reason", "")
        print(f"  [R{round_n}] Evidence domain gate: rel={ev_topic_rel:.3f} ({ev_reason[:80]})")
        # P0a evidence-domain gate threshold = ev_threshold = min(0.40, max(0.30, min_topic_relevance-0.10))
        # ~= 0.40 with default min_topic_relevance=0.50. NOTE: min_topic_relevance(0.50) is the SEPARATE
        # writer-accept bar (see accept check below); the evidence gate is intentionally looser (~0.40).
        ev_threshold = min(0.40, max(0.30, min_topic_relevance - 0.10))
        if ev_topic_rel < ev_threshold:
            if round_n < max_rounds:
                must_cover_hint = ", ".join(spec.must_cover_terms[:4]) or spec.title
                avoid_hint = ", ".join(spec.avoid_terms[:4]) or "none"
                current_hint = (
                    f"Evidence domain mismatch (relevance={ev_topic_rel:.2f}). "
                    f"Problem: {ev_reason}. "
                    f"Section must cover: {must_cover_hint}. "
                    f"Avoid retrieving adjacent domains: {avoid_hint}. "
                    f"Use query terms that explicitly name the target domain '{spec.title}' and canonical concepts."
                )
                print(f"  [R{round_n}] RETRY: evidence domain mismatch")
                continue
            else:
                # HARD BLOCK: section with wrong-domain evidence cannot be written.
                # Writer would produce rlhf_v3-style garbage (143w of RAG for RLHF section).
                # Mark as blocked so the run can be audited and retried with correct queries.
                raise RuntimeError(
                    f"[P0a HARD BLOCK] Section '{spec.title}' blocked: "
                    f"topic_relevance={ev_topic_rel:.3f} < {ev_threshold:.2f} after {max_rounds} rounds. "
                    f"Evidence domain mismatch: {ev_reason}. "
                    f"Section requires retry with correct domain queries. "
                    f"DO NOT write -- mark as BLOCKED."
                )

        # --- Build context block ---
        # Rank9: strip markdown headings and "(ChN.M...)" outline tags from the tail first,
        # so the next section doesn't echo a heading/title it sees in continuation context.
        ctx_parts = []
        for ps in prior_sections[-2:]:
            raw_tail = " ".join(ps.get("content", "").split()[-80:])
            # Strip headings (## or ### lines) and outline-disambiguation tags
            cleaned_tail = re.sub(r"(?m)^\s*#{1,6}\s.*$", "", raw_tail)
            cleaned_tail = re.sub(r"\((?:Ch|Chapter)\s*\d+\.\d+[^)]*\)", "", cleaned_tail)
            if cleaned_tail.strip():
                ctx_parts.append(f"Prior section: {ps.get('title','')}\n{cleaned_tail}")
        context_block = "\n\n".join(ctx_parts)
        if context_block:
            context_block += "\n\n"

        # Prior concepts warning
        if prior_concepts:
            covered = ", ".join(prior_concepts[:5])
            context_block += (
                f"[NOTE: These concepts were already covered earlier: {covered}. "
                f"Do NOT redefine them -- reference and build on them.]\n\n"
            )

        # --- Write ---
        t_w = time.time()
        writer_prompt = _build_writer_prompt(
            spec, chapter_title, topic_context, context_block, evidence_block,
            prior_sections=prior_sections
        )

        content = _ollama_chat(
            writer_model,
            [{"role": "user", "content": writer_prompt}],
            temperature=0.7,
            num_predict=2600,  # #1 headroom for step-by-step + formulas + pseudocode
        )

        if not content:
            print(f"  [R{round_n}] Writer returned empty content")
            continue

        # Citation cleanup
        content, n_dropped = _notes.clean_citations(content, len(ranked))
        if n_dropped:
            print(f"  [R{round_n}] Cleaned {n_dropped} bad citations")
        # Rank2 CITE-GUARD: a citation-shaped [N..] surviving clean_citations means the
        # cleaner regex has a gap -- surface it without eating legit math like [N=512].
        _resid = re.findall(r"\[\s*[Nn]\d*(?:\s*,\s*[Nn]?\d+)*\s*\]", content)
        if _resid:
            print(f"  [CITE-GUARD] WARN residual placeholder survived cleaner: {_resid[:3]}")

        w = len(content.split())
        cite_markers = re.findall(r"\[(\d+)\]", content)
        n_cites = len(cite_markers)
        print(f"  [R{round_n}] Written: {w}w, {n_cites} citations in {time.time()-t_w:.1f}s")

        # GATE-6: Cross-reference check (RULES Stage B - book coherence)
        cross_refs_found = 0  # Initialize for return statement
        cross_ref_result = {"refs_found": 0, "pass": True, "orphan": False}
        if prior_sections:
            try:
                from . import verify as _verify
                cross_ref_result = _verify.verify_cross_references_v2(
                    section_content=content,
                    prior_sections=prior_sections,
                    min_refs=min_cross_refs
                )
                cross_refs_found = cross_ref_result.get("refs_found", 0)
                print(f"  [R{round_n}] Cross-refs to prior sections: {cross_refs_found} {'✅' if cross_ref_result.get('pass') else '❌'}")
                if cross_ref_result.get("orphan"):
                    print(f"  [R{round_n}] ⚠️  ORPHAN section - no references to prior sections!")
            except Exception as e:
                # Fallback to simple check
                prior_titles_lower = [s.get("title", "").lower() for s in prior_sections]
                cross_refs_found = sum(1 for pt in prior_titles_lower if pt and pt.lower() in content.lower())
                print(f"  [R{round_n}] Cross-refs (fallback): {cross_refs_found}")

        # F5: Cross-reference gate -- ONE dynamic rule. Fixes the old bug where this block
        # used a flat min_cross_refs(=2) and ran BEFORE the prior-count-aware min_refs_needed,
        # forcing a 1-prior section to find 2 refs. Compute min_refs_needed HERE and use it.
        min_refs_needed = 2 if len(prior_sections) >= 2 else (1 if len(prior_sections) == 1 else 0)
        if prior_sections and cross_refs_found < min_refs_needed:
            if round_n < max_rounds:
                prior_title_1 = prior_sections[-1].get("title", "prior section")
                prior_title_2 = prior_sections[-2].get("title", "earlier section") if len(prior_sections) >= 2 else prior_sections[0].get("title", "earlier section")
                current_hint = (
                    f"CRITICAL: Section lacks cross-references ({cross_refs_found}/{min_refs_needed} found). "
                    f"You MUST add at least {min_refs_needed} cross-reference(s) to prior sections using their EXACT TITLES. "
                    f"Add sentences like: 'As discussed in [{prior_title_1}]...' "
                    f"and 'Building on [{prior_title_2}]...'. "
                    f"Use section TITLES, NOT numeric refs like 'Section 2.1'. "
                    f"Rewrite the section NOW with these cross-references included."
                )
                print(f"  [R{round_n}] RETRY: cross-refs={cross_refs_found}/{min_refs_needed} -- hint added")
                continue  # Skip expensive verification, retry immediately
            else:
                raise RuntimeError(
                    f"[CROSS-REF BLOCK] Section '{spec.title}': "
                    f"only {cross_refs_found}/{min_refs_needed} cross-references after {round_n} rounds. "
                    f"Writer must reference prior sections by TITLE. "
                    f"DO NOT write sections without cross-references."
                )

        # --- Verify (HHEM grounding) ---
        t_v = time.time()
        try:
            from . import faithfulness as _f
            claims = _f.decompose_claims(content, None)
            grounding_res = _f.grounding_score(claims, ranked)
            grounding = grounding_res.get("grounding", 0.0)
            _g_cited = grounding_res.get("grounding_cited")
            if _g_cited is not None:
                print(f"  [R{round_n}] [#4 WARN] grounding(per-source)={grounding:.3f} vs "
                      f"grounding(cited)={_g_cited:.3f} on {grounding_res.get('n_cited_claims', 0)} cited claims "
                      f"-- gate uses per-source until re-baselined")
        except Exception as e:
            print(f"  [R{round_n}] Grounding failed: {e}")
            grounding = 0.5  # Optimistic default
            grounding_res = {}

        try:
            from . import verify as _verify
            def _judge_llm(p):  # LOCAL gemma judge via Ollama -- never Claude/external
                return _ollama_chat(judge_model, [{"role": "user", "content": p}],
                                    temperature=0.1, num_predict=200)
            topic_res = _verify.topic_relevance_check(
                section_title=spec.title,
                goal=spec.goal,
                must_cover_terms=spec.must_cover_terms,
                avoid_terms=spec.avoid_terms,
                content=content,
                prior_sections=prior_sections,
                model=judge_model,
                llm_call_fn=_judge_llm,
            )
            topic_relevance = float(topic_res.get("topic_relevance", 0.5))
        except Exception as e:
            print(f"  [R{round_n}] Topic relevance failed: {e}")
            topic_relevance = 0.5
            topic_res = {"reason": "topic relevance failed"}

        print(f"  [R{round_n}] Grounding: {grounding:.3f} (advisory; min_grounding={min_grounding} NOT enforced -- P0) in {time.time()-t_v:.1f}s")
        print(f"  [R{round_n}] Topic relevance: {topic_relevance:.3f}")
        print(f"  [R{round_n}] Total time: {time.time()-t_r:.1f}s")

        # Track query/source IDs for overlap detection in next round
        prior_query_sigs.update(q_sigs)
        for s in raw_sources:
            if getattr(s, "id", ""):
                prior_source_ids.add(getattr(s, "id", ""))
            elif getattr(s, "url", ""):
                prior_source_ids.add(getattr(s, "url", ""))

        # --- Best round selection ---
        # Score = (topic_relevance, grounding, has_citations).
        # P0: topic (the LIVE signal) wins first; grounding is advisory tie-break only
        # (was grounding-first, but grounding is now ~noise at 0-0.46, so it must not pick
        # which degraded draft ships when no round cleanly accepts).
        def _round_tuple():
            t = topic_relevance
            g = grounding
            c = 1 if n_cites > 0 else 0
            return (t, g, c)
        if best_score == 0.0 or _round_tuple() > _best_round_tuple:
            best_content = content
            best_score = grounding
            best_topic_relevance = topic_relevance
            best_sources = ranked
            best_evidence = evidence_block
            best_n_cites = n_cites
            best_cite_markers = cite_markers
            best_cross_refs = cross_refs_found
            _best_round_tuple = _round_tuple()

        # --- Accept gates ---
        # Cross-ref requirement (min_refs_needed already computed at the F5 gate above;
        # by here cross_refs_found >= min_refs_needed, so this is effectively satisfied).
        has_min_cross_refs = cross_refs_found >= min_refs_needed

        # G2 CITATION INTEGRITY: does each [N] actually support its own cited claim?
        # verify_section() = bge-m3 cosine prefilter + LOCAL gemma batch judge (never
        # Claude/external). P0 (2026-06-22): DECOUPLED from grounding -- run whenever the
        # section has citations AND is on-topic, REGARDLESS of grounding. (Old gate put
        # grounding>=0.70 inside base_ok, but per-source-max HHEM maxes ~0.46 on synthesized
        # prose, so base_ok was never true and G2 NEVER ran -> cite_precision was a 1.0
        # default. Grounding is now LOG-ONLY/advisory; the live faithfulness gate is this
        # real per-[N] cite_precision.)
        topic_ok = topic_relevance >= min_topic_relevance
        cite_precision = None  # None = not measured this round (keeps the 1.0 default out of logs)
        if n_cites > 0 and topic_ok:
            try:
                cite_res = _verify.verify_section(content, ranked, model=judge_model)
                cite_precision = float(cite_res.get("grounding", 1.0))
                print(f"  [R{round_n}] Citation integrity (G2): {cite_precision:.3f}")
            except Exception as e:
                # fail-CLOSED: on error do NOT auto-pass -- 0.0 means this round won't
                # clean-accept (retries / ships best-effort).
                print(f"  [R{round_n}] Citation integrity UNVERIFIED (fail-closed): {e}")
                cite_precision = 0.0

        # ACCEPT (clean, quality='ok') when topic + cites + cross-refs + a REAL measured
        # cite_precision all pass. Grounding is logged but is NOT a gate anymore (P0).
        gate_ok = (n_cites > 0 and topic_ok and has_min_cross_refs)
        if gate_ok and cite_precision is not None and cite_precision >= min_cite_precision:
            accepted = True
            # BUGFIX (accept-coupling): best-round selection is TOPIC-FIRST, so the best-topic
            # round may NOT be this accepting round -- and the function returns best_content.
            # Without this override a section could ship an EARLIER, higher-topic round's body
            # that FAILED this very cite_precision gate, mislabelled quality='ok'. Pin best_* to
            # the round that actually passed G2 so the shipped body is the verified one. (n_cites /
            # cite_markers / cross_refs_found are already this round's because we break here.)
            best_content = content
            best_sources = ranked
            best_topic_relevance = topic_relevance
            best_score = grounding
            best_n_cites = n_cites
            best_cite_markers = cite_markers
            best_cross_refs = cross_refs_found
            print(f"  [R{round_n}] ACCEPT: topic={topic_relevance:.3f}, cite_prec={cite_precision:.3f}, "
                  f"cross-refs={cross_refs_found} (grounding={grounding:.3f} advisory)")
            break

        # (StageE topic-drift hard block moved to AFTER the loop -- it gates on the BEST
        #  round's topic, not the last round's, so last-round variance can't discard an
        #  otherwise-on-topic draft.)

        # Retry for grounding / topic / citation issues
        if round_n < max_rounds:
            weak_summary = grounding_res.get("weak_summary", "") if isinstance(grounding_res, dict) else ""
            drift_terms = ", ".join(topic_res.get("drift_terms", [])[:4]) if isinstance(topic_res, dict) else ""
            missing_terms = ", ".join(topic_res.get("missing_terms", [])[:4]) if isinstance(topic_res, dict) else ""
            # G5b: numeric cross-refs ("Section 2.1") are fabrication-prone -- the book
            # cannot verify them; nudge the writer to use exact section TITLES instead.
            numeric_refs = re.findall(r"\b(?:Section|Chapter)\s+\d+(?:\.\d+)?", content)
            numeric_hint = (f" Replace numeric refs ({', '.join(numeric_refs[:3])}) with exact section TITLES."
                            if numeric_refs else "")
            cite_hint = (" Some citations do not support their claim -- attach the correct [N] to each fact."
                         if (cite_precision is not None and cite_precision < min_cite_precision) else "")
            cite_str = f", cite_prec={cite_precision:.3f}" if cite_precision is not None else ""
            current_hint = (
                f"Previous draft: grounding={grounding:.3f} (advisory), topic={topic_relevance:.3f}{cite_str}. "
                f"Focus on: {missing_terms or 'none'}. Avoid: {drift_terms or 'none'}. "
                f"{weak_summary[:140] if weak_summary else 'cite more specific facts'}.{cite_hint}{numeric_hint}"
            )
            print(f"  [R{round_n}] RETRY with refined queries")

        # RULES Stage D: min word count >= 120 hard rule
        # Sections shorter than 120 words are "garbage" (rlhf_v3-style 143w sections)
        # Treat as evidence failure: retry or block
        if w < 120:
            if round_n < max_rounds:
                current_hint = (
                    f"Section too short ({w}w < 120 minimum). "
                    f"Evidence insufficient for a substantive section. "
                    f"Retrieve more specific sources and write with depth."
                )
                print(f"  [R{round_n}] RETRY: word_count={w} < 120 (garbage threshold)")
                continue
            else:
                raise RuntimeError(
                    f"[StageD HARD BLOCK] Section '{spec.title}' blocked: "
                    f"word_count={w} < 120. "
                    f"RULES StageD: filler word count is not quality. "
                    f"DO NOT write thin sections."
                )

    # StageE TOPIC-DRIFT hard block (P0): no round cleanly accepted AND even the BEST round
    # reads off-topic (best topic < min) -> drift -> block, don't ship. Gating on the best
    # round's topic (not the last round's) avoids discarding an on-topic draft to last-round
    # variance. Grounding is NOT part of this decision anymore (it is advisory/log-only).
    if not accepted and best_topic_relevance < min_topic_relevance:
        raise RuntimeError(
            f"[StageE HARD BLOCK] Section '{spec.title}' blocked: "
            f"best topic_purity={best_topic_relevance:.3f} < {min_topic_relevance}. "
            f"topic drift detected. DO NOT write."
        )

    # G6 (warn-first): flag near-duplicate sections via bge-m3 cosine vs prior bodies.
    # Soft signal ONLY -- logs, does NOT block (a blocking threshold needs calibration
    # from a real run). Gives the bge-m3 content-dedup signal the product wants.
    try:
        if best_content and prior_sections:
            from .embeddings import embed as _g6_embed, cosine as _g6_cos
            _g6_prior = prior_sections[-12:]
            _g6_texts = [best_content[:1500]] + [(ps.get("content", "") or "")[:1500] for ps in _g6_prior]
            _g6_vecs = _g6_embed(_g6_texts, model="bge-m3:latest")
            if _g6_vecs and len(_g6_vecs) == len(_g6_texts):
                _g6_sims = [(_g6_cos(_g6_vecs[0], _g6_vecs[k + 1]), _g6_prior[k].get("title", ""))
                            for k in range(len(_g6_prior))]
                if _g6_sims:
                    _g6_mx, _g6_who = max(_g6_sims, key=lambda x: x[0])
                    if _g6_mx >= 0.85:
                        print(f"  [G6 DEDUP-WARN] section ~{_g6_mx:.2f} cosine to prior "
                              f"'{_g6_who[:50]}' (>=0.85; warn-first, not blocked)")
    except Exception as e:
        print(f"  [G6 DEDUP] skipped: {e}")

    # --- Extract new concepts ---
    discovered = _concept_decomposition(best_content)
    all_concepts.extend(discovered)
    # Filter out already-covered concepts
    new_concepts = [c for c in discovered if c.lower() not in
                    [pc.lower() for pc in prior_concepts]]
    print(f"  [INVESTIGATE] Discovered {len(discovered)} concepts, {len(new_concepts)} new")

    # P0c: update seen_counts with sources from the best accepted round
    # so subsequent sections penalize over-represented sources
    for s in best_sources:
        sid = getattr(s, "id", "") or getattr(s, "url", "") or ""
        if sid:
            seen_counts[sid] = seen_counts.get(sid, 0) + 1
    if run_seen_counts is not None:
        for k, v in seen_counts.items():
            run_seen_counts[k] = v  # propagate back to orchestrator

    # Notify concept callback if provided
    if concept_callback:
        for concept in new_concepts:
            concept_callback(concept, section_title)

    return SectionResult(
        content=best_content,
        sources=best_sources,
        grounding_score=best_score,
        topic_relevance_score=best_topic_relevance,
        n_citations=best_n_cites,
        new_concepts=new_concepts,
        research_rounds=research_rounds,
        citation_markers=best_cite_markers,
        quality="ok" if accepted else "degraded",
        cross_ref_count=best_cross_refs,  # GATE-6: Cross-reference count (pinned to best_content's round)
    )
