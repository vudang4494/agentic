"""
Stage 0: DISCOVERY
Broad scoping of a topic to produce a TopicProfile.

Input:  raw topic string
Output: TopicProfile dataclass with name, description, scope, key_concepts,
        initial_queries, canonical_papers, target_depth, estimated_sections

The researcher:
  1. Runs axis-aware scoping queries across multiple providers
  2. Gathers diverse sources (arxiv, wikipedia, web)
  3. Builds a topic kernel with canonical anchors and out-of-scope guards
  4. LLM synthesizes a structured topic profile from the sources

Why this matters: the outline depends on what you discover, not what you assumed.
"""
from .config import DISCOVERY_MODEL
import httpx, json, re, time
from dataclasses import dataclass, field, asdict
from typing import Dict, List

OLLAMA_BASE = "http://localhost:11434"
TIMEOUT = 300.0

# R7: Reject wrapper/redirect URLs in canonical papers
_DDG_REDIRECT_RE = re.compile(r"https?://duckduckgo\.com/l/\?uddg=", re.IGNORECASE)
_PREFERRED_URL_RE = re.compile(
    r"^(https?://)?(arxiv\.org/(abs|pdf|html)|en\.wikipedia\.org/|"
    r"(www\.)?[a-z0-9-]+\.(io|com|org|edu|gov|ai)(/|$))",
    re.IGNORECASE,
)

def _validate_canonical_urls(papers: list) -> list:
    """Filter canonical papers: prefer direct links, reject wrapper/redirect URLs."""
    if not papers:
        return papers
    validated = []
    dropped_redirect = 0
    for p in papers:
        url = p.get("url", "")
        title = p.get("title", "")
        if not title:
            continue  # a canonical paper with no title is unusable
        # R7: only drop GENUINE DDG wrapper/redirect URLs. A canonical paper with a title
        # but no/empty URL is still useful (coverage + canonical_seeds resolution) and is
        # NOT a redirect -- KEEP it. (The old code dropped every no-URL paper and mislabeled
        # them "wrapper/redirect", silently losing canonical coverage -- see validate_v37.)
        if url and _DDG_REDIRECT_RE.search(url):
            dropped_redirect += 1
            continue
        validated.append(p)
    if dropped_redirect:
        print(f"[DISCOVERY] R7 canonical filter: {len(papers)} -> {len(validated)} papers "
              f"(removed {dropped_redirect} DDG redirect URLs)")
    return validated


@dataclass
class TopicProfile:
    name: str              # Refined book title
    subtitle: str           # 1-line description
    description: str        # 2-3 sentence scope description
    scope: str              # "introductory" | "comprehensive" | "expert"
    key_concepts: List[str] = field(default_factory=list)
    initial_queries: List[str] = field(default_factory=list)
    canonical_papers: List[dict] = field(default_factory=list)  # [{title, url, year}]
    canonical_terms: List[str] = field(default_factory=list)
    must_cover: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    seed_queries_by_axis: Dict[str, List[str]] = field(default_factory=dict)
    estimated_sections: int = 12
    sections_per_chapter: int = 8
    # P0b: canonical arxiv IDs injected by discover_topic (from gold yaml or discovery)
    canonical_arxiv_ids: List[str] = field(default_factory=list)
    # P0b: all protected source IDs (canonical + user-provided) -- passed to investigation
    protected_source_ids: List[str] = field(default_factory=list)
    # Raw synthesis text kept for debugging
    _synthesis: str = ""


def _ollama_chat(model: str, messages: list, temperature: float = 0.3,
                 num_predict: int = 2000, timeout: float = TIMEOUT) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    content = (data.get("message") or {}).get("content", "").strip()
    if not content:
        content = (data.get("message") or {}).get("thinking", "").strip()
    return content


def _scope_queries(topic: str) -> Dict[str, List[str]]:
    """Axis-aware scoping queries to keep discovery anchored to the topic kernel."""
    return {
        "definition": [
            f"what is {topic}",
            f"{topic} introduction overview",
        ],
        "canonical": [
            f"{topic} seminal paper",
            f"{topic} landmark papers",
        ],
        "math": [
            f"{topic} mathematical foundations",
            f"{topic} objective derivation",
        ],
        "architecture": [
            f"{topic} architecture implementation",
            f"{topic} model design",
        ],
        "evaluation": [
            f"{topic} benchmarks evaluation",
        ],
        "applications": [
            f"{topic} applications use cases",
        ],
    }


def _flatten_axis_queries(axis_queries: Dict[str, List[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for queries in axis_queries.values():
        for q in queries:
            qn = q.strip()
            if qn and qn not in seen:
                seen.add(qn)
                ordered.append(qn)
    return ordered


def _infer_out_of_scope(topic: str, unique_sources: List) -> List[str]:
    """Infer drift-prone terms that should be treated as out-of-scope unless strongly evidenced."""
    text_blob = " \n ".join(
        filter(None, [topic] + [getattr(s, "title", "") + " " + (getattr(s, "excerpt", "") or "") for s in unique_sources[:12]])
    ).lower()
    guards = []
    candidate_terms = [
        "retrieval-augmented generation",
        "rag",
        "enterprise agents",
        "llm trust stack",
        "knowledge base orchestration",
        "prompt engineering",
    ]
    diffusion_like = any(k in topic.lower() for k in ["diffusion", "score-based", "denoising"])
    if diffusion_like:
        for term in candidate_terms:
            if term not in text_blob:
                guards.append(term)
    return guards[:8]


def discover_topic(
    topic: str,
    target_depth: str = "comprehensive",
    providers: tuple = ("arxiv", "wikipedia", "ddg"),
    model: str = DISCOVERY_MODEL,
    canonical_arxiv_ids: List[str] = None,
) -> TopicProfile:
    """Main entry point. Run full discovery pipeline.

    Args:
        topic: Raw topic string.
        target_depth: "introductory" | "comprehensive" | "expert".
        providers: Search provider tuple.
        model: Ollama model for LLM synthesis.
        canonical_arxiv_ids: P0b: known foundational arxiv IDs to force-fetch.
            These bypass search retrieval and are injected directly into the evidence pool
            so the pipeline always contains canonical references regardless of search recency bias.
    """
    from . import search as _search
    from .types import Query

    print(f"[DISCOVERY] Topic: {topic}")
    print(f"[DISCOVERY] Depth: {target_depth}")

    # --- P0b: Canonical injection step ---
    # Force-fetch known foundational papers so they are always present in the evidence pool.
    # These bypass search retrieval entirely and survive the cosine gate via protected_ids.
    injected_sources: List = []
    canonical_ids_found: List[str] = []
    if canonical_arxiv_ids:
        injected = _search.arxiv_by_id(canonical_arxiv_ids)
        for s in injected:
            canonical_ids_found.append(getattr(s, "id", ""))
        injected_sources.extend(injected)
        print(f"[DISCOVERY] P0b injected {len(injected)} canonical papers: "
              f"{[getattr(s,'id','?') for s in injected]}")

    # --- Step 1: Axis-aware scoping queries ---
    axis_queries = _scope_queries(topic)
    queries = _flatten_axis_queries(axis_queries)
    print(f"[DISCOVERY] Scoping axes: {len(axis_queries)} | queries: {len(queries)}")

    # --- Step 2: Gather diverse sources ---
    all_sources = []
    gather_errors = []
    for axis, axis_qs in axis_queries.items():
        for q in axis_qs:
            try:
                sources = _search.gather([Query(q=q, intent=axis)], providers=providers, per_provider_k=3)
                all_sources.extend(sources)
            except Exception as e:
                gather_errors.append(f"{axis}:{q} -> {e}")
                print(f"[DISCOVERY] Gather warn: {axis}:{q[:40]} -> {e}")
            time.sleep(0.3)

    # --- Step 2b: seed-grounded canonical auto-injection (P0b self-injection) ---
    # Even WITHOUT --canonical-arxiv-ids, ground canonical anchors in the curated
    # SEED_MAP: resolve well-known methods named in the topic+evidence digest to their
    # real arxiv IDs and force-fetch them, so canonical_papers are arxiv-verified rather
    # than gemma free-text guesses. Topic-agnostic (SEED_MAP is a fixed alias->id registry);
    # fires only when a known method is actually mentioned, so no-op for unknown domains.
    try:
        from . import canonical_seeds as _seeds
        _already = set(canonical_arxiv_ids or [])
        _digest = topic + " " + " ".join(
            ((getattr(s, "title", "") or "") + " " + (getattr(s, "excerpt", "") or ""))
            for s in all_sources[:20])
        _seed_ids = [a for a in _seeds.resolve_seeds(_digest, max_seeds=4) if a not in _already]
        if _seed_ids:
            _seed_src = _search.arxiv_by_id(_seed_ids)
            for s in _seed_src:
                canonical_ids_found.append(getattr(s, "id", ""))
            injected_sources.extend(_seed_src)
            print(f"[DISCOVERY] P0b seed-grounded auto-inject {len(_seed_src)} canonical "
                  f"papers from SEED_MAP: {_seed_ids}")
    except Exception as e:
        print(f"[DISCOVERY] seed-grounding skipped: {e}")

    # --- Step 3: Inject canonical sources into pool ---
    # Canonical sources are prepended so they appear first in dedup (seen set starts empty).
    # They will be marked as protected_ids in investigation so they survive the cosine gate.
    seen = set()
    unique = []
    # First: injected canonical sources (force-include, even if duplicates)
    for s in injected_sources:
        key = getattr(s, "url", None) or getattr(s, "id", "")
        unique.append(s)
        if key:
            seen.add(key)
    # Then: regular gathered sources (dedup by URL)
    for s in all_sources:
        key = getattr(s, "url", None) or getattr(s, "id", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    print(f"[DISCOVERY] Gathered {len(unique)} unique sources "
          f"(+ {len(injected_sources)} canonical injected)")

    # --- Step 3: Build source context for LLM ---
    topic_kernel = {
        "canonical_terms": getattr(topic, "split", lambda: [])() if False else [],
        "must_cover": [],
        "out_of_scope": _infer_out_of_scope(topic, unique),
        "seed_queries_by_axis": axis_queries,
    }
    source_texts = []
    # P0b: prepend injected canonical sources to the digest so the LLM knows they are present
    for i, s in enumerate(unique[:20], 1):
        prefix = " [CANONICAL] " if getattr(s, "id", "") in canonical_ids_found else ""
        text = f"[{i}]{prefix}{s.title}\nURL: {s.url}\n{s.excerpt or ''}"
        source_texts.append(text)
    context = "\n\n".join(source_texts)

    # P0b: canonical IDs list for topic profile
    canon_ids = list(canonical_ids_found)

    # --- Step 4: LLM synthesizes topic profile ---
    synthesis_prompt = f"""You are a research librarian. Given the following sources about \"{topic}\",
produce a structured TOPIC PROFILE as a JSON object (no markdown fences, no prose around it):

{{
  \"name\": \"Refined book title (max 60 chars)\",
  \"subtitle\": \"One-line book subtitle (max 80 chars)\",
  \"description\": \"2-3 sentence scope description of what this book covers\",
  \"scope\": \"comprehensive\",
  \"key_concepts\": [\"concept 1\", \"concept 2\", \"...\"],
  \"initial_queries\": [\"specific research query 1\", \"specific research query 2\", \"...\"],
  \"canonical_papers\": [{{\"title\": \"...\", \"url\": \"...\", \"year\": N}}],
  \"canonical_terms\": [\"term 1\", \"term 2\", \"...\"],
  \"must_cover\": [\"theme 1\", \"theme 2\", \"...\"],
  \"out_of_scope\": [\"drift term 1\", \"drift term 2\"],
  \"seed_queries_by_axis\": {{\"definition\": [\"...\"], \"canonical\": [\"...\"]}},
  \"estimated_sections\": N,
  \"sections_per_chapter\": 8
}}

Rules:
- key_concepts: 8-15 core concepts this topic covers
- initial_queries: 6-10 specific search queries a researcher should follow
- canonical_papers: 3-5 landmark papers (with URLs if available)
- canonical_terms: 6-12 anchor terms that must appear somewhere in the final book
- must_cover: 6-12 non-overlapping topical areas the book must cover
- out_of_scope: concepts that are adjacent but should be excluded unless the evidence strongly justifies them
- seed_queries_by_axis: preserve the research axes and refine them if needed
- estimated_sections: how many sections (8-16 makes sense for a book)
- description should make clear what is IN scope and what is NOT
- scope field: \"introductory\" | \"comprehensive\" | \"expert\"

IMPORTANT -- Priority canonical papers:
The following sources are marked [CANONICAL] and represent foundational papers for this topic.
You MUST include them in canonical_papers if they appear in the source list.

{json.dumps(topic_kernel['out_of_scope'], ensure_ascii=False)}

Sources:
# PRIORITY canonical sources (these must appear in canonical_papers and canonical_terms):
{chr(10).join(f"# CANONICAL: {s.title} (arxiv:{s.id.split(':')[-1] if ':' in s.id else '?'})" for s in injected_sources[:8]) if injected_sources else "# (no canonical injections)"}
# Other sources:
{context[:8000]}
"""

    print("[DISCOVERY] Synthesizing topic profile...")
    try:
        synthesis = _ollama_chat(model, [{"role": "user", "content": synthesis_prompt}],
                                 temperature=0.3, num_predict=2200)
    except Exception as e:
        print(f"[DISCOVERY] Synthesis failed on model {model}: {e}")
        synthesis = ""
    print(f"[DISCOVERY] Synthesis: {len(synthesis)} chars")

    # --- Step 5: Parse JSON ---
    try:
        # Try to extract JSON block
        m = re.search(r"\{[\s\S]*\}", synthesis)
        if m:
            parsed = json.loads(m.group())
        else:
            parsed = json.loads(synthesis)
    except json.JSONDecodeError as e:
        print(f"[DISCOVERY] JSON parse failed: {e}")
        print(f"[DISCOVERY] Raw: {synthesis[:500]}")
        parsed = {
            "name": topic, "subtitle": "",
            "description": f"Research topic: {topic}. Discovery used retrieval-first fallback due to incomplete synthesis.",
            "scope": target_depth, "key_concepts": [],
            "initial_queries": queries, "canonical_papers": [],
            "canonical_terms": [], "must_cover": [],
            "out_of_scope": topic_kernel["out_of_scope"],
            "seed_queries_by_axis": axis_queries,
            "estimated_sections": 12, "sections_per_chapter": 8,
        }

    # P0b: collect all protected source IDs from injected canonical papers
    protected_ids = list(set(
        getattr(s, "id", "") for s in injected_sources if getattr(s, "id", "")
    ))

    profile = TopicProfile(
        name=parsed.get("name", topic),
        subtitle=parsed.get("subtitle", ""),
        description=parsed.get("description", ""),
        scope=parsed.get("scope", target_depth),
        key_concepts=parsed.get("key_concepts", []),
        initial_queries=parsed.get("initial_queries", queries),
        canonical_papers=_validate_canonical_urls(parsed.get("canonical_papers", [])),
        canonical_terms=parsed.get("canonical_terms", parsed.get("key_concepts", [])[:8]),
        must_cover=parsed.get("must_cover", parsed.get("key_concepts", [])[:8]),
        out_of_scope=parsed.get("out_of_scope", topic_kernel["out_of_scope"]),
        seed_queries_by_axis=parsed.get("seed_queries_by_axis", axis_queries),
        estimated_sections=parsed.get("estimated_sections", 12),
        sections_per_chapter=parsed.get("sections_per_chapter", 8),
        canonical_arxiv_ids=canon_ids,        # P0b: IDs of injected canonical papers
        protected_source_ids=protected_ids,      # P0b: all protected IDs for investigation stage
        _synthesis=synthesis,
    )

    print(f"[DISCOVERY] Profile: {profile.name}")
    print(f"[DISCOVERY] Concepts: {len(profile.key_concepts)}")
    print(f"[DISCOVERY] Queries: {len(profile.initial_queries)}")
    print(f"[DISCOVERY] Papers: {len(profile.canonical_papers)}")
    print(f"[DISCOVERY] Canonical terms: {len(profile.canonical_terms)}")
    print(f"[DISCOVERY] Must-cover axes: {len(profile.must_cover)}")
    print(f"[DISCOVERY] Out-of-scope guards: {len(profile.out_of_scope)}")
    print(f"[DISCOVERY] Sections: {profile.estimated_sections} chapters x "
          f"{profile.sections_per_chapter} = "
          f"{profile.estimated_sections * profile.sections_per_chapter}")
    if profile.protected_source_ids:
        print(f"[DISCOVERY] P0b protected IDs: {len(profile.protected_source_ids)} canonical sources injected")

    return profile
