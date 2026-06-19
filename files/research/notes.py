"""Dedup, rank, and format research sources into an EVIDENCE block for the writer."""
import math
import re
from typing import List
from urllib.parse import urlparse

from .embeddings import embed, cosine
from .fetch import fetch_full_text
from .types import Source


def dedup(sources: List[Source]) -> List[Source]:
    """Drop duplicate sources by URL, keeping the first occurrence."""
    seen: set = set()
    out: List[Source] = []
    for s in sources:
        key = s.url or s.id
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ---- BM25 (sparse) for RRF fusion ----
# Pure-Python BM25 — no external deps required.
# Used as the sparse arm in RRF fusion alongside dense cosine.

def _tokenize(text: str) -> List[str]:
    # Rank6: math is NOISE for the sparse arm. The old tokenizer turned `\frac{a}{b}` into
    # ['frac','a','b'] and LaTeX control words (frac/sqrt/beta/sum/mathrm...) into high-frequency
    # junk terms shared by EVERY formula-heavy section -> inflated BM25 similarity -> false
    # near-duplicates and wrong RRF fusion. Strip $...$/$$...$$ spans + LaTeX commands BEFORE
    # tokenizing; prose terms that matter ("softmax", "attention", "quantization") survive in prose.
    text = text.lower()
    text = re.sub(r"\$\$.+?\$\$|\$[^$\n]+?\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\[a-z]+", " ", text)
    return re.findall(r"[a-z0-9]+", text)


def _bm25_score(doc_tokens: List[str], query_tokens: List[str],
                 avg_dl: float, doc_lens: List[int], doc_id: int,
                 k1: float = 1.5, b: float = 0.75) -> float:
    """Okapi BM25 score for one doc. Returns 0 if doc has no overlap with query."""
    dlen = doc_lens[doc_id]
    doc_set: set = {}
    for t in doc_tokens:
        doc_set[t] = doc_set.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        if qt not in doc_set:
            continue
        tf = doc_set[qt]
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (dlen / avg_dl))
        score += numerator / (denominator + 1e-10)
    return score


# Rank-1 fix (bookv6 eval: 82 forbidden-domain hits, all cleared the old soft
# 0.60 threshold because well-written blogs embed as topically relevant).
# Two tiers now:
#   FORBIDDEN -- blog/social/news platforms that are NEVER a primary technical
#                source. Hard-dropped by host-suffix BEFORE the cosine gate,
#                regardless of relevance. Mirror this list in the
#                `forbidden_domains` field of files/eval/topics/*.yaml.
#   GREY      -- aggregators/tutorials: allowed only if they clear the higher
#                `noisy_min_relevance` soft threshold.
_FORBIDDEN_DOMAINS = (
    "medium.com", "substack.com", "techcrunch.com",
    "youtube.com", "vimeo.com", "tiktok.com",
    "reddit.com", "facebook.com", "twitter.com", "x.com",
    "linkedin.com", "pinterest.com", "quora.com",
)
_GREY_DOMAINS = (
    "towardsdatascience.com",
    "duckduckgo.com",   # caught the "DuckDuckGo's history" false-match in Ch11.4
)


def _host(url: str) -> str:
    """Parsed lowercase host without leading www. ('' on parse failure)."""
    if not url:
        return ""
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return h[4:] if h.startswith("www.") else h


def _host_suffix_match(host: str, domains) -> bool:
    """True if host == d or is a subdomain of d. Suffix-match (not substring)
    so 'x.com' does NOT spuriously match 'matrix.com', while
    'cameronrwolfe.substack.com' correctly matches 'substack.com'."""
    return any(host == d or host.endswith("." + d) for d in domains)


def _is_forbidden_domain(url: str) -> bool:
    return _host_suffix_match(_host(url), _FORBIDDEN_DOMAINS)


def _is_noisy_domain(url: str) -> bool:
    """GREY tier -- subject to the higher soft relevance threshold."""
    return _host_suffix_match(_host(url), _GREY_DOMAINS)


def prefilter(sources: List[Source], section_prompt: str,
              min_relevance: float = 0.48,   # Rank7: 0.45->0.48 trims the weakest off-topic tail
              noisy_min_relevance: float = 0.65,
              embed_model: str = "bge-m3:latest",
              protected_ids: set = None) -> List[Source]:
    """Drop obviously off-topic sources BEFORE the main rank().

    Two thresholds:
      - any source below `min_relevance` cosine to the section prompt is dropped
      - sources from `_NOISY_DOMAINS` (YouTube, social, etc.) must clear the
        higher `noisy_min_relevance` -- otherwise they're dropped

    This prevents Tavily's occasional off-domain match from polluting the top-8
    that the writer sees. Run BEFORE rank() so rank() chooses from a clean pool.
    """
    sources = dedup(sources)
    if not sources:
        return []
    # Tier 1: hard-drop FORBIDDEN domains by host-suffix BEFORE any embedding.
    # These never qualify as primary sources regardless of cosine relevance.
    dropped_forbidden = sum(1 for s in sources if _is_forbidden_domain(s.url))
    sources = [s for s in sources if not _is_forbidden_domain(s.url)]
    if not sources:
        if dropped_forbidden:
            print(f"[research/notes] prefilter hard-dropped {dropped_forbidden} forbidden-domain "
                  f"(0 sources left)", flush=True)
        return []
    texts = [section_prompt] + [f"{s.title}. {s.excerpt}" for s in sources]
    vectors = embed(texts, model=embed_model)
    if len(vectors) != len(texts):
        # embedding failed -- pass through, rank() will still try
        return sources
    qv = vectors[0]
    protected_ids = protected_ids or set()
    kept = []
    dropped_noisy = 0
    dropped_offtopic = 0
    for s, v in zip(sources, vectors[1:]):
        rel = cosine(qv, v)
        s.relevance = rel  # cache for rank() to reuse
        # Rank5: canonical seeds bypass the cosine gate -- they are known-good
        # primary sources injected on purpose, kept even if the descriptive
        # section prompt embeds at a lower cosine than the paper's abstract.
        if s.id in protected_ids:
            kept.append(s)
            continue
        threshold = noisy_min_relevance if _is_noisy_domain(s.url) else min_relevance
        if rel < threshold:
            if _is_noisy_domain(s.url):
                dropped_noisy += 1
            else:
                dropped_offtopic += 1
            continue
        kept.append(s)
    if dropped_forbidden or dropped_noisy or dropped_offtopic:
        print(f"[research/notes] prefilter dropped {dropped_forbidden} forbidden + "
              f"{dropped_offtopic} off-topic + {dropped_noisy} grey-domain "
              f"(kept {len(kept)}/{len(kept)+dropped_offtopic+dropped_noisy})",
              flush=True)
    return kept


_PRIMARY_PROVIDERS = ("arxiv", "wikipedia")


def _apply_primary_quota(scored: List[Source], top_k: int, primary_floor: int,
                          protected_ids: set = None) -> List[Source]:
    """Guarantee up to `primary_floor` of the top_k slots for primary sources
    (arxiv/wikipedia) before filling the rest with the best-cosine remainder.
    Canonical seeds (protected_ids) are forced into top_k FIRST regardless of
    cosine score -- they are known-good primary sources injected on purpose
    and must not be demoted by cosine competition.

    Rank6 (primary_floor): bge-m3 cosine has no provider awareness, so well-written
    Tavily blogs out-ranked primary papers -- 114/150 bookv6 sections retained ZERO arxiv.
    Rank5+ (protected_ids): canonical seeds survive the cosine gate but still lose
    to higher-cosine non-seed results in the quota-reserve step.

    This function fixes the Rank5 gap: protected_ids are placed at the FRONT of
    the result list (up to top_k), then the primary_quota fills the remainder
    from the cosine-sorted pool, so seeds are never demoted."""
    protected_ids = protected_ids or set()
    if primary_floor <= 0 and not protected_ids:
        return scored[:top_k]
    # Step 1: extract protected sources that were already sorted into `scored`
    protected_sources = [s for s in scored if s.id in protected_ids]
    protected_ids_seen = {s.id for s in protected_sources}
    rest = [s for s in scored if s.id not in protected_ids_seen]
    # Step 2: reserve primary (arxiv/wiki) from the non-protected rest
    primary = [s for s in rest if (s.provider or "") in _PRIMARY_PROVIDERS]
    reserved = primary[:primary_floor]
    reserved_ids = {id(s) for s in reserved}
    rest_remainder = [s for s in rest if id(s) not in reserved_ids]
    # Step 3: protected sources go FIRST, then primary quota, then cosine remainder
    return (protected_sources + reserved + rest_remainder)[:top_k]


def rank_rrf(sources: List[Source], section_prompt: str, top_k: int = 20,
              embed_model: str = "bge-m3:latest",
              rrf_k: int = 60,
              primary_floor: int = 0,
              protected_ids: set = None,
              # P0c: penalize sources that have appeared in many prior sections this run
              seen_counts: dict = None,
              max_sections_seen: int = 50,
              # Agentic evidence-pool rescue: IDs reused for a STARVED section -> exempt from the P0c
              # diversity penalty (reusing a sibling source IS the rescue; penalizing it defeats it).
              # These still pass the cosine prefilter, so faithfulness is unaffected.
              p0c_exempt_ids: set = None,
              ) -> List[Source]:
    """Score and rank sources using Reciprocal Rank Fusion (RRF) of sparse (BM25) and dense (cosine) retrieval.

    RRF combines the ranking signals from two independent retrieval methods:
      - Sparse: BM25 over tokenized (title + excerpt)
      - Dense:  cosine similarity via bge-m3 embeddings

    Why RRF:
      - Sparse (BM25) captures exact/partial term matches that dense misses
        (e.g. "LLM hallucination", "RLHF", acronyms, code fragments).
      - Dense captures semantic similarity when exact terms differ.
      - RRF is parameter-free and robust: neither signal dominates.

    The fused score for each doc is:
      score = sum(1 / (rrf_k + rank_in_arm)) across all arms
    A doc ranked 1st gets 1/(60+1), ranked 2nd gets 1/(60+2), etc.

    Args:
        sources: Deduplicated list of Source objects.
        section_prompt: The query for dense embedding.
        top_k: How many sources to return. Default 20 (RRK then caps at 8).
        embed_model: Ollama embed model for dense arm.
        rrf_k: RRF k-parameter (default 60 from literature).
        primary_floor: Reserve N slots for arxiv/wikipedia.
        protected_ids: Force-include these IDs regardless of score.

    Returns:
        List of Source objects with `dense_score`, `sparse_score`, and
        `rrf_score` attributes set.
    """
    sources = dedup(sources)
    if not sources:
        return []

    # ---- Sparse arm: BM25 ----
    query_tokens = _tokenize(section_prompt)
    doc_tokens_list = [_tokenize(f"{s.title}. {s.excerpt}") for s in sources]
    doc_lens = [len(tokens) for tokens in doc_tokens_list]
    avg_dl = sum(doc_lens) / max(len(doc_lens), 1)
    n = len(sources)

    sparse_scores = []
    for i in range(n):
        score = _bm25_score(doc_tokens_list[i], query_tokens, avg_dl, doc_lens, i)
        sparse_scores.append(score)

    # Normalize sparse scores to [0,1] using softmax
    max_sparse = max(sparse_scores) if sparse_scores else 1.0
    sparse_norm = [s / max_sparse for s in sparse_scores]

    # ---- Dense arm: bge-m3 cosine ----
    texts = [section_prompt] + [f"{s.title}. {s.excerpt}" for s in sources]
    vectors = embed(texts, model=embed_model)
    if len(vectors) == len(texts):
        query_vec = vectors[0]
        dense_scores = []
        for v in vectors[1:]:
            dense_scores.append(cosine(query_vec, v))
    else:
        print(f"[research/notes] RRF embed failed; using sparse only", flush=True)
        dense_scores = [0.0] * n

    # Attach per-arm scores
    for i, s in enumerate(sources):
        s.sparse_score = sparse_norm[i]
        s.dense_score = dense_scores[i]

    # ---- RRF fusion ----
    # Build rank lists for each arm
    dense_ranked = sorted(range(n), key=lambda i: dense_scores[i], reverse=True)
    sparse_ranked = sorted(range(n), key=lambda i: sparse_scores[i], reverse=True)

    rrf_scores = [0.0] * n
    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(sparse_ranked):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

    # Attach fused score and sort
    # Provider boost: arxiv/wikipedia get a 3x RRF score multiplier so they
    # rise above DDG surface-level results even if cosine/sparse rank is equal.
    # The _apply_primary_quota still guarantees min(primary_floor) slots on top,
    # but the boost ensures they don't just barely scrape in at slot 8.
    BOOST_MULTIPLIER = 3.0
    _PRIMARY_DOMAINS = {"arxiv.org", "en.wikipedia.org"}
    # P0c: seen-count penalty -- sources that appeared in many prior sections get
    # down-ranked to diversify the evidence pool across the run.
    # Formula: score *= max(0.05, (1 - seen_count/max_sections_seen)^2)
    # e.g. seen_count=5/50 -> 0.81x; seen_count=14/16 -> 0.20x; seen_count=30/50 -> 0.06x
    # Protected canonical sources are exempt.
    # Note: this penalty is applied BEFORE _apply_primary_quota reorders by cosine.
    # The final top_k is still controlled by _apply_primary_quota which does NOT
    # re-apply seen_count penalty -- so the penalty reduces RRF score but quota
    # can still override. Primary quota is intentional (arxiv/wikipedia preference).
    seen_counts = seen_counts or {}
    protected_ids = protected_ids or set()
    p0c_exempt_ids = p0c_exempt_ids or set()
    max_sections_seen = max(1, max_sections_seen)
    for i, s in enumerate(sources):
        url = (s.url or "").lower()
        if any(d in url for d in _PRIMARY_DOMAINS):
            rrf_scores[i] *= BOOST_MULTIPLIER
        # P0c: seen-count penalty (skip for protected canonical seeds)
        sid = getattr(s, "id", "") or getattr(s, "url", "") or ""
        seen_count = seen_counts.get(sid, 0)
        if seen_count > 0 and sid not in protected_ids and sid not in p0c_exempt_ids:
            fraction = min(1.0, seen_count / max_sections_seen)
            penalty = max(0.05, (1.0 - fraction) ** 2)
            rrf_scores[i] *= penalty
        s.rrf_score = rrf_scores[i]

    scored = sorted(sources, key=lambda s: s.rrf_score, reverse=True)

    # ---- Primary quota + protected ----
    return _apply_primary_quota(scored, top_k, primary_floor, protected_ids)


def enrich_top_sources(sources: List[Source], top_n: int = 2,
                       max_words_per: int = 350) -> List[Source]:
    """Fetch full text for the top-N highest-relevance sources and replace their
    short search excerpt with a longer extracted body (up to max_words_per).

    Rationale: search APIs return 80-word excerpts which are too thin for the
    writer to quote specifics. Pulling the top-2 sources' full text gives the
    writer 600-700 words of real evidence (vs. ~160 words across all sources).

    Mutates sources in-place AND returns them (chainable). Failures degrade
    silently -- the original short excerpt is kept if full-text fetch fails.
    """
    if not sources:
        return sources
    for s in sources[:top_n]:
        try:
            body = fetch_full_text(s.url, max_words=max_words_per)
        except Exception as e:
            print(f"[research/notes] enrich failed for {s.url}: {e}", flush=True)
            body = ""
        if body and len(body) > len(s.excerpt or ""):
            s.excerpt = body
    return sources


def clean_citations(content: str, n_sources: int) -> tuple:
    """Strip writer-side citation pathologies surfaced by the bookv3 audit:

    1. `[N]` markers where N is outside [1, n_sources] (writer hallucinated index)
    2. `[N]` LITERAL placeholder -- the writer copied the variable name from the
       system prompt instead of filling in a real number
    3. "Tavily (2024)" / "DuckDuckGo (2023)" / "Brave (...)" -- writer treats
       the search provider name as if it were a research-paper author. Replaced
       with neutral "(source)" so the prose doesn't read like the search engine
       authored the paper.

    Returns (cleaned_content, n_dropped). All replacements remove the offending
    marker / attribution and collapse the surrounding whitespace.
    """
    import re as _re
    if not content:
        return content, 0
    dropped = 0

    # MATH/CODE SAFETY: mask $$...$$ / $...$ math spans + ``` code fences so the citation
    # regexes below never strip ML notation like [N] (index set {1..N}), [n], [N,k] (dim
    # list) out of a formula. Verified bug: `$[N]$` -> `$$` and `$$\sum_{i\in[N]}$$` ->
    # broken `\in ` -> tectonic crash, silently. Only inter-span PROSE is cleaned; spans
    # are restored verbatim at the end.
    _spans = []
    def _mask(m):
        _spans.append(m.group(0))
        return f"\x00MATHSPAN{len(_spans) - 1}\x00"
    content = _re.sub(r"```[\s\S]*?```|\$\$[\s\S]+?\$\$|\$[^$\n]+\$", _mask, content)

    # 1: N-PREFIXED PLACEHOLDER markers leaked from the SYS template. bookv6
    #    leaked 77 of these into the rendered book ([N], [N1], [N10], [N3, N7],
    #    [N2, 5]) because the old regex `\[([Nn]|\d+)\]` only caught the bare
    #    `[N]` / `[n]` form, not N fused with a digit. N is never a valid source
    #    index, so any bracket whose tokens are N/n + digits + commas is dropped.
    #    Constrained to citation-shaped content (N, digits, commas, spaces) so it
    #    never eats legit bracketed prose like "[Note 3]".
    _NPLACEHOLDER_RE = _re.compile(r"\[\s*[Nn]\d*(?:\s*,\s*[Nn]?\d+)*\s*\]")
    cleaned, n_nplace = _NPLACEHOLDER_RE.subn("", content)
    dropped += n_nplace

    # 2: numeric [N] out-of-range (writer hallucinated index beyond source count)
    def _repl_num(m):
        nonlocal dropped
        try:
            n = int(m.group(1))
        except ValueError:
            return m.group(0)
        if 1 <= n <= max(n_sources, 1):
            return m.group(0)
        dropped += 1
        return ""
    cleaned = _re.sub(r"\[(\d+)\]", _repl_num, cleaned)

    # 3: provider-as-author attributions -- "Tavily (2024)", "DDG (2023)", etc.
    _PROVIDER_AUTHOR_RE = _re.compile(
        r"\b(?:Tavily|DuckDuckGo|DDG|Brave(?:\s+Search)?|Google\s+Search|Bing)\s*"
        r"(?:et\s+al\.?\s*)?\(\s*\d{4}\s*\)",
        _re.IGNORECASE,
    )
    cleaned, n_provider = _PROVIDER_AUTHOR_RE.subn("(source)", cleaned)
    dropped += n_provider

    # Cleanup: collapse double-spaces / orphan trailing punctuation
    cleaned = _re.sub(r" {2,}", " ", cleaned)
    cleaned = _re.sub(r"\s+([,.;:])", r"\1", cleaned)
    # restore masked math/code spans verbatim
    for _i, _sp in enumerate(_spans):
        cleaned = cleaned.replace(f"\x00MATHSPAN{_i}\x00", _sp)
    return cleaned, dropped


def format_for_prompt(sources: List[Source]) -> str:
    """Render an EVIDENCE block the writer can quote from.

    Output looks like:
        EVIDENCE (cite as [N]; do NOT invent papers or URLs):

        [1] Vaswani et al. (2017). "Attention Is All You Need" -- arxiv:1706.03762
            Excerpt: "We propose a new simple network architecture, the Transformer..."

        [2] Wikipedia. "Transformer (deep learning architecture)"
            Excerpt: "...self-attention computes a weighted sum of values..."

        ---

    Total length is bounded (each excerpt <= 80 words from search.py).
    """
    if not sources:
        return ""

    n_sources = len(sources)
    lines = [
        f"EVIDENCE -- exactly {n_sources} numbered sources are available. Cite inline as [N] "
        f"where 1 <= N <= {n_sources}. Do NOT use any other index. Aim for 5-8 citations "
        "anchored to specific factual claims (numbers, dates, named methods, paper findings). "
        "If a point lacks evidence here, hedge instead of omitting -- write 'recent work suggests "
        "...' without a citation rather than dropping the point.",
        "",
    ]
    for i, s in enumerate(sources, start=1):
        # display_author() never exposes a search-tool brand (Tavily/DDG/Brave).
        authors = s.display_author()
        year = f" ({s.year})" if s.year else ""
        attribution = f"{authors}{year}. " if authors else ""
        lines.append(f'[{i}] {attribution}"{s.title}" -- {s.id}  <{s.url}>')
        if s.excerpt:
            label = "Full text" if len(s.excerpt.split()) > 120 else "Excerpt"
            lines.append(f"    {label}: {s.excerpt}")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# P0a: Section Topic Relevance Gate
# Evaluates whether the evidence pool itself matches the section's domain
# BEFORE the writer ever sees the evidence. This prevents sections from being
# written from entirely wrong-domain sources (e.g., RAG papers for RLHF sections).
# ============================================================================

def check_evidence_domain(
    sources: List[Source],
    section_title: str,
    section_goal: str,
    must_cover_terms: List[str],
    avoid_terms: List[str],
    model: str = "gemma4:e4b",
) -> dict:
    """Gate: does the evidence pool match the section's domain?

    Runs BEFORE the writer, so a failing gate can redirect the query generation
    loop to try different search terms. This is the primary defense against
    topic purity failures where investigation returns evidence from the wrong domain.

    Args:
        sources: ranked list of Source objects (top-8 after rerank)
        section_title: short title of the section
        section_goal: what the section should cover
        must_cover_terms: terms the section must address
        avoid_terms: terms the section should avoid
        model: judge model for LLM-based scoring

    Returns:
        (gate threshold lives at the call site: deep_investigate.py ev_threshold ~= 0.40)
        dict: {topic_relevance: float [0,1], verdict: str, reason: str, score_breakdown: dict}
    """
    if not sources:
        return {
            "topic_relevance": 0.0,
            "verdict": "no_evidence",
            "reason": "no sources provided to gate",
            "score_breakdown": {},
        }

    # Fast path: keyword overlap between section terms and evidence titles/excerpts.
    # This costs zero LLM calls and catches the obvious failures.
    section_keywords = set(
        t.lower() for t in (must_cover_terms or [])
        if t and len(t) > 3
    )
    # Also include title words (>=3 chars)
    for word in re.findall(r"[A-Za-z0-9]{3,}", section_title.lower()):
        section_keywords.add(word)

    evidence_texts = " ".join(
        f"{s.title} {s.excerpt or ''}".lower()
        for s in sources[:8]
    )

    matched = sum(1 for kw in section_keywords if kw in evidence_texts)
    keyword_score = matched / max(len(section_keywords), 1) if section_keywords else 0.5

    # Provider domain audit: check if the evidence pool is dominated by one domain.
    # If top-3 providers are the same family, that's a contamination signal.
    provider_counts: dict = {}
    for s in sources[:8]:
        prov = s.provider or "unknown"
        provider_counts[prov] = provider_counts.get(prov, 0) + 1
    top_provider_pct = max(provider_counts.values()) / max(len(sources[:8]), 1) if sources else 1.0

    # Build evidence digest for LLM
    source_digest = "\n".join(
        f"- [{i+1}] {s.title} ({s.provider}, {s.year or '?'}) -- {s.excerpt[:120]}"
        for i, s in enumerate(sources[:8])
    )

    prompt = f"""You are a domain-alignment judge. Given a section brief and its retrieved evidence,
decide whether the evidence pool actually covers the intended section topic.

SECTION TITLE: {section_title}
SECTION GOAL: {section_goal}
MUST COVER: {', '.join(must_cover_terms) if must_cover_terms else '(none specified)'}
AVOID: {', '.join(avoid_terms) if avoid_terms else '(none specified)'}

EVIDENCE POOL:
{source_digest}

Respond with ONLY a single JSON object (no markdown fences, no prose):
{{"topic_relevance": <float 0-1>, "verdict": "on_topic|partial|off_topic", "reason": "<one short sentence explaining the verdict>"}}

Rules:
- topic_relevance must be anchored to how well the evidence pool matches the section title and goal
- A score >= 0.70 means the evidence is clearly about the right topic
- A score <= 0.30 means the evidence is about the wrong topic entirely
- Scores 0.30-0.70 are partial: evidence mentions related topics but misses the core focus
"""

    try:
        import httpx
        payload = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.1, "num_predict": 200},
            "think": False,
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post("http://localhost:11434/api/chat", json=payload)
            r.raise_for_status()
        raw = (r.json().get("message") or {}).get("content", "").strip()

        # Parse JSON
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            import json as _json
            data = _json.loads(m.group(0))
            llm_score = float(data.get("topic_relevance", 0.5))
            llm_verdict = str(data.get("verdict", "partial")).lower()
            llm_reason = str(data.get("reason", ""))

            # Blend fast-path keyword score with LLM judgment
            # Keyword score is a prior; LLM overrides it when they disagree significantly
            if abs(keyword_score - llm_score) > 0.4:
                # Strong disagreement: weight toward LLM (it sees the full excerpt)
                final_score = llm_score * 0.7 + keyword_score * 0.3
            else:
                final_score = (llm_score + keyword_score) / 2.0

            # Provider contamination penalty: if top provider > 70%, penalize
            if top_provider_pct > 0.50:
                final_score = final_score * 0.85
                llm_reason += f" (provider concentration {top_provider_pct:.0%} may indicate cluster)"

            final_score = max(0.0, min(1.0, round(final_score, 3)))

            verdict = llm_verdict if llm_verdict in ("on_topic", "partial", "off_topic") else "partial"
            if final_score >= 0.70:
                verdict = "on_topic"
            elif final_score < 0.50:
                verdict = "off_topic"
            else:
                verdict = "partial"

            return {
                "topic_relevance": final_score,
                "verdict": verdict,
                "reason": llm_reason,
                "score_breakdown": {
                    "keyword_score": round(keyword_score, 3),
                    "llm_score": llm_score,
                    "provider_pct": round(top_provider_pct, 3),
                },
            }
    except Exception as e:
        print(f"[research/notes] check_evidence_domain LLM failed: {e}", flush=True)

    # Fallback: keyword-only scoring
    final_score = round(keyword_score, 3)
    verdict = "on_topic" if final_score >= 0.60 else "partial" if final_score >= 0.50 else "off_topic"
    return {
        "topic_relevance": final_score,
        "verdict": verdict,
        "reason": f"fallback keyword match: {matched}/{len(section_keywords)} terms found",
        "score_breakdown": {
            "keyword_score": round(keyword_score, 3),
            "llm_score": None,
            "provider_pct": round(top_provider_pct, 3),
        },
    }
