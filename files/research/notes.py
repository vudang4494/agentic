"""Dedup, rank, and format research sources into an EVIDENCE block for the writer."""
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
              min_relevance: float = 0.45,
              noisy_min_relevance: float = 0.60,
              embed_model: str = "bge-m3:latest") -> List[Source]:
    """Drop obviously off-topic sources BEFORE the main rank().

    Two thresholds (tightened 2026-05-27 after eval found off-topic papers
    like "Hyper Loop Algebras" and "HEVC Encoding Energy" sneaking past at
    0.30/0.55):
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
    kept = []
    dropped_noisy = 0
    dropped_offtopic = 0
    for s, v in zip(sources, vectors[1:]):
        rel = cosine(qv, v)
        s.relevance = rel  # cache for rank() to reuse
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


def _apply_primary_quota(scored: List[Source], top_k: int, primary_floor: int) -> List[Source]:
    """Guarantee up to `primary_floor` of the top_k slots for primary sources
    (arxiv/wikipedia) before filling the rest with the best-cosine remainder.

    Rank6: bge-m3 cosine has no provider awareness, so well-written Tavily blogs
    out-ranked primary papers -- 114/150 bookv6 sections retained ZERO arxiv.
    This reserves slots for primary sources when they exist, without wasting a
    slot when they don't (degrades gracefully to pure cosine)."""
    if primary_floor <= 0:
        return scored[:top_k]
    primary = [s for s in scored if (s.provider or "") in _PRIMARY_PROVIDERS]
    reserved = primary[:primary_floor]
    reserved_ids = {id(s) for s in reserved}
    rest = [s for s in scored if id(s) not in reserved_ids]
    return (reserved + rest)[:top_k]


def rank(sources: List[Source], section_prompt: str, top_k: int = 8,
         embed_model: str = "bge-m3:latest", precomputed: bool = False,
         primary_floor: int = 0) -> List[Source]:
    """Score sources by cosine similarity to the section prompt, return top_k.

    Rank13: precomputed=True reuses prefilter's cached s.relevance (skips a
    redundant embed). Rank6: primary_floor reserves N of top_k slots for
    arxiv/wikipedia primary sources.

    Falls back to insertion order if the embedding call fails.
    """
    sources = dedup(sources)
    if not sources:
        return []
    if precomputed and all(getattr(s, "relevance", 0.0) for s in sources):
        scored = sorted(sources, key=lambda s: s.relevance, reverse=True)
        return _apply_primary_quota(scored, top_k, primary_floor)

    texts = [section_prompt] + [f"{s.title}. {s.excerpt}" for s in sources]
    vectors = embed(texts, model=embed_model)
    if len(vectors) != len(texts):
        print(f"[research/notes] embedding failed; falling back to insertion order", flush=True)
        return sources[:top_k]

    query_vec = vectors[0]
    scored = []
    for s, v in zip(sources, vectors[1:]):
        s.relevance = cosine(query_vec, v)
        scored.append(s)
    scored.sort(key=lambda s: s.relevance, reverse=True)
    return _apply_primary_quota(scored, top_k, primary_floor)


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
