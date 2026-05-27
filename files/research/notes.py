"""Dedup, rank, and format research sources into an EVIDENCE block for the writer."""
from typing import List

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


# Domains Tavily occasionally returns that are almost never useful for technical
# research synthesis (corporate marketing, low-quality aggregators, video-only).
# We don't HARD-block them -- if no other source matches, they can still appear --
# but we boost their similarity threshold so they have to be highly relevant.
_NOISY_DOMAINS = (
    "youtube.com", "vimeo.com", "tiktok.com",
    "duckduckgo.com",   # caught the "DuckDuckGo's history" false-match in Ch11.4
    "facebook.com", "twitter.com", "x.com", "linkedin.com",
    "reddit.com",       # often discussion, low signal-per-token
    "pinterest.com",
    # Blog/aggregator platforms -- 2026-05-27 bookv3_pilot eval caught
    # medium.com x5, cameronrwolfe.substack.com x1, vizuara.substack.com x1
    # leaking past at cosine 0.45-0.55 threshold. They're substring-matched,
    # so "*.substack.com" / "*.medium.com" subdomains are covered.
    "medium.com", "substack.com", "techcrunch.com",
    "towardsdatascience.com",  # also a medium subsite, but bare domain too
)


def _is_noisy_domain(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(d in u for d in _NOISY_DOMAINS)


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
    if dropped_noisy or dropped_offtopic:
        print(f"[research/notes] prefilter dropped {dropped_offtopic} off-topic + "
              f"{dropped_noisy} noisy-domain (kept {len(kept)}/{len(sources)})",
              flush=True)
    return kept


def rank(sources: List[Source], section_prompt: str, top_k: int = 8,
         embed_model: str = "bge-m3:latest") -> List[Source]:
    """Score sources by cosine similarity to the section prompt, return top_k.

    Falls back to keyword overlap if the embedding call fails (no relevance
    scores assigned in that case; sources returned in their original order
    truncated to top_k).
    """
    sources = dedup(sources)
    if not sources:
        return []
    if len(sources) <= top_k:
        # Still compute relevance scores even if we'd take them all -- the
        # writer uses these to know which sources matter most.
        pass

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
    return scored[:top_k]


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

    # 1 + 2: numeric [N] handling (out-of-range AND literal "[N]")
    def _repl_num(m):
        nonlocal dropped
        n_text = m.group(1)
        if n_text.upper() == "N":
            # literal "[N]" placeholder leaked from SYS prompt
            dropped += 1
            return ""
        try:
            n = int(n_text)
        except ValueError:
            return m.group(0)
        if 1 <= n <= max(n_sources, 1):
            return m.group(0)
        dropped += 1
        return ""
    cleaned = _re.sub(r"\[([Nn]|\d+)\]", _repl_num, content)

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
        authors = ", ".join(s.authors[:3]) if s.authors else s.provider.capitalize()
        if s.authors and len(s.authors) > 3:
            authors += " et al."
        year = f" ({s.year})" if s.year else ""
        lines.append(f'[{i}] {authors}{year}. "{s.title}" -- {s.id}  <{s.url}>')
        if s.excerpt:
            label = "Full text" if len(s.excerpt.split()) > 120 else "Excerpt"
            lines.append(f"    {label}: {s.excerpt}")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)
