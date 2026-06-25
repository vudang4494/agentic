"""Shared data types for the research layer."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from urllib.parse import urlparse


# Search-provider/tool names that must NEVER appear as a citation author.
# bookv6 leaked ~220 "Tavily (2024)"-style attributions because the author
# fallback used `provider.capitalize()`. Tavily/DDG/Brave are retrieval tools,
# not researchers.
_PROVIDER_TOOL_NAMES = {"tavily", "ddg", "duckduckgo", "brave", "bing", "google"}


def _host_label(url: str) -> str:
    """Neutral publisher label from a URL host: 'arxiv.org' -> 'arxiv.org',
    'cameronrwolfe.substack.com' -> 'cameronrwolfe.substack.com'. '' on failure."""
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return h[4:] if h.startswith("www.") else h


def neutral_author(authors, provider: str, url: str) -> str:
    """Citation author label that NEVER exposes a search-tool brand. Works on
    raw fields so it is reusable from both Source objects and state.json dicts.
    Priority: real authors -> Wikipedia -> publisher host -> '' (rely on title)."""
    authors = authors or []
    if authors:
        label = ", ".join(authors[:3])
        if len(authors) > 3:
            label += " et al."
        return label
    if (provider or "").lower() == "wikipedia":
        return "Wikipedia"
    if (provider or "").lower() in _PROVIDER_TOOL_NAMES:
        host = _host_label(url)
        return host if host and "tavily" not in host else ""
    host = _host_label(url)
    return host if host and "tavily" not in host else ""


@dataclass
class Source:
    """A single retrieved source (arxiv paper, wikipedia article, web page)."""
    id: str               # "arxiv:1706.03762" | "wiki:Transformer_(deep_learning)" | "url:<sha1>"
    title: str
    url: str
    excerpt: str          # cleaned text excerpt fed to the writer (~80 words)
    provider: str         # "arxiv" | "wikipedia" | "ddg"
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    relevance: float = 0.0  # cosine similarity to section prompt, populated by notes.rank()

    def to_dict(self) -> dict:
        return asdict(self)

    def display_author(self) -> str:
        """Citation author label that NEVER exposes a search-tool brand.
        Priority: real authors -> Wikipedia -> publisher host -> '' (rely on title)."""
        return neutral_author(self.authors, self.provider, self.url)

    def citation(self) -> str:
        """Author-year inline citation form: '(Vaswani et al., 2017)' or '(Wikipedia, 2024)'."""
        if self.authors:
            first = self.authors[0].split()[-1] if self.authors else ""
            etal = " et al." if len(self.authors) > 1 else ""
            year_part = f", {self.year}" if self.year else ""
            return f"({first}{etal}{year_part})"
        if self.provider == "wikipedia":
            return f"(Wikipedia{', ' + str(self.year) if self.year else ''})"
        return f"({self.id})"

    def reference_line(self, n: int) -> str:
        """Single-line bibliography entry for the References page.
        Author label never exposes a search-tool brand (uses display_author)."""
        authors_str = self.display_author()
        year_str = f" ({self.year})" if self.year else ""
        # Build the attribution segment so an empty author + present year never
        # produces "[n]  (2024)_Title_." (double space + year glued to title).
        attribution = (authors_str + year_str).strip()
        prefix = f"{attribution}. " if attribution else ""
        return f"[{n}] {prefix}_{self.title}_. {self.url}"


@dataclass
class Query:
    """A search query proposed by the query generator."""
    q: str               # the actual search string
    intent: str = ""     # short tag: "primary source" | "supporting" | "definition" | etc.

    def to_dict(self) -> dict:
        return asdict(self)
