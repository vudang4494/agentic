"""Shared data types for the research layer."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


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
        """Single-line bibliography entry for the References page."""
        authors_str = ", ".join(self.authors) if self.authors else self.provider.capitalize()
        year_str = f" ({self.year})" if self.year else ""
        return f"[{n}] {authors_str}{year_str}. _{self.title}_. {self.url}"


@dataclass
class Query:
    """A search query proposed by the query generator."""
    q: str               # the actual search string
    intent: str = ""     # short tag: "primary source" | "supporting" | "definition" | etc.

    def to_dict(self) -> dict:
        return asdict(self)
