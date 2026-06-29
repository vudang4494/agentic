"""Intra-book citation cleaner (Stage-F hygiene).

The writer (Qwen iq3) is fed the titles of sibling/prior sections as cross-reference
context. It then NAME-DROPS those section titles inline as if they were external cited
papers -- e.g. "As noted in *Data Collection for On-Policy Trajectories in LLMs*, the
quality of advantages depends ...". On book_900 this polluted ~1282 inline phrases
across 98% of sections, inflating cross_refs and corrupting citation hygiene
(a section "cites" the book's own other sections as authorities).

This is NOT a hallucinated EXTERNAL paper -- it is the book's own section title. So the
fix is deterministic: detect a delimited phrase that EXACTLY matches one of the book's
section titles, and strip the citation-framing clause ("As <verb> in *Title*,"), keeping
the underlying claim as plain prose. Real external citations (`[N]` -> retrieved sources)
and real external paper titles are untouched because they never match a section title.

Doctrine: this is an ORCHESTRATION post-process (assemble-time hygiene), not a model
change. Verifier != writer is unaffected. Single source of truth lives here.
"""
import re

# Framing verbs the writer uses before a name-dropped section title.
_FRAME_VERBS = (
    r"(?:noted|discussed|detailed|highlighted|established|described|outlined|explored|"
    r"seen|demonstrated|identified|mentioned|analyzed|shown|explained|illustrated|"
    r"presented|examined|introduced|covered|reviewed|explained|argued|emphasized)"
)

# Noun-framings: "findings in *Title*", "the results of *Title*", "insights from *Title*".
_FRAME_NOUNS = (
    r"(?:findings?|results?|insights?|conclusions?|observations?|discussions?|sections?|"
    r"analysis|treatment|methods?|techniques?|approaches?|concepts?|principles?)"
)

# Optional leading ", as" / "As" + verb + (in|by|from) + delimited title + optional [N] + trailing comma.
_FRAME_RE = re.compile(
    r"(?:,\s*)?\b(?:[Aa]s\s+)?" + _FRAME_VERBS + r"\s+(?:in|by|from)\s+"
    r"([*'\"])(?P<title>\[?[^*'\"\n]{15,170}?\]?)\1"
    r"(?:\s*\[\d+\])?\s*,?\s*"
)

# Secondary: noun-framed name-drops "findings in/of/from *Title*" (+ optional prior/earlier).
_BARE_RE = re.compile(
    r"(?:,\s*)?\b(?:prior\s+|earlier\s+|previous\s+|the\s+)?" + _FRAME_NOUNS +
    r"\s+(?:on|of|in|from)\s+([*'\"])(?P<title>\[?[^*'\"\n]{15,170}?\]?)\1"
    r"(?:\s*\[\d+\])?\s*,?\s*"
)

# Subject-position: "*Title* demonstrates/shows/... that <claim>" -> drop the fake attribution,
# keep the that-clause as a direct statement. REQUIRES "that" so we never leave a noun fragment.
_REPORT_VERBS = (
    r"(?:demonstrates?|shows?|highlights?|illustrates?|notes?|argues?|establishes?|"
    r"suggests?|finds?|reveals?|describes?|presents?|introduces?|emphasizes?|indicates?|"
    r"reports?|confirms?|proposes?|outlines?|explores?|examines?|discusses?|posits?|"
    r"underscores?|advocates?|states?|asserts?|explains?|concludes?)"
)
_SUBJECT_RE = re.compile(
    r"(?:^|(?<=[\s.,;:]))([*'\"])(?P<title>\[?[^*'\"\n]{15,170}?\]?)\1\s+"
    r"(?:" + _REPORT_VERBS + r"|\w+(?:s|ed))\s+that\s+"
)

# Tertiary: connector framings "related/compared/central/applied <conn> *Title*".
_CONN_RE = re.compile(
    r"(?:,\s*)?\b(?:related|compared|comparable|central|core|applied|relative|similar|akin|"
    r"connected|linked|tied|analogous|contrasted)\s+(?:to|with)\s+(?:the\s+)?"
    r"([*'\"])(?P<title>\[?[^*'\"\n]{15,170}?\]?)\1"
    r"(?:\s*\[\d+\])?\s*,?\s*"
)

# Locative/object preposition + (optional article) + title (+ optional trailing "section/
# chapter/framework" noun) + optional [N] + comma. Removing a prepositional/object phrase is
# grammatically safe. Guarded by title membership so external refs are untouched.
_LOC_RE = re.compile(
    r"(?:,\s*)?\b(?:[Ii]n|[Ww]ithin|[Tt]hroughout|[Aa]cross|[Vv]ia|[Uu]sing|[Pp]er|[Ff]ollowing|"
    r"[Uu]nlike|[Ll]ike|[Aa]nd|[Oo]r|[Ss]pecifically|[Ss]ee)\s+(?:the\s+)?"
    r"([*'\"])(?P<title>\[?[^*'\"\n]{15,170}?\]?)\1"
    r"(?:\s+(?:section|chapter|framework|approach|model|paper|work|study|pipeline))?"
    r"(?:\s*\[\d+\])?\s*,?\s*"
)

# "the '<Title>' section/framework <verb>" subject built on a noun -> drop the title, keep noun.
_SECNOUN_RE = re.compile(
    r"\bthe\s+([*'\"])(?P<title>\[?[^*'\"\n]{15,170}?\]?)\1\s+"
    r"(?=(?:section|chapter|framework|approach|model|method|paper|work|study))"
)


def _norm(s: str) -> str:
    # strip wrapping markdown/brackets the writer sometimes adds: *[Title]*, "Title", (Title)
    s = (s or "").strip().strip("[]()").strip()
    return re.sub(r"\s+", " ", s).strip().lower()


def _capitalize_sentence_starts(text: str) -> str:
    # Capitalize the first alphabetic char after a sentence-ending mark + space (fixes a
    # clause we removed that began a sentence: "...simultaneously. recent work" -> "Recent").
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def clean_intrabook_citations(content: str, title_set):
    """Strip inline name-drops of the book's own section titles.

    Args:
        content: section markdown body.
        title_set: iterable of ALL section titles in the book (the cross-ref pool).

    Returns:
        (cleaned_content, n_removed)
    """
    if not content or not title_set:
        return content, 0
    norm_titles = {_norm(t) for t in title_set if len(_norm(t)) >= 18}
    if not norm_titles:
        return content, 0

    removed = [0]

    def _repl(m):
        if _norm(m.group("title")) in norm_titles:
            removed[0] += 1
            return " "
        return m.group(0)

    out = _FRAME_RE.sub(_repl, content)
    out = _BARE_RE.sub(_repl, out)
    out = _CONN_RE.sub(_repl, out)
    out = _SUBJECT_RE.sub(_repl, out)
    out = _SECNOUN_RE.sub(_repl, out)
    out = _LOC_RE.sub(_repl, out)

    if removed[0]:
        # Whitespace/punct cleanup -- spaces/tabs ONLY (never newlines: keep paragraph breaks).
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"[ \t]+([,.;:])", r"\1", out)
        out = re.sub(r",\s*,", ", ", out)
        out = re.sub(r"\(\s*\)", "", out)
        out = re.sub(r"[ \t]+\n", "\n", out)
        out = _capitalize_sentence_starts(out)
    return out, removed[0]


def clean_book_sections(sections: dict, title_set) -> int:
    """In-place clean of a {key: {content,...}} map. Returns total phrases removed."""
    total = 0
    for v in sections.values():
        body = v.get("content")
        if not body:
            continue
        cleaned, n = clean_intrabook_citations(body, title_set)
        if n:
            v["content"] = cleaned
            total += n
    return total
