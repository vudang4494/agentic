"""Disk-cached HTTP fetcher for research sources.

Cache layout: research/cache/<sha1(url)>.json containing
    {"url": ..., "fetched_at": ..., "status": int, "content": str, "headers": {...}}

The cache is content-addressed by URL so a re-run never re-hits the network for
the same source. Use clear_cache() to wipe and force-refresh.
"""
import hashlib
import json
import json as _json
import re
import time
from pathlib import Path
from typing import Optional

import httpx

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
USER_AGENT = "AgentDeepLearning/0.2 (research layer; local pipeline; contact via repo)"
TIMEOUT = 10.0  # arxiv.org is fast; 30s causes 60s wait (interval+sleep+timeout)


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"


def fetch(url: str, accept: Optional[str] = None, force: bool = False, timeout: float = 10.0) -> Optional[dict]:
    """Fetch URL, returning {url, status, content, fetched_at, headers}.

    Reads from cache unless force=True. Returns None on network failure with no
    cached copy. Network/HTTP errors are swallowed -- the research layer must
    degrade gracefully.
    """
    cp = _cache_path(url)
    if cp.exists() and not force:
        try:
            return json.loads(cp.read_text())
        except Exception:
            pass  # corrupt cache entry, fall through to refetch

    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.get(url, headers=headers)
        record = {
            "url": url,
            "status": r.status_code,
            "content": r.text if r.status_code < 400 else "",
            "fetched_at": time.time(),
            "headers": dict(r.headers),
        }
    except Exception as e:
        print(f"[research/fetch] WARN: {url} -> {e}", flush=True)
        return None

    try:
        cp.write_text(json.dumps(record, ensure_ascii=False))
    except Exception:
        pass
    return record


def clear_cache() -> int:
    """Wipe the on-disk cache. Returns number of files removed."""
    n = 0
    for f in CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            n += 1
        except Exception:
            pass
    return n


# ---------------------------------------------------------------------------
# Full-text extraction
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Heuristic HTML -> readable text. Tries trafilatura if installed, falls back
    to a script/style strip + tag stripper. Good enough for the writer to read
    a few extra paragraphs of evidence beyond the 80-word search excerpts."""
    try:
        import trafilatura
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted and len(extracted) > 200:
            return extracted
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: strip script/style blocks, drop tags, collapse whitespace
    import re as _re
    text = _re.sub(r"<script[\s\S]*?</script>", " ", html, flags=_re.IGNORECASE)
    text = _re.sub(r"<style[\s\S]*?</style>", " ", text, flags=_re.IGNORECASE)
    text = _re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                 .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    text = _re.sub(r"\s+", " ", text).strip()
    return text


_BINARY_EXTS = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
                 ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".mp4")


def _looks_binary(record: dict) -> bool:
    """Detect PDF / office / image / archive content even when the URL extension is missing.

    Heuristics: HTTP Content-Type, leading magic bytes, and content control-char ratio.
    Used to short-circuit fetch_full_text before we try to render PDF bytes as text
    (which is what produced the 'binary leak' on Ch4.3 in the May-23 run).
    """
    if not record:
        return True
    ctype = (record.get("headers", {}) or {}).get("content-type", "").lower()
    if any(t in ctype for t in ("pdf", "msword", "octet-stream",
                                 "image/", "video/", "audio/", "zip")):
        return True
    content = record.get("content", "") or ""
    if content.startswith("%PDF") or content.startswith("PK\x03\x04"):
        return True
    # control-char ratio sniff
    if content:
        sample = content[:2000]
        ctrl = sum(1 for c in sample if c < " " and c not in "\t\r\n")
        if ctrl / max(len(sample), 1) > 0.05:
            return True
    return False


def _mathml_to_latex(html: str) -> str:
    """arxiv /html/ pages embed equations as <math>...<annotation
    encoding="application/x-tex">LATEX</annotation>...</math>. Replace each <math> block
    with its TeX annotation as inline $...$ so the equation SURVIVES _html_to_text (which
    would otherwise flatten MathML to noise). This is what lets the writer + grounding
    scorer see the paper's REAL formulas instead of the writer inventing them from memory."""
    import re as _re

    def _repl(m):
        a = _re.search(r'<annotation[^>]*encoding="application/x-tex"[^>]*>(.*?)</annotation>',
                       m.group(0), _re.DOTALL)
        if not a:
            return " "
        # Map &lt;/&gt; to math-safe \lt/\gt (valid in math mode), NOT raw < > -- otherwise
        # the downstream _html_to_text tag-stripper `<[^>]+>` eats the math '<' PLUS all
        # prose up to the next real '>' (verified on 1706.03762: k<n destroyed ~40 words +
        # 2 formulas). So the inlined $...$ must carry no bare angle brackets.
        tex = (a.group(1).replace("&lt;", r" \lt ").replace("&gt;", r" \gt ")
               .replace("&amp;", "&").replace("&quot;", '"')).strip()
        return f" $ {tex} $ " if tex else " "

    return _re.sub(r"<math\b[^>]*>.*?</math>", _repl, html, flags=_re.DOTALL | _re.IGNORECASE)


def _pick_math_window(text: str, max_words: int) -> str:
    """When the body exceeds max_words, return the window RICHEST in math/equation tokens
    so the writer gets the method/equation region, not the title/ToC/intro."""
    import re as _re
    words = text.split()
    if len(words) <= max_words:
        return text
    best_i, best = 0, -1
    step = max(1, max_words // 2)
    for i in range(0, len(words) - max_words + 1, step):
        chunk = " ".join(words[i:i + max_words])
        score = len(_re.findall(r"\\(?:frac|sqrt|sum|prod|softmax|text|mathbf|cdot)|[=^_]|\$", chunk))
        if score > best:
            best, best_i = score, i
    return " ".join(words[best_i:best_i + max_words])


def fetch_full_text(url: str, max_words: int = 350) -> str:
    """Fetch a page and return up to max_words of readable body text.

    Optimisations:
    - arxiv abstract pages: jump to the /pdf/ or /html/ form for cleaner text.
    - Wikipedia /wiki/X URLs: hit the REST plain-extract endpoint instead.
    - Everything else: fetch the URL, extract via trafilatura or tag-strip.
    - PDF / office / image / archive URLs: skip entirely (return ""). Writer
      shouldn't see binary bytes; the short search-excerpt stays the source's
      contribution.

    On any failure returns an empty string -- callers must be ready for that.
    """
    if not url:
        return ""
    # Cheap pre-check by extension
    if any(url.lower().split("?")[0].endswith(ext) for ext in _BINARY_EXTS):
        return ""

    # Wikipedia -- use plain text extract endpoint (gives clean readable content)
    if "wikipedia.org/wiki/" in url:
        import urllib.parse as _up
        raw_slug = url.split("/wiki/", 1)[1].split("#")[0]
        # Title to the MW API must be the human-readable form, not double-encoded.
        title = _up.unquote(raw_slug).replace("_", " ")
        slug = _up.quote(title.replace(" ", "_"))
        plain = fetch(
            f"https://en.wikipedia.org/w/api.php?format=json&action=query"
            f"&prop=extracts&explaintext=1&exsectionformat=plain&titles={slug}",
            accept="application/json",
        )
        if plain and plain.get("status", 0) < 400 and plain.get("content"):
            try:
                data = _json.loads(plain["content"])
                pages = (data.get("query") or {}).get("pages") or {}
                for p in pages.values():
                    txt = p.get("extract", "")
                    if txt:
                        words = txt.split()
                        return " ".join(words[:max_words])
            except Exception:
                pass

    # arxiv -- PREFER the /html/ full paper: it carries the Method section + equations as
    # MathML/TeX, whereas /abs/ is only the abstract (0 equations -> the writer would invent
    # formulas from memory). Additive: on 404/empty (pre-2023 papers have no /html/) we fall
    # back to the /abs/ abstract path below, so this never loses information or hard-fails.
    m_aid = re.search(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url, re.IGNORECASE)
    if m_aid:
        hrec = fetch(f"https://arxiv.org/html/{m_aid.group(1)}", accept="text/html")
        if hrec and hrec.get("status", 0) < 400 and hrec.get("content") and not _looks_binary(hrec):
            htext = _html_to_text(_mathml_to_latex(hrec["content"]))  # MathML -> TeX BEFORE tag-strip
            htext = _pick_math_window(htext, max_words)                # equation-dense region, not ToC
            hwords = htext.split()
            if len(hwords) >= 40:  # got a real body
                return " ".join(hwords[:max_words])
        # else: fall through to the /abs/ abstract path

    rec = fetch(url, accept="text/html")
    if not rec or rec.get("status", 0) >= 400 or not rec.get("content"):
        return ""
    # Don't try to render PDF / image / archive bytes as text.
    if _looks_binary(rec):
        return ""

    text = _html_to_text(rec["content"])
    if not text:
        return ""

    # For arxiv pages, lop off the standard nav/header boilerplate by jumping
    # to the abstract block if we can find it.
    if "arxiv.org" in url:
        m = re.search(r"Abstract[:\s]+([\s\S]+?)Subjects[:\s]", text, flags=re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    words = text.split()
    return " ".join(words[:max_words])
