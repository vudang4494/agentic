"""Search provider adapters for the research layer.

Each provider function takes a query string and returns a list of Source. All
provider calls degrade silently on network/parse errors -- callers should
expect possibly-empty results.

Providers:
  - tavily    : api.tavily.com (AI-friendly web search)            -- on if TAVILY_API_KEY set
  - arxiv     : export.arxiv.org/api/query (Atom XML)              -- on by default
  - wikipedia : en.wikipedia.org/w/api.php + REST page summary      -- on by default
  - ddg       : DuckDuckGo HTML scrape                             -- OFF by default
"""
import json as _json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Iterable, List

import httpx

from .fetch import fetch
from .types import Query, Source

ARXIV_API = "https://export.arxiv.org/api/query"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"
DDG_HTML = "https://html.duckduckgo.com/html/"
TAVILY_API = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 30.0
BRAVE_API = "https://api.search.brave.com/res/v1/web/search"
BRAVE_TIMEOUT = 20.0

# Session-level kill switch: once tavily returns HTTP 432 (rate-limit) we stop
# calling it for the rest of the process so we don't burn ~30s/section retrying.
_TAVILY_DISABLED_THIS_SESSION = False
_TAVILY_FAILURE_COUNT = 0
_TAVILY_FAILURE_THRESHOLD = 3

# Be polite -- arxiv ToS asks for >= 3s between requests.
_LAST_ARXIV_CALL = 0.0
ARXIV_MIN_INTERVAL = 3.0


def _excerpt(text: str, max_words: int = 80) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    words = text.split(" ")
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


# ---------------------------------------------------------------------------
# arxiv
# ---------------------------------------------------------------------------

_ATOM_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def arxiv_search(query: str, k: int = 3) -> List[Source]:
    """Query arxiv's Atom API. Returns up to k Sources with abstract excerpts."""
    global _LAST_ARXIV_CALL
    elapsed = time.time() - _LAST_ARXIV_CALL
    if elapsed < ARXIV_MIN_INTERVAL:
        time.sleep(ARXIV_MIN_INTERVAL - elapsed)
    _LAST_ARXIV_CALL = time.time()

    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": k,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    rec = fetch(url, accept="application/atom+xml")
    if not rec or rec.get("status", 0) >= 400 or not rec.get("content"):
        return []

    try:
        root = ET.fromstring(rec["content"])
    except ET.ParseError:
        return []

    out: List[Source] = []
    for entry in root.findall("a:entry", _ATOM_NS):
        title_el = entry.find("a:title", _ATOM_NS)
        summary_el = entry.find("a:summary", _ATOM_NS)
        id_el = entry.find("a:id", _ATOM_NS)
        pub_el = entry.find("a:published", _ATOM_NS)
        if title_el is None or id_el is None:
            continue
        title = (title_el.text or "").strip()
        abs_url = (id_el.text or "").strip()
        arxiv_id = abs_url.rsplit("/", 1)[-1] if abs_url else ""
        year = None
        if pub_el is not None and pub_el.text:
            try:
                year = int(pub_el.text[:4])
            except ValueError:
                pass
        authors = [
            (a.findtext("a:name", default="", namespaces=_ATOM_NS) or "").strip()
            for a in entry.findall("a:author", _ATOM_NS)
        ]
        authors = [a for a in authors if a]
        excerpt = _excerpt(summary_el.text if summary_el is not None else "", max_words=80)
        out.append(Source(
            id=f"arxiv:{arxiv_id}",
            title=title,
            url=abs_url,
            excerpt=excerpt,
            provider="arxiv",
            authors=authors,
            year=year,
        ))
    return out


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def wiki_search(query: str, k: int = 2) -> List[Source]:
    """Search Wikipedia and pull the page summary for the top-k results."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "format": "json",
        "srsearch": query, "srlimit": k, "srprop": "snippet|timestamp",
    })
    rec = fetch(f"{WIKI_SEARCH}?{params}", accept="application/json")
    if not rec or rec.get("status", 0) >= 400:
        return []
    try:
        import json as _json
        data = _json.loads(rec["content"])
    except Exception:
        return []
    hits = data.get("query", {}).get("search", [])
    out: List[Source] = []
    for hit in hits[:k]:
        title = hit.get("title", "")
        if not title:
            continue
        slug = urllib.parse.quote(title.replace(" ", "_"))
        summary_rec = fetch(f"{WIKI_SUMMARY}/{slug}", accept="application/json")
        excerpt = ""
        year = None
        if summary_rec and summary_rec.get("status", 0) < 400:
            try:
                sd = _json.loads(summary_rec["content"])
                excerpt = _excerpt(sd.get("extract", ""), max_words=80)
            except Exception:
                pass
        if not excerpt:
            # Fall back to the search-result snippet (has <span class="searchmatch"> tags).
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            excerpt = _excerpt(snippet, max_words=80)
        ts = hit.get("timestamp", "")
        if ts and len(ts) >= 4:
            try:
                year = int(ts[:4])
            except ValueError:
                pass
        out.append(Source(
            id=f"wiki:{title.replace(' ', '_')}",
            title=title,
            url=f"https://en.wikipedia.org/wiki/{slug}",
            excerpt=excerpt,
            provider="wikipedia",
            authors=[],
            year=year,
        ))
    return out


# ---------------------------------------------------------------------------
# Tavily (web search built for AI agents)
# ---------------------------------------------------------------------------

def _tavily_api_key() -> str:
    """Look up Tavily key from env. Empty string disables the provider silently."""
    return os.environ.get("TAVILY_API_KEY", "").strip()


def tavily_search(query: str, k: int = 5, depth: str = "advanced") -> List[Source]:
    """Tavily AI-friendly web search. Returns up to k Sources with content excerpts.

    Falls back to an empty list on any failure (missing key, network, parse).
    Auto-disables itself for the rest of the session after 3 HTTP 432 (rate-limit)
    failures so a quota-exhausted Tavily account doesn't add 30s of retries to
    every section for the remainder of a multi-hour run.
    """
    global _TAVILY_DISABLED_THIS_SESSION, _TAVILY_FAILURE_COUNT
    if _TAVILY_DISABLED_THIS_SESSION:
        return []
    key = _tavily_api_key()
    if not key:
        return []
    payload = {
        "api_key": key,
        "query": query,
        "search_depth": depth,
        "max_results": k,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    try:
        with httpx.Client(timeout=TAVILY_TIMEOUT) as c:
            r = c.post(TAVILY_API, json=payload)
            r.raise_for_status()
            data = r.json()
        _TAVILY_FAILURE_COUNT = 0
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 432:
            _TAVILY_FAILURE_COUNT += 1
            if _TAVILY_FAILURE_COUNT >= _TAVILY_FAILURE_THRESHOLD:
                _TAVILY_DISABLED_THIS_SESSION = True
                print(f"[research/search] tavily rate-limit (HTTP 432) hit {_TAVILY_FAILURE_THRESHOLD}x -- "
                      "auto-disabled for this session. Re-enable by restarting the process.",
                      flush=True)
            else:
                print(f"[research/search] tavily 432 ({_TAVILY_FAILURE_COUNT}/{_TAVILY_FAILURE_THRESHOLD})", flush=True)
        else:
            print(f"[research/search] tavily HTTP {e.response.status_code}: {e}", flush=True)
        return []
    except Exception as e:
        print(f"[research/search] tavily failed: {e}", flush=True)
        return []

    out: List[Source] = []
    for hit in (data.get("results") or [])[:k]:
        url = hit.get("url", "")
        title = (hit.get("title") or "").strip()
        excerpt = _excerpt(hit.get("content") or "", max_words=160)  # tavily gives richer excerpts
        if not url or not title:
            continue
        # Best-effort year extraction from URL or content (e.g. /2024/, "2024-")
        year = None
        for src in (url, hit.get("content") or ""):
            m = re.search(r"\b(20\d{2})\b", src)
            if m:
                try:
                    y = int(m.group(1))
                    if 1990 <= y <= 2030:
                        year = y
                        break
                except ValueError:
                    pass
        out.append(Source(
            id=f"tavily:{abs(hash(url)) & 0xFFFFFF:06x}",
            title=title,
            url=url,
            excerpt=excerpt,
            provider="tavily",
            authors=[],
            year=year,
        ))
    return out


# ---------------------------------------------------------------------------
# Brave Search (free tier ~2000 queries/month at https://brave.com/search/api/)
# ---------------------------------------------------------------------------

def _brave_api_key() -> str:
    return os.environ.get("BRAVE_API_KEY", "").strip()


def brave_search(query: str, k: int = 5) -> List[Source]:
    """Brave Search API. AI-friendly results with rich snippets; closest free
    substitute for Tavily. Requires BRAVE_API_KEY env (get one at brave.com/search/api/)."""
    key = _brave_api_key()
    if not key:
        return []
    params = {"q": query, "count": k, "result_filter": "web"}
    headers = {"Accept": "application/json", "X-Subscription-Token": key}
    try:
        with httpx.Client(timeout=BRAVE_TIMEOUT) as c:
            r = c.get(BRAVE_API, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"[research/search] brave failed: {e}", flush=True)
        return []
    out: List[Source] = []
    for hit in (data.get("web", {}).get("results") or [])[:k]:
        url = hit.get("url", "")
        title = (hit.get("title") or "").strip()
        excerpt = _excerpt(hit.get("description") or hit.get("snippet") or "", max_words=140)
        if not url or not title:
            continue
        year = None
        page_age = hit.get("page_age", "")
        m = re.search(r"\b(20\d{2})\b", page_age + " " + url)
        if m:
            try:
                year = int(m.group(1))
                if not (1990 <= year <= 2030):
                    year = None
            except ValueError:
                pass
        out.append(Source(
            id=f"brave:{abs(hash(url)) & 0xFFFFFF:06x}",
            title=title, url=url, excerpt=excerpt,
            provider="brave", authors=[], year=year,
        ))
    return out


# ---------------------------------------------------------------------------
# DuckDuckGo HTML (zero-key fallback, free)
# ---------------------------------------------------------------------------

_DDG_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


def ddg_search(query: str, k: int = 3) -> List[Source]:
    """DuckDuckGo HTML scrape -- intentionally simple, off by default."""
    rec = fetch(f"{DDG_HTML}?q={urllib.parse.quote(query)}", accept="text/html")
    if not rec or rec.get("status", 0) >= 400:
        return []
    out: List[Source] = []
    for m in _DDG_RESULT_RE.finditer(rec["content"]):
        url, title_html, snippet_html = m.group(1), m.group(2), m.group(3)
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        if not title or not url.startswith("http"):
            continue
        out.append(Source(
            id=f"ddg:{hash(url) & 0xFFFFFF:06x}",
            title=title,
            url=url,
            excerpt=_excerpt(snippet, max_words=80),
            provider="ddg",
            authors=[],
            year=None,
        ))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_PROVIDER_FUNCS = {
    "tavily":    tavily_search,
    "brave":     brave_search,
    "arxiv":     arxiv_search,
    "wikipedia": wiki_search,
    "ddg":       ddg_search,
}


def available_providers(requested: Iterable[str]) -> List[str]:
    """Filter requested providers down to ones whose prerequisites are met.

    `tavily` needs TAVILY_API_KEY AND must not be session-disabled (rate-limit).
    `brave`  needs BRAVE_API_KEY.
    arxiv / wikipedia / ddg have no creds.
    """
    out = []
    for p in requested:
        if p == "tavily":
            if not _tavily_api_key() or _TAVILY_DISABLED_THIS_SESSION:
                continue
        elif p == "brave" and not _brave_api_key():
            continue
        if p in _PROVIDER_FUNCS:
            out.append(p)
    return out


def gather(queries: Iterable[Query], providers: Iterable[str] = ("tavily", "arxiv", "wikipedia"),
           per_provider_k: int = 3) -> List[Source]:
    """Run each query across each provider, return concatenated raw results.

    Deduplication and ranking happen in notes.rank(), not here.
    Providers whose prerequisites aren't met (e.g. tavily without a key) are
    silently skipped -- the pipeline degrades to whatever IS available.
    """
    active = available_providers(providers)
    out: List[Source] = []
    for q in queries:
        qstr = q.q if isinstance(q, Query) else str(q)
        for p in active:
            fn = _PROVIDER_FUNCS.get(p)
            if not fn:
                continue
            try:
                results = fn(qstr, k=per_provider_k)
            except Exception as e:
                print(f"[research/search] {p}({qstr!r}) failed: {e}", flush=True)
                results = []
            out.extend(results)
    return out
