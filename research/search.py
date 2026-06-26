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
import hashlib
import json as _json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Iterable, List, Optional

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


def _canonical_url(url: str) -> str:
    """Topic-agnostic URL canonicalizer (no model, no network) so the SAME page
    surfaced by different providers OR different rounds collapses to one identity.
    Unifies arxiv abs/pdf/html + version suffix, wikipedia slug host, and strips
    tracking params / fragment / trailing slash. Returns input unchanged on any
    parse failure (never raises). Used for de-dup and P0c seen-counting."""
    try:
        u = (url or "").strip()
        if not u:
            return url
        # arxiv: /abs/<id>, /pdf/<id>(vN)(.pdf), /html/<id> -> one canonical abs (version stripped)
        m = re.search(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", u, re.I)
        if m:
            return f"https://arxiv.org/abs/{m.group(1)}"
        p = urllib.parse.urlsplit(u)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if "wikipedia.org" in host and "/wiki/" in (p.path or ""):
            host = "en.wikipedia.org"
        keep = []
        for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=False):
            kl = k.lower()
            if kl.startswith("utm_") or kl.startswith("mc_") or kl in {"ref", "fbclid", "gclid", "igshid", "source"}:
                continue
            keep.append((k, v))
        path = (p.path or "/").rstrip("/") or "/"
        return urllib.parse.urlunsplit(("https", host, path, urllib.parse.urlencode(keep), ""))
    except Exception:
        return url


def _url_id(url: str) -> str:
    """Deterministic id derived from the canonical URL so cross-provider /
    cross-round duplicates share ONE id (notes.dedup keys on url-or-id; P0c
    seen_counts key on id-or-url -- both now collapse to the same identity)."""
    return "url:" + hashlib.sha1(_canonical_url(url).encode("utf-8", "ignore")).hexdigest()[:12]

# Session-level kill switch: once tavily returns a recurring 4xx error we stop
# calling it for the rest of the process so we don't burn ~30s/section retrying.
# HTTP 432 = rate-limit; 402 = quota exceeded; 403 = forbidden/auth; 429 = too many.
_TAVILY_DISABLED_THIS_SESSION = False
_TAVILY_FAILURE_COUNT = 0
_TAVILY_FAILURE_THRESHOLD = 3

# arxiv: the export.arxiv.org Atom API connects but its READS often time out under
# load. Fail FAST (short read timeout) and, after a few timeouts, auto-disable the
# direct API for the session and route academic queries through Tavily instead --
# Tavily(include_domains=arxiv.org) is far more reliable. (User-requested 2026-06.)
ARXIV_READ_TIMEOUT = 7.0
_ARXIV_TIMEOUT_COUNT = 0
_ARXIV_TIMEOUT_THRESHOLD = 3
# Codes that indicate a recurring/retryable error (not transient).
_TAVILY_RETRY_CODES = {402, 403, 429, 432}

# Be polite -- arxiv ToS asks for >= 3s between requests.
_LAST_ARXIV_CALL = 0.0
ARXIV_MIN_INTERVAL = 4.0  # bumped from 3.0 since arxiv_search now makes 2 calls/query


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


# Two-phase arxiv search:
#   Phase A: ti: (title-field) match -- biases toward canonical/foundational
#            papers, whose titles tend to carry the technique name verbatim
#            ("Attention Is All You Need", "Layer Normalization", "BERT: ...").
#   Phase B: all: (relevance) match -- existing behavior, surfaces topically
#            relevant papers including recent derivative work.
# Phase-A hits are prepended so the prefilter/ranker sees them first; dedup
# is by arxiv id. Eval (2026-05-27, transformer topic) confirmed phase A is
# necessary -- without it, must_cite_recall was 0.0 on canonical papers
# because relevance-ranking favored 2024 derivative papers over Vaswani 2017.

_ARXIV_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "with", "to",
    "from", "using", "based", "into", "via", "by", "is", "are", "as",
    "section", "write", "include", "cover",  # query-gen artifacts
}


def _arxiv_title_query(q: str) -> str:
    """Reduce a free-text query to a tight title-field arxiv query.

    arxiv's ti: field needs short keyword runs to be useful -- long natural
    sentences match almost nothing. We strip stopwords and cap at 4 terms.
    Returns "" if too few signal terms survive (caller should skip phase A).
    """
    import re as _re
    tokens = [w for w in _re.findall(r"[A-Za-z][A-Za-z0-9-]+", q.lower())
              if w not in _ARXIV_TITLE_STOPWORDS and len(w) > 2]
    if len(tokens) < 2:
        return ""
    return " ".join(tokens[:4])


def _arxiv_raw_query_params(extra: dict) -> List[Source]:
    """Single arxiv API call with arbitrary params (search_query OR id_list)."""
    global _LAST_ARXIV_CALL, _ARXIV_TIMEOUT_COUNT, _ARXIV_AVAILABLE
    if not _ARXIV_AVAILABLE:
        return []
    elapsed = time.time() - _LAST_ARXIV_CALL
    if elapsed < ARXIV_MIN_INTERVAL:
        time.sleep(ARXIV_MIN_INTERVAL - elapsed)
    _LAST_ARXIV_CALL = time.time()

    base = {"start": 0, "max_results": 5, "sortBy": "relevance", "sortOrder": "descending"}
    # id_list queries must NOT carry sortBy=relevance with no search_query.
    if "id_list" in extra:
        base.pop("sortBy", None)
        base.pop("sortOrder", None)
    base.update(extra)
    params = urllib.parse.urlencode(base)
    url = f"{ARXIV_API}?{params}"
    rec = fetch(url, accept="application/atom+xml", timeout=ARXIV_READ_TIMEOUT)
    if not rec or rec.get("status", 0) >= 400 or not rec.get("content"):
        # Count read-timeouts/failures; after a few, auto-disable the flaky direct
        # API for the session so arxiv_search falls back to Tavily (no more 7s stalls).
        if not rec:
            _ARXIV_TIMEOUT_COUNT += 1
            if _ARXIV_TIMEOUT_COUNT >= _ARXIV_TIMEOUT_THRESHOLD and _ARXIV_AVAILABLE:
                _ARXIV_AVAILABLE = False
                print("[search] export.arxiv.org timing out repeatedly -- direct API "
                      "auto-disabled; routing academic queries via Tavily(arxiv.org)", flush=True)
        return []
    _ARXIV_TIMEOUT_COUNT = 0  # a success resets the streak
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


def _arxiv_raw_query(search_query: str, k: int) -> List[Source]:
    """Keyword arxiv query (ti:/all:). Thin wrapper over _arxiv_raw_query_params."""
    return _arxiv_raw_query_params({"search_query": search_query, "max_results": k})


def arxiv_by_id(arxiv_ids) -> List[Source]:
    """Fetch specific arxiv papers by id (Rank5 canonical-seed retrieval).

    Uses the arxiv API id_list param so a named canonical paper is retrieved
    directly instead of hoping a keyword search surfaces it.
    If arxiv.org is unreachable, returns an empty list."""
    if not _ARXIV_AVAILABLE:
        return []
    # Strip "arxiv:" prefix and "vN" version suffix from IDs
    ids = [re.sub(r"^arxiv:", "", str(i).strip()) for i in (arxiv_ids or []) if i]
    ids = [re.sub(r"v\d+$", "", i) for i in ids]
    ids = [i for i in ids if i]
    if not ids:
        return []
    return _arxiv_raw_query_params({"id_list": ",".join(ids), "max_results": len(ids)})


def _arxiv_via_tavily(query: str, k: int = 3) -> List[Source]:
    """Reliable arxiv fallback: when the export.arxiv.org Atom API times out, fetch
    arxiv papers through Tavily(include_domains=arxiv.org) -- far more reliable than
    the flaky direct endpoint. Re-tags arxiv.org hits as provider='arxiv' so they
    still count as primary sources downstream. (User-requested 2026-06.)"""
    out: List[Source] = []
    for s in tavily_search(query, k=max(3, k), include_domains=["arxiv.org"]):
        if "arxiv.org" in (getattr(s, "url", "") or ""):
            try:
                s.provider = "arxiv"
            except Exception:
                pass
            out.append(s)
    return out


def arxiv_search(query: str, k: int = 3) -> List[Source]:
    """Two-phase arxiv search: title-field (canonical bias) then all-field.
    Falls back to Tavily(arxiv.org) when the direct API is down/timing out."""
    if not _ARXIV_AVAILABLE:
        # direct API auto-disabled (repeated timeouts) -> Tavily takeover
        return _arxiv_via_tavily(query, k)
    # Phase A: title-field match -- canonical-paper bias.
    title_q = _arxiv_title_query(query)
    title_hits = _arxiv_raw_query(f"ti:{title_q}", k=max(2, k)) if title_q else []
    # Phase B: original all-field relevance search.
    all_hits = _arxiv_raw_query(f"all:{query}", k=k)
    # Merge: title-hits first so canonical papers are seen by the ranker
    # before recent derivatives. Dedup by arxiv id.
    seen: set[str] = set()
    merged: List[Source] = []
    for s in title_hits + all_hits:
        if s.id in seen:
            continue
        seen.add(s.id)
        merged.append(s)
    # Direct API returned nothing (timeout/empty) -> recover via Tavily(arxiv.org).
    if not merged:
        merged = _arxiv_via_tavily(query, k)
    return merged


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


def tavily_search(query: str, k: int = 5, depth: str = "advanced",
                  include_domains: Optional[List[str]] = None) -> List[Source]:
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
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        with httpx.Client(timeout=TAVILY_TIMEOUT) as c:
            r = c.post(TAVILY_API, json=payload)
            r.raise_for_status()
            data = r.json()
        _TAVILY_FAILURE_COUNT = 0
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in _TAVILY_RETRY_CODES:
            _TAVILY_FAILURE_COUNT += 1
            if _TAVILY_FAILURE_COUNT >= _TAVILY_FAILURE_THRESHOLD:
                _TAVILY_DISABLED_THIS_SESSION = True
                print(f"[research/search] tavily HTTP {status} hit {_TAVILY_FAILURE_THRESHOLD}x -- "
                      "auto-disabled for this session. Re-enable by restarting the process.",
                      flush=True)
            else:
                print(f"[research/search] tavily HTTP {status} ({_TAVILY_FAILURE_COUNT}/"
                      f"{_TAVILY_FAILURE_THRESHOLD})", flush=True)
        else:
            print(f"[research/search] tavily HTTP {status}: {e}", flush=True)
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
            id=_url_id(url),
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
            id=_url_id(url),
            title=title, url=url, excerpt=excerpt,
            provider="brave", authors=[], year=year,
        ))
    return out


# ---------------------------------------------------------------------------
# DuckDuckGo HTML (zero-key fallback, free)
# ---------------------------------------------------------------------------

# DuckDuckGo HTML structure shifts frequently. We use two fallback patterns:
# Primary: class="result__a" for links + class="result__snippet" for excerpts.
# Fallback: open-search <a> tags with data-testid="result-extras-url-link".
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>.*?</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_DDG_FALLBACK_RE = re.compile(
    r'<a[^>]+class="[^"]*result[^"]*"[^>]+href="(https?://[^"]+)"[^>]*>([^<]{5,100}?)</a>',
    re.DOTALL,
)


def ddg_search(query: str, k: int = 3) -> List[Source]:
    """DuckDuckGo HTML scrape -- intentionally simple, off by default."""
    rec = fetch(f"{DDG_HTML}?q={urllib.parse.quote(query)}", accept="text/html")
    if not rec or rec.get("status", 0) >= 400:
        return []
    html = rec["content"]
    out: List[Source] = []
    seen_urls: set = set()

    # Primary: try the structured result class first
    for m in _DDG_RESULT_RE.finditer(html):
        url, snippet_html = m.group(1), m.group(2)
        title = re.sub(r"<[^>]+>", "", html[m.start():m.end()]).strip()
        title = re.sub(r"<[^>]+>", "", title)
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        # DDG uses protocol-relative URLs: //example.com/... -> https://example.com/...
        if not url:
            continue
        if url.startswith("//"):
            url = "https:" + url
        if not (url.startswith("http") or url.startswith("https")):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(Source(
            id=_url_id(url),
            title=title[:200] if title else url,
            url=url,
            excerpt=_excerpt(snippet, max_words=80),
            provider="ddg",
            authors=[],
            year=None,
        ))
        if len(out) >= k:
            return out

    # Fallback: open-search links (handles when DDG HTML structure shifts)
    for m in _DDG_FALLBACK_RE.finditer(html):
        url, title = m.group(1), m.group(2).strip()
        if not url or url in seen_urls or "duckduckgo" in url.lower():
            continue
        seen_urls.add(url)
        out.append(Source(
            id=_url_id(url),
            title=title[:200] if title else url,
            url=url,
            excerpt=_excerpt(title, max_words=80),
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


def _arxiv_reachable() -> bool:
    """Check if arxiv.org is reachable within 5 seconds."""
    import socket as _socket
    try:
        _socket.create_connection(("export.arxiv.org", 443), timeout=5)
        return True
    except Exception:
        return False


_ARXIV_AVAILABLE: bool = _arxiv_reachable()
if not _ARXIV_AVAILABLE:
    import sys as _sys
    print("[search] arxiv.org unreachable -- auto-disabled for this session",
          file=_sys.stderr, flush=True)


def available_providers(requested: Iterable[str]) -> List[str]:
    """Filter requested providers down to ones whose prerequisites are met.

    `tavily` needs TAVILY_ENABLED=1, a valid TAVILY_API_KEY, AND must not be
    session-disabled (rate-limit/402). `brave` needs BRAVE_API_KEY.
    arxiv / wikipedia / ddg have no creds. arxiv is auto-disabled if unreachable.
    """
    import os as _os
    tavily_enabled = _os.environ.get("TAVILY_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    out = []
    for p in requested:
        if p == "tavily":
            if not tavily_enabled:
                continue
            if not _tavily_api_key() or _TAVILY_DISABLED_THIS_SESSION:
                continue
        elif p == "brave" and not _brave_api_key():
            continue
        elif p == "arxiv" and not _ARXIV_AVAILABLE:
            continue
        if p in _PROVIDER_FUNCS:
            out.append(p)
    return out


def gather(queries: Iterable[Query], providers: Iterable[str] = ("tavily", "arxiv", "wikipedia"),
           per_provider_k: int = 3) -> List[Source]:
    """Run each query across each provider; collapse cross-provider/cross-round
    duplicate pages by canonical URL, return the deduped pool.

    Ranking happens in notes.rank(); the canonical-URL collapse here ensures the
    SAME page from two providers (or two queries/rounds) becomes ONE Source with
    a deterministic id, so P0c seen-counting and notes.dedup key on one identity.
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

    # Collapse duplicates by canonical URL identity (keep first = provider/trust
    # order). Canonicalize the survivor's url so downstream dedup/seen-counting
    # key on the same identity. Non-URL sources fall back to their id.
    seen_keys = set()
    deduped: List[Source] = []
    for s in out:
        raw = getattr(s, "url", "") or ""
        key = _canonical_url(raw) if raw else (getattr(s, "id", "") or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            if raw:
                s.url = _canonical_url(raw)
        except Exception:
            pass
        deduped.append(s)
    return deduped
