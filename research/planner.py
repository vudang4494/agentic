"""Planner agent: turn a topic string into a research-informed CHAPTERS outline.

This is Stage 3 of the Agentic Deep Research roadmap. When the pipeline is
launched with `--topic "..."`, the planner:

  1. Does a brief scoping research pass over the topic (3 broad queries -> ~10 sources).
  2. Asks an LLM to produce a 12-chapter / 8-section-per-chapter outline (96
     total sections), each section with a 1-2 line generation prompt.
  3. Returns a CHAPTERS list compatible with deep_research.py's existing format.

Output schema matches the hardcoded CHAPTERS:
    [{"n": int, "t": str, "passes": [{"p": int, "t": str, "w": int, "pr": str}, ...]}, ...]

The planner is best-effort: if generation fails or parses poorly, the caller
falls back to the hardcoded CHAPTERS list.
"""
import json
import re
from typing import List, Optional

import httpx

from . import search as _search
from .query_gen import _strip_think
from .types import Query

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_PLANNER_MODEL = "batiai/qwen3.6-35b:iq3"  # MoE 35B / 3B active, ~15 tok/s on M4
# qwen35:9b removed -- not available in this Ollama install.
# If qwen3.6-35b fails (OOM, timeout), planner falls back to hardcoded outline.
DEFAULT_TIMEOUT = 480.0

DEFAULT_N_CHAPTERS = 12
DEFAULT_N_PASSES = 8
DEFAULT_WORD_BUDGET = 4200

PLANNER_SYS = (
    "You are a research book planner. Given a topic and a short scoping summary "
    "of recent literature, produce a comprehensive outline for a technical book.\n\n"
    "Output ONLY a JSON object with this exact schema (no prose, no markdown fences):\n"
    "{\n"
    '  "title":    "<book title>",\n'
    '  "chapters": [\n'
    "    {\n"
    '      "n": 1,\n'
    '      "t": "<chapter title>",\n'
    '      "passes": [\n'
    '        {"p": 1, "t": "<section title>", "pr": "<1-2 sentence generation directive>"},\n'
    "        ...\n"
    "      ]\n"
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Produce exactly the requested number of chapters and passes per chapter.\n"
    "- Each chapter must cover a distinct, non-overlapping sub-area of the topic.\n"
    "- Each section title should be specific enough that a writer knows what to cover.\n"
    "- The `pr` field is the prompt the writer sees. Make it concrete: list key sub-topics, "
    "formulas, methods, or case studies to include.\n"
    "- Order chapters from foundational -> intermediate -> advanced -> applications -> frontiers."
)


def _scope_topic(topic: str, providers=("tavily", "arxiv", "wikipedia")) -> str:
    """Run a brief research pass to give the planner real context.

    Returns a compact bullet list of titles + 1-line summaries from up to 10
    sources across 3 broad scoping queries. Empty string if everything fails.
    """
    scoping_queries = [
        Query(q=f"{topic} survey overview"),
        Query(q=f"{topic} fundamentals introduction"),
        Query(q=f"{topic} state of the art 2024 2025"),
    ]
    sources = _search.gather(scoping_queries, providers=providers, per_provider_k=3)
    if not sources:
        return ""
    # Dedup by URL
    seen = set()
    uniq = []
    for s in sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        uniq.append(s)
        if len(uniq) >= 10:
            break
    lines = ["Recent literature scoped during planning:"]
    for s in uniq:
        year = f" ({s.year})" if s.year else ""
        snippet = (s.excerpt or "").split(". ")[0][:160]
        lines.append(f"  - {s.title}{year}: {snippet}")
    return "\n".join(lines)


def _parse_outline(raw: str, n_chapters: int, n_passes: int,
                   word_budget: int) -> Optional[List[dict]]:
    """Extract the JSON outline from a model response. Returns CHAPTERS-format list
    or None on parse failure. Pads/truncates to exact (n_chapters, n_passes) so the
    runner/state code never has to deal with off-shape outlines."""
    raw = _strip_think(raw or "")
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None

    chapters_in = data.get("chapters") or []
    if not isinstance(chapters_in, list) or not chapters_in:
        return None

    # Sections with these title patterns get replaced -- the pipeline already
    # auto-builds a References page from the collected sources at assembly time,
    # so a section that just says "References and Further Reading" leads the
    # writer to invent another prose page with hallucinated citations
    # (observed in bookv3 Ch7.10).
    _META_SECTION_RE = re.compile(
        r"\b(references|bibliography|further\s+reading|works\s+cited|"
        r"acknowledg(e?)ments?|appendix|index|glossary)\b",
        re.IGNORECASE,
    )

    out: List[dict] = []
    for i, ch in enumerate(chapters_in[:n_chapters], start=1):
        if not isinstance(ch, dict):
            continue
        ct = str(ch.get("t") or ch.get("title") or f"Chapter {i}").strip()
        passes_in = ch.get("passes") or []
        passes_out = []
        for j, pp in enumerate(passes_in[:n_passes], start=1):
            if isinstance(pp, dict):
                pt = str(pp.get("t") or pp.get("title") or f"Section {i}.{j}").strip()
                pr = str(pp.get("pr") or pp.get("prompt") or pt).strip()
            elif isinstance(pp, str):
                pt, pr = pp[:80], pp
            else:
                continue
            # Skip meta sections -- the assembler handles them.
            if _META_SECTION_RE.search(pt):
                print(f"[planner] skipping meta section Ch{i}.{j}: {pt!r} "
                      f"(pipeline assembles refs/index automatically)", flush=True)
                continue
            passes_out.append({"p": j, "t": pt, "w": word_budget, "pr": pr})
        if not passes_out:
            continue
        # Re-number passes after meta-skip so the sequence stays 1..N
        for new_p, pp in enumerate(passes_out, start=1):
            pp["p"] = new_p
        # Pad to n_passes if model under-produced
        while len(passes_out) < n_passes:
            j = len(passes_out) + 1
            passes_out.append({
                "p": j, "t": f"{ct} -- additional aspect {j}", "w": word_budget,
                "pr": f"Continue the {ct.lower()} discussion with an additional important sub-topic not yet covered in earlier passes.",
            })
        out.append({"n": i, "t": ct, "passes": passes_out})

    if not out:
        return None
    # Chapter shortfall is a hard fail -- we used to pad with "Additional Chapter N" /
    # "Aspect J" placeholders, but the bookv3 run revealed that the writer takes
    # those as legitimate prompts and emits ~80 sections of generic filler. Better
    # to return None and let plan_outline() retry with a different temperature
    # (and possibly higher num_predict) than to silently truncate the topic.
    if len(out) < n_chapters:
        print(f"[planner] chapter shortfall: model emitted {len(out)}/{n_chapters} -- "
              f"refusing to pad with placeholders (would dilute the book).", flush=True)
        return None
    return out


def _chat(messages: list, model: str, timeout: float,
          num_predict: int = 2000, temperature: float = 0.4) -> str:
    """One Ollama /api/chat round. Returns content string ('' on failure)."""
    payload = {
        "model": model, "stream": False, "messages": messages,
        "options": {"temperature": temperature, "num_predict": num_predict, "top_p": 0.9},
        "think": False,  # Force content into "message.content", not "thinking" block
    }
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            r.raise_for_status()
            raw = (r.json().get("message") or {}).get("content", "")
            return _strip_think(raw)  # Strip any remaining think blocks defensively
    except Exception as e:
        print(f"[planner] _chat error: {e}", flush=True)
        return ""


_CHAPTERS_SYS = (
    "You are a research book planner. Given a topic and a short scoping summary, "
    "propose chapter titles for a technical book. Output ONLY a JSON array of "
    "strings (no prose, no markdown fences, no objects):\n"
    '["Chapter 1 title", "Chapter 2 title", ...]\n\n'
    "Rules:\n"
    "- Produce EXACTLY the requested number of chapter titles.\n"
    "- Each chapter covers a distinct, non-overlapping sub-area.\n"
    "- Order: foundational -> intermediate -> advanced -> applications -> frontiers.\n"
    "- Titles must be specific to the topic, not generic ('Introduction', 'Conclusion' are too weak)."
)

_SECTIONS_SYS = (
    "You are a research book planner detailing ONE chapter. Output ONLY a JSON array "
    "of section objects (no prose, no markdown fences):\n"
    '[{"t": "<section title>", "pr": "<1-2 sentence writer directive>"}, ...]\n\n'
    "Rules:\n"
    "- Produce EXACTLY the requested number of sections.\n"
    "- Each `pr` is concrete: name the key sub-topics, formulas, methods, papers, or "
    "case studies the writer should cover.\n"
    "- Sections within the chapter must not overlap each other.\n"
    "- Do NOT create a 'References', 'Summary', or 'Conclusion' section."
)


def _extract_json_array(raw: str):
    """Pull the first top-level JSON array out of a model response."""
    raw = _strip_think(raw or "")
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _plan_chapter_titles(topic, scope, n_chapters, model, timeout) -> List[str]:
    """Phase 1: just the chapter titles. Small output -> reliable. Retries x3."""
    user = (f"TOPIC: {topic}\nProduce EXACTLY {n_chapters} chapter titles.\n\n"
            f"{scope}\n\nReturn the JSON array of {n_chapters} titles now.")
    for attempt in range(1, 4):
        raw = _chat(
            [{"role": "system", "content": _CHAPTERS_SYS},
             {"role": "user", "content": user}],
            model=model, timeout=timeout, num_predict=2000,
            temperature=0.4 + (attempt - 1) * 0.15,
        )
        arr = _extract_json_array(raw)
        titles = [str(t).strip() for t in arr if str(t).strip()] if isinstance(arr, list) else []
        if len(titles) >= n_chapters:
            return titles[:n_chapters]
        print(f"[planner] phase1 attempt {attempt}: got {len(titles)}/{n_chapters} titles, retrying", flush=True)
    return titles[:n_chapters] if 'titles' in dir() else []


def _plan_chapter_sections(topic, chapter_title, n_passes, prior_titles, model, timeout) -> List[dict]:
    """Phase 2: section titles + prompts for ONE chapter. Small output -> reliable."""
    avoid = ""
    if prior_titles:
        avoid = ("\nAlready-covered chapters (do NOT repeat their material, build on it):\n- "
                 + "\n- ".join(prior_titles[-8:]) + "\n")
    user = (f"TOPIC: {topic}\nCHAPTER: {chapter_title}\n{avoid}\n"
            f"Produce EXACTLY {n_passes} sections for THIS chapter only.\n"
            f"Return the JSON array of {n_passes} section objects now.")
    for attempt in range(1, 4):
        raw = _chat(
            [{"role": "system", "content": _SECTIONS_SYS},
             {"role": "user", "content": user}],
            model=model, timeout=timeout, num_predict=4000,
            temperature=0.4 + (attempt - 1) * 0.15,
        )
        arr = _extract_json_array(raw)
        if isinstance(arr, list):
            secs = []
            for pp in arr:
                if isinstance(pp, dict) and (pp.get("t") or pp.get("title")):
                    secs.append({
                        "t": str(pp.get("t") or pp.get("title")).strip(),
                        "pr": str(pp.get("pr") or pp.get("prompt") or pp.get("t") or "").strip(),
                    })
            if len(secs) >= n_passes:
                return secs[:n_passes]
        print(f"[planner] phase2 '{chapter_title[:30]}' attempt {attempt}: "
              f"got {len(arr) if isinstance(arr, list) else 0}/{n_passes}, retrying", flush=True)
    # Best-effort pad with chapter-relevant prompts (NOT generic "Aspect N")
    secs = secs if 'secs' in dir() and secs else []
    while len(secs) < n_passes:
        k = len(secs) + 1
        secs.append({
            "t": f"{chapter_title}: dimension {k}",
            "pr": f"Cover an additional important aspect of {chapter_title.lower()} "
                  f"not addressed in the other sections of this chapter.",
        })
    return secs[:n_passes]


def plan_outline(topic: str,
                 n_chapters: int = DEFAULT_N_CHAPTERS,
                 n_passes: int = DEFAULT_N_PASSES,
                 word_budget: int = DEFAULT_WORD_BUDGET,
                 model: str = DEFAULT_PLANNER_MODEL,
                 timeout: float = DEFAULT_TIMEOUT,
                 max_attempts: int = 3) -> Optional[List[dict]]:
    """Two-phase planner -- the robust replacement for one-shot generation.

    Phase 1: ask for the chapter titles only (small output, never truncates).
    Phase 2: for each chapter, ask for its N section prompts in a separate call.

    The bookv3/bookv4 runs proved that asking ANY local model (4B or 9B) for a
    full 15x10 = 150-section JSON in one response truncates -- the output is
    just too long to stay coherent. Decomposing into 1 + N small calls makes
    each generation reliable regardless of model size. Falls back to None (and
    thus the hardcoded CHAPTERS) only if Phase 1 itself can't produce enough
    chapter titles after retries.
    """
    print(f"[planner] scoping research for topic: {topic!r}", flush=True)
    scope = _scope_topic(topic)
    if scope:
        print(f"[planner] scoped {scope.count(chr(10))} sources", flush=True)

    # Phase 1: chapter titles
    titles = _plan_chapter_titles(topic, scope, n_chapters, model, timeout)
    if len(titles) < n_chapters:
        print(f"[planner] phase1 failed: only {len(titles)}/{n_chapters} chapter titles -- "
              f"falling back to hardcoded outline.", flush=True)
        return None
    print(f"[planner] phase1 OK: {len(titles)} chapter titles", flush=True)

    # Phase 2: per-chapter sections
    out: List[dict] = []
    for i, ct in enumerate(titles, start=1):
        secs = _plan_chapter_sections(topic, ct, n_passes, [c["t"] for c in out], model, timeout)
        passes = [{"p": j, "t": s["t"], "w": word_budget, "pr": s["pr"]}
                  for j, s in enumerate(secs, start=1)]
        out.append({"n": i, "t": ct, "passes": passes})
        print(f"[planner] phase2 {i}/{n_chapters}: {ct[:50]} -> {len(passes)} sections", flush=True)

    outline = _self_correct(out)
    total = sum(len(c["passes"]) for c in outline)
    print(f"[planner] outline OK (two-phase): {len(outline)} chapters, {total} sections", flush=True)
    return outline


def _self_correct(outline: List[dict]) -> List[dict]:
    """Cheap deterministic self-correction over a freshly-planned outline:

    1. Disambiguate any literally-duplicated section titles.
    2. Disambiguate NEAR-duplicates via bge-m3 cosine on titles (catches
       cases like "Tokenization Strategies" vs "Tokenizer Methods" which
       the literal-match pass misses but the writer will duplicate).
    3. Audit for cross-chapter concept overlap; report it to the caller.

    Concept-overlap removal is deferred to the runtime dedupe_outline() pass in
    deep_research.run() -- that pass needs runtime access to the full key-term
    list and is idempotent, so we don't duplicate the logic here.
    """
    # 1. Literal-duplicate disambiguation
    seen_titles = {}
    for ch in outline:
        for pp in ch["passes"]:
            t = pp["t"].strip()
            key = t.lower()
            if key in seen_titles:
                pp["t"] = f"{t} (Ch{ch['n']}.{pp['p']} -- {ch['t'].split()[0]} perspective)"
            else:
                seen_titles[key] = (ch["n"], pp["p"])

    # 2. Near-duplicate detection via bge-m3 cosine. Annotate the LATER
    # section with a "DISTINCT FROM Ch X.Y" directive so the writer is
    # primed to differentiate from the earlier coverage. Cheap (1 batch
    # embedding call, ~150 titles -> ~2s on M-series).
    try:
        from .embeddings import embed, cosine  # local import: planner runs without it on Ollama failures
        all_pps = [(ch["n"], pp["p"], pp) for ch in outline for pp in ch["passes"]]
        titles = [pp["t"] for _, _, pp in all_pps]
        vecs = embed(titles, model="bge-m3:latest") if len(titles) > 1 else []
        if len(vecs) == len(titles):
            NEAR_DUP_THRESHOLD = 0.85
            for i in range(1, len(vecs)):
                for j in range(i):
                    sim = cosine(vecs[i], vecs[j])
                    if sim < NEAR_DUP_THRESHOLD:
                        continue
                    later_n, later_p, later = all_pps[i]
                    earlier_n, earlier_p, earlier = all_pps[j]
                    directive = (
                        f"\n\n[OUTLINE-DEDUPE] This section title is near-duplicate "
                        f"(cosine={sim:.2f}) of Ch{earlier_n}.{earlier_p} "
                        f"'{earlier['t']}'. Write THIS section from a clearly "
                        f"distinct angle -- do NOT re-derive the same content. "
                        f"Reference the earlier section if you must touch the "
                        f"overlapping concept."
                    )
                    if "[OUTLINE-DEDUPE]" not in later["pr"]:
                        later["pr"] = later["pr"].rstrip() + directive
                    break  # one near-dup callout per later-section is enough
            n_near = sum(1 for _, _, pp in all_pps if "[OUTLINE-DEDUPE]" in pp["pr"])
            if n_near:
                print(f"[planner] near-duplicate audit: {n_near} sections flagged "
                      f"with bge-m3 cosine >= {NEAR_DUP_THRESHOLD}", flush=True)
    except Exception as e:
        print(f"[planner] near-dup audit skipped ({type(e).__name__}: {e})", flush=True)

    # 3. Audit concept overlap (informational)
    from collections import Counter
    key_terms = ["attention", "embedding", "scaling laws", "fine-tuning", "transformer",
                 "RAG", "quantization", "LoRA", "RLHF", "DPO", "tokeniz"]
    counts = Counter()
    for ch in outline:
        for pp in ch["passes"]:
            blob = (pp["t"] + " " + pp["pr"]).lower()
            for t in key_terms:
                if t.lower() in blob:
                    counts[t] += 1
    repeats = [(t, n) for t, n in counts.items() if n >= 3]
    if repeats:
        print(f"[planner] outline concept-overlap audit: " +
              ", ".join(f"{t}x{n}" for t, n in repeats) +
              " -- runtime dedupe_outline() will inject 'already covered' directives.",
              flush=True)
    return outline
