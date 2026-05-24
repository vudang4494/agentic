"""Citation grounding verifier.

After the writer produces a section with `[N]` citation markers, this module
extracts each (claim, source) pair and asks a small LLM judge:

    "Does the evidence in source [N] support the cited claim?"

The judge returns one of {supports, partial, contradicts, unrelated, no_evidence}
plus a one-line reason. The aggregate score guides whether the section needs a
re-research-and-rewrite pass.

Default judge: qwen3.5:4b (better JSON compliance than gemma3:4b in practice).
"""
import json
import re
from typing import Dict, List, Tuple

import httpx

from .types import Source

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "qwen3.5:4b"
DEFAULT_TIMEOUT = 60.0
MAX_CLAIM_CHARS = 600
MAX_EVIDENCE_CHARS = 1200

JUDGE_SYS = (
    "You are a strict citation-grounding judge. You receive a claim from a research book "
    "and the EVIDENCE that the author cites in support of that claim. Decide whether the "
    "evidence actually supports the claim and respond with ONLY a single JSON object on "
    "one line, no prose, no markdown fences:\n"
    '{"verdict":"supports|partial|contradicts|unrelated|no_evidence","reason":"<one short sentence>"}\n'
    "Verdicts:\n"
    "  supports     -- evidence clearly states or implies the claim\n"
    "  partial      -- evidence supports part of the claim but not all\n"
    "  contradicts  -- evidence says the opposite\n"
    "  unrelated    -- evidence is on a different topic\n"
    "  no_evidence  -- the evidence text is empty or too generic to judge\n"
    "Be strict: 'supports' requires a direct match, not just topical overlap."
)

# Map verdicts to numeric grounding scores (0..1).
_VERDICT_SCORE = {
    "supports":    1.0,
    "partial":     0.5,
    "no_evidence": 0.3,    # missing data, not the writer's fault
    "unrelated":   0.0,
    "contradicts": 0.0,
}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(s: str) -> str:
    return _THINK_RE.sub("", s or "").strip()


def _extract_claim(text: str, marker_pos: int, window_chars: int = 400) -> str:
    """Pull the sentence(s) leading up to a `[N]` marker, up to ~window_chars.

    Walks back from marker_pos to the previous sentence boundary, capping at the
    paragraph boundary. The claim is the chunk of text the citation is attached to.
    """
    start = max(0, marker_pos - window_chars)
    # Prefer a sentence break in the back-window
    chunk = text[start:marker_pos]
    # Cut at the latest period+space within the chunk
    m = re.search(r"[.!?]\s+(?=[A-Z(\"'])", chunk)
    if m:
        chunk = chunk[m.end():]
    # Cut at the latest paragraph break
    nn = chunk.rfind("\n\n")
    if nn >= 0:
        chunk = chunk[nn + 2:]
    return chunk.strip()[:MAX_CLAIM_CHARS]


def _judge_one(client: httpx.Client, model: str, claim: str, source: Source) -> Dict:
    """Ask the judge model whether `source` supports `claim`. Returns parsed JSON
    or a permissive 'no_evidence' verdict if the call/parse fails."""
    evidence = (source.excerpt or "")[:MAX_EVIDENCE_CHARS]
    user_prompt = (
        f"CLAIM:\n{claim}\n\n"
        f"EVIDENCE (source [{source.id}], title: \"{source.title}\"):\n{evidence}\n\n"
        "Return the JSON object now."
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": 200, "top_p": 0.9},
        "think": False,
    }
    try:
        r = client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        raw = _strip_think((r.json().get("message") or {}).get("content", ""))
    except Exception as e:
        return {"verdict": "no_evidence", "reason": f"judge call failed: {e}", "_skipped": True}

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"verdict": "no_evidence", "reason": "judge returned non-JSON", "_skipped": True}
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {"verdict": "no_evidence", "reason": "judge JSON parse failed", "_skipped": True}
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in _VERDICT_SCORE:
        verdict = "no_evidence"
    return {"verdict": verdict, "reason": str(data.get("reason", ""))[:200]}


def verify_section(content: str, sources: List[Source],
                   model: str = DEFAULT_JUDGE_MODEL,
                   timeout: float = DEFAULT_TIMEOUT) -> Dict:
    """Verify every `[N]` citation in content against its source.

    Returns:
        {
          "grounding": float in [0,1]  -- average score across all citations,
          "n_citations": int,
          "verdicts": [{"n": int, "claim": str, "verdict": str, "reason": str}, ...],
          "weak_citations": [n for n where verdict is unrelated/contradicts],
          "weak_summary": str  -- aggregated reason string for re-research feedback
        }

    Empty/no-citation sections get grounding=1.0 (nothing to verify).
    """
    # Find every [N] marker with its byte offset
    markers: List[Tuple[int, int]] = []  # (cite_index, position_in_text)
    for m in re.finditer(r"\[(\d+)\]", content):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        markers.append((n, m.start()))

    if not markers:
        # Penalize zero-citation sections HARD when evidence was available -- otherwise
        # the writer learns to drop all `[N]` markers to auto-pass verification (the
        # "writer gaming" pathology). If no sources were retrieved either, the section
        # genuinely has nothing to cite -> neutral score.
        if sources:
            return {
                "grounding": 0.0,
                "n_citations": 0,
                "verdicts": [],
                "weak_citations": [],
                "weak_summary": (
                    "section dropped ALL citations despite evidence being provided -- "
                    "writer must use [N] markers anchored to specific factual claims"
                ),
            }
        return {"grounding": 1.0, "n_citations": 0, "verdicts": [],
                "weak_citations": [], "weak_summary": ""}

    verdicts: List[Dict] = []
    scores: List[float] = []
    weak: List[int] = []

    with httpx.Client(timeout=timeout) as client:
        for n, pos in markers:
            if not (1 <= n <= len(sources)):
                # Citation refers to a non-existent source -- hardest possible failure
                verdicts.append({"n": n, "claim": "", "verdict": "unrelated",
                                 "reason": f"citation [{n}] but only {len(sources)} sources available"})
                scores.append(0.0)
                weak.append(n)
                continue
            claim = _extract_claim(content, pos)
            v = _judge_one(client, model, claim, sources[n - 1])
            scores.append(_VERDICT_SCORE.get(v["verdict"], 0.0))
            verdicts.append({"n": n, "claim": claim[:200], **v})
            if v["verdict"] in ("unrelated", "contradicts"):
                weak.append(n)

    grounding = sum(scores) / max(len(scores), 1)
    # Aggregate weak reasons for the re-research hint
    weak_reasons = [v["reason"] for v in verdicts
                    if v["verdict"] in ("unrelated", "contradicts") and v.get("reason")]
    weak_summary = "; ".join(weak_reasons[:3])

    return {
        "grounding": round(grounding, 3),
        "n_citations": len(markers),
        "verdicts": verdicts,
        "weak_citations": weak,
        "weak_summary": weak_summary,
    }
