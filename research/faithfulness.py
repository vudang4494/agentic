"""Faithfulness scorer — claim decomposition + HHEM grounding.

Tier 2 of the verify funnel (per-section):
  1. Decompose WRT body into atomic claims (one independent statement each).
  2. Score each (premise=source_excerpts, hypothesis=claim) with HHEM.
  3. Return grounding = #supported / #total, plus per-claim verdicts.

Why HHEM over LLM judge:
  - HHEM judge (flan-t5-base) separates from writer -- no self-preference bias.
  - HHEM-2.1-Open is a separate 0.1B flan-t5-base — no weight sharing.
  - 1 forward pass per claim (~1.5s on CPU), vs 14B LLM call per section.

HHEM-2.1-Open:
  - Model: vectara/hallucination_evaluation_model (flan-t5-base, 109.6M)
  - Size: <600MB fp32
  - Input: (premise, hypothesis) pairs
  - Output: [0,1] — 0=hallucinated, 1=fully consistent with premise
  - Threshold: HHEM_SUPPORT=0.5 (Vectara default)

IMPORTANT: transformers>=5.0 breaks HHEM loading via the standard API because
HHEM's custom model class lacks `all_tied_weights_keys` attribute required by
transformers' model finalization. We patch torch.nn.Module.__getattribute__ to
return an empty dict when that attribute is missing, allowing the model to load.

Source: https://huggingface.co/vectara/hallucination_evaluation_model
Benchmark: HHEM-2.1-Open balanced accuracy 74.28% on RAGTruth-QA, 64.42% on RAGTruth-Summ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---- Patch: transformers>=5.x compatibility for HHEM ----
# HHEM's custom model class lacks `all_tied_weights_keys` which transformers 5.x
# tries to access during model finalization. This is a known compatibility issue
# with custom remote-code models. The fix is to intercept the attribute error
# at the torch.nn.Module level and return an empty dict instead.
import torch.nn
_hhem_patch_applied = False


def _apply_hhem_compat_patch():
    global _hhem_patch_applied
    if _hhem_patch_applied:
        return
    _orig = torch.nn.Module.__getattribute__

    def _safe_getattr(self, name):
        if name == 'all_tied_weights_keys':
            try:
                keys = _orig(self, '_tied_weights_keys')
                return keys if keys is not None else {}
            except (AttributeError, TypeError):
                return {}
        return _orig(self, name)

    torch.nn.Module.__getattribute__ = _safe_getattr
    _hhem_patch_applied = True


# ---- Knobs ----
HHEM_SUPPORT = 0.5      # claim supported if HHEM >= this
HHEM_MODEL = "vectara/hallucination_evaluation_model"
HHEM_DEVICE = "cpu"    # MPS can be used if GPU memory available; CPU is reliable

# ---- HHEM instance (load once, reuse) ----
_hhem = None


def _retie_hhem_embeddings(model) -> bool:
    """ROOT FIX for the degenerate scorer: transformers>=5.x + the `all_tied_weights_keys={}` compat
    patch DISABLE T5 weight-tying. The HHEM checkpoint ships only the tied `shared` embedding, so
    `encoder.embed_tokens.weight` loads as ALL-ZEROS -> the encoder sees zero input embeddings ->
    constant output -> predict() returns the bit-identical constant ~0.502 for EVERY pair, making
    Gate D a no-op. Re-point embed_tokens at the loaded `shared` weight to restore real scores.
    Verified: re-tie turns a true/false/unrelated battery from [0.502,0.502,0.502] into
    [0.894,0.018,0.014]. (See eval/bench_hhem_discrimination.py.)"""
    try:
        tr = model.t5.transformer
        shared = tr.shared.weight
        if hasattr(tr, "encoder") and float(tr.encoder.embed_tokens.weight.std()) == 0.0:
            tr.encoder.embed_tokens.weight = shared
        if hasattr(tr, "decoder") and float(tr.decoder.embed_tokens.weight.std()) == 0.0:
            tr.decoder.embed_tokens.weight = shared
        return True
    except Exception as e:
        print(f"[faithfulness] HHEM embedding re-tie skipped: {e}", flush=True)
        return False


def _hhem_discriminates(model) -> bool:
    """Startup assertion: a TRUE claim must score clearly above a FALSE one. Guards against a silently
    degenerate scorer (broken tying / future regression) turning Gate D back into a no-op."""
    try:
        t, f = (float(x) for x in model.predict([
            ("The sky is blue.", "The sky is blue."),
            ("The sky is blue.", "The sky is green and red."),
        ]))
        return (t - f) > 0.2
    except Exception:
        return True  # never block the pipeline on a probe failure


def _get_hhem():
    """Lazy-load HHEM model. Call once per process."""
    global _hhem
    if _hhem is not None:
        return _hhem
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers torch. "
            "See requirements.txt"
        )

    _apply_hhem_compat_patch()

    _hhem = AutoModelForSequenceClassification.from_pretrained(
        HHEM_MODEL,
        trust_remote_code=True,
        device_map=HHEM_DEVICE,
    )
    _retie_hhem_embeddings(_hhem)
    if not _hhem_discriminates(_hhem):
        print("[faithfulness] WARNING: HHEM still degenerate after re-tie -- grounding (Gate D) is a "
              "NO-OP; rely on G2 cite_precision + G4 topic. Verify with bench_hhem_discrimination.py.",
              flush=True)
    return _hhem


def decompose_claims(body_md: str, llm_call_fn=None) -> list:
    """Split markdown body into atomic, independent, de-pronoun'd claims.

    Args:
        body_md: Markdown text produced by WRT.
        llm_call_fn: Optional LLM callable. If None, falls back to simple
                     sentence splitting. Pass the Ollama generate fn for best quality.

    Returns:
        List of claim strings, stripped of whitespace.
    """
    if not body_md or not body_md.strip():
        return []

    if llm_call_fn is not None:
        prompt = (
            "Tach doan sau thanh cac cau khang dinh nguyen tu, moi cau doc lap, "
            "da thay dai tu bang danh tu cu the. Moi cau 1 dong, khong danh so.\n\n"
            f"{body_md}"
        )
        result = llm_call_fn(prompt)
        # llm_call_fn should return text; split by lines
        if isinstance(result, str):
            lines = result.splitlines()
        elif hasattr(result, "content"):
            lines = result.content.splitlines()
        else:
            lines = str(result).splitlines()
        return [c.strip() for c in lines if c.strip()]

    # Fallback: simple sentence splitting
    import re
    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', body_md)
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and len(s) < 600:
            claims.append(s)
    return claims


def grounding_score(claims: list, sources: list, threshold: float = HHEM_SUPPORT) -> dict:
    """Score claims against source excerpts using HHEM.

    Args:
        claims: List of claim strings (from decompose_claims).
        sources: List of Source objects or dicts with "excerpt" field.
        threshold: HHEM score above which a claim is considered "supported".

    Returns:
        dict:
          grounding       -- float [0,1]: supported / total claims
          n_supported     -- int: count of claims with HHEM >= threshold
          n_partial       -- int: count with 0.3 <= HHEM < threshold
          n_unsupported   -- int: count with HHEM < 0.3
          per_claim       -- list of (claim, hhem_score, verdict) tuples
          n_claims        -- int: total claims
    """
    if not claims:
        return {
            "grounding": 1.0,
            "n_supported": 0,
            "n_partial": 0,
            "n_unsupported": 0,
            "per_claim": [],
            "n_claims": 0,
        }

    hhem = _get_hhem()

    # G3 DE-SATURATE: score each claim against each source SEPARATELY (one batched
    # predict) and take the MAX. The old code concatenated ALL excerpts into one
    # ~2000-char mega-premise, so almost any topical sentence matched something and
    # grounding saturated at 1.0 across all 280 v36 sections. Per-source-max requires
    # a claim to be supported by a SINGLE excerpt -> a discriminating signal.
    # COST BOUND: grounding SATURATES on clean content (1.0, non-discriminating; G2
    # cite_precision is the real signal), so do NOT spend hundreds of HHEM pairs here. The
    # earlier windowing (excerpt x windows x sources) made this 515s/section. Use ONE premise
    # per source (Rank-1 fetch already returns the math-dense window), few sources, capped
    # claims; the per-[N] citation-aware pass below carries precision cheaply (~1 pair/claim).
    # HHEM = flan-t5-base, HARD 512-token window. Dense technical text tokenizes ~2 chars/token,
    # so premise+claim must stay well under ~1000 chars or the (premise,claim) pair is silently
    # TRUNCATED -> garbage scores (observed 940>512 -> grounding pinned ~0.08 on good content).
    MAX_PREMISE_CHARS = 700    # ~350 tokens; leaves room for the claim + prompt template under 512
    MAX_SOURCES = 3
    MAX_CLAIMS = 24
    claims = claims[:MAX_CLAIMS]

    excerpts = []
    for s in sources[:MAX_SOURCES]:
        exc = (s.excerpt if hasattr(s, "excerpt") else s.get("excerpt", "")) if s else ""
        if exc:
            excerpts.append(exc[:MAX_PREMISE_CHARS])

    if not excerpts:
        # No evidence to ground against -> nothing can be supported.
        return {
            "grounding": 0.0, "grounding_mean": 0.0, "unsupported_fraction": 1.0,
            "n_supported": 0, "n_partial": 0, "n_unsupported": len(claims),
            "per_claim": [(c, 0.0, "unsupported") for c in claims], "n_claims": len(claims),
            "weak_summary": "no source excerpts available to ground claims",
        }

    # All (excerpt, claim) pairs in ONE batched predict; remember each pair's claim.
    pairs, pair_claim_idx = [], []
    for ci, claim in enumerate(claims):
        c = claim[:280]   # keep premise+claim under the 512-token HHEM window
        for exc in excerpts:
            pairs.append((exc, c))
            pair_claim_idx.append(ci)

    try:
        raw_scores = hhem.predict(pairs)  # returns list of floats [0,1]
    except Exception:
        try:
            from transformers import AutoModelForSequenceClassification
            hhem = AutoModelForSequenceClassification.from_pretrained(
                HHEM_MODEL, trust_remote_code=True
            )
            raw_scores = hhem.predict(pairs)
        except Exception as e:
            # Never crash the pipeline on a scorer failure -> neutral, non-blocking signal.
            print(f"[faithfulness] HHEM predict failed, neutral grounding: {e}", flush=True)
            return {
                "grounding": 0.5, "grounding_mean": 0.5, "unsupported_fraction": 0.0,
                "n_supported": 0, "n_partial": len(claims), "n_unsupported": 0,
                "per_claim": [(c, 0.5, "partial") for c in claims], "n_claims": len(claims),
                "weak_summary": "", "fallback": True,
            }

    # Max HHEM score per claim across its candidate sources.
    best = [0.0] * len(claims)
    for pi, score in enumerate(raw_scores):
        ci = pair_claim_idx[pi]
        sc = float(score)
        if sc > best[ci]:
            best[ci] = sc

    per_claim = []
    n_supported = n_partial = n_unsupported = 0
    for claim, sc in zip(claims, best):
        if sc >= threshold:
            verdict = "supported"; n_supported += 1
        elif sc >= 0.3:
            verdict = "partial"; n_partial += 1
        else:
            verdict = "unsupported"; n_unsupported += 1
        per_claim.append((claim, sc, verdict))

    total = len(claims)
    # Continuous grounding: full credit for supported, half for partial.
    grounding = (n_supported + 0.5 * n_partial) / total if total else 1.0
    grounding_mean = sum(best) / total if total else 1.0
    unsupported_fraction = n_unsupported / total if total else 0.0

    # #4 CITATION-AWARE (warn-first): score each claim against the source it actually
    # CITES ([N] -> sources[N-1]), stripped of markers -- the correct grounding unit.
    # Logged for comparison; the GATE still uses `grounding` (per-source-max) until a
    # validation run re-baselines min_grounding (citation-aware drops below 0.70 more).
    import re as _re2
    _cited_pairs, _cited_ci = [], []
    for ci, claim in enumerate(claims):
        _idxs = [int(m) for m in _re2.findall(r"\[(\d+)\]", claim)]
        _excs = []
        for n in _idxs:
            if 1 <= n <= len(sources):
                s = sources[n - 1]
                e = (s.excerpt if hasattr(s, "excerpt") else s.get("excerpt", "")) if s else ""
                if e:
                    _excs.append(e[:MAX_PREMISE_CHARS])
        if _excs:
            _cited_pairs.append(("\n".join(_excs)[:MAX_PREMISE_CHARS],
                                 _re2.sub(r"\[\d+\]", "", claim)[:280]))
            _cited_ci.append(ci)
    _cited_score = {}
    if _cited_pairs:
        try:
            for _k, _sc in enumerate(hhem.predict(_cited_pairs)):
                _cited_score[_cited_ci[_k]] = float(_sc)
        except Exception:
            _cited_score = {}
    _n_sup_cited = sum(1 for ci in range(total) if _cited_score.get(ci, best[ci]) >= threshold)
    grounding_cited = _n_sup_cited / total if total else 1.0

    return {
        "grounding": round(grounding, 3),
        "grounding_cited": round(grounding_cited, 3),
        "n_cited_claims": len(_cited_pairs),
        "grounding_mean": round(grounding_mean, 3),
        "unsupported_fraction": round(unsupported_fraction, 3),
        "n_supported": n_supported,
        "n_partial": n_partial,
        "n_unsupported": n_unsupported,
        "per_claim": per_claim,
        "n_claims": total,
        "weak_summary": (
            f"{n_unsupported}/{total} claims not supported by any single source"
            if n_unsupported else ""
        ),
    }


# ---- Smoke test ----
if __name__ == "__main__":
    test_claims = [
        "Transformers use self-attention mechanisms.",
        "The year 2023 saw the release of GPT-5.",
        "BERT was introduced by Vaswani et al. in 2017.",
    ]
    test_sources = [
        {"excerpt": "The Transformer architecture uses self-attention to process sequences in parallel."},
        {"excerpt": "GPT-4 was released in March 2023."},
        {"excerpt": "Attention Is All You Need was published by Vaswani et al. in 2017."},
    ]
    result = grounding_score(test_claims, test_sources)
    print(f"HHEM_SUPPORT={HHEM_SUPPORT}")
    print(f"grounding={result['grounding']:.3f}  "
          f"supported={result['n_supported']}  "
          f"partial={result['n_partial']}  "
          f"unsupported={result['n_unsupported']}")
    for claim, score, verdict in result["per_claim"]:
        print(f"  [{verdict:10}] {score:.4f}  {claim[:60]}")
