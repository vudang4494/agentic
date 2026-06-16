"""Deterministic unit tests for the G2/G3/G4 verify-layer optimization.

Mocks the LOCAL models (HHEM, gemma judge) so the pure logic is validated WITHOUT
a slow Ollama run. Run: python3 files/eval/test_verify_optim.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> files/

import research.faithfulness as faithfulness
import research.verify as verify

_fail = 0


def check(name, cond, detail=""):
    global _fail
    status = "PASS" if cond else "FAIL"
    if not cond:
        _fail += 1
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


# ---------------------------------------------------------------------------
# G3: grounding must DE-SATURATE (per-source max, not mega-premise)
# ---------------------------------------------------------------------------
class FakeHHEM:
    """predict(pairs) -> high score iff claim shares >=2 long words with the excerpt."""
    def predict(self, pairs):
        out = []
        for premise, claim in pairs:
            pw = set(re.findall(r"[a-z]{4,}", premise.lower()))
            cw = set(re.findall(r"[a-z]{4,}", claim.lower()))
            out.append(0.95 if len(pw & cw) >= 2 else 0.05)
        return out


def test_grounding_desaturate():
    print("G3 grounding de-saturation:")
    faithfulness._hhem = FakeHHEM()  # bypass real model load
    claims = [
        "Transformers use self-attention layers.",   # supported by src1
        "Quantum chromodynamics binds quarks tightly.",  # supported by NOTHING
    ]
    sources = [
        {"excerpt": "The Transformer architecture relies on self-attention layers to mix tokens."},
        {"excerpt": "Retrieval augmented generation fetches documents for grounding."},
    ]
    res = faithfulness.grounding_score(claims, sources)
    check("grounding NOT saturated at 1.0", res["grounding"] < 1.0, f"grounding={res['grounding']}")
    check("grounding == 0.5 (1 of 2 supported)", abs(res["grounding"] - 0.5) < 1e-6, f"grounding={res['grounding']}")
    check("unsupported_fraction == 0.5", abs(res["unsupported_fraction"] - 0.5) < 1e-6, f"uf={res['unsupported_fraction']}")
    check("weak_summary populated", bool(res["weak_summary"]), f"weak_summary={res['weak_summary']!r}")
    check("new keys present", {"grounding_mean", "unsupported_fraction"} <= set(res), f"keys={sorted(res)}")

    # No-evidence => grounding 0.0 (cannot ground without sources)
    res2 = faithfulness.grounding_score(["any claim here"], [{"excerpt": ""}])
    check("no-excerpt sources => grounding 0.0", res2["grounding"] == 0.0, f"grounding={res2['grounding']}")


# ---------------------------------------------------------------------------
# G4: topic blend with local judge + StageE protection floor
# ---------------------------------------------------------------------------
def fake_judge(score, verdict):
    def _fn(prompt):
        return '{"score": %s, "verdict": "%s", "reason": "mock"}' % (score, verdict)
    return _fn


def test_topic_g4():
    print("G4 topic judge blend + StageE floor:")
    # Case A: judge says off_topic AND a must_cover term is missing -> score drops below 0.5
    r = verify.topic_relevance_check(
        section_title="RoPE", goal="explain rotary position embedding",
        must_cover_terms=["rotary", "MISSINGTERM"], avoid_terms=[],
        content="This section discusses rotary embeddings in depth.",
        llm_call_fn=fake_judge(0.1, "off_topic"),
    )
    check("drift (judge off_topic + missing term) -> < 0.50", r["topic_relevance"] < 0.50,
          f"topic={r['topic_relevance']}")
    check("answer_relevance_score recorded", r["answer_relevance_score"] == 0.1, f"ar={r['answer_relevance_score']}")

    # Case B: judge says off_topic BUT all must_cover present and no drift -> floored to >= 0.50
    r2 = verify.topic_relevance_check(
        section_title="RoPE", goal="explain rotary position embedding",
        must_cover_terms=["rotary", "embedding"], avoid_terms=[],
        content="Rotary embedding rotates query and key vectors; rotary embedding is relative.",
        llm_call_fn=fake_judge(0.05, "off_topic"),
    )
    check("term-covered section protected from false StageE block (>= 0.50)",
          r2["topic_relevance"] >= 0.50, f"topic={r2['topic_relevance']}")

    # Case C: no llm -> pure heuristic (quantized), still works
    r3 = verify.topic_relevance_check(
        section_title="RoPE", goal="g", must_cover_terms=["rotary"], avoid_terms=[],
        content="rotary stuff " * 40, llm_call_fn=None,
    )
    check("fallback to heuristic when llm_call_fn is None", r3["answer_relevance_score"] is None,
          f"ar={r3['answer_relevance_score']}")

    # Case D: judge relevant -> continuous (de-quantized) score, not stuck on {0.5,0.75,1.0}
    r4 = verify.topic_relevance_check(
        section_title="RoPE", goal="g", must_cover_terms=["rotary", "embedding"], avoid_terms=[],
        content="rotary embedding " * 40, llm_call_fn=fake_judge(0.83, "relevant"),
    )
    check("blended score is continuous", round(r4["topic_relevance"], 3) not in (0.5, 0.75, 1.0),
          f"topic={r4['topic_relevance']}")


# ---------------------------------------------------------------------------
# G5b: numeric cross-ref detection regex (fabrication-prone)
# ---------------------------------------------------------------------------
def test_numeric_refs():
    print("G5b numeric cross-ref regex:")
    content = "As shown in Section 2.1 and discussed in Chapter 3, and Section 10..."
    found = re.findall(r"\b(?:Section|Chapter)\s+\d+(?:\.\d+)?", content)
    check("detects numeric Section/Chapter refs", found == ["Section 2.1", "Chapter 3", "Section 10"],
          f"found={found}")
    none = re.findall(r"\b(?:Section|Chapter)\s+\d+(?:\.\d+)?",
                      "As discussed in 'Attention head: Core Definitions'...")
    check("title-based refs are NOT flagged", none == [], f"found={none}")


def test_grounding_citation_aware():
    print("#4 citation-aware grounding (vs per-source-max):")
    faithfulness._hhem = FakeHHEM()
    # Claim cites [2], but its real support lives in source 1 (uncited).
    claims = ["Gradient descent converges reliably [2]."]
    sources = [
        {"excerpt": "Gradient descent converges reliably under standard assumptions."},
        {"excerpt": "Completely unrelated cooking recipes and kitchen techniques."},
    ]
    res = faithfulness.grounding_score(claims, sources)
    check("per-source-max credits it (lenient)", res["grounding"] >= 0.99, f"g={res['grounding']}")
    check("citation-aware does NOT (cited src2 unrelated)", res["grounding_cited"] <= 0.01,
          f"g_cited={res['grounding_cited']}")
    check("n_cited_claims counted", res["n_cited_claims"] == 1, f"n={res['n_cited_claims']}")


if __name__ == "__main__":
    test_grounding_desaturate()
    test_topic_g4()
    test_numeric_refs()
    test_grounding_citation_aware()
    print()
    if _fail:
        print(f"RESULT: {_fail} check(s) FAILED")
        sys.exit(1)
    print("RESULT: all checks PASSED")
