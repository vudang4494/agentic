"""P0-2b benchmark: does the G2 citation judge DISCRIMINATE, or just floor/inflate?

Companion to bench_hhem_discrimination.py, but for the *citation-integrity* gate (G2,
verify.verify_section, local gemma). Calls the REAL pipeline path -- no reimplementation --
against labeled sections whose faithfulness is known a priori:

  GOOD          -- every [N] claim is a faithful PARAPHRASE of its cited source -> must score HIGH
  BAD_UNRELATED -- same claims, sources swapped to off-topic excerpts          -> must score LOW
  BAD_CONTRADICT-- same claims, sources assert the OPPOSITE                     -> must score LOW

The pre-P0-2b judge was prompted for strict "direct match, not topical overlap", so faithful
paraphrases scored partial/no_evidence and cite_precision floored ~0.3-0.4 < 0.45 -> 0 section
ever clean-accepted. P0-2b softens the prompt to accept paraphrase/implication. This test is the
GUARD against over-softening: it must show GOOD passes the 0.45 gate AND stays well above BAD
(gap >= MIN_GAP), i.e. the judge discriminates rather than rubber-stamping everything to 1.0.

Run:  python3 files/eval/bench_cite_discrimination.py
Exit: 0 if discriminating (GOOD>=GATE and GOOD-BAD>=MIN_GAP for both bad variants), else 1.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> files/

from research import verify as V
from research.types import Source

GATE = 0.45      # min_cite_precision in deep_investigate.py -- GOOD must clear this
MIN_GAP = 0.30   # GOOD must outscore each BAD variant by at least this (real discrimination)

# Faithful PARAPHRASES (writer-style restatement) of each source -- this is the case the
# pre-P0-2b strict judge wrongly floored. Markers [1]..[5] map to sources 1..5 below.
GOOD_CONTENT = (
    "A Transformer relies on self-attention so that every token can be weighed against every "
    "other token in a single parallel pass rather than sequentially [1]. This architecture was "
    "first introduced in 2017 [2]. RLHF aligns a policy by optimizing it against a reward model "
    "that is fit to human preference comparisons [3]. Layer normalization rescales each sample's "
    "activations across the feature dimension [4]. The softmax turns a vector of logits into a "
    "normalized probability distribution that sums to one [5]."
)

# The five evidence excerpts the GOOD claims paraphrase (truthfully).
GOOD_EXCERPTS = [
    "The Transformer architecture uses self-attention to process all tokens in parallel.",
    "Attention Is All You Need was published by Vaswani et al. in 2017.",
    "RLHF fine-tunes a policy against a reward model trained on human preference comparisons.",
    "Layer normalization normalizes activations across the feature dimension within each sample.",
    "The softmax function turns a vector of logits into a probability distribution that sums to one.",
]

# Off-topic excerpts (same claims cite these) -> citations are UNRELATED.
UNRELATED_EXCERPTS = [
    "Bananas are an excellent source of potassium.",
    "The Eiffel Tower is located in Paris, France.",
    "Coffee is brewed from roasted coffee beans.",
    "The capital of Japan is Tokyo.",
    "The Pacific is the largest ocean on Earth.",
]

# Excerpts that assert the OPPOSITE of each claim -> citations CONTRADICT.
CONTRADICT_EXCERPTS = [
    "Transformers process tokens strictly one at a time, never in parallel.",
    "Attention Is All You Need was published in 1995.",
    "RLHF requires no human feedback and uses no reward model at all.",
    "Layer normalization normalizes across the batch dimension like batch norm.",
    "Softmax returns negative values that sum to minus one.",
]


def _mk_sources(excerpts):
    return [
        Source(id=f"test:{i}", title=f"Test source {i}", url=f"https://example.org/{i}",
               excerpt=ex, provider="arxiv")
        for i, ex in enumerate(excerpts, start=1)
    ]


def _score(label, content, excerpts):
    res = V.verify_section(content, _mk_sources(excerpts))
    g = res["grounding"]
    verds = [(v["n"], v["verdict"]) for v in res["verdicts"]]
    print(f"\n[{label}] cite_precision={g:.3f}  n_cit={res['n_citations']}")
    for n, vd in sorted(verds):
        print(f"    [{n}] -> {vd}")
    if res.get("weak_summary"):
        print(f"    weak: {res['weak_summary'][:120]}")
    return g


def main():
    print("=" * 78)
    print("P0-2b G2 citation-judge discrimination benchmark")
    print(f"model={V.DEFAULT_JUDGE_MODEL}  GATE(min_cite_precision)={GATE}  MIN_GAP={MIN_GAP}")
    print(f"cosine auto: supports>= {V.AUTO_SUPPORT_COS}  unrelated<= {V.AUTO_UNRELATED_COS}")
    print("=" * 78)

    try:
        good = _score("GOOD (faithful paraphrase)", GOOD_CONTENT, GOOD_EXCERPTS)
        bad_u = _score("BAD_UNRELATED (off-topic sources)", GOOD_CONTENT, UNRELATED_EXCERPTS)
        bad_c = _score("BAD_CONTRADICT (opposite sources)", GOOD_CONTENT, CONTRADICT_EXCERPTS)
    except Exception as e:
        print(f"\nERROR: judge call failed ({e}). Is Ollama up with {V.DEFAULT_JUDGE_MODEL}?")
        return 2

    gap_u = good - bad_u
    gap_c = good - bad_c
    print("\n" + "-" * 78)
    print(f"GOOD            = {good:.3f}   (gate {GATE}: {'PASS' if good >= GATE else 'FAIL -- still floored'})")
    print(f"BAD_UNRELATED   = {bad_u:.3f}   gap GOOD-BAD = {gap_u:+.3f}  (need >= {MIN_GAP})")
    print(f"BAD_CONTRADICT  = {bad_c:.3f}   gap GOOD-BAD = {gap_c:+.3f}  (need >= {MIN_GAP})")
    print("-" * 78)

    discriminates = good >= GATE and gap_u >= MIN_GAP and gap_c >= MIN_GAP
    bad_blocked = bad_u < GATE and bad_c < GATE
    if discriminates and bad_blocked:
        print("VERDICT: DISCRIMINATING -- faithful paraphrase clears the gate, off-source/contradicting")
        print("         citations score below it. The G2 gate is now a real signal (not floor, not 1.0).")
        rc = 0
    elif good < GATE:
        print("VERDICT: STILL FLOORED -- faithful paraphrase fails the gate. Soften the judge further")
        print("         (or recalibrate GATE downward only AFTER confirming the GOOD-BAD gap holds).")
        rc = 1
    else:
        print("VERDICT: NOT DISCRIMINATING ENOUGH -- GOOD-BAD gap too small / bad not blocked. Risk of")
        print("         rubber-stamping (1.0-giả). Do NOT lower the gate; tighten unrelated/contradicts.")
        rc = 1
    print("=" * 78)
    return rc


if __name__ == "__main__":
    sys.exit(main())
