"""Rank6 benchmark: how badly do the claim-splitter and BM25 tokenizer mangle MATH?

Two real pipeline functions are math-blind:
  1. faithfulness.decompose_claims(body, llm_call_fn=None)  -- fallback splits on (?<=[.!?])\\s+,
     so a '.' INSIDE a formula or after a display block severs the formula -> a claim carries an
     odd number of '$' (broken math) that HHEM/citation then scores as garbage.
  2. notes._tokenize(text) = re.findall(r"[A-Za-z0-9]+", lower)  -- drops every math symbol and
     turns LaTeX control words into bare junk tokens (\\frac -> 'frac', \\mathbb -> 'mathbb'),
     polluting the BM25 sparse index and dropping the operators that carry meaning.

This quantifies the damage so Rank6 can be scoped. No fix here -- measurement only.

Run:  python3 eval/bench_math_split_bm25.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> repo root (research/ importable)

from research.faithfulness import decompose_claims
from research.notes import _tokenize

# Math-bearing bodies the writer realistically emits (display + inline + decimals + abbreviations).
BODIES = [
    ("attention",
     "The scaled dot-product attention is $$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}"
     "\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V.$$ Here $d_k$ is the key dimension. "
     "Scaling by $1/\\sqrt{d_k}$ keeps the variance near $1.0$ so gradients stay stable. "
     "In practice $h=8$ heads are used, i.e. the model splits $d_{model}=512$ into 8 parts."),
    ("layernorm",
     "Layer normalization computes $\\mu=\\frac{1}{H}\\sum_{i=1}^{H} x_i$ and "
     "$\\sigma^2=\\frac{1}{H}\\sum_{i=1}^{H}(x_i-\\mu)^2$. The output is "
     "$y=\\gamma\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}+\\beta$, where $\\epsilon=10^{-5}$. "
     "Unlike batch norm, statistics are per-sample, e.g. for one token at a time."),
    ("adam",
     "Adam updates first moment $m_t=\\beta_1 m_{t-1}+(1-\\beta_1) g_t$ and second moment "
     "$v_t=\\beta_2 v_{t-1}+(1-\\beta_2) g_t^2$. With $\\beta_1=0.9$, $\\beta_2=0.999$, the "
     "step is $\\theta_t=\\theta_{t-1}-\\alpha \\hat m_t / (\\sqrt{\\hat v_t}+\\epsilon)$."),
]


def _dollar_parity_ok(s):
    # A well-formed claim has balanced inline '$' (even count of single-$ after removing $$).
    no_disp = s.replace("$$", "")
    return no_disp.count("$") % 2 == 0


def bench_claim_splitter():
    print("=" * 78)
    print("Rank6a: claim-splitter (fallback) severs formulas")
    print("=" * 78)
    total_claims = total_broken = total_with_math = 0
    for name, body in BODIES:
        claims = decompose_claims(body, llm_call_fn=None)
        broken = [c for c in claims if "$" in c and not _dollar_parity_ok(c)]
        withmath = [c for c in claims if "$" in c]
        total_claims += len(claims)
        total_broken += len(broken)
        total_with_math += len(withmath)
        print(f"\n[{name}] {len(claims)} claims, {len(withmath)} carry math, "
              f"{len(broken)} have BROKEN $-parity (severed formula):")
        for c in claims:
            flag = "  <-- BROKEN $" if ("$" in c and not _dollar_parity_ok(c)) else ""
            print(f"   | {c[:72]}{flag}")
    print("\n" + "-" * 78)
    rate = total_broken / total_with_math if total_with_math else 0.0
    print(f"TOTAL: {total_broken}/{total_with_math} math-bearing claims have a SEVERED formula "
          f"({rate:.0%}).  These reach HHEM/citation as half-formulas.")
    return total_broken, total_with_math


def bench_bm25_tokenizer():
    print("\n" + "=" * 78)
    print("Rank6b: BM25 _tokenize drops math symbols + injects LaTeX-command junk tokens")
    print("=" * 78)
    # LaTeX control words that become meaningless high-freq tokens in the sparse index.
    JUNK = {"frac", "sqrt", "mathrm", "mathbb", "mathcal", "softmax", "sum", "beta", "alpha",
            "gamma", "mu", "sigma", "epsilon", "theta", "hat", "left", "right", "cdot", "times"}
    total_tokens = total_junk = total_symbols_dropped = 0
    for name, body in BODIES:
        toks = _tokenize(body)
        junk = [t for t in toks if t in JUNK]
        # math operator/symbol chars present in the source but absent from any token
        symbols = re.findall(r"[=^_+\-*/<>(){}\\$]", body)
        total_tokens += len(toks)
        total_junk += len(junk)
        total_symbols_dropped += len(symbols)
        print(f"\n[{name}] {len(toks)} tokens; {len(junk)} are LaTeX-command junk; "
              f"{len(symbols)} math/operator chars dropped entirely.")
        print(f"   tokens : {toks}")
        print(f"   junk   : {sorted(set(junk))}")
    print("\n" + "-" * 78)
    jr = total_junk / total_tokens if total_tokens else 0.0
    print(f"TOTAL: {total_junk}/{total_tokens} BM25 tokens ({jr:.0%}) are LaTeX-command junk; "
          f"{total_symbols_dropped} operator/symbol chars dropped across {len(BODIES)} bodies.")
    # Show the index-pollution effect: a math doc matching a query purely on junk overlap.
    q = "frac sqrt"  # a query would never legitimately contain these, but a math doc indexes them
    qd_overlap = [t for t in _tokenize(BODIES[0][1]) if t in set(_tokenize(q))]
    print(f"pollution demo: query _tokenize('{q}') overlaps body['attention'] on {qd_overlap} "
          f"-> LaTeX control words are first-class index terms, not math.")
    return total_junk, total_tokens


def main():
    b, m = bench_claim_splitter()
    j, t = bench_bm25_tokenizer()
    jr = j / t if t else 0.0
    print("\n" + "=" * 78)
    print("Rank6 VERDICT (measurement-driven; severity from the numbers, not assumed):")
    if b == 0:
        print(f"  claim-splitter: {b}/{m} math claims severed -> LOW severity. The (?<=[.!?])\\s+ regex")
        print("         needs a SPACE after the period; periods inside $...$ (decimals, $$...$$) have")
        print("         none, so formulas survive. Real splitter bug is over-splitting i.e./e.g. (not math).")
        print("         => Rank6a: an abbreviation guard would help quality, but math-masking is NOT urgent.")
    else:
        print(f"  claim-splitter: {b}/{m} math claims severed -> mask $...$/$$...$$ before the sentence")
        print("         regex (mirror notes.clean_citations), restore after.")
    sev = "REAL" if jr >= 0.10 else "LOW"
    print(f"  BM25 tokenizer: {j}/{t} tokens ({jr:.0%}) are LaTeX-command junk + all operators dropped")
    print(f"         -> {sev} severity. 'frac/sqrt/beta/sum' become shared index terms, inflating BM25")
    print("         similarity between ANY two formula-heavy sections (false near-duplicates / wrong RRF).")
    print("         => Rank6b: strip LaTeX spans (or mask to one atom) before _tokenize so math stops")
    print("         polluting the sparse arm. Decide: drop-math vs keep-as-single-atom.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
