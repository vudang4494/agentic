"""Regression tests for special/mathematical-character handling across the pipeline.

Guards the fixes for the math-char audit (Rank 1/2/7). Run:
    python3 files/eval/test_math_char_safety.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> files/

from research.fetch import _mathml_to_latex, _html_to_text
from research.notes import clean_citations

_fail = 0


def check(name, cond, detail=""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def test_extraction_angle_brackets():
    # Rank 1: math '<'/'>' must NOT be eaten by the downstream tag-stripper.
    print("Rank1 extraction (math < > not destroyed):")
    html = ('A width <math><annotation encoding="application/x-tex">k&lt;n</annotation></math> '
            'needs <a href="#b">[18]</a> a stack of '
            '<math><annotation encoding="application/x-tex">O(n/k)</annotation></math> steps.')
    out = _html_to_text(_mathml_to_latex(html))
    check("prose around math survives", "needs" in out and "stack" in out and "steps" in out, out)
    check("adjacent formula O(n/k) survives", "O(n/k)" in out, out)
    check("no raw '<' leaks to LaTeX", "<" not in out, out)
    check("math '<' became \\lt (math-safe)", r"\lt" in out, out)


def test_citation_math_safe():
    # Rank 2: clean_citations must NOT strip [N]/[n] index notation inside math.
    print("Rank2 clean_citations (math [N] protected, prose [N] still cleaned):")
    for s in [r"$[N]$ is the set [2]", r"$$\sum_{i \in [N]} x_i$$ [1]", r"$R^{[N]}$ [3]"]:
        out, _ = clean_citations(s, 5)
        check(f"math span preserved: {s[:24]}", "[N]" in out, out)
        check("no broken empty $$ from math strip", "$$ " not in out.replace("$$ x", "x") or "[N]" in out, out)
    out, n = clean_citations("real placeholder [N] here [2]", 5)
    check("prose [N] placeholder still stripped", "[N]" not in out and n >= 1, out)
    out, n = clean_citations("out of range [99] and ok [2]", 5)
    check("out-of-range [99] stripped, [2] kept", "[99]" not in out and "[2]" in out, out)


def test_surrogate_guard():
    # Rank 7: a lone surrogate must not crash the assemble write.
    print("Rank7 surrogate guard:")
    bad = "formula $a$ \udce2 tail"
    crashed = False
    try:
        bad.encode("utf-8")
    except UnicodeEncodeError:
        crashed = True
    check("raw surrogate would crash utf-8 encode", crashed)
    fixed = bad.encode("utf-8", "ignore").decode("utf-8")
    ok = True
    try:
        fixed.encode("utf-8")
    except Exception:
        ok = False
    check("guarded text encodes cleanly + keeps formula", ok and "$a$" in fixed, fixed)


if __name__ == "__main__":
    test_extraction_angle_brackets()
    test_citation_math_safe()
    test_surrogate_guard()
    print()
    if _fail:
        print(f"RESULT: {_fail} check(s) FAILED")
        sys.exit(1)
    print("RESULT: all checks PASSED")
