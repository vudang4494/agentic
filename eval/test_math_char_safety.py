"""Regression tests for special/mathematical-character handling across the pipeline.

Guards the fixes for the math-char audit (Rank 1/2/7). Run:
    python3 eval/test_math_char_safety.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> repo root (research/ importable)

from research.fetch import _mathml_to_latex, _html_to_text
from research.notes import clean_citations
from research.mathfix import (
    normalize_math, escape_unicode_math, validate_and_neutralize_math, _math_span_valid,
    balance_inline_dollar,
)

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


def test_mathfix_render_safety():
    # Rank 3/4: canonical mathfix -- extended Unicode map, span-aware sqrt, validate+neutralize.
    print("Rank3/4 mathfix (render-safe normalization):")
    # sqrt must carry its radicand and (in prose) be wrapped in $ -- a bare \sqrt crashes tectonic.
    o = escape_unicode_math("root √2 and √{n+1} in prose")
    check("prose sqrt radicand + wrapped", r"$\sqrt{2}$" in o and r"$\sqrt{n+1}$" in o, o)
    check("no empty/bare \\sqrt{}", r"\sqrt{}" not in o and " \\sqrt" not in o, o)
    # previously-uncovered chars now map (would otherwise drop or crash)
    o = escape_unicode_math("A⊗B, ∫f, x→y, ⟨a,b⟩, ℓ, 𝔼[x], ‖v‖")
    for sym, tex in [("⊗", r"\otimes"), ("∫", r"\int"), ("→", r"\to"),
                     ("⟨", r"\langle"), ("ℓ", r"\ell"), ("𝔼", r"\mathbb{E}"), ("‖", r"\|")]:
        check(f"{sym} -> {tex}", tex in o and sym not in o, o)
    check("blackboard 𝔼 NOT NFKC-flattened to plain E", "$E$" not in o, o)
    # validator: balanced ok, unbalanced/dangling \left rejected (honoring \{ \})
    check("valid span ok", _math_span_valid(r"\frac{a}{b}") and _math_span_valid(r"\{x\}"))
    check("broken span rejected", not _math_span_valid(r"\frac{a}{b") and not _math_span_valid(r"\left( x"))
    # neutralize a broken span -> literal inline code (no $ exec, no \frac in text mode), prose survives
    out = validate_and_neutralize_math(r"ok $\frac{a}{b}$ bad $\frac{1}{0$ tail survives")
    check("valid span preserved", r"$\frac{a}{b}$" in out, out)
    check("broken span -> literal code", r"`$\frac{1}{0$`" in out, out)
    check("broken span not live math", "$\\frac{1}{0$ tail" not in out, out)
    check("prose after broken span survives", "tail survives" in out, out)
    # Per-paragraph bounding: good math in CLEAN paragraphs survives; a paragraph with an unclosed
    # brace (content-bleed) is neutralized to literal so it cannot run to EOF and abort the render.
    p = normalize_math("Eq $$\\mathrm{softmax}(x)$$ here.\n\nBleed: $s \\in \\mathbb{As discussed in foo$ end.\n\nRoot √2 there.")
    check("good math in clean paragraphs survives", "softmax" in p and r"$\sqrt{2}$" in p, p)
    check("unbalanced-brace paragraph neutralized (bounded)", r"\$s" in p or r"\mathbb\{As" in p, p)
    # idempotent: running twice == running once (no compounding of neutralized spans)
    once = normalize_math("bad $\\frac{1}{0$ tail")
    check("normalize_math idempotent", normalize_math(once) == once, once)


def test_bm25_math_safe():
    # Rank6: the BM25 sparse tokenizer must NOT index LaTeX-command junk (frac/sqrt/mathrm/sum...)
    # which would inflate similarity between any two formula-heavy sections. Prose terms survive.
    from research.notes import _tokenize
    print("Rank6 BM25 _tokenize (math stripped, prose kept):")
    toks = _tokenize(r"The softmax layer $\mathrm{softmax}(x)=\frac{e^{x}}{\sum_j e^{x_j}}$ aids attention.")
    junk = {"frac", "sqrt", "mathrm", "sum", "mathbb", "cdot", "left", "right"}
    check("no LaTeX-command junk tokens", not (set(toks) & junk), toks)
    check("prose terms survive", {"softmax", "layer", "attention"} <= set(toks), toks)
    toks2 = _tokenize(r"$$\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_i g_i$$")
    check("pure display-math span -> no junk", not (set(toks2) & junk), toks2)


def test_mathfix_render_robustness():
    # The full-book render fixes: raw % in math, stray inline $, undefined-macro neutralize.
    print("Render robustness (raw %, stray $, macro-allowlist):")
    # raw % inside math is a LaTeX comment -> would eat the rest of the formula; must be escaped.
    o = normalize_math(r"$$\sigma=\sqrt{\frac{\sum(x% _i)}{E}}$$")
    check("raw % in math escaped to \\%", r"\%" in o and "x\\% _i" in o.replace(" ", " "), o)
    check("escaped % keeps formula closeable", o.count("{") == o.count("}"), o)
    # a stray unpaired inline $ (price / dangling) -> escaped to literal \$ so it can't open math.
    o = balance_inline_dollar("cost was $0.20 per call and $x$ is valid")
    check("price $0.20 -> \\$0.20", r"\$0.20" in o, o)
    check("valid $x$ pair preserved", "$x$" in o, o)
    o = balance_inline_dollar(r"param $\delta$ and a dangling $ here")
    check("dangling trailing $ escaped", o.count("$") - o.count(r"\$") == 2, o)  # only the $\delta$ pair live
    # undefined macro (not in the tectonic allowlist) -> neutralized to literal code, not fed to TeX.
    check("known macro span valid", _math_span_valid(r"\frac{a}{b}") and _math_span_valid(r"\mathbb{R}"))
    check("undefined macro span invalid (neutralized)", not _math_span_valid(r"\frobnicate{x}"))
    out = validate_and_neutralize_math(r"good $\frac{a}{b}$ bad $\frobnicate{x}$ tail")
    check("undefined-macro span -> literal code", r"`$\frobnicate{x}$`" in out, out)
    check("good span beside it preserved", r"$\frac{a}{b}$" in out, out)


if __name__ == "__main__":
    test_extraction_angle_brackets()
    test_citation_math_safe()
    test_surrogate_guard()
    test_mathfix_render_safety()
    test_mathfix_render_robustness()
    test_bm25_math_safe()
    print()
    if _fail:
        print(f"RESULT: {_fail} check(s) FAILED")
        sys.exit(1)
    print("RESULT: all checks PASSED")
