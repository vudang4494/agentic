"""Canonical math/special-character normalization for the whole pipeline.

ONE source of truth (was duplicated + drifted across deep_research_v3.py and
scripts/render_book.py). Pipeline: split glued display -> balance $$ -> escape
Unicode-math -> validate+neutralize invalid LaTeX. Goal: a math char never gets
silently dropped, and ONE bad formula never crashes the whole render.
"""
import re
import unicodedata

# --- code-fence + math span detectors ---
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_GLUED_DISPLAY_RE = re.compile(r"(?<=[}\)\]A-Za-z0-9])\$\$(?=\s*\\[A-Za-z])")
_MATH_SPAN_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.DOTALL)
_SQRT_RE = re.compile(r"√\s*(\{[^{}]*\}|\([^()]*\)|[A-Za-z0-9]+)")


def _code_fence_spans(text):
    return [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]


_GREEK_TEX = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi", "χ": r"\chi",
    "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}

# UNION of the two old (drifted) maps + the 46 commonly-emitted chars that neither covered.
_MATH_UNICODE = {
    # blackboard / calligraphic (map BEFORE NFKC so they are not flattened to plain letters)
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}",
    "ℂ": r"\mathbb{C}", "𝔼": r"\mathbb{E}", "𝔻": r"\mathbb{D}", "𝔽": r"\mathbb{F}",
    "ℒ": r"\mathcal{L}", "ℋ": r"\mathcal{H}", "𝒩": r"\mathcal{N}", "𝒟": r"\mathcal{D}",
    "𝒜": r"\mathcal{A}", "ℓ": r"\ell",
    # relations / operators
    "∈": r"\in", "∉": r"\notin", "∞": r"\infty", "∑": r"\sum", "∏": r"\prod",
    "∫": r"\int", "∮": r"\oint", "∇": r"\nabla", "∂": r"\partial",
    "≈": r"\approx", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≡": r"\equiv",
    "≅": r"\cong", "≃": r"\simeq", "∼": r"\sim", "∝": r"\propto", "≜": r"\triangleq",
    "×": r"\times", "⋅": r"\cdot", "·": r"\cdot", "±": r"\pm", "∓": r"\mp",
    "⊗": r"\otimes", "⊙": r"\odot", "⊕": r"\oplus", "⊖": r"\ominus", "⊥": r"\perp",
    "∥": r"\parallel", "‖": r"\|", "⊤": r"\top", "⊨": r"\models",
    "∘": r"\circ", "∗": r"\ast", "≪": r"\ll", "≫": r"\gg", "⪅": r"\lesssim",
    # arrows
    "→": r"\to", "←": r"\gets", "↔": r"\leftrightarrow", "⟶": r"\longrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "↦": r"\mapsto",
    # sets / logic
    "⊆": r"\subseteq", "⊂": r"\subset", "⊇": r"\supseteq", "⊃": r"\supset",
    "∪": r"\cup", "∩": r"\cap", "∖": r"\setminus", "∅": r"\emptyset",
    "∀": r"\forall", "∃": r"\exists", "¬": r"\neg", "∧": r"\land", "∨": r"\lor",
    # delimiters / misc
    "⟨": r"\langle", "⟩": r"\rangle", "⌊": r"\lfloor", "⌋": r"\rfloor",
    "⌈": r"\lceil", "⌉": r"\rceil", "…": r"\dots", "⋯": r"\cdots",
    # sub/superscripts
    "²": r"^{2}", "³": r"^{3}", "⁴": r"^{4}", "ⁿ": r"^{n}",
    "₀": r"_{0}", "₁": r"_{1}", "₂": r"_{2}", "₃": r"_{3}",
}

_ALL = {**_GREEK_TEX, **_MATH_UNICODE}


def split_glued_display(content):
    new, n = _GLUED_DISPLAY_RE.subn("$$\n\n$$", content)
    if n:
        print(f"[normalize_math] split {n} glued display-math delimiter(s)", flush=True)
    return new


def balance_display_math(content):
    """Repair odd `$$` counts (outside fenced code) that crash tectonic."""
    fences = _code_fence_spans(content)
    def in_fence(pos):
        return any(a <= pos < b for a, b in fences)
    positions = [m.start() for m in re.finditer(r"\$\$", content) if not in_fence(m.start())]
    if len(positions) % 2 == 0:
        return content
    last = positions[-1]
    prev = positions[-2] if len(positions) >= 2 else None
    before = content[prev + 2:last] if prev is not None else ""
    after = content[last + 2:]
    has_latex = lambda s: bool(re.search(r"\\[A-Za-z]+", s))
    if has_latex(after) and not has_latex(before):
        m = re.search(r"\n\s*\n", after)
        cut = (last + 2 + m.start()) if m else len(content)
        print("[normalize_math] odd $$; closing unclosed trailing display formula", flush=True)
        return content[:cut] + "$$" + content[cut:]
    if prev is not None and has_latex(before):
        print("[normalize_math] odd $$; wrapping dangling display formula (missing opener)", flush=True)
        return content[:prev + 2] + "\n\n$$" + before.strip() + "$$\n\n" + content[last + 2:]
    print(f"[normalize_math] odd $$ ({len(positions)}); dropping orphan display delimiter", flush=True)
    return content[:last] + content[last + 2:]


def _apply_sqrt(text, wrap):
    """√x -> \\sqrt{x} (radicand: braced/parened/token). `wrap` adds the surrounding $...$ when
    the √ sits in prose (outside a math span); inside a span we must NOT add $ (would nest)."""
    o, c = ("$", "$") if wrap else ("", "")
    text = _SQRT_RE.sub(lambda m: o + r"\sqrt{" + m.group(1).strip("{}()") + "}" + c, text)
    return text.replace("√", o + r"\sqrt{}" + c)  # lone √ with no radicand


def balance_inline_dollar(content):
    """Escape a STRAY unpaired inline `$` (outside $$ blocks / code fences) -> literal `\\$`.
    A lone `$` (a price like `$0.20`, or a dangling delimiter after a content-bleed) opens TeX math
    that runs to the next `$` -- often swallowing a later \\frac and aborting the WHOLE render with
    'File ended while scanning use of \\frac'. balance_display_math only fixes `$$`; this fixes `$`."""
    masks = []
    def _mask(m):
        masks.append(m.group(0))
        return f"\x00M{len(masks)-1}\x00"
    masked = re.sub(r"```[\s\S]*?```|\$\$[\s\S]*?\$\$", _mask, content)
    n = 0
    out = []
    for line in masked.split("\n"):
        ds = [m.start() for m in re.finditer(r"(?<!\\)\$", line)]
        if len(ds) % 2 == 1:
            # prefer escaping a price-like `$` (followed by a digit); else the last (unpaired) one.
            price = [p for p in ds if line[p + 1:p + 2].isdigit()]
            t = price[0] if len(price) == 1 else ds[-1]
            line = line[:t] + r"\$" + line[t + 1:]
            n += 1
        out.append(line)
    masked = "\n".join(out)
    for i, mk in enumerate(masks):
        masked = masked.replace(f"\x00M{i}\x00", mk)
    if n:
        print(f"[normalize_math] escaped {n} stray unpaired inline $ -> literal (render-safe)", flush=True)
    return masked


def escape_unicode_math(content):
    # inside $...$ spans: replace bare Unicode-math with its TeX (NO extra $)
    def _fix_span(m):
        span = _apply_sqrt(m.group(0), wrap=False)
        # A raw `%` inside math is a LaTeX COMMENT -> it eats the rest of the formula (every closing
        # brace), making \frac/\sqrt scan to EOF and aborting the WHOLE render. Escape it to \%.
        span = re.sub(r"(?<!\\)%", r"\\%", span)
        for ch, tex in _ALL.items():
            if ch in span:
                span = span.replace(ch, tex + " ")
        return span
    content = _MATH_SPAN_RE.sub(_fix_span, content)
    # outside spans: a bare Unicode-math char in prose must be WRAPPED in inline math, else a
    # bare \sqrt/\otimes lands in text mode and crashes tectonic with 'Missing $ inserted'.
    content = _apply_sqrt(content, wrap=True)
    for ch, tex in _ALL.items():
        if ch in content:
            content = content.replace(ch, f"${tex}$")
    # NFKC only for leftover math-alphanumeric (bold) NOT in the map
    content = "".join(
        unicodedata.normalize("NFKC", c) if 0x1D400 <= ord(c) <= 0x1D7FF else c
        for c in content
    )
    return content


# Macros tectonic can render (base TeX + amsmath + amssymb + the preamble \providecommand set in
# scripts/render_book.py). A span using a control word OUTSIDE this set (e.g. a writer-invented or
# package-only macro) is NEUTRALIZED -- otherwise ONE "Undefined control sequence" aborts the whole
# tectonic render (which is what forced the weasyprint fallback that cannot typeset LaTeX math).
_MACRO_ALLOWLIST = frozenset("""
text textbf textit textrm texttt textsf textsc textnormal textstyle textsuperscript mbox hbox
mathnormal mathbb mathcal mathfrak mathscr mathbf mathrm mathit mathsf mathtt boldsymbol bm
mathds mathbbm symbb symbf symcal
frac dfrac tfrac binom dbinom tbinom sqrt root over
hat widehat tilde widetilde bar overline vec dot ddot acute grave check breve mathring
overbrace underbrace overrightarrow overleftarrow underrightarrow underline not
left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr middle
quad qquad enspace thinspace negthinspace hspace vspace phantom qqaud
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi
pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
sin cos tan cot sec csc sinh cosh tanh coth arcsin arccos arctan log ln lg exp
lim limsup liminf max min sup inf det dim ker deg gcd hom arg Pr mod bmod pmod
softmax argmax argmin sign operatorname operatornamewithlimits DeclareMathOperator
forall exists nexists neg lnot land lor wedge vee implies iff therefore because
in notin ni subset supset subseteq supseteq subsetneq supsetneq cup cap bigcup bigcap
emptyset varnothing setminus complement
to gets rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow longrightarrow
longleftarrow uparrow downarrow updownarrow mapsto longmapsto hookrightarrow hookleftarrow
rightharpoonup leftharpoondown xrightarrow xleftarrow leadsto
top bot vdash dashv models vDash Vdash perp parallel mid nmid
ldots cdots vdots ddots dots cdot bullet star ast circ
infty partial nabla ell hbar imath jmath wp Re Im aleph prime
langle rangle lfloor rfloor lceil rceil lbrace rbrace lbrack rbrack vert Vert backslash
sum prod coprod int oint iint iiint bigsqcup bigvee bigwedge bigodot bigotimes bigoplus biguplus
pm mp times div ast star dagger ddagger amalg uplus sqcap sqcup wr diamond
triangleleft triangleright oplus ominus otimes oslash odot bigcirc
leq geq le ge ll gg leqslant geqslant lneq gneq neq ne prec preceq succ succeq sim simeq cong
approx equiv propto asymp doteq triangleq lesssim gtrsim approxeq
quad mid arg lim
begin end nonumber label tag notag split aligned align alignat gather gathered multline cases
array matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix smallmatrix substack
displaystyle scriptstyle scriptscriptstyle limits nolimits stackrel overset underset
angle triangle square diamond Box blacksquare flat natural sharp surd checkmark
bf rm it sf tt cal sl em scriptsize footnotesize small large Large
lt gt
""".split())


def _math_span_valid(inner):
    """Render-safe check: braces balanced (honoring \\{ \\}), \\left/\\right paired, and every macro
    is in the tectonic-renderable allowlist. A span failing ANY of these is neutralized to literal
    code so it cannot crash the render."""
    s = inner.replace(r"\{", "").replace(r"\}", "")
    if s.count("{") != s.count("}"):
        return False
    if inner.count(r"\left") != inner.count(r"\right"):
        return False
    for mac in set(re.findall(r"\\([A-Za-z]+)", inner)):
        if mac not in _MACRO_ALLOWLIST:
            return False
    return True


def _neutralize_span(whole):
    """Render a broken math span as LITERAL monospaced source via a markdown inline-code span.
    Pandoc turns backtick code into \\texttt{...} with every LaTeX-active char escaped, so neither
    the `$` delimiters NOR the inner commands (\\frac, \\left, ...) execute -> tectonic can't crash.
    Just escaping the `$` is NOT enough: it leaves bare \\frac in text mode, which itself errors
    'Missing $ inserted'. The source text stays visible, so no information is lost."""
    longest = run = 0
    for ch in whole:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if (whole.startswith("`") or whole.endswith("`")) else ""
    return f"{fence}{pad}{whole}{pad}{fence}"


def validate_and_neutralize_math(content):
    """A structurally-broken LaTeX span (unbalanced { or \\left) crashes tectonic and fails the
    WHOLE book. Neutralize such a span -> render its source as literal code, so it never crashes
    and no information is lost."""
    fences = _code_fence_spans(content)
    n = [0]
    def repl(m):
        if any(a <= m.start() < b for a, b in fences):
            return m.group(0)
        whole = m.group(0)
        inner = whole[2:-2] if whole.startswith("$$") else whole[1:-1]
        if _math_span_valid(inner):
            return whole
        n[0] += 1
        return _neutralize_span(whole)
    out = _MATH_SPAN_RE.sub(repl, content)
    if n[0]:
        print(f"[normalize_math] neutralized {n[0]} invalid LaTeX span(s) to literal code (render-safe)", flush=True)
    return out


def normalize_math(content):
    """Canonical math normalization: split glued -> balance $$ -> escape Unicode -> validate."""
    if not content:
        return content
    content = split_glued_display(content)
    content = balance_display_math(content)
    content = balance_inline_dollar(content)
    content = escape_unicode_math(content)
    content = validate_and_neutralize_math(content)
    return content
