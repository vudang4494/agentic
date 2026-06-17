"""BAER -- transparent Benchmark / Analyze / Eval / Report for a finished book run.

Deterministic, reproducible quality assessment so a finished run can be judged by NUMBERS,
not by guessing. Reads only the run artifacts (state.json + book.md + book.pdf + the run log),
computes objective metrics, and writes book_eval.json + book_eval_report.md into the run dir.

It does NOT call any model -- every number here is mechanically derived from the output, so the
same run always yields the same report. Where a signal is known to be inert (HHEM grounding, see
Rank5 / bench_hhem_discrimination.py) it is reported AND flagged as non-discriminating, never used
as a quality score.

Usage:
    python3 files/eval/benchmark_book.py <run-name-or-dir>
    python3 files/eval/benchmark_book.py _struct_verify
"""
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "files/output/runs"

# ---- thresholds (the REAL accept gate; code is truth -- deep_investigate.py) ----
TH_GROUNDING = 0.70      # INERT (HHEM dead) -- reported but not trusted
TH_TOPIC = 0.50         # G4, real discriminating signal
TH_CITES = 1            # n_cites > 0
TH_REF_RELEVANCE = 0.50  # a source counts as on-topic at/above this rerank relevance
REDUND_SHINGLE = 8       # k-word shingles for cross-section near-duplicate detection
REDUND_FLAG = 0.30      # Jaccard above this between two sections = near-duplicate (v36 failure mode)


# ============================== load ==============================
def load_run(name):
    run_dir = Path(name) if os.path.isdir(name) else RUNS / name
    if not run_dir.is_dir():
        sys.exit(f"run dir not found: {run_dir}")
    state = json.loads((run_dir / "state.json").read_text())
    profile = state.get("profile", {}) or {}
    if not profile and (run_dir / "topic_profile.json").exists():
        profile = json.loads((run_dir / "topic_profile.json").read_text())
    book_md = (run_dir / "book.md").read_text() if (run_dir / "book.md").exists() else ""
    log = ""
    log_path = run_dir.parent / f"{run_dir.name}.log"
    if log_path.exists():
        log = log_path.read_text(errors="ignore")
    return run_dir, state, profile, book_md, log


def pdf_pages(run_dir):
    pdf = run_dir / "book.pdf"
    if not pdf.exists():
        return None
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


# ============================== text helpers ==============================
def _words(s):
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())


def _shingles(s, k=REDUND_SHINGLE):
    w = _words(s)
    return set(tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1)))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_MATH_INLINE = re.compile(r"(?<!\$)\$[^$\n]+\$(?!\$)")
_MATH_DISPLAY = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_ALGO = re.compile(r"\b(algorithm|pseudocode|step\s*\d|procedure|for each|repeat until|initialize)\b", re.I)


def math_depth(content):
    return {
        "inline": len(_MATH_INLINE.findall(content or "")),
        "display": len(_MATH_DISPLAY.findall(content or "")),
        "code": len(_CODE_FENCE.findall(content or "")),
        "algo": len(_ALGO.findall(content or "")),
    }


# ============================== BENCHMARK ==============================
def benchmark(run_dir, state, sections):
    rows = []
    for key in sorted(sections, key=lambda k: [int(x) for x in k.split(".")]):
        s = sections[key]
        md = math_depth(s.get("content", ""))
        rows.append({
            "key": key,
            "title": s.get("title", "")[:48],
            "words": len(_words(s.get("content", ""))),
            "cites": s.get("n_citations", 0),
            "grounding": s.get("grounding", 0.0),
            "topic": s.get("topic_relevance", 0.0),
            "xrefs": s.get("cross_refs", 0),
            "quality": s.get("quality", "?"),
            "concepts": len(s.get("new_concepts", []) or []),
            "n_sources": len(s.get("sources", []) or []),
            "formulas": md["inline"] + md["display"],
            "code": md["code"],
            "algo": md["algo"],
        })
    total_words = sum(r["words"] for r in rows)
    accepted = [r for r in rows if str(r["quality"]).lower() not in ("blocked", "?", "")]
    blocked = [r for r in rows if str(r["quality"]).lower() == "blocked"]
    pages = pdf_pages(run_dir)
    return {
        "sections": len(rows),
        "accepted": len(accepted),
        "blocked": len(blocked),
        "total_words": total_words,
        "avg_words": round(total_words / len(rows), 1) if rows else 0,
        "pages_pdf": pages,
        "pages_est": round(total_words / 450),  # ~450 words/page fallback when no PDF
        "total_cites": sum(r["cites"] for r in rows),
        "avg_cites": round(sum(r["cites"] for r in rows) / len(rows), 1) if rows else 0,
        "render_ok": (run_dir / "book.pdf").exists(),
        "rows": rows,
    }


# ============================== ANALYZE ==============================
def analyze(state, profile, sections, book_md):
    rows_keys = list(sections.keys())

    # -- title uniqueness / anti-matrix --
    titles = [sections[k].get("title", "") for k in rows_keys]
    norm_titles = [" ".join(_words(t)) for t in titles]
    dup_titles = [t for t, c in Counter(norm_titles).items() if c > 1 and t]
    # templated-pattern sniff: same trailing label across many titles ("X: foo", "Y: foo")
    suffixes = Counter(t.split(":")[-1].strip() for t in titles if ":" in t)
    matrix_suffix = [s for s, c in suffixes.items() if c >= 3 and s]

    # -- cross-section redundancy (the v36 duplication failure) --
    shs = {k: _shingles(sections[k].get("content", "")) for k in rows_keys}
    dup_pairs = []
    keys = rows_keys
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = _jaccard(shs[keys[i]], shs[keys[j]])
            if sim >= REDUND_FLAG:
                dup_pairs.append((keys[i], keys[j], round(sim, 3)))
    dup_pairs.sort(key=lambda x: -x[2])
    sims = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sims.append(_jaccard(shs[keys[i]], shs[keys[j]]))
    max_sim = round(max(sims), 3) if sims else 0.0
    mean_sim = round(sum(sims) / len(sims), 3) if sims else 0.0

    # -- reference relevance + provider mix + canonical recall --
    protected = set(state.get("protected_source_ids", []) or profile.get("protected_source_ids", []) or [])
    prov = Counter()
    rel_on = rel_total = 0
    cited_ids = set()
    for k in rows_keys:
        for src in sections[k].get("sources", []) or []:
            if not isinstance(src, dict):
                continue
            prov[src.get("provider", "?")] += 1
            cited_ids.add(src.get("id", ""))
            r = src.get("relevance")
            if isinstance(r, (int, float)):
                rel_total += 1
                if r >= TH_REF_RELEVANCE:
                    rel_on += 1
    canon_recall = (len(protected & cited_ids) / len(protected)) if protected else None

    # -- coverage of must_cover / canonical_terms; out_of_scope leakage --
    book_lc = book_md.lower()
    book_wset = set(_words(book_md))
    _STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "its", "via",
             "e", "g", "eg", "ie", "other", "within", "role", "application", "impact", "topics",
             "advanced", "analyzing", "comparison", "methods", "foundational"}
    def _covered_substr(terms):
        terms = terms or []
        hit = [t for t in terms if t and t.lower() in book_lc]
        return len(hit), len(terms), [t for t in terms if t and t.lower() not in book_lc]
    def _covered_phrase(terms):
        # must_cover entries are long requirement sentences; substring never matches. Count an entry
        # covered when >= 60% of its CONTENT words (non-stopword, len>2) appear anywhere in the book.
        terms = terms or []
        hit, miss = 0, []
        for t in terms:
            cw = [w for w in _words(t) if w not in _STOP and len(w) > 2]
            if not cw:
                continue
            frac = sum(1 for w in cw if w in book_wset) / len(cw)
            if frac >= 0.6:
                hit += 1
            else:
                miss.append(t)
        return hit, len(terms), miss
    mc_hit, mc_n, mc_miss = _covered_phrase(profile.get("must_cover"))
    ct_hit, ct_n, _ = _covered_substr(profile.get("canonical_terms"))
    oos = profile.get("out_of_scope") or []
    oos_leak = [t for t in oos if t and book_lc.count(t.lower()) >= 3]  # appears 3+ times = leaked

    # -- technical depth roll-up --
    md_tot = {"inline": 0, "display": 0, "code": 0, "algo": 0}
    sec_with_formula = 0
    for k in rows_keys:
        d = math_depth(sections[k].get("content", ""))
        for kk in md_tot:
            md_tot[kk] += d[kk]
        if d["inline"] + d["display"] > 0:
            sec_with_formula += 1

    return {
        "dup_titles": dup_titles,
        "matrix_suffix": matrix_suffix,
        "redundancy": {"max_sim": max_sim, "mean_sim": mean_sim,
                        "near_dup_pairs": dup_pairs[:15], "n_near_dup": len(dup_pairs)},
        "references": {"provider_mix": dict(prov),
                        "on_topic_frac": round(rel_on / rel_total, 3) if rel_total else None,
                        "n_rated": rel_total,
                        "canonical_recall": round(canon_recall, 3) if canon_recall is not None else None,
                        "n_protected": len(protected)},
        "coverage": {"must_cover": f"{mc_hit}/{mc_n}", "must_cover_missing": mc_miss,
                      "canonical_terms": f"{ct_hit}/{ct_n}", "out_of_scope_leak": oos_leak},
        "tech_depth": {"sections_with_formula": sec_with_formula, "n_sections": len(rows_keys),
                        "pct_with_formula": round(100 * sec_with_formula / len(rows_keys)) if rows_keys else 0,
                        **md_tot},
    }


# ============================== EVAL (gates, honest) ==============================
def evaluate(bench, log):
    rows = bench["rows"]
    n = len(rows)
    def frac(pred):
        return round(sum(1 for r in rows if pred(r)) / n, 3) if n else 0.0
    # cite_precision is NOT in state.json -> parse from the run log if present.
    cps = [float(x) for x in re.findall(r"cite_prec=(\d+\.\d+)", log or "")]
    grd = [r["grounding"] for r in rows]
    grounding_inert = (len(set(round(g, 4) for g in grd)) <= 1) if grd else False
    return {
        "topic_ge_0.50": frac(lambda r: r["topic"] >= TH_TOPIC),
        "topic_mean": round(sum(r["topic"] for r in rows) / n, 3) if n else 0,
        "topic_distinct": len(set(round(r["topic"], 3) for r in rows)),
        "cites_ge_1": frac(lambda r: r["cites"] >= TH_CITES),
        "has_xref": frac(lambda r: r["xrefs"] >= 1),
        "accept_rate": round(bench["accepted"] / n, 3) if n else 0,
        "grounding_mean": round(sum(grd) / n, 3) if n else 0,
        "grounding_inert": grounding_inert,
        "cite_precision_mean": round(sum(cps) / len(cps), 3) if cps else None,
        "cite_precision_n": len(cps),
    }


# ============================== REPORT ==============================
def _bar(x, width=20):
    x = max(0.0, min(1.0, x or 0.0))
    return "#" * round(x * width) + "-" * (width - round(x * width))


def report(run_dir, state, bench, ana, ev):
    L = []
    A = L.append
    A(f"# Book Eval Report -- {state.get('topic','?')}")
    A(f"_run: {run_dir.name}_  \n")

    A("## 1. BENCHMARK (volume + coverage)")
    A(f"- Pages: **{bench['pages_pdf'] if bench['pages_pdf'] else '(no PDF) ~'+str(bench['pages_est'])+' est'}**"
      f" | Words: **{bench['total_words']:,}** | Sections: **{bench['sections']}**"
      f" (accepted {bench['accepted']}, blocked {bench['blocked']})")
    A(f"- Avg {bench['avg_words']} words/section | Citations: **{bench['total_cites']}** "
      f"(avg {bench['avg_cites']}/section) | Render PDF: {'YES' if bench['render_ok'] else 'NO'}")
    A("")
    A("| sec | title | words | cites | topic | xref | g(inert) | formulas | algo | quality |")
    A("|---|---|--:|--:|--:|--:|--:|--:|--:|---|")
    for r in bench["rows"][:60]:
        A(f"| {r['key']} | {r['title']} | {r['words']} | {r['cites']} | {r['topic']:.2f} | "
          f"{r['xrefs']} | {r['grounding']:.2f} | {r['formulas']} | {r['algo']} | {r['quality']} |")
    if len(bench["rows"]) > 60:
        A(f"| ... | (+{len(bench['rows'])-60} more sections) | | | | | | | | |")
    A("")

    A("## 2. ANALYZE (structure quality -- deterministic)")
    rd = ana["redundancy"]
    A(f"- **Redundancy** (cross-section {REDUND_SHINGLE}-gram Jaccard): mean {rd['mean_sim']}, "
      f"max {rd['max_sim']}, near-duplicate pairs (>= {REDUND_FLAG}): **{rd['n_near_dup']}**")
    for a, b, s in rd["near_dup_pairs"][:8]:
        A(f"    - {a} ~ {b}: {s}")
    A(f"- **Title uniqueness**: duplicate titles {len(ana['dup_titles'])}; "
      f"matrix-suffix pattern {ana['matrix_suffix'] or 'none'}")
    rf = ana["references"]
    A(f"- **References**: on-topic frac {rf['on_topic_frac']} (n={rf['n_rated']}); "
      f"canonical recall {rf['canonical_recall']} (of {rf['n_protected']} protected); "
      f"providers {rf['provider_mix']}")
    cov = ana["coverage"]
    A(f"- **Coverage**: must_cover {cov['must_cover']} (missing: {cov['must_cover_missing'] or 'none'}); "
      f"canonical_terms {cov['canonical_terms']}; out-of-scope leak {cov['out_of_scope_leak'] or 'none'}")
    td = ana["tech_depth"]
    A(f"- **Technical depth**: {td['pct_with_formula']}% sections carry a formula "
      f"({td['sections_with_formula']}/{td['n_sections']}); totals -> inline {td['inline']}, "
      f"display {td['display']}, code-blocks {td['code']}, algo-markers {td['algo']}")
    A("")

    A("## 3. EVAL (gates -- REAL vs inert)")
    A(f"- topic >= {TH_TOPIC} (G4, REAL): **{ev['topic_ge_0.50']:.0%}** `{_bar(ev['topic_ge_0.50'])}` "
      f"(mean {ev['topic_mean']}, distinct values {ev['topic_distinct']})")
    A(f"- cite_precision (G2, REAL, from log): mean {ev['cite_precision_mean']} (n={ev['cite_precision_n']})")
    A(f"- cites >= 1: **{ev['cites_ge_1']:.0%}** | has cross-ref: {ev['has_xref']:.0%} | "
      f"accept rate {ev['accept_rate']:.0%}")
    A(f"- grounding (HHEM): mean {ev['grounding_mean']} -- "
      f"{'**INERT / NO-OP** (constant; do NOT use as quality, see Rank5)' if ev['grounding_inert'] else 'varies'}")
    A("")

    A("## 4. VERDICT")
    issues = []
    if ana["redundancy"]["n_near_dup"] > 0:
        issues.append(f"{ana['redundancy']['n_near_dup']} near-duplicate section pair(s) -> redundancy risk")
    if ana["matrix_suffix"]:
        issues.append(f"matrix-suffix titles {ana['matrix_suffix']} -> outline may be templated")
    if td["pct_with_formula"] < 50:
        issues.append(f"only {td['pct_with_formula']}% sections have a formula -> technical depth thin")
    if rf["on_topic_frac"] is not None and rf["on_topic_frac"] < 0.7:
        issues.append(f"reference on-topic frac {rf['on_topic_frac']} < 0.70 -> sourcing drift")
    if cov["out_of_scope_leak"]:
        issues.append(f"out-of-scope leak: {cov['out_of_scope_leak']}")
    if ev["topic_ge_0.50"] < 0.9:
        issues.append(f"{ev['topic_ge_0.50']:.0%} sections pass topic>=0.50 (G4)")
    if issues:
        A("Issues to inspect:")
        for it in issues:
            A(f"- {it}")
    else:
        A("No deterministic red flags (redundancy, matrix, depth, sourcing, coverage all within bounds).")
    A("")
    A("> Note: grounding is reported but INERT (HHEM degenerate, Rank5). The real quality signals "
      "are topic (G4), cite_precision (G2), redundancy, technical-depth, and reference on-topic frac.")
    return "\n".join(L)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 files/eval/benchmark_book.py <run-name-or-dir>")
    run_dir, state, profile, book_md, log = load_run(sys.argv[1])
    sections = {k: v for k, v in (state.get("sections") or {}).items()}
    if not sections:
        sys.exit("no sections in state.json")
    bench = benchmark(run_dir, state, sections)
    ana = analyze(state, profile, sections, book_md)
    ev = evaluate(bench, log)
    rep = report(run_dir, state, bench, ana, ev)

    (run_dir / "book_eval.json").write_text(json.dumps(
        {"benchmark": {k: v for k, v in bench.items() if k != "rows"},
         "per_section": bench["rows"], "analyze": ana, "eval": ev}, indent=2, ensure_ascii=False))
    (run_dir / "book_eval_report.md").write_text(rep)
    print(rep)
    print(f"\n[BAER] wrote {run_dir/'book_eval_report.md'} + book_eval.json")


if __name__ == "__main__":
    main()
