#!/usr/bin/env python3
"""Quick eval for v3 pipeline runs against gold yaml standards."""
import json, re, yaml
from collections import defaultdict
from pathlib import Path
from datetime import datetime

ROOT = Path("output/runs")
TOPICS_DIR = Path("eval/topics")

RUNS = [
    ("transformer_v3",     "transformer"),
    ("llm_agentic_2026_v3", "llm_agentic_2026"),
    ("diffusion_v3",        None),
    ("agentic_v3",          None),
    ("rag_v3",              None),
    ("rlhf_v3",             None),
    ("longctx_v3",          None),
]

def compute_stats(run_dir: Path):
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        state = json.load(f)

    sections = state.get("sections", {})
    book_path = run_dir / "book.md"
    book_content = book_path.read_text() if book_path.exists() else ""

    # Per-section stats
    groundings = []
    n_citations_list = []
    word_counts = []
    zero_cite_count = 0
    all_sources = []
    all_source_domains = []
    quality_ok = 0
    topic_relevances = []

    for sk, sv in sections.items():
        g = sv.get("grounding", 0)
        if g is not None:
            groundings.append(g)
        tr = sv.get("topic_relevance", 0)
        if tr is not None:
            topic_relevances.append(tr)
        nc = sv.get("n_citations", 0)
        n_citations_list.append(nc)
        wc = sv.get("content", "") or ""
        word_counts.append(len(wc.split()))
        if nc == 0:
            zero_cite_count += 1
        for src in sv.get("sources", []):
            all_sources.append(src)
            url = src.get("url", "")
            all_source_domains.append(url)

    # Grounding
    avg_g = sum(groundings) / len(groundings) if groundings else 0
    median_g = sorted(groundings)[len(groundings)//2] if groundings else 0

    # Citation: extract arxiv IDs from source IDs (strip version suffix)
    arxiv_ids = set()
    for src in all_sources:
        sid = src.get("id", "")
        if "arxiv" in sid:
            aid = sid.split(":", 1)[-1].strip()
            # Strip version suffix (e.g. 1706.03762v1 -> 1706.03762)
            aid = re.sub(r'v\d+$', '', aid)
            arxiv_ids.add(aid)

    # Forbidden domain hits
    forbidden = {
        "techcrunch.com", "medium.com", "youtube.com", "reddit.com",
        "twitter.com", "linkedin.com", "x.com", "substack.com",
        "quora.com", "venturebeat.com", "huggingface.co/blog",
        "news.ycombinator.com"
    }
    fbd_hits = sum(1 for d in all_source_domains if any(b in d for b in forbidden))

    # Subtopic coverage (book-level content scan)
    # Approximate using the first section's content for expected_subtopics pattern
    # We'll do book-level scan for covered keywords
    subtopic_coverage = None  # computed if gold yaml exists

    # Word stats
    median_wc = sorted(word_counts)[len(word_counts)//2] if word_counts else 0

    n_sections = len(sections)

    return {
        "n_sections": n_sections,
        "total_words": state.get("total_words", 0),
        "avg_grounding": round(avg_g, 4),
        "median_grounding": round(median_g, 4),
        "avg_topic_relevance": round(sum(topic_relevances)/len(topic_relevances), 4) if topic_relevances else 0,
        "median_wc": median_wc,
        "avg_wc": round(sum(word_counts)/len(word_counts), 1) if word_counts else 0,
        "zero_cite_sections": zero_cite_count,
        "zero_cite_pct": round(zero_cite_count / n_sections * 100, 1) if n_sections else 0,
        "total_citations": sum(n_citations_list),
        "avg_citations": round(sum(n_citations_list)/len(n_citations_list), 1) if n_citations_list else 0,
        "arxiv_ids_found": sorted(arxiv_ids),
        "n_arxiv_found": len(arxiv_ids),
        "forbidden_domain_hits": fbd_hits,
        "quality_ok": quality_ok,
        "groundings": groundings,
        "word_counts": word_counts,
        "n_citations_list": n_citations_list,
    }


def gold_eval(stats: dict, gold: dict, book_content: str):
    """Evaluate stats against gold yaml thresholds."""
    results = {}
    t = gold.get("thresholds", {})

    # must_cite recall
    must_ids = {m["arxiv"] for m in gold.get("must_cite", [])}
    found_must = sum(1 for aid in must_ids if aid in stats.get("arxiv_ids_found", []))
    must_recall = found_must / len(must_ids) if must_ids else 0
    results["must_cite_recall"] = round(must_recall, 3)
    results["must_cite_found"] = f"{found_must}/{len(must_ids)}"
    results["must_cite_pass"] = must_recall >= t.get("must_cite_recall_min", 0)

    # should_cite recall
    should_ids = {s["arxiv"] for s in gold.get("should_cite", [])}
    found_should = sum(1 for aid in should_ids if aid in stats.get("arxiv_ids_found", []))
    should_recall = found_should / len(should_ids) if should_ids else 0
    results["should_cite_recall"] = round(should_recall, 3)
    results["should_cite_found"] = f"{found_should}/{len(should_ids)}"
    results["should_cite_pass"] = should_recall >= t.get("should_cite_recall_min", 0)

    # Grounding
    avg_g = stats["avg_grounding"]
    results["grounding_pass"] = avg_g >= t.get("grounding_mean_min", 0)

    # Zero cite sections
    results["zero_cite_pass"] = stats["zero_cite_sections"] <= t.get("zero_cite_sections_max", 99)

    # Forbidden domains
    results["forbidden_pass"] = stats["forbidden_domain_hits"] <= t.get("forbidden_domain_hits_max", 99)

    # Subtopic coverage (book-level)
    expected = gold.get("expected_subtopics", [])
    if expected:
        covered = sum(1 for kw in expected if kw.lower() in book_content.lower())
        coverage = covered / len(expected)
        results["subtopic_coverage"] = round(coverage, 3)
        results["subtopic_found"] = f"{covered}/{len(expected)}"
        results["subtopic_pass"] = coverage >= t.get("subtopic_coverage_min", 0)
    else:
        results["subtopic_coverage"] = None
        results["subtopic_pass"] = True

    # Median word count
    mwc = stats["median_wc"]
    lo = t.get("median_words_min", 0)
    hi = t.get("median_words_max", 99999)
    results["median_wc_pass"] = lo <= mwc <= hi

    # Overall
    checks = [
        results.get("must_cite_pass", False),
        results.get("grounding_pass", False),
        results.get("zero_cite_pass", False),
        results.get("forbidden_pass", False),
        results.get("subtopic_pass", True),
        results.get("median_wc_pass", True),
    ]
    results["overall_pass"] = sum(checks) >= len(checks) - 1  # allow 1 fail

    return results


def main():
    print("=" * 80)
    print(f"V3 PIPELINE EVAL REPORT  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    all_rows = []

    for run_name, topic_yaml in RUNS:
        run_dir = ROOT / run_name
        if not run_dir.exists():
            print(f"\n[SKIP] {run_name} -- directory not found")
            continue

        stats = compute_stats(run_dir)
        if stats is None:
            print(f"\n[SKIP] {run_name} -- no state.json")
            continue

        print(f"\n{'─'*80}")
        print(f"  RUN: {run_name}")
        print(f"  Sections: {stats['n_sections']}  |  Words: {stats['total_words']:,}")
        print(f"  Grounding: avg={stats['avg_grounding']}  median={stats['median_grounding']}  topic_rel={stats['avg_topic_relevance']}")
        print(f"  Citations: total={stats['total_citations']}  avg={stats['avg_citations']}/sec  zero-cite={stats['zero_cite_sections']} ({stats['zero_cite_pct']}%)")
        print(f"  Words/sec: median={stats['median_wc']}  avg={stats['avg_wc']}")
        print(f"  arXiv IDs found: {stats['n_arxiv_found']}")
        print(f"  Forbidden domain hits: {stats['forbidden_domain_hits']}")

        if topic_yaml:
            gold_path = TOPICS_DIR / f"{topic_yaml}.yaml"
            if gold_path.exists():
                with open(gold_path) as f:
                    gold = yaml.safe_load(f)

                book_path = run_dir / "book.md"
                book_content = book_path.read_text() if book_path.exists() else ""

                gres = gold_eval(stats, gold, book_content)
                print(f"\n  GOLD EVAL vs {topic_yaml}:")
                print(f"    must_cite_recall: {gres['must_cite_found']} ({gres['must_cite_recall']:.0%})  [{'PASS' if gres['must_cite_pass'] else 'FAIL'}]")
                print(f"    should_cite_recall: {gres['should_cite_found']} ({gres['should_cite_recall']:.0%})  [{'PASS' if gres['should_cite_pass'] else 'FAIL'}]")
                print(f"    grounding_mean: {stats['avg_grounding']}  [{'PASS' if gres['grounding_pass'] else 'FAIL'}]")
                print(f"    zero_cite_secs: {stats['zero_cite_sections']}  [{'PASS' if gres['zero_cite_pass'] else 'FAIL'}]")
                print(f"    forbidden_hits: {stats['forbidden_domain_hits']}  [{'PASS' if gres['forbidden_pass'] else 'FAIL'}]")
                if gres['subtopic_coverage'] is not None:
                    print(f"    subtopic_cov: {gres['subtopic_found']} ({gres['subtopic_coverage']:.0%})  [{'PASS' if gres['subtopic_pass'] else 'FAIL'}]")
                print(f"    median_wc: {stats['median_wc']}  [{'PASS' if gres['median_wc_pass'] else 'FAIL'}]")
                print(f"    OVERALL: {'PASS' if gres['overall_pass'] else 'FAIL'}")
            else:
                print(f"  [No gold yaml: {gold_path}]")
        else:
            print(f"  (no gold yaml for this topic)")

        all_rows.append({
            "run": run_name,
            "sections": stats["n_sections"],
            "words": stats["total_words"],
            "avg_g": stats["avg_grounding"],
            "med_wc": stats["median_wc"],
            "zero_cite": stats["zero_cite_sections"],
            "arxiv": stats["n_arxiv_found"],
            "fbd_hits": stats["forbidden_domain_hits"],
        })

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'─'*80}")
    header = f"  {'Run':<25} {'Sec':>4} {'Words':>7} {'Avg G':>6} {'Med WC':>6} {'0-cite':>6} {'arXiv':>5} {'Fbd':>4}"
    print(header)
    print(f"  {'─'*24} {'─'*4} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*4}")
    for r in all_rows:
        print(f"  {r['run']:<24} {r['sections']:>4} {r['words']:>7,} {r['avg_g']:>6.3f} {r['med_wc']:>6} {r['zero_cite']:>6} {r['arxiv']:>5} {r['fbd_hits']:>4}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
