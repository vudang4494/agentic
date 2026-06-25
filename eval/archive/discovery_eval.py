#!/usr/bin/env python3
"""
Discovery / Outline evaluation harness for the v3 Deep Research pipeline.

Purpose:
- Evaluate Stage 0 (Discovery) and Stage 1 (Outline) artifacts directly
- Produce concrete scores for TopicProfile completeness, outline specificity,
  prompt quality, fallback behavior, and source authority mix
- Support Tier 1 execution of `plan.md`

Usage:
  python3 eval/discovery_eval.py --run attention_v3
  python3 eval/discovery_eval.py --run attention_v3 --format markdown
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

try:
    from research.discovery import TopicProfile
    from research.outline_from_research import OutlineProfile
except Exception:
    TopicProfile = None
    OutlineProfile = None

GENERIC_TITLE_RE = re.compile(r"^(chapter|part|section)\s+\d+\b", re.IGNORECASE)
WORD_RE = re.compile(r"\b\w+\b")
PRIMARY_HOSTS = {
    "arxiv.org",
    "en.wikipedia.org",
    "wikipedia.org",
    "aclanthology.org",
    "openreview.net",
    "proceedings.mlr.press",
    "jmlr.org",
    "nature.com",
    "science.org",
    "semanticscholar.org",
}
SECONDARY_HINTS = (
    "blog",
    "medium.com",
    "substack.com",
    "towardsdatascience.com",
    "analyticsvidhya.com",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _maybe_bool_to_int(v: bool) -> int:
    return 1 if v else 0


def _completeness_score(topic_profile: dict) -> dict:
    required = {
        "name": bool(_normalize_text(topic_profile.get("name"))),
        "subtitle": bool(_normalize_text(topic_profile.get("subtitle"))),
        "description": bool(_normalize_text(topic_profile.get("description"))),
        "scope": bool(_normalize_text(topic_profile.get("scope"))),
        "key_concepts": len(topic_profile.get("key_concepts", []) or []) >= 4,
        "initial_queries": len(topic_profile.get("initial_queries", []) or []) >= 4,
        "canonical_papers": len(topic_profile.get("canonical_papers", []) or []) >= 1,
        "canonical_terms": len(topic_profile.get("canonical_terms", []) or []) >= 4,
        "must_cover": len(topic_profile.get("must_cover", []) or []) >= 4,
        "out_of_scope": len(topic_profile.get("out_of_scope", []) or []) >= 1,
        "seed_queries_by_axis": len(topic_profile.get("seed_queries_by_axis", {}) or {}) >= 3,
    }
    present = sum(_maybe_bool_to_int(v) for v in required.values())
    total = len(required)
    return {
        "present": present,
        "total": total,
        "ratio": round(present / total, 4) if total else 0.0,
        "checks": required,
    }


def _title_specificity_score(title: str, topic_terms: list[str]) -> float:
    t = _normalize_text(title)
    if not t:
        return 0.0
    lower = t.lower()
    score = 1.0
    if GENERIC_TITLE_RE.match(t):
        score -= 0.6
    n_words = _word_count(t)
    if n_words <= 2:
        score -= 0.25
    if ":" in t or " of " in lower or " for " in lower:
        score += 0.15
    if any(term.lower() in lower for term in topic_terms if term):
        score += 0.2
    if any(k in lower for k in ("introduction", "overview", "basics")) and n_words <= 3:
        score -= 0.15
    return max(0.0, min(1.0, score))


def _prompt_quality_score(prompt: str, section_title: str) -> dict:
    p = _normalize_text(prompt)
    if not p:
        return {"score": 0.0, "flags": ["empty_prompt"]}
    flags = []
    score = 1.0
    wc = _word_count(p)
    if wc < 8:
        score -= 0.35
        flags.append("too_short")
    if p.lower().startswith("write a section on "):
        score -= 0.15
        flags.append("template_prompt")
    if section_title and p.strip().lower() == f"write a section on {section_title.lower()}.":
        score -= 0.30
        flags.append("title_only_prompt")
    if not any(k in p.lower() for k in ("cover", "explain", "compare", "analyze", "evidence", "avoid")):
        score -= 0.15
        flags.append("weak_directive")
    return {"score": max(0.0, min(1.0, score)), "flags": flags}


def _source_authority_stats(state: dict) -> dict:
    sections = state.get("sections", {}) or {}
    hosts = Counter()
    primary = 0
    secondary = 0
    total = 0
    for sec in sections.values():
        for src in sec.get("sources", []) or []:
            total += 1
            sid = (src.get("id") or "").lower()
            url = src.get("url") or ""
            host = _host(url)
            if host:
                hosts[host] += 1
            is_primary = (
                sid.startswith("arxiv:")
                or sid.startswith("wiki:")
                or host in PRIMARY_HOSTS
                or host.endswith(".wikipedia.org")
            )
            is_secondary = any(h in host for h in SECONDARY_HINTS) if host else False
            if is_primary:
                primary += 1
            elif is_secondary:
                secondary += 1
    return {
        "total_sources": total,
        "primary_sources": primary,
        "secondary_sources": secondary,
        "primary_source_pct": round(primary / total, 4) if total else 0.0,
        "secondary_source_pct": round(secondary / total, 4) if total else 0.0,
        "top_hosts": dict(hosts.most_common(12)),
    }


def evaluate_run(run_dir: Path) -> dict:
    topic_profile = _read_json(run_dir / "topic_profile.json")
    outline_profile = _read_json(run_dir / "outline_profile.json")
    state = _read_json(run_dir / "state.json") if (run_dir / "state.json").exists() else {}

    completeness = _completeness_score(topic_profile)
    topic_terms = list(topic_profile.get("canonical_terms", []) or []) + list(topic_profile.get("must_cover", []) or [])

    chapters = outline_profile.get("chapters", []) or []
    chapter_titles = []
    section_titles = []
    prompt_scores = []
    prompt_flags = Counter()
    chapter_scores = []
    section_scores = []
    generic_chapters = []
    generic_sections = []
    empty_prompts = 0
    total_sections = 0

    for ch in chapters:
        ch_title = _normalize_text(ch.get("t"))
        chapter_titles.append(ch_title)
        ch_score = _title_specificity_score(ch_title, topic_terms)
        chapter_scores.append(ch_score)
        if GENERIC_TITLE_RE.match(ch_title):
            generic_chapters.append(ch_title)
        for sec in ch.get("sections", []) or []:
            total_sections += 1
            sec_title = _normalize_text(sec.get("t"))
            sec_prompt = _normalize_text(sec.get("pr"))
            section_titles.append(sec_title)
            sec_score = _title_specificity_score(sec_title, topic_terms)
            section_scores.append(sec_score)
            if GENERIC_TITLE_RE.match(sec_title):
                generic_sections.append(sec_title)
            pq = _prompt_quality_score(sec_prompt, sec_title)
            prompt_scores.append(pq["score"])
            prompt_flags.update(pq["flags"])
            if "empty_prompt" in pq["flags"]:
                empty_prompts += 1

    outline_audit = outline_profile.get("outline_audit", {}) or {}
    evidence_map = outline_profile.get("evidence_map", []) or []
    fallback_triggered = not bool(outline_audit.get("ok", True))
    missing_pr_audit = "missing_section_directives" in (outline_audit.get("issues") or [])

    authority = _source_authority_stats(state)

    discovery = {
        "topic_profile_completeness": completeness,
        "canonical_terms_count": len(topic_profile.get("canonical_terms", []) or []),
        "must_cover_count": len(topic_profile.get("must_cover", []) or []),
        "out_of_scope_count": len(topic_profile.get("out_of_scope", []) or []),
        "initial_queries_count": len(topic_profile.get("initial_queries", []) or []),
        "canonical_papers_count": len(topic_profile.get("canonical_papers", []) or []),
        "seed_axes_count": len(topic_profile.get("seed_queries_by_axis", {}) or {}),
    }

    outline = {
        "n_chapters": len(chapters),
        "n_sections": total_sections,
        "chapter_specificity_avg": round(sum(chapter_scores) / len(chapter_scores), 4) if chapter_scores else 0.0,
        "section_specificity_avg": round(sum(section_scores) / len(section_scores), 4) if section_scores else 0.0,
        "prompt_quality_avg": round(sum(prompt_scores) / len(prompt_scores), 4) if prompt_scores else 0.0,
        "generic_chapter_rate": round(len(generic_chapters) / len(chapter_titles), 4) if chapter_titles else 0.0,
        "generic_section_rate": round(len(generic_sections) / len(section_titles), 4) if section_titles else 0.0,
        "empty_prompt_rate": round(empty_prompts / total_sections, 4) if total_sections else 0.0,
        "duplicate_section_titles": len(section_titles) - len(set(section_titles)),
        "fallback_triggered": fallback_triggered,
        "audit_ok": bool(outline_audit.get("ok", False)),
        "audit_issues": outline_audit.get("issues", []) or [],
        "missing_prompt_issue_flag": missing_pr_audit,
        "evidence_buckets": len(evidence_map),
        "generic_chapters": generic_chapters[:10],
        "generic_sections": generic_sections[:12],
        "prompt_flags": dict(prompt_flags),
    }

    scorecard = {
        "topic_profile_complete_pass": completeness["ratio"] >= 0.90,
        "chapter_specificity_pass": outline["chapter_specificity_avg"] >= 0.80,
        "section_specificity_pass": outline["section_specificity_avg"] >= 0.80,
        "prompt_quality_pass": outline["prompt_quality_avg"] >= 0.75,
        "generic_title_rate_pass": outline["generic_section_rate"] <= 0.10 and outline["generic_chapter_rate"] <= 0.10,
        "empty_prompt_rate_pass": outline["empty_prompt_rate"] == 0.0,
        "fallback_rate_pass": not outline["fallback_triggered"],
        "primary_source_mix_pass": authority["primary_source_pct"] >= 0.30 if authority["total_sources"] else False,
    }

    # R0 trigger detection
    r0_triggers = []
    # Trigger 1: final chapter titles == first section titles (destructive overwrite)
    for ch in chapters:
        ch_t = _normalize_text(ch.get("t"))
        first_sec_t = _normalize_text(ch.get("sections", [{}])[0].get("t") if ch.get("sections") else "")
        if ch_t and first_sec_t and ch_t.lower() == first_sec_t.lower():
            r0_triggers.append("chapter_equals_section_title")
            break
    # Trigger 2: duplicate sections but raw had none (dedup corruption)
    raw_has_dups = outline_profile.get("_raw", "") and (
        outline.get("duplicate_section_titles", 0) > 0
    )
    if raw_has_dups:
        r0_triggers.append("duplicate_sections_post_processing")
    # Trigger 3: coverage notes all empty (P3c data loss)
    coverage_notes = [_normalize_text(ch.get("coverage_note")) for ch in chapters]
    if coverage_notes and all(not n for n in coverage_notes):
        r0_triggers.append("all_coverage_notes_empty")
    overall_pass = all(scorecard.values())

    return {
        "run": run_dir.name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "topic_profile": str(run_dir / "topic_profile.json"),
            "outline_profile": str(run_dir / "outline_profile.json"),
            "state": str(run_dir / "state.json") if (run_dir / "state.json").exists() else "",
        },
        "discovery": discovery,
        "outline": outline,
        "source_authority": authority,
        "scorecard": scorecard,
        "r0_triggers": r0_triggers,
        "overall_pass": overall_pass,
    }


def format_markdown(report: dict) -> str:
    d = report["discovery"]
    o = report["outline"]
    a = report["source_authority"]
    s = report["scorecard"]
    lines = [
        f"# Discovery Eval -- {report['run']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Overall pass: **{report['overall_pass']}**",
        f"- TopicProfile completeness: **{d['topic_profile_completeness']['ratio']:.2%}**",
        f"- Chapter specificity avg: **{o['chapter_specificity_avg']:.2f}**",
        f"- Section specificity avg: **{o['section_specificity_avg']:.2f}**",
        f"- Prompt quality avg: **{o['prompt_quality_avg']:.2f}**",
        f"- Primary source pct: **{a['primary_source_pct']:.2%}**",
        f"- Fallback triggered: **{o['fallback_triggered']}**",
        "",
        "## R0 Pipeline Corruption Triggers",
        "",
        f"- Triggers detected: **{len(report.get('r0_triggers', []))}**",
    ]
    for t in report.get("r0_triggers", []):
        lines.append(f"  - `{t}`: **pipeline corruption** (not model quality)")
    lines.extend([
        "",
        "## Scorecard",
        "",
    ])
    for k, v in s.items():
        lines.append(f"- {k}: **{v}**")
    lines.extend([
        "",
        "## Outline issues",
        "",
        f"- Audit issues: {', '.join(o['audit_issues']) if o['audit_issues'] else '(none)'}",
        f"- Generic chapter rate: {o['generic_chapter_rate']:.2%}",
        f"- Generic section rate: {o['generic_section_rate']:.2%}",
        f"- Empty prompt rate: {o['empty_prompt_rate']:.2%}",
        f"- Duplicate section titles: {o['duplicate_section_titles']}",
        "",
        "## Source authority",
        "",
        f"- Total sources: {a['total_sources']}",
        f"- Primary sources: {a['primary_sources']}",
        f"- Secondary sources: {a['secondary_sources']}",
        f"- Top hosts: {a['top_hosts']}",
    ])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Discovery/Outline artifacts for a v3 run")
    ap.add_argument("--run", required=True, help="Run folder name under output/runs/")
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--save", action="store_true", help="Save report beside the run artifacts")
    args = ap.parse_args()

    run_dir = ROOT / "output/runs" / args.run
    if not run_dir.exists():
        print(f"[ERR] Run not found: {run_dir}", file=sys.stderr)
        return 2
    for required in ("topic_profile.json", "outline_profile.json"):
        if not (run_dir / required).exists():
            print(f"[ERR] Missing required artifact: {required}", file=sys.stderr)
            return 2

    report = evaluate_run(run_dir)
    if args.format == "markdown":
        output = format_markdown(report)
    else:
        output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)

    if args.save:
        suffix = "md" if args.format == "markdown" else "json"
        out_path = run_dir / f"discovery_eval.{suffix}"
        out_path.write_text(output, encoding="utf-8")
        print(f"\n[SAVED] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
