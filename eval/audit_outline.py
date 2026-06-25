#!/usr/bin/env python3
"""
Outline + content audit for an Agentic-Deep-Research run.

Loads a `state.json` and answers three questions the eval metrics don't:

  1. **Title duplication (near-dup, not just exact)**: cluster section titles by
     bge-m3 cosine similarity. Anything >= 0.85 is flagged. Catches the case
     where planner emits "Tokenization Strategies" in Ch3 and "Tokenizer
     Methods" in Ch9 -- exact-match dedupe misses these but the writer will
     duplicate content.

  2. **Topic drift**: cosine of each section title to the input topic string.
     Titles below `min_topic_cosine` are flagged as off-mission. Catches the
     case in the 2026-05-27 transformer eval where planner emitted
     "Time Series and Financial Applications" for a "Transformer" topic.

  3. **Content duplication**: cosine between full section content (when
     available). Anything >= 0.80 is a content-level duplicate -- the writer
     produced the same prose twice despite different titles.

Pure-Python except for bge-m3 embeddings via the running Ollama. Safe to run
against a state.json that is still being written (we read once and analyze).

Usage:
  python3 eval/audit_outline.py --state output/runs/bookv7/state.json --topic "Large Language Models"
  python3 eval/audit_outline.py --state output/runs/bookv7/state.json \
      --topic "Large Language Models" --out eval/reports/audit_bookv7.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from research.embeddings import embed, cosine  # noqa: E402


def _section_titles(state: dict) -> list[tuple[str, str]]:
    """Return [(key, title)] sorted by chapter.section order."""
    passes = state.get("passes", {})
    items = []
    for k, v in passes.items():
        title = v.get("title") or v.get("pp_t") or ""
        items.append((k, title))
    items.sort(key=lambda kv: tuple(int(p) for p in kv[0].split(".")))
    return items


def _section_contents(state: dict) -> list[tuple[str, str]]:
    passes = state.get("passes", {})
    items = []
    for k, v in passes.items():
        content = v.get("content", "")
        if content:
            items.append((k, content))
    items.sort(key=lambda kv: tuple(int(p) for p in kv[0].split(".")))
    return items


def title_cluster_dups(titles: list[tuple[str, str]],
                       threshold: float = 0.85,
                       embed_model: str = "bge-m3:latest") -> list[dict]:
    """Return clusters of titles whose pairwise cosine >= threshold."""
    if len(titles) < 2:
        return []
    keys = [k for k, _ in titles]
    texts = [t for _, t in titles]
    vecs = embed(texts, model=embed_model)
    if len(vecs) != len(texts):
        print("[audit] WARN: embedding failed -- skipping title cluster check",
              file=sys.stderr)
        return []

    clusters: list[set[int]] = []
    for i in range(len(vecs)):
        merged = False
        for cl in clusters:
            if any(cosine(vecs[i], vecs[j]) >= threshold for j in cl):
                cl.add(i)
                merged = True
                break
        if not merged:
            clusters.append({i})
    out = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        members = sorted(cl)
        max_pair_cos = max(
            cosine(vecs[a], vecs[b])
            for a in members for b in members if a < b
        )
        out.append({
            "members": [{"key": keys[i], "title": texts[i]} for i in members],
            "max_pair_cosine": round(max_pair_cos, 3),
        })
    return out


def topic_drift(titles: list[tuple[str, str]],
                topic: str,
                min_topic_cosine: float = 0.45,
                embed_model: str = "bge-m3:latest") -> list[dict]:
    """Flag titles whose cosine to the topic string is below threshold."""
    if not titles:
        return []
    texts = [topic] + [t for _, t in titles]
    vecs = embed(texts, model=embed_model)
    if len(vecs) != len(texts):
        return []
    topic_v = vecs[0]
    flagged = []
    for (k, t), v in zip(titles, vecs[1:]):
        c = cosine(topic_v, v)
        if c < min_topic_cosine:
            flagged.append({"key": k, "title": t, "cosine": round(c, 3)})
    return sorted(flagged, key=lambda x: x["cosine"])


def content_dups(contents: list[tuple[str, str]],
                 threshold: float = 0.80,
                 sample_chars: int = 1500,
                 embed_model: str = "bge-m3:latest") -> list[dict]:
    """Pairwise content cosine. Embed the FIRST `sample_chars` of each section
    (bge-m3 has a hard cap; full sections often exceed it) -- cheap proxy
    that still catches identical openings and rehashes."""
    if len(contents) < 2:
        return []
    keys = [k for k, _ in contents]
    snippets = [c[:sample_chars] for _, c in contents]
    vecs = embed(snippets, model=embed_model)
    if len(vecs) != len(contents):
        print("[audit] WARN: content embedding failed", file=sys.stderr)
        return []
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            c = cosine(vecs[i], vecs[j])
            if c >= threshold:
                pairs.append({"a": keys[i], "b": keys[j], "cosine": round(c, 3)})
    return sorted(pairs, key=lambda p: -p["cosine"])


def _render_md(report: dict, path: Path) -> None:
    L: list[str] = []
    L.append(f"# Outline + Content Audit -- `{report['state_file']}`")
    L.append("")
    L.append(f"- Topic: `{report['topic']}`")
    L.append(f"- Sections scored: {report['n_sections']}")
    L.append("")

    L.append("## Topic drift (titles < 0.45 cosine to topic)")
    L.append("")
    if not report["topic_drift"]:
        L.append("None.")
    else:
        L.append("| Section | Title | Cosine |")
        L.append("|---|---|---|")
        for f in report["topic_drift"]:
            L.append(f"| `{f['key']}` | {f['title']} | {f['cosine']} |")
    L.append("")

    L.append("## Title near-duplicates (cosine >= 0.85)")
    L.append("")
    if not report["title_clusters"]:
        L.append("None.")
    else:
        for i, cl in enumerate(report["title_clusters"], 1):
            L.append(f"### Cluster {i} (max pair cosine {cl['max_pair_cosine']})")
            L.append("")
            for m in cl["members"]:
                L.append(f"- `{m['key']}` -- {m['title']}")
            L.append("")

    L.append("## Content duplicates (cosine >= 0.80 on first 1500 chars)")
    L.append("")
    if not report["content_dups"]:
        L.append("None.")
    else:
        L.append("| A | B | Cosine |")
        L.append("|---|---|---|")
        for p in report["content_dups"][:50]:
            L.append(f"| `{p['a']}` | `{p['b']}` | {p['cosine']} |")
    L.append("")

    L.append("## Verdict")
    L.append("")
    L.append(f"- Topic drift sections: **{len(report['topic_drift'])}**")
    L.append(f"- Title clusters: **{len(report['title_clusters'])}**")
    L.append(f"- Content duplicate pairs: **{len(report['content_dups'])}**")
    drift_pct = len(report["topic_drift"]) / max(1, report["n_sections"])
    if drift_pct > 0.20 or report["title_clusters"] or report["content_dups"]:
        L.append("")
        L.append("**NEEDS ATTENTION** -- outline or content has duplication / drift issues.")
    else:
        L.append("")
        L.append("OK -- no significant drift or duplication detected.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


def main() -> int:
    p = argparse.ArgumentParser(description="Outline + content audit for state.json")
    p.add_argument("--state", required=True, help="path to <out>.state.json")
    p.add_argument("--topic", required=True, help="the topic the run targeted")
    p.add_argument("--out", help="markdown report output path (default: stdout summary)")
    p.add_argument("--title-cluster-threshold", type=float, default=0.85)
    p.add_argument("--topic-drift-threshold", type=float, default=0.45)
    p.add_argument("--content-dup-threshold", type=float, default=0.80)
    args = p.parse_args()

    state_path = Path(args.state)
    state = json.loads(state_path.read_text())
    titles = _section_titles(state)
    contents = _section_contents(state)

    print(f"[audit] {len(titles)} sections in {state_path.name} -- analyzing...",
          file=sys.stderr)
    drift = topic_drift(titles, args.topic, args.topic_drift_threshold)
    print(f"[audit] topic-drift candidates: {len(drift)}", file=sys.stderr)
    clusters = title_cluster_dups(titles, args.title_cluster_threshold)
    print(f"[audit] title clusters: {len(clusters)}", file=sys.stderr)
    dups = content_dups(contents, args.content_dup_threshold)
    print(f"[audit] content duplicate pairs: {len(dups)}", file=sys.stderr)

    report = {
        "state_file": str(state_path),
        "topic": args.topic,
        "n_sections": len(titles),
        "topic_drift": drift,
        "title_clusters": clusters,
        "content_dups": dups,
    }

    out_path = Path(args.out) if args.out else None
    if out_path:
        _render_md(report, out_path)
        print(f"[audit] wrote {out_path}", file=sys.stderr)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    needs_attention = bool(drift) or bool(clusters) or bool(dups)
    return 1 if needs_attention else 0


if __name__ == "__main__":
    sys.exit(main())
