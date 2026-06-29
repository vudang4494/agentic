"""Unit/acceptance test for research/decite.py (intra-book citation cleaner).

Asserts the three invariants that make the cleaner SAFE to run at assemble time:
  1. Intra-book name-drops (a section's title cited as an external paper) are removed.
  2. Real EXTERNAL citations ([N] markers + real author/paper names) are PRESERVED.
  3. No grammar regression: removal never leaves a lowercase sentence-start or stray comma.

Run: python3 eval/test_decite.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.decite import clean_intrabook_citations

TITLES = [
    "Data Collection for On-Policy Trajectories in LLMs",
    "The KL Divergence Constraint: Preventing Model Drift",
    "High-Throughput Pipelines: System Design for Labeling Scale",
    "Implementation Details: PPO in Deep Learning Framework Pipelines",
]

CASES = [
    # (input, must_NOT_contain, must_contain)
    ("As noted in *Data Collection for On-Policy Trajectories in LLMs*, the quality of advantages depends on rollout [2].",
     "Data Collection for On-Policy Trajectories", "quality of advantages depends on rollout [2]"),
    ("In 'The KL Divergence Constraint: Preventing Model Drift', we detailed how the KL term constrains policy.",
     "KL Divergence Constraint: Preventing Model Drift", "we detailed how the KL term"),
    ("Similarly, 'High-Throughput Pipelines: System Design for Labeling Scale' established that without infrastructure, noise grows.",
     "High-Throughput Pipelines", "without infrastructure, noise grows"),
    ("As formalized by Jaques et al. [7], the objective maximizes reward while penalizing KL drift.",
     None, "Jaques et al. [7]"),  # EXTERNAL ref must survive untouched
    ("The transformer architecture (Vaswani et al., 2017) introduced self-attention [1].",
     None, "Vaswani et al., 2017"),  # external paper name must survive
]


def main():
    fails = 0
    for i, (inp, banned, keep) in enumerate(CASES, 1):
        out, n = clean_intrabook_citations(inp, TITLES)
        ok = True
        if banned and banned.lower() in out.lower():
            ok = False; print(f"[{i}] FAIL: banned intra-book title still present -> {out!r}")
        if keep and keep.lower() not in out.lower():
            ok = False; print(f"[{i}] FAIL: dropped content that must survive ({keep!r}) -> {out!r}")
        # grammar: no lowercase right after a sentence period, no ' ,' / ',,'
        import re
        if re.search(r"[.!?]\s+[a-z]", out) and "e.g." not in out and "i.e." not in out and "vs." not in out:
            ok = False; print(f"[{i}] FAIL: lowercase sentence-start after cleaning -> {out!r}")
        if " ," in out or ",," in out:
            ok = False; print(f"[{i}] FAIL: stray comma artifact -> {out!r}")
        if banned is None and n != 0:
            ok = False; print(f"[{i}] FAIL: removed something from an external-only sentence (n={n}) -> {out!r}")
        if ok:
            print(f"[{i}] PASS (removed={n})")
        else:
            fails += 1

    # Real-book smoke: run over book_900 state if present, assert >=90% reduction + word loss <6%
    sp = Path(__file__).resolve().parent.parent / "output/runs/book_900/state.json"
    if sp.exists():
        import json, re
        st = json.load(open(sp)); secs = st["sections"]
        titles = [v.get("title", "") for v in secs.values() if v.get("title")]
        norm = {re.sub(r"\s+", " ", t.strip()).lower() for t in titles if len(t) > 18}
        def nd(getter):
            n = 0
            for v in secs.values():
                if v.get("quality") == "BLOCKED":
                    continue
                body = getter(v)
                for m in re.finditer(r"([*'\"])([^*'\"\n]{18,180}?)\1", body or ""):
                    if re.sub(r"\s+", " ", m.group(2).strip().strip("[]()")).lower() in norm:
                        n += 1
            return n
        before = nd(lambda v: v.get("content"))
        cleaned = {k: clean_intrabook_citations(v.get("content") or "", titles)[0] for k, v in secs.items() if v.get("quality") != "BLOCKED"}
        after = sum(len([m for m in re.finditer(r"([*'\"])([^*'\"\n]{18,180}?)\1", c)
                         if re.sub(r"\s+", " ", m.group(2).strip().strip("[]()")).lower() in norm]) for c in cleaned.values())
        red = 100 * (before - after) / max(before, 1)
        wb = sum(len((v.get("content") or "").split()) for v in secs.values() if v.get("quality") != "BLOCKED")
        wa = sum(len(c.split()) for c in cleaned.values())
        wl = 100 * (wb - wa) / max(wb, 1)
        print(f"\n[book_900] name-drops {before} -> {after} ({red:.1f}% reduction), word-loss {wl:.1f}%")
        if red < 90: fails += 1; print("  FAIL: reduction < 90%")
        if wl > 6: fails += 1; print("  FAIL: word-loss > 6% (too aggressive)")

    print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
