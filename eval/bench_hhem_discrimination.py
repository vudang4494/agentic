"""Rank5 benchmark: is the HHEM grounding scorer actually DISCRIMINATING, or degenerate?

Explicit, reproducible evidence for the Gate-D-is-dead hypothesis. Calls the REAL pipeline
path (faithfulness._get_hhem / grounding_score) -- no reimplementation -- against a labeled
battery of (premise, hypothesis) pairs whose truth is known a priori:

  ENTAILED    -- hypothesis IS supported by premise   -> a working scorer must score HIGH
  CONTRADICTED-- hypothesis CONTRADICTS premise        -> must score LOW
  UNRELATED   -- hypothesis off-topic vs premise       -> must score LOW

A working HHEM separates ENTAILED from {CONTRADICTED, UNRELATED}. A degenerate one returns a
~constant -> zero spread -> Gate D (grounding >= 0.70) is a NO-OP. The numbers here decide the
Rank5 fix (pin transformers vs repair load path) and set the startup discrimination-assertion
threshold.

Run:  python3 eval/bench_hhem_discrimination.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> repo root (research/ importable)

from research import faithfulness as F

# (premise, hypothesis, label) -- 8 premises x 3 = 24 labeled pairs.
BATTERY = [
    # premise, entailed-hyp, contradicted-hyp, unrelated-hyp
    ("The Transformer architecture uses self-attention to process all tokens in parallel.",
     "Transformers use self-attention.",
     "Transformers process tokens strictly one at a time, never in parallel.",
     "Bananas are an excellent source of potassium."),
    ("Attention Is All You Need was published by Vaswani et al. in 2017.",
     "The paper introducing the Transformer appeared in 2017.",
     "Attention Is All You Need was published in 1995.",
     "The Eiffel Tower is located in Paris, France."),
    ("BERT is a bidirectional encoder pretrained with masked language modeling.",
     "BERT uses masked language modeling during pretraining.",
     "BERT is a left-to-right autoregressive decoder with no masking.",
     "Photosynthesis converts sunlight into chemical energy in plants."),
    ("Layer normalization normalizes activations across the feature dimension within each sample.",
     "Layer norm normalizes over features for each individual sample.",
     "Layer normalization normalizes across the batch dimension like batch norm.",
     "The capital of Japan is Tokyo."),
    ("Dropout randomly zeroes a fraction of activations during training to reduce overfitting.",
     "Dropout zeroes some activations at training time to fight overfitting.",
     "Dropout multiplies every activation by two during inference.",
     "Whales are the largest mammals in the ocean."),
    ("Adam combines momentum and per-parameter adaptive learning rates.",
     "Adam uses adaptive per-parameter learning rates together with momentum.",
     "Adam is a fixed global learning rate with no momentum whatsoever.",
     "Mount Everest is the tallest mountain above sea level."),
    ("RLHF fine-tunes a policy against a reward model trained on human preference comparisons.",
     "RLHF optimizes a policy using a reward model learned from human preferences.",
     "RLHF requires no human feedback and uses no reward model at all.",
     "Coffee is brewed from roasted coffee beans."),
    ("The softmax function turns a vector of logits into a probability distribution that sums to one.",
     "Softmax maps logits to probabilities summing to one.",
     "Softmax returns negative values that sum to minus one.",
     "The Pacific is the largest ocean on Earth."),
]


def _stats(xs):
    n = len(xs)
    mean = sum(xs) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in xs) / n if n else 0.0
    return mean, var ** 0.5, min(xs), max(xs)


def main():
    import torch, transformers
    print("=" * 78)
    print("Rank5 HHEM discrimination benchmark")
    print(f"transformers={transformers.__version__}  torch={torch.__version__}  "
          f"model={F.HHEM_MODEL}  HHEM_SUPPORT={F.HHEM_SUPPORT}")
    print("=" * 78)

    hhem = F._get_hhem()

    pairs, labels, meta = [], [], []
    for premise, ent, con, unr in BATTERY:
        for hyp, lab in ((ent, "ENTAILED"), (con, "CONTRADICTED"), (unr, "UNRELATED")):
            pairs.append((premise, hyp))
            labels.append(lab)
            meta.append((premise, hyp))

    scores = [float(s) for s in hhem.predict(pairs)]

    print("\nper-pair scores:")
    for (p, h), lab, sc in zip(meta, labels, scores):
        print(f"  {sc:.6f}  [{lab:12}]  {h[:58]}")

    uniq = sorted(set(round(s, 6) for s in scores))
    by = {L: [sc for sc, lab in zip(scores, labels) if lab == L]
          for L in ("ENTAILED", "CONTRADICTED", "UNRELATED")}

    print("\n" + "-" * 78)
    print(f"distinct score values: {len(uniq)}  ->  {uniq if len(uniq) <= 6 else uniq[:6]}")
    all_mean, all_sd, all_lo, all_hi = _stats(scores)
    print(f"all pairs   : mean={all_mean:.6f} sd={all_sd:.6f} min={all_lo:.6f} max={all_hi:.6f}"
          f"  spread={all_hi - all_lo:.6f}")
    for L in ("ENTAILED", "CONTRADICTED", "UNRELATED"):
        m, sd, lo, hi = _stats(by[L])
        print(f"{L:12}: mean={m:.6f} sd={sd:.6f} min={lo:.6f} max={hi:.6f}")

    # Pairwise discrimination on the SAME premise: does ENTAILED outscore its contradiction/unrelated?
    n_pair = len(BATTERY)
    ent = by["ENTAILED"]; con = by["CONTRADICTED"]; unr = by["UNRELATED"]
    win_con = sum(1 for e, c in zip(ent, con) if e > c)
    win_unr = sum(1 for e, u in zip(ent, unr) if e > u)
    tie_con = sum(1 for e, c in zip(ent, con) if e == c)
    print("-" * 78)
    print(f"ENTAILED > its CONTRADICTED (same premise): {win_con}/{n_pair}   (ties: {tie_con})")
    print(f"ENTAILED > its UNRELATED   (same premise): {win_unr}/{n_pair}")
    sep = by_sep = _stats(ent)[0] - _stats(con + unr)[0]
    print(f"separation mean(ENTAILED) - mean(CONTRA+UNREL) = {sep:+.6f}")

    # Now route through the REAL gate function to show Gate D behavior end-to-end.
    print("-" * 78)
    mixed_claims = [BATTERY[i][2] for i in range(4)]  # 4 CONTRADICTED claims (all false)
    src = [{"excerpt": BATTERY[i][0]} for i in range(4)]
    g = F.grounding_score(mixed_claims, src)
    print(f"grounding_score() on 4 FALSE claims vs their premises: grounding={g['grounding']}  "
          f"supported={g['n_supported']}/{g['n_claims']}  (a working gate would NOT support false claims)")

    print("=" * 78)
    degenerate = len(uniq) == 1 or all_sd < 1e-4
    if degenerate:
        print("VERDICT: DEGENERATE -- scorer returns a (near-)constant; spread ~0; Gate D is a NO-OP.")
        print(f"         constant value = {uniq[0] if len(uniq)==1 else f'~{all_mean:.6f}'};"
              f" >= HHEM_SUPPORT({F.HHEM_SUPPORT}) so every claim 'supported' -> grounding pins ~1.0.")
        print("         Rank5 needed: repair load path / pin transformers, + startup discrimination assert.")
    else:
        acc = (win_con + win_unr) / (2 * n_pair)
        print(f"VERDICT: DISCRIMINATING -- pairwise accuracy {acc:.2%}, separation {sep:+.4f}.")
        print(f"         Suggested startup assertion: mean(ENTAILED) - mean(false) must be >= "
              f"{max(0.05, sep/2):.3f} (half observed); fail loudly otherwise.")
    print("=" * 78)
    return 0 if True else 1


if __name__ == "__main__":
    sys.exit(main())
