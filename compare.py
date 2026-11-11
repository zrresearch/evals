#!/usr/bin/env python3
"""Compare two runs item by item.

  compare.py <label A> <label B> [--boot 10000]

For each test both runs finished: mean score of each, the difference (A - B) with a 95% bootstrap interval
that resamples whole items, an exact sign test over items where the two differ, and median output tokens per
item (a model that writes twice as much for the same score costs twice the time). Items are matched by id;
epochs are averaged within an item first.
"""
import json, math, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(label, test):
    f = RESULTS / label / f"{test}.jsonl"
    if not f.exists():
        return None
    score, tok = defaultdict(list), defaultdict(list)
    for l in f.read_text().splitlines():
        r = json.loads(l)
        if r["score"] is not None and not r["error"]:
            score[str(r["id"])].append(r["score"])
            if r["output_tokens"]:
                tok[str(r["id"])].append(r["output_tokens"])
    return {k: st.mean(v) for k, v in score.items()}, {k: st.mean(v) for k, v in tok.items()}


def sign_test(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    args = sys.argv[1:]
    boot = int(args[args.index("--boot") + 1]) if "--boot" in args else 10000
    A, B = [x for x in args if not x.startswith("--") and not x.isdigit()][:2]
    tests = sorted({f.stem for f in (RESULTS / A).glob("*.jsonl")} & {f.stem for f in (RESULTS / B).glob("*.jsonl")})
    rng = random.Random(2026)
    print(f"{'test':18} {'items':>5} {A[:12]:>12} {B[:12]:>12} {'diff':>7} {'95% CI':>17} {'A>B':>4} {'B>A':>4} {'p':>6}  tokens A / B")
    out = {}
    for t in tests:
        (sa, ta), (sb, tb) = load(A, t), load(B, t)
        ids = sorted(set(sa) & set(sb))
        if not ids:
            continue
        d = [sa[i] - sb[i] for i in ids]
        means = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(boot))
        lo, hi = means[int(0.025 * boot)], means[int(0.975 * boot)]
        wa, wb = sum(x > 0 for x in d), sum(x < 0 for x in d)
        p = sign_test(wa, wb)
        tka = st.median([ta[i] for i in ids if i in ta]) if ta else None
        tkb = st.median([tb[i] for i in ids if i in tb]) if tb else None
        ma, mb = st.mean(sa[i] for i in ids), st.mean(sb[i] for i in ids)
        print(f"{t:18} {len(ids):5} {100 * ma:11.1f}% {100 * mb:11.1f}% {100 * st.mean(d):+6.1f} "
              f"({100 * lo:+6.1f} to {100 * hi:+5.1f}) {wa:4} {wb:4} {p:6.3f}  {tka or 0:,.0f} / {tkb or 0:,.0f}")
        out[t] = {"items": len(ids), "a": ma, "b": mb, "diff": st.mean(d), "ci95": [lo, hi], "a_better": wa,
                  "b_better": wb, "p_sign": p, "median_output_tokens": [tka, tkb]}
    (RESULTS / f"compare-{A}-vs-{B}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
