"""bci/arousal analysis: state decoding + odor detectability per state.

1. State decode: dark trials, quiet (0.3x) vs active (1.0x) OU noise,
   spatial-rms LOO + permutation.  HONEST framing: the manipulation is
   synthetic (noise scaling), so amplitude separation is expected --
   the number quantifies the floor difference, not a discovered state.
2. Odor detectability per state: net response to the 75 pA pulse
   relative to that state's dark floor -- the exp018 premise
   (signal-to-floor) made dynamic.

Usage:
    python bci/arousal/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (400, 1400)
BASE = (0, 280)


def rms(p, a, b):
    return float(np.sqrt((p[a:b] ** 2).mean()))


def loo_acc(X, y):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0) for c in (0, 1)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    runs = meta["runs"]
    phis = {}
    for state in meta["states"]:
        for stim in meta["stimuli"]:
            phis[(state, stim)] = [
                np.load(OUT / f"{state}_{stim}_r{r}.npy") * 1e6 * 1.7
                for r in range(runs)
                if (OUT / f"{state}_{stim}_r{r}.npy").exists()]
    lo, hi = meta["states"]
    print(f"== state decode (dark trials, {lo} vs {hi} base-scale) ==")
    X = np.array([np.sqrt((p[WIN[0]:WIN[1]] ** 2).mean(axis=0))
                  for state in meta["states"]
                  for p in phis[(state, "dark")]])
    y = np.array([0] * len(phis[(lo, "dark")])
                 + [1] * len(phis[(hi, "dark")]))
    acc = loo_acc(X, y)
    rng = np.random.default_rng(1)
    perm = [loo_acc(X, rng.permutation(y)) for _ in range(2000)]
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    floors = {s: float(np.mean([rms(q, *BASE)
                                for q in phis[(s, "dark")]]))
              for s in meta["states"]}
    print(f"dark floor rms: {floors[lo]:.4f} vs {floors[hi]:.4f} uV | "
          f"state LOO {acc:.0%} perm-p={p:.4f}")
    print("== odor detectability per state (75 pA pulse) ==")
    for state in meta["states"]:
        nets = [rms(q, WIN[0], WIN[1]) - rms(q, *BASE)
                for q in phis[(state, "odor")]]
        print(f"  {state:6s}: net odor response "
              f"{np.mean(nets):+.4f}+-{np.std(nets):.4f} uV "
              f"(per-trial baseline; floor {floors[state]:.4f})")


if __name__ == "__main__":
    main()
