"""bci/dstate10 analysis: duration -> lock map at consolidating dose."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (10000, 29000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145)}


def classify(alpn, std):
    return min(REFS, key=lambda k: (abs(alpn - REFS[k][0]) / 20.0
                                    + abs(std - REFS[k][1]) / 0.05))


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== dstate10: duration -> state (150 pA, window [10,29] s) ==")
    print("   refs: lock ALPN 20 std 0.0145 | dark 0 / 0.005")
    print(f"   {'arm':>6}{'seed':>6}{'ALPN':>9}{'MBON':>8}"
          f"{'std uV':>9}{'mean uV':>9}  state")
    for tag, spec in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        a, b = WIN
        alpn = float(z["ALPN"][a:b].mean())
        mbon = float(z["MBON"][a:b].mean())
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        seg = e[a:b]
        std = float(seg.std()) * 1e6 * 1.7
        mean = float(seg.mean()) * 1e6 * 1.7
        st = classify(alpn, std)
        res[tag] = {"chem": spec["chem"], "seed": spec["seed"],
                    "alpn": round(alpn, 2), "mbon": round(mbon, 3),
                    "std": round(std, 4), "mean": round(mean, 3),
                    "state": st}
        print(f"   {tag:>6}{spec['seed']:>6}{alpn:9.2f}{mbon:8.2f}"
              f"{std:9.4f}{mean:9.3f}  {st}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate10] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
