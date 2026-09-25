"""bci/dstate8 analysis: lock survival + window shift under AL
subsampling.  Per arm: ALPN rate, scalp std/mean in [10,29] s,
classified vs the full-population refs.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (10000, 29000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145),
        "high_lock": (100.0, 0.017)}


def classify(alpn, std):
    best, bd = "other", 1e9
    for name, (ra, rs) in REFS.items():
        d = abs(alpn - ra) / 20.0 + abs(std - rs) / 0.05
        if d < bd:
            best, bd = name, d
    return best


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== dstate8: AL subsampling vs the lock (window [10,29] s) ==")
    print("   refs full-pop: lock ALPN 20 std 0.0145 | "
          "window abs 0.0016-0.0023")
    for tag, spec in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        a, b = WIN
        alpn = float(z["ALPN"][a:b].mean())
        alon = float(z["ALON"][a:b].mean())
        alln = float(z["ALLN"][a:b].mean())
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        seg = e[a:b]
        std = float(seg.std()) * 1e6 * 1.7
        mean = float(seg.mean()) * 1e6 * 1.7
        st = classify(alpn, std)
        res[tag] = {"F": spec["F"], "gain": spec["gain"],
                    "alpn": round(alpn, 2), "alon": round(alon, 2),
                    "alln": round(alln, 2), "std": round(std, 4),
                    "mean": round(mean, 3), "state": st}
        print(f"   {tag:>12} (F={spec['F']:g}, g={spec['gain']}): "
              f"ALPN {alpn:7.2f}  ALON {alon:7.2f}  ALLN {alln:7.2f}  "
              f"std {std:.4f}  mean {mean:7.3f}  -> {st}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate8] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
