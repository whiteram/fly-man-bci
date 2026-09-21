"""dstate3 analysis: lock-window width + AL-local engine test.

Part 1: the fine CEN_C scan -> ALPN/ALON/ALLN/MBON plateau rates vs
absolute gain (window width + thresholds).  Part 2: the split arms
-> does killing ALL (0.0001) kill the lock, does boosting it (0.004)
produce the broad attractor at the working point?
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
KEYS = ("ALPN", "ALLN", "ALON", "ALIN", "MBON")


def row(tag):
    z = np.load(OUT / f"{tag}_pop.npz")
    e = np.load(OUT / f"{tag}_scalp.npy")
    e = e.mean(axis=1) if e.ndim == 2 else e
    uv = np.abs(e) * 1e6 * 1.7
    rec = {k: round(float(z[k][10000:60000].mean()), 2) for k in KEYS}
    rec["mean_abs"] = round(float(uv[10000:60000].mean()), 4)
    rec["std"] = round(float(uv[10000:60000].std()), 4)
    return rec


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== Part 1: fine CEN_C scan ==")
    print(f"{'gain':>8s} " + " ".join(f"{k:>7s}" for k in KEYS)
          + "  |mean|   std")
    for tag, g in sorted(meta["scan_abs"].items(),
                         key=lambda kv: kv[1]):
        t = f"da1_dc60_{tag}_s{meta['seed']}"
        if not (OUT / f"{t}_pop.npz").exists():
            continue
        rec = row(t)
        res[f"scan_{g:g}"] = rec
        print(f"{g:8.4f} " + " ".join(f"{rec[k]:7.2f}" for k in KEYS)
              + f"  {rec['mean_abs']:7.4f} {rec['std']:6.4f}")
    print("\n== Part 2: AL-local split (CEN_C at 0.002) ==")
    print(f"{'arm':>10s} " + " ".join(f"{k:>7s}" for k in KEYS)
          + "  |mean|   std")
    for tag, g in sorted(meta["splits"].items(), key=lambda kv: kv[1]):
        t = f"da1_dc60_{tag}_s{meta['seed']}"
        if not (OUT / f"{t}_pop.npz").exists():
            continue
        rec = row(t)
        res[f"split_{g:g}"] = rec
        print(f"{g:10.4f} " + " ".join(f"{rec[k]:7.2f}" for k in KEYS)
              + f"  {rec['mean_abs']:7.4f} {rec['std']:6.4f}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[dstate3] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
