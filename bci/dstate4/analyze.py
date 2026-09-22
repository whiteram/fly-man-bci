"""dstate4 analysis: the matching matrix -- state per (entry, gain,
STD) cell.  Rates (10-60 s) + |mean|/std + a state label from the
dstate-series landmarks (quiet lock: ALPN ~20, std ~0.015, MBON 0;
attractor: ALON/ALLN 300+, MBON >50; latch: MBON 300+; post-collapse
plateau: ALPN ~38, MBON 3-5, std ~0.18).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
KEYS = ("ALPN", "ALLN", "ALON", "ALIN", "MBON", "Kenyon_Cell")


def label(rec):
    if rec["ALPN"] < 2 and rec["MBON"] < 2:
        return "dark"
    if rec["MBON"] >= 200:
        return "latch"
    if rec["ALON"] >= 100 and rec["MBON"] >= 50:
        return "attractor"
    if 5 <= rec["MBON"] <= 50 and rec["ALPN"] >= 25:
        return "post-collapse plateau"
    if rec["MBON"] < 5:
        return "quiet lock"
    return "mixed"


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print(f"{'arm':>14s} " + " ".join(f"{k[:6]:>6s}" for k in KEYS)
          + "  |mean|   std  state")
    for arm in meta["arms"]:
        p = OUT / f"{arm}_pop.npz"
        if not p.exists():
            continue
        z = np.load(p)
        e = np.load(OUT / f"{arm}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        uv = np.abs(e) * 1e6 * 1.7
        rec = {k: round(float(z[k][10000:60000].mean()), 2) for k in KEYS}
        rec["mean_abs"] = round(float(uv[10000:60000].mean()), 4)
        rec["std"] = round(float(uv[10000:60000].std()), 4)
        rec["state"] = label(rec)
        res[arm] = rec
        print(f"{arm:>14s} " + " ".join(f"{rec[k]:6.1f}" for k in KEYS)
              + f"  {rec['mean_abs']:7.4f} {rec['std']:6.4f}  "
              f"{rec['state']}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[dstate4] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
