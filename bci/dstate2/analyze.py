"""dstate2 analysis: ALPN plateau rate vs CEN_C gain.

Per arm: dark ALPN (0-300 ms), plateau ALPN (10-60 s), every other
population's plateau rate (is anything else awake at low gains?),
scalp mean/std.  Plotted against the x1.0 reference from dstate1:
a monotone rate-vs-gain curve = the lock is synapse-driven through
CEN_C; flat = ALPN-intrinsic.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print(f"{'arm':>6s} {'gain':>8s} {'ALPN_lock':>9s} {'ALPN_dark':>9s} "
          + " ".join(f"{g:>6s}" for g in meta["groups"].split(",")
                     if g != "ALPN")
          + "   mean_uv  std_uv")
    for arm, mult in meta["arms"].items():
        tag = f"da1_dc60_{arm}_s{meta['seed']}"
        f = OUT / f"{tag}_pop.npz"
        if not f.exists():
            continue
        z = np.load(f)
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        uv = np.abs(e) * 1e6 * 1.7
        row = {"alpn_lock": round(float(z["ALPN"][10000:60000].mean()), 2),
               "alpn_dark": round(float(z["ALPN"][100:300].mean()), 2)}
        for g in meta["groups"].split(","):
            if g != "ALPN":
                row[g] = round(float(z[g][10000:60000].mean()), 2)
        row["mean_uv"] = round(float(uv[10000:60000].mean()), 4)
        row["std_uv"] = round(float(uv[10000:60000].std()), 4)
        res[arm] = row
        print(f"{arm:>6s} {0.002*mult:8.5f} {row['alpn_lock']:9.2f} "
              f"{row['alpn_dark']:9.2f} "
              + " ".join(f"{row[g]:6.2f}" for g in meta["groups"].split(",")
                         if g != "ALPN")
              + f"  {row['mean_uv']:+8.4f} {row['std_uv']:7.4f}")
    ref = OUT.parent / "dstate1" / "outputs" / "summary.json"
    if ref.exists():
        s1 = json.loads(ref.read_text(encoding="utf-8"))
        b1 = s1["da1_dc60_s62"]["blocks"]["10-20s"]
        print(f"   x1.0  0.00200  {b1['ALPN']:9.2f}  (dstate1 reference; "
              f"MBON {b1['MBON']}, std {b1['std_uv']})")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[dstate2] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
