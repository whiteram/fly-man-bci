"""bci/dstate5 analysis: did the probe exit the lock?

Per arm x seed: pre [10,29] s vs post [31,50] s windows -- 1-s scalp
std, mean uV, class rates.  Classification against the dstate refs:
dark (ALPN ~0, std ~0.10), lock (ALPN ~20, std 0.0145), plateau
(ALPN ~38, std 0.18), broad attractor (ALPN 300+).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PRE, POST = (10000, 29000), (31000, 50000)
REFS = {"dark": (0.0, 0.100), "lock": (19.8, 0.0145),
        "plateau": (38.0, 0.18)}


def classify(alpn, std):
    best, bd = "attractor/other", 1e9
    for name, (ra, rs) in REFS.items():
        d = abs(alpn - ra) / 20.0 + abs(std - rs) / 0.05
        if d < bd:
            best, bd = name, d
    return best


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for arm in meta["arms"]:
        for seed in meta["seeds"]:
            sp = OUT / f"{arm}_s{seed}_scalp.npy"
            pp = OUT / f"{arm}_s{seed}_pop.npz"
            if not sp.exists():
                continue
            e = np.load(sp)
            e = e.mean(axis=1) if e.ndim == 2 else e
            row = {}
            for lab, (a, b) in (("pre", PRE), ("post", POST)):
                seg = e[a:b]
                alpn = float(np.load(pp)["ALPN"][a:b].mean()) \
                    if pp.exists() else np.nan
                row[lab] = {"std_uv": round(float(seg.std()) * 1e6 * 1.7, 4),
                            "mean_uv": round(float(seg.mean()) * 1e6 * 1.7, 4),
                            "alpn_hz": round(alpn, 2)}
                if pp.exists():
                    z = np.load(pp)
                    row[lab].update({k: round(float(z[k][a:b].mean()), 2)
                                     for k in ("Kenyon_Cell", "MBON", "DAN")
                                     if k in z.files})
            row["pre_state"] = classify(row["pre"]["alpn_hz"],
                                        row["pre"]["std_uv"])
            row["post_state"] = classify(row["post"]["alpn_hz"],
                                         row["post"]["std_uv"])
            res[f"{arm}_s{seed}"] = row
            print(f"\n-- {arm} s{seed} --  pre ({row['pre_state']}) -> "
                  f"post ({row['post_state']})")
            for lab in ("pre", "post"):
                r = row[lab]
                print(f"   {lab}: std {r['std_uv']:7.4f} uV  "
                      f"mean {r['mean_uv']:7.3f} uV  "
                      f"ALPN {r['alpn_hz']:7.2f}  MBON {r.get('MBON', 0):7.2f}"
                      f"  KC {r.get('Kenyon_Cell', 0):6.2f}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[dstate5] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
