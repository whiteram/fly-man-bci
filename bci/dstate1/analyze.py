"""dstate1 analysis: DC persistence + in-state rate signature.

Per seed: the scalp mean/std in 10-s blocks over the 60 s trial
(dark block 0-300 ms as the reference) and the population rates in
matching blocks.  Verdict: does the odor-DC target state match the
sz plateau signature (mean -1.825, std 0.18, MBON 3-5 Hz, ALPN 38)?
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for seed in meta["seeds"]:
        tag = f"da1_dc60_s{seed}"
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        uv = e * 1e6 * 1.7
        z = np.load(OUT / f"{tag}_pop.npz")
        blocks = {}
        print(f"\n-- {tag} --  block    mean     std   "
              + "  ".join(f"{g:>6s}" for g in meta["groups"].split(",")))
        for a in range(0, 60000, 10000):
            b = min(a + 10000, 60000)
            w = uv[a:b]
            row = {"mean_uv": round(float(w.mean()), 4),
                   "std_uv": round(float(w.std()), 4)}
            for g in meta["groups"].split(","):
                row[g] = round(float(z[g][a:b].mean()), 2)
            blocks[f"{a // 1000}-{b // 1000}s"] = row
            print(f"  {a // 1000:3d}-{b // 1000:<3d}s {row['mean_uv']:+8.4f} "
                  f"{row['std_uv']:7.4f}  "
                  + "  ".join(f"{row[g]:6.1f}" for g in row if g != "mean_uv"
                              and g != "std_uv"))
        dark = uv[0:300]
        post = uv[10000:60000]
        rec = {"dark_mean": round(float(dark.mean()), 4),
               "dark_std": round(float(dark.std()), 4),
               "post_mean": round(float(post.mean()), 4),
               "post_std": round(float(post.std()), 4),
               "dc_step": round(float(post.mean() - dark.mean()), 4),
               "blocks": blocks}
        # signature match vs the sz plateau reference
        ref = meta["sz_plateau_reference"]
        rec["match_sz_plateau"] = {
            "mean_within_0.01": abs(rec["post_mean"]
                                    - ref["mean_uv"]) < 0.01,
            "std_within_0.02": abs(rec["post_std"]
                                   - ref["std_uv"]) < 0.02,
            "MBON_low": rec["blocks"]["10-20s"]["MBON"] < 10,
            "ALPN_elevated": rec["blocks"]["10-20s"]["ALPN"] > 20}
        res[tag] = rec
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[dstate1] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
