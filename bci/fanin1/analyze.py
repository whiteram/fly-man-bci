"""bci/fanin1 analysis: density vs learning visibility + stability.

Per arm, per trial: pool size, w_all / w_trained (KCg-m = the A
odor's KC subtype), grown_frac (|w-w0| > 0.005), mod integral over
the train window (the in-circuit RW decrement), A-probe rms
(4.2-4.7 s), and baseline/max scalp rms (stability -- density
changes can latch, the mb-mod DPM lesson).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
W0 = 0.03
TRAIN = (500, 2500)
PROBE = (4200, 4700)


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for arm, a in meta["arms"].items():
        rows = []
        for k in range(meta["n_trials"]):
            tag = f"{arm}_t{k}"
            st = OUT / "states" / f"{tag}.npz"
            if not st.exists():
                continue
            d = np.load(st)
            w = d["w_KCM"]
            pt = d["pre_type"]
            rec = {"pool": int(w.size),
                   "w_all": float(w.mean()),
                   "w_A": float(w[pt == "KCg-m"].mean()),
                   "grown_frac": float((np.abs(w - W0) > 0.005).mean())}
            dt_ = OUT / f"{tag}_dan.npy"
            if dt_.exists():
                tr = np.load(dt_)
                inwin = ((tr[:, 0] >= TRAIN[0])
                         & (tr[:, 0] < TRAIN[1]))
                rec["mod_ms"] = float(tr[inwin, 2].sum() * 0.5)
            eg = OUT / f"{tag}.npy"
            if eg.exists():
                e = np.load(eg)
                e = e.mean(axis=1) if e.ndim == 2 else e
                base = e[int(3000):int(4000)]
                rec["probe_uv"] = float(np.sqrt(
                    ((e[PROBE[0]:PROBE[1]] - base.mean()) ** 2).mean())
                    * 1e6 * 1.7)
                rec["base_uv"] = float(np.sqrt(
                    (e[:500] ** 2).mean()) * 1e6 * 1.7)
                rec["max_uv"] = float(np.abs(e).max() * 1e6 * 1.7)
            rows.append(rec)
        res[arm] = rows
        print(f"\n-- {arm} (lr {a['lr']}) --  pool  w_all  w_A  "
              f"grown%  mod_ms  probe_uv  base/max_uv")
        for k, r_ in enumerate(rows):
            print(f"   t{k}  {r_['pool']:6d}  {r_['w_all']:.4f}  "
                  f"{r_['w_A']:.4f}  {r_['grown_frac']:6.1%}  "
                  f"{r_.get('mod_ms', float('nan')):7.1f}  "
                  f"{r_.get('probe_uv', float('nan')):7.3f}  "
                  f"{r_.get('base_uv', float('nan')):.2f}/"
                  f"{r_.get('max_uv', float('nan')):.1f}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[fanin1] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
