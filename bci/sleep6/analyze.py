"""bci/sleep6 analysis: the resource curve d(t) against sleep5's R(t).

Per trial (t_end = delay moment): the std-gated CEN_C pool's final
per-edge d -- count and mean of ever-depleted edges (d < 0.9999),
the no-redepletion recovery prediction 1-exp(-(t-8)/tau), and the
matching R_mbon/R_scalp from sleep5's summary.  d recovers while R
caps -> plateau-state ceiling; d tracks R -> resource ceiling.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
S5 = ROOT = Path(__file__).resolve().parents[2] / "bci" / "sleep5" \
    / "outputs" / "summary.json"
T_COLL_S = 8.0
TAU_S = 10.0


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    try:
        s5 = json.loads(S5.read_text(encoding="utf-8"))
        rmap = {p["delay_s"]: p for p in s5["points"]}
    except (OSError, KeyError):
        rmap = {}
    res = {}
    print(" t_end  depleted-edges  mean_d(depleted)  pred_d  "
          "R_mbon  R_scalp")
    for t_end in meta["t_ends_ms"]:
        tag = f"szl1_e{t_end / 1000:g}s_s{meta['seed']}"
        f = OUT / f"{tag}_stddebug.npz"
        if not f.exists():
            continue
        d = np.load(f)["d"]
        dep = d < 0.9999
        t_s = t_end / 1000.0
        pred = 1.0 - np.exp(-max(t_s - T_COLL_S, 0.0) / TAU_S)
        mean_dep = float(d[dep].mean()) if dep.any() else 1.0
        r = rmap.get(int(t_s), {})
        rec = {"t_end_s": t_s, "pool": int(d.size),
               "depleted_frac": float(dep.mean()),
               "mean_d_depleted": mean_dep,
               "min_d": float(d.min()),
               "pred_d": round(float(pred), 3),
               "R_mbon": r.get("R_mbon"), "R_scalp": r.get("R_scalp")}
        res[tag] = rec
        print(f" {t_s:5.0f}s  {rec['depleted_frac']:11.3%}  "
              f"{mean_dep:15.4f}  {pred:6.3f}  "
              f"{r.get('R_mbon', '-'):>6}  {r.get('R_scalp', '-'):>6}")
    # verdict
    if res:
        late = [v for v in res.values() if v["t_end_s"] >= 50]
        if late:
            v = late[-1]
            d_ok = v["mean_d_depleted"] > 0.95
            r_cap = (v["R_mbon"] or 1.0) < 0.7
            verdict = ("STATE ceiling (d recovered, R capped)"
                       if d_ok and r_cap else
                       "RESOURCE ceiling (d not recovered)"
                       if not d_ok else "no ceiling at late delays")
            print(f"\n[sleep6] verdict: {verdict}")
            res["_verdict"] = verdict
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[sleep6] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
