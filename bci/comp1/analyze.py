"""bci/comp1 analysis: compartment gate vs global gate.

Per arm x condition, per trial: mean w of the trained odor's KC
subtype, fraction of edges grown (|w-w0| > 0.005 -- the architectural
restriction: comp should cap at the PAM07 compartment fraction ~13.7%,
global spreads over every eligible edge), mod integral in the train
window, probe rms.  Contrasts: blocking ratio (w_B final blocked vs
control) and error suppression (mod_ms p2, blocked vs control) per
arm; the global arm doubles as the dangate2 reproduction check.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SUB = {"A": "KCg-m", "B": "KCab-m", "C": "KCab-s"}
TRAIN = (500.0, 2500.0)
PROBE = (4200.0, 4700.0)
W0 = 0.03


def state_stats(npz):
    d = np.load(npz)
    pt = d["pre_type"]
    w = d["w_KCM"]
    by = {}
    for sub in set(str(s) for s in pt):
        m = pt == sub
        by[sub] = float(w[m].mean())
    grown = float((np.abs(w - W0) > 0.005).mean())
    return by, float(w.mean()), grown


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    n_p2 = meta["n_p2"]
    res = {}
    for arm in meta["arms"]:
        res[arm] = {}
        for cond in meta["conditions"]:
            rows = {}
            for f in sorted((OUT / "states").glob(
                    f"{arm}_{cond}_r*_p*.npz")):
                tag = f.name[:-4]
                by, wall, grown = state_stats(f)
                ph = "p1" if "_p1" in tag else "p2"
                sub = SUB["A"] if ph == "p1" else SUB["B"]
                rec = {"w_trained": by.get(sub, np.nan),
                       "w_all": wall, "grown_frac": grown}
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
                    base = e[3000:4000]
                    rec["probe_uv"] = float(np.sqrt(
                        ((e[4200:4700] - base.mean()) ** 2).mean())
                        * 1e6 * 1.7)
                rows[tag] = rec
            res[arm][cond] = rows
            print(f"\n-- {arm}/{cond} --  trial  w_trained  w_all  "
                  f"grown%  mod_ms  probe_uv")
            for tag in sorted(rows):
                r_ = rows[tag]
                print(f"   {tag:>26}  {r_['w_trained']:9.4f} "
                      f"{r_['w_all']:7.4f}  {r_['grown_frac']:6.1%}  "
                      f"{r_.get('mod_ms', float('nan')):7.1f}  "
                      f"{r_.get('probe_uv', float('nan')):7.3f}")
    summary = {}
    print("\n== contrasts ==")
    for arm in res:
        b, c = res[arm].get("blocked", {}), res[arm].get("control", {})
        fb = [v["w_trained"] for t, v in b.items()
              if "_p2" in t and t.endswith(f"p2{n_p2 - 1}")]
        fc = [v["w_trained"] for t, v in c.items()
              if "_p2" in t and t.endswith(f"p2{n_p2 - 1}")]
        ratio = (float(np.mean(fc) / np.mean(fb))
                 if fb and fc and np.mean(fb) > 0 else np.nan)
        mb = np.nanmean([v["mod_ms"] for t, v in b.items() if "_p2" in t]
                        or [np.nan])
        mc = np.nanmean([v["mod_ms"] for t, v in c.items() if "_p2" in t]
                        or [np.nan])
        gb = np.nanmean([v["grown_frac"] for t, v in b.items()
                         if "_p2" in t] or [np.nan])
        gc = np.nanmean([v["grown_frac"] for t, v in c.items()
                         if "_p2" in t] or [np.nan])
        summary[arm] = {"w_B_blocked_final": float(np.mean(fb)) if fb
                        else None,
                        "w_B_control_final": float(np.mean(fc)) if fc
                        else None,
                        "blocking_ratio": ratio,
                        "mod_ms_p2_blocked": float(mb),
                        "mod_ms_p2_control": float(mc),
                        "error_suppression":
                            float(1 - mb / mc) if mc else None,
                        "grown_frac_p2_blocked": float(gb),
                        "grown_frac_p2_control": float(gc)}
        print(f"  {arm}: w_B final {np.mean(fb) if fb else '-'} vs "
              f"{np.mean(fc) if fc else '-'} (blocking {ratio:.2f}x), "
              f"mod {mb:.1f}/{mc:.1f} (suppression "
              f"{1 - mb / mc if mc else float('nan'):.0%}), "
              f"grown {gb:.1%}/{gc:.1%}")
    (OUT / "summary.json").write_text(json.dumps(
        {"per_arm": summary,
         "raw": {a: {c: rows for c, rows in cc.items()}
                 for a, cc in res.items()}}, indent=1))
    print(f"\n[comp1] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
