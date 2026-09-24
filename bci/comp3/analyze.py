"""bci/comp3 analysis: co-presence dual reward -- the 2x2 completes.

Per arm: w_A (KCg-m), w_B (KCab-m), w_all, grown_frac, per-trial mod
integral, B-probe rms.  Headline: B-learning ratio = dw_B(trained) /
dw_B(control) per gate type under A+B co-presence with BOTH rewards
firing -- comp ~1 (private gate blind to A's V) vs glob >1 (aggregate
gate suppressed by A's co-expressed V: blocking of B's OWN reward).
Prints the full 2x2 (presence x gate) with the comp1/comp2 cells.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
W0 = 0.03
TRAIN = (500, 2500)
PROBE = (4200, 4700)


def state_stats(npz):
    d = np.load(npz)
    w = d["w_KCM"]
    pt = d["pre_type"]
    return (float(w[pt == "KCg-m"].mean()), float(w[pt == "KCab-m"].mean()),
            float(w.mean()), float((np.abs(w - W0) > 0.005).mean()))


def mod_ms(tag, out):
    p = out / f"{tag}_dan.npy"
    if not p.exists():
        return np.nan
    tr = np.load(p)
    inwin = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
    return float(tr[inwin, 2].sum() * 0.5)


def probe_uv(tag, out):
    p = out / f"{tag}.npy"
    if not p.exists():
        return np.nan
    e = np.load(p)
    e = e.mean(axis=1) if e.ndim == 2 else e
    base = e[3000:4000]
    return float(np.sqrt(((e[PROBE[0]:PROBE[1]] - base.mean()) ** 2)
                         .mean()) * 1e6 * 1.7)


def main():
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--outdir", type=str, default=None)
    _args = _ap.parse_args()
    out = Path(_args.outdir) if _args.outdir else OUT
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for arm in meta["arms"]:
        rows = []
        for r in range(meta.get("sessions", 1)):
            for i in range(meta["n_p1"] + meta["n_p2"]):
                ph = "p1" if i < meta["n_p1"] else "p2"
                tag = f"{arm}_r{r}_{ph}{i if ph == 'p1' else i - meta['n_p1']}"
                st = out / "states" / f"{tag}.npz"
                if not st.exists():
                    continue
                wa, wb, wall, gf = state_stats(st)
                rows.append({"tag": tag, "w_A": wa, "w_B": wb,
                             "w_all": wall, "grown_frac": gf,
                             "mod_ms": mod_ms(tag, out),
                             "probe_uv": probe_uv(tag, out)})
        res[arm] = rows
        print(f"\n-- {arm} --  trial  w_A  w_B  w_all  grown%  "
              f"mod_ms  probe_uv")
        for row in rows:
            print(f"   {row['tag']:>16}  {row['w_A']:.4f} "
                  f"{row['w_B']:.4f} {row['w_all']:.4f} "
                  f"{row['grown_frac']:6.1%} {row['mod_ms']:8.1f} "
                  f"{row['probe_uv']:7.3f}")

    print("\n== co-presence dual-reward contrast (dw_B trained/ctl) ==")
    summary = {}
    for gate in ("comp", "glob"):
        tr, ct = res.get(f"{gate}_tr"), res.get(f"{gate}_ctl")
        if not tr or not ct:
            continue
        np1 = meta["n_p1"]
        per_s = {}
        sessions = sorted({r["tag"].split("_r")[1][0] for r in tr})
        for s in sessions:
            trs = [x for x in tr if f"_r{s}_" in x["tag"]]
            cts = [x for x in ct if f"_r{s}_" in x["tag"]]
            dA = trs[np1 - 1]["w_A"] - W0
            dB_tr = trs[-1]["w_B"] - W0
            dB_ct = cts[-1]["w_B"] - W0
            per_s[s] = {"dw_A_p1": dA, "dw_B_trained": dB_tr,
                        "dw_B_control": dB_ct,
                        "B_ratio": dB_tr / dB_ct if dB_ct > 0 else np.nan}
        dB_tr = float(np.mean([v["dw_B_trained"] for v in per_s.values()]))
        dB_ct = float(np.mean([v["dw_B_control"] for v in per_s.values()]))
        dA = float(np.mean([v["dw_A_p1"] for v in per_s.values()]))
        ratio = dB_tr / dB_ct if dB_ct > 0 else np.nan
        summary[gate] = {"dw_A_p1": dA, "dw_B_trained": dB_tr,
                         "dw_B_control": dB_ct, "B_ratio": ratio,
                         "per_session": per_s}
        print(f"  {gate}: dw_A(p1) {dA:.4f}  dw_B tr/ctl "
              f"{dB_tr:.4f}/{dB_ct:.4f} -> B ratio {ratio:.2f} "
              f"({'blocked' if ratio < 0.8 else 'independent'})")

    print("\n== 2x2: B's own reward, by A-presence x gate ==")
    print("  A absent  (comp2):  glob 0.99   comp 1.00   "
          "(sequential, no crosstalk either gate)")
    print("  A present, B unrewarded (comp1): glob 1.41 w_B ratio; "
          "comp w_B flat (nothing to learn)")
    for gate in ("glob", "comp"):
        if gate in summary:
            r = summary[gate]["B_ratio"]
            print(f"  A CO-PRESENT, dual reward (comp3): "
                  f"{gate} {r:.2f}")
    (out / "summary.json").write_text(json.dumps(
        {"per_arm": res, "contrast": summary}, indent=1))
    print(f"\n[comp3] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
