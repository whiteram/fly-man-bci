"""bci/dangate analysis: circuit-level error coding + emergent blocking.

Primary: w_B (KCab-m -> MBON edges) at the END of each arm's AB+ phase,
blocked vs control -- did B learn less when A already carried V?
Secondary: per-trial DAN-gate mod integral (the in-circuit error):
should shrink across AB+ trials in blocked but not control.  Also the
B-probe scalp response (net rms 4.2-4.7 s) as the readout-side check.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SUB = {"A": "KCg-m", "B": "KCab-m", "C": "KCab-s"}
TRAIN = (500.0, 2500.0)
PROBE = (4200.0, 4700.0)


def w_by_subtype(npz, prefix=""):
    d = np.load(npz)
    pt = d[f"{prefix}pre_type"]
    w = d[f"{prefix}w"] if f"{prefix}w" in d.files else d[f"{prefix}w_KCM"]
    out = {}
    for sub in set(str(s) for s in pt):
        m = pt == sub
        out[sub] = (float(w[m].mean()), float(w[m].std()), int(m.sum()))
    return out, float(w.mean())


def main(prefix=""):
    meta = json.loads((OUT / f"{prefix}meta.json").read_text())
    n_p1, n_p2 = meta["n_p1"], meta["n_p2"]
    reps = meta["sessions"]
    res = {}
    for cond in meta["conditions"]:
        rows, mods, probes = [], [], []
        for r in range(reps):
            for i, (ph, k) in enumerate(
                    [("p1", k) for k in range(n_p1)]
                    + [("p2", k) for k in range(n_p2)]):
                tag = f"{prefix}{cond}_r{r}_{ph}{k}"
                st = OUT / "states" / f"{tag}.npz"
                if not st.exists():
                    continue
                by, wall = w_by_subtype(st)
                sub = SUB["A"] if ph == "p1" else SUB["B"]
                rows.append({"trial": f"{ph}{k}", "arm": cond, "r": r,
                             "w_trained": by.get(sub, (np.nan, 0, 0))[0],
                             "w_all": wall})
                dt_ = OUT / f"{tag}_dan.npy"
                if dt_.exists():
                    tr = np.load(dt_)
                    m_ = tr[:, 2]
                    inwin = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
                    mods.append(float(m_[inwin].sum() * 0.5))  # DT 0.5 ms
                eg = OUT / f"{tag}.npy"
                if eg.exists():
                    e = np.load(eg)
                    e = e.mean(axis=0) if e.ndim == 2 else e
                    seg = slice(int(PROBE[0]), int(PROBE[1]))
                    base = e[int(3000):int(4000)]
                    probes.append(float(np.sqrt(
                        ((e[seg] - base.mean()) ** 2).mean()) * 1e6 * 1.7))
        res[cond] = {"rows": rows, "mod_ms": mods, "probe_uv": probes}

    print(f"[dangate] {'PILOT ' if prefix else ''}"
          f"lr={meta['lr']} w0={meta['w0']} kcm={meta['kcm_gain']:g} "
          f"mbmd={meta['mbmd_gain']:g} gate={meta['dan_gate']}")
    for cond, rr in res.items():
        print(f"\n-- {cond} --   trial  w_trained   w_all   mod_ms   probe_uv")
        for row, m_, p_ in zip(rr["rows"], rr["mod_ms"] or [None] * 9,
                               rr["probe_uv"] or [None] * 9):
            print(f"   {row['trial']:>7}  {row['w_trained']:9.4f} "
                  f"{row['w_all']:8.4f}  "
                  f"{(f'{m_:9.1f}' if m_ is not None else '        -')}  "
                  f"{(f'{p_:8.3f}' if p_ is not None else '       -')}")

    summary = {"per_cond": {}}
    if "blocked" in res and "control" in res and not prefix:
        wb = [r["w_trained"] for r in res["blocked"]["rows"]
              if r["trial"].startswith("p2") and r["trial"] != "p20"]
        wc = [r["w_trained"] for r in res["control"]["rows"]
              if r["trial"].startswith("p2") and r["trial"] != "p20"]
        # final-trial w_B (state AFTER last AB+ pairing)
        fb = [r["w_trained"] for r in res["blocked"]["rows"]
              if r["trial"] == f"p2{n_p2 - 1}"]
        fc = [r["w_trained"] for r in res["control"]["rows"]
              if r["trial"] == f"p2{n_p2 - 1}"]
        ratio = (float(np.mean(fc) / np.mean(fb))
                 if fb and fc and np.mean(fb) > 0 else np.nan)
        # error signature: mean mod over LAST p2 trials, blocked vs control
        mb = np.mean([res["blocked"]["mod_ms"][i]
                      for i, r in enumerate(res["blocked"]["rows"])
                      if r["trial"].startswith("p2")]) \
            if res["blocked"]["mod_ms"] else np.nan
        mc = np.mean([res["control"]["mod_ms"][i]
                      for i, r in enumerate(res["control"]["rows"])
                      if r["trial"].startswith("p2")]) \
            if res["control"]["mod_ms"] else np.nan
        print(f"\n[dangate] w_B final: blocked {np.mean(fb):.4f} vs "
              f"control {np.mean(fc):.4f}  -> ratio {ratio:.2f}x "
              f"(>1 = blocking emerged)")
        print(f"[dangate] mod_ms p2 mean: blocked {mb:.1f} vs control "
              f"{mc:.1f}  -> error suppression "
              f"{(1 - mb / mc if mc else np.nan):.0%}"
              f"  (circuit-level prediction error)")
        summary["per_cond"] = {
            "w_B_blocked": float(np.mean(fb)), "w_B_control": float(np.mean(fc)),
            "ratio": ratio, "mod_ms_p2_blocked": float(mb),
            "mod_ms_p2_control": float(mc)}
    (OUT / f"{prefix}summary.json").write_text(json.dumps(
        {**summary, "raw": {
            c: {"rows": rr["rows"], "mod_ms": rr["mod_ms"],
                "probe_uv": rr["probe_uv"]} for c, rr in res.items()}},
        indent=1))
    print(f"[dangate] summary -> {OUT / f'{prefix}summary.json'}")


if __name__ == "__main__":
    main("pilot_" if (OUT / "pilot_meta.json").exists()
         and not (OUT / "meta.json").exists() else "")
