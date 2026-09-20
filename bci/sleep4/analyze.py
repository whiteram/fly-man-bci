"""bci/sleep4 analysis: endogenous re-ignition + tau dwell scaling.

Per arm (from meta.json): 1-s scalp rms classified up(>3 uV) /
down(<1.5 uV) as in sleep3.  Key readouts:
  - single-pulse arms (sz_l1): up-episodes starting after the re-arm
    horizon (collapse onset + 3*tau) = spontaneous re-ignition
    (endogenous slow oscillation); max level in that window; micro
    flares (bins > 2 uV).
  - two-pulse arms (sz_l3): level just before pulse 2 (re-arm
    completeness, [p2-10s, p2)), collapse delay after pulse 2, and
    the observed cycle period (pulse-2 onset -> collapse 2).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
UP_UV, DOWN_UV = 3.0, 1.5


def classify(rms1s):
    st = np.zeros(len(rms1s), dtype=int)
    st[rms1s > UP_UV] = 1
    st[rms1s < DOWN_UV] = -1
    return st


def episodes(st):
    out, i = [], 0
    while i < len(st):
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        if st[i] != 0 and (j - i + 1) >= 2:
            out.append((int(st[i]), i, j - i + 1))
        i = j + 1
    return out


def collapse_after(eps, onset_ms):
    w0 = max(1, int(onset_ms // 1000))
    ups = [e for e in eps if e[0] == 1 and e[1] >= w0]
    if not ups:
        return None
    s0 = min(e[1] for e in ups)
    run = [e[2] for e in ups if e[1] == s0][0]
    return s0 + run - w0


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for tag, info in sorted(meta["arms"].items()):
        f = OUT / f"{tag}_scalp.npy"
        if not f.exists():
            continue
        e = np.load(f)
        e = e.mean(axis=1) if e.ndim == 2 else e
        rms = np.array([np.sqrt((e[a:a + 1000] ** 2).mean())
                        for a in range(0, len(e) - 999, 1000)]) * 1e6 * 1.7
        st = classify(rms)
        eps = episodes(st)
        tau_s = info["tau_ms"] / 1000.0
        two_pulse = info["chem"] == "sz_l3"
        onsets = [800.0] + ([80000.0] if two_pulse else [])
        coll = [collapse_after(eps, on) for on in onsets]
        peaks = [round(float(rms[int(on // 1000):int(on // 1000) + 5].max()), 2)
                 for on in onsets]
        rec = {"chem": info["chem"], "tau_ms": info["tau_ms"],
               "std": info["std"], "seed": info["seed"],
               "rms_uv": [round(float(v), 3) for v in rms],
               "episodes": [(k, s, L) for k, s, L in eps],
               "collapse_s_after_pulse": coll,
               "pulse_transient_peak_uv": peaks,
               "plateau_10_30s_uv": round(float(rms[10:30].mean()), 2)}
        c1 = coll[0]
        if not two_pulse:
            horizon = (c1 + 3 * tau_s) if c1 is not None else 3 * tau_s
            w0 = max(int(horizon) + 1, 0)
            w = rms[w0:]
            late = [ep for ep in eps if ep[0] == 1 and ep[1] >= w0]
            rec["rearm_horizon_s"] = round(horizon, 1)
            rec["post_rearm_max_uv"] = round(float(w.max()), 2) \
                if len(w) else None
            rec["spontaneous_reignition"] = bool(late)
            rec["late_up_episodes"] = late
            rec["flares_gt2uv_post"] = int((w > 2.0).sum()) if len(w) else 0
            verdict = "ENDOGENOUS" if late else "externally driven"
            print(f"-- {tag} -- pulse1 peak {peaks[0]} uV, plateau "
                  f"{rec['plateau_10_30s_uv']} uV, horizon {horizon:.0f}s, "
                  f"post max {rec['post_rearm_max_uv']} uV -> {verdict}")
        else:
            i2 = int(onsets[1] // 1000)
            pre2 = rms[max(0, i2 - 10):i2]
            rec["pre_pulse2_level_uv"] = round(float(pre2.mean()), 2)
            c2 = coll[1]
            period = (onsets[1] / 1000.0 + c2) if c2 is not None else None
            rec["cycle_period_s"] = round(period, 1) if period else None
            print(f"-- {tag} -- p1 peak {peaks[0]} uV, plateau "
                  f"{rec['plateau_10_30s_uv']} uV, p2 peak {peaks[1]} uV "
                  f"(re-arm test vs 6.6), pre-p2 level "
                  f"{rec['pre_pulse2_level_uv']} uV, collapse2 {c2}s, "
                  f"period {rec['cycle_period_s']}s")
        pop = OUT / f"{tag}_pop.npz"
        if pop.exists():
            z = np.load(pop)
            rec["rate_hz_last20s"] = {
                k: round(float(z[k][-20000:].mean()), 2) for k in z.files}
        res[tag] = rec
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[sleep4] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
