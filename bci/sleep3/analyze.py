"""bci/sleep3 analysis: up/down bistability + hysteresis cycling.

For each arm: 1-s scalp rms (17-electrode mean, the sleep2 observable)
classified into up (>3 uV) / down (<1.5 uV); collapse delay after each
ignition pulse; dwell times; and whether pulse 2 re-latches (the
hysteresis test).  Class rates give the mechanism view.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PULSES = [(800.0, 2800.0), (35000.0, 37300.0)]
UP_UV, DOWN_UV = 3.0, 1.5


def classify(rms1s):
    """up(1)/down(-1)/mid(0) per 1-s bin, ignoring the pulse windows."""
    st = np.zeros(len(rms1s), dtype=int)
    st[rms1s > UP_UV] = 1
    st[rms1s < DOWN_UV] = -1
    return st


def episode_stats(st):
    """ contiguous same-sign runs >= 2 s (up or down), as (kind, start, len)"""
    eps, i = [], 0
    while i < len(st):
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        if st[i] != 0 and (j - i + 1) >= 2:
            eps.append((int(st[i]), i, j - i + 1))
        i = j + 1
    return eps


def main():
    res = {}
    for f in sorted(OUT.glob("*_scalp.npy")):
        tag = f.name[:-10]
        e = np.load(f)
        e = e.mean(axis=1) if e.ndim == 2 else e
        rms = np.array([np.sqrt((e[a:a + 1000] ** 2).mean())
                        for a in range(0, len(e) - 999, 1000)]) * 1e6 * 1.7
        st = classify(rms)
        eps = episode_stats(st)
        # collapse delay after each pulse: last up bin following pulse end
        coll = []
        for p0, p1 in PULSES:
            w0 = int(p1 // 1000) + 1
            ups = [s for k, s, L in eps if k == 1 and s >= w0 - 1]
            if not ups:
                coll.append(None)
                continue
            s0 = min(ups)
            run = [L for k, s, L in eps if k == 1 and s == s0][0]
            coll.append(s0 + run - w0)          # s the up-state survived
        res[tag] = {"rms_uv": [round(float(v), 3) for v in rms],
                    "frac_up": float((st == 1).mean()),
                    "level_10_30s_uv": round(float(rms[10:30].mean()), 2),
                    "episodes": [(k, s, L) for k, s, L in eps],
                    "collapse_s_after_pulse": coll}
        pop = OUT / f"{tag}_pop.npz"
        if pop.exists():
            z = np.load(pop)
            res[tag]["rate_hz_10_30s"] = {
                k: round(float(z[k][10000:30000].mean()), 2)
                for k in z.files}
        print(f"\n-- {tag} --  frac_up {res[tag]['frac_up']:.2f}  "
              f"level(10-30s) {res[tag]['level_10_30s_uv']:.2f} uV  "
              f"collapse {coll}")
        print("   rms(uV, 1s): " + " ".join(f"{v:.1f}" for v in rms[::4]))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[sleep3] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
