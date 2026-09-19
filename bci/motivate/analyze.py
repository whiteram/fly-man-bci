"""bci/motivate analysis: single-shot learning depth vs gate strength.

Mechanism: KCM w_scale mean after one paired trial, per mod-scale arm.
Scalp: probe net response delta vs the lr=0 reference.  Trend test:
Spearman rho of w (and probe delta) against mod-scale, permutation p
over shuffles of the arm labels; plus monotonicity check.

Usage:
    python bci/motivate/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
A, B = 4300, 4700


def net(phi, a, b, pre=200):
    return float(np.sqrt((phi[a:b] ** 2).mean())
                 - np.sqrt((phi[a - pre:a] ** 2).mean()))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    runs = meta["runs"]
    nolr = np.array([net(np.load(OUT / f"nolr_r{r}.npy") * 1e6 * 1.7,
                         A, B) for r in range(runs)
                     if (OUT / f"nolr_r{r}.npy").exists()])
    mods = np.array(meta["mods"], dtype=float)
    w_mean, pdelta = [], []
    print("mod-scale  w mean (sd)          probe net (sd)         "
          "delta vs nolr")
    for mod in meta["mods"]:
        ws, ps = [], []
        for r in range(runs):
            stem = f"m{mod:g}_r{r}"
            if (OUT / f"{stem}.npy").exists():
                ws.append(float(np.load(OUT / "states" /
                                        f"{stem}.npz")["w"].mean()))
                ps.append(net(np.load(OUT / f"{stem}.npy") * 1e6 * 1.7,
                              A, B))
        w_mean.append(np.mean(ws))
        pdelta.append(np.mean(ps) - nolr.mean())
        print(f"{mod:8.3f}  {np.mean(ws):.4f} ({np.std(ws):.4f})   "
              f"{np.mean(ps):+.4f} ({np.std(ps):.4f})   "
              f"{np.mean(ps) - nolr.mean():+.4f}")
    print(f"nolr reference probe net: {nolr.mean():+.4f} "
          f"({nolr.std():.4f})  n={len(nolr)}")
    w_mean = np.array(w_mean)
    pdelta = np.array(pdelta)
    rho_w = spearman(mods, w_mean)
    rho_p = spearman(mods, pdelta)
    rng = np.random.default_rng(11)
    pw = pp = 0.0
    for _ in range(5000):
        pm = rng.permutation(mods)
        if abs(spearman(pm, w_mean)) >= abs(rho_w):
            pw += 1
        if abs(spearman(pm, pdelta)) >= abs(rho_p):
            pp += 1
    pw = (pw + 1) / 5001
    pp = (pp + 1) / 5001
    mono = bool(np.all(np.diff(w_mean) < 0))
    print(f"\nw vs mod-scale: Spearman rho {rho_w:+.3f}, perm-p {pw:.4f}"
          f"  (monotone decreasing: {mono})")
    print(f"probe-delta vs mod-scale: Spearman rho {rho_p:+.3f}, "
          f"perm-p {pp:.4f}")
    (OUT / "summary.json").write_text(json.dumps(
        {"mods": mods.tolist(), "w_mean": w_mean.tolist(),
         "probe_delta": pdelta.tolist(), "rho_w": rho_w, "p_w": pw,
         "rho_p": rho_p, "p_p": pp, "monotone_w": mono,
         "nolr_probe": float(nolr.mean())}, indent=1))
    print(f"-> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
