"""bci/noise_robustness: per-paradigm sensor-noise robustness table.

ANALYSIS-ONLY (zero acquisition): re-analyzes every classification
paradigm's saved trial outputs (bci/*/outputs/*.npy) with a UNIFORM
decoder -- per-trial spatial rms (17 electrodes) over the paradigm's
canonical window, z-scored, nearest-centroid LOO -- while adding white
Gaussian electrode noise at ratios r in {0.3, 1, 3, 10} x the
paradigm's own median signal AC rms (relative scale -> comparable
across paradigms; the ssvep branch's clinical 1.5/3.79 sensor/signal
ratio ~ 0.4x sits inside this range).  20 noise draws per level.

The uniform decoder is deliberately NOT each paradigm's bespoke
analyzer (templates/CCA/perm tests -- see the per-paradigm READMEs);
it lower-bounds robustness and makes the columns comparable.

Usage:
    python bci/noise_robustness/analyze.py
"""
import json
from pathlib import Path

import numpy as np

BCI = Path(__file__).resolve().parents[1]

# (dir, classes-key, file-fmt, window-key fallbacks)
PARADIGMS = [
    ("motor_imagery", "classes", "{c}_r{r}.npy", ("imagery_ms",)),
    ("olfactory", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("auditory", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("touch", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("stereo", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("visualfield", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("dose", "levels", "d{c}_r{r}.npy", ("drive_ms",)),
    ("odormix", "conditions", "{c}_r{r}.npy", ("drive_ms",)),
    ("gustatory", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("seizure", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("looming", "classes", "{c}_r{r}.npy", ("drive_ms",)),
    ("direction", "classes", "d{c}_r{r}.npy", ("drive_ms",)),
    ("mrcp", "classes", "{c}_r{r}.npy", ("imagery_ms", "ramp_ms")),
    ("cnv", "classes", "{c}_r{r}.npy", ("anticipation_ms", "s2_ms")),
    ("cvep", "targets", "eeg_t{c:02d}_r{r}.npy", ()),
]
RATIOS = [0.3, 1.0, 3.0, 10.0]
DRAWS = 20


def features(phi, w0, w1):
    return np.sqrt((phi[w0:w1] ** 2).mean(0))


def loo_acc(X, y):
    Zs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Zs[m & (y == c)].mean(0)
                          for c in np.unique(y)])
        ok += int(np.argmin(((cents - Zs[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    rng = np.random.default_rng(7)
    rows = []
    for dname, ckey, fmt, wkeys in PARADIGMS:
        out = BCI / dname / "outputs"
        meta_p = out / "meta.json"
        if not meta_p.exists():
            print(f"[skip] {dname}: no meta")
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        classes = meta.get(ckey)
        if classes is None:
            print(f"[skip] {dname}: no {ckey}")
            continue
        X0, y = [], []
        for ci, c in enumerate(classes):
            for r in range(int(meta.get("repeats", 1))):
                p = out / fmt.format(c=c, r=r)
                if p.exists():
                    X0.append(np.load(p) * 1e6 * 1.7)
                    y.append(ci)
        if len(y) < len(classes) * 2:
            print(f"[skip] {dname}: too few trials ({len(y)})")
            continue
        dur = int(meta.get("dur_ms", 3600))
        w = next((meta[k] for k in wkeys if k in meta), None)
        w0, w1 = (int(w[0]), int(w[1])) if w else (300, dur - 100)
        F = np.array([features(phi, w0, w1) for phi in X0])
        y = np.array(y)
        sig = float(np.median([np.sqrt(
            ((phi - phi.mean(0)) ** 2).mean()) for phi in X0]))
        accs = [loo_acc(F, y)]

        def noisy_feats(sigma):
            # sensor noise on the RAW window signal, then re-extract
            # the rms feature (quadrature bias included -- the honest
            # feature-level consequence of electrode noise)
            return np.array([
                features(phi[w0:w1]
                         + rng.normal(0.0, sigma, (w1 - w0, phi.shape[1])),
                         0, w1 - w0) for phi in X0])

        for ratio in RATIOS:
            a = [loo_acc(noisy_feats(ratio * sig), y)
                 for _ in range(DRAWS)]
            accs.append(float(np.mean(a)))
        rows.append({"paradigm": dname, "k": len(classes),
                     "n": len(y), "window_ms": [w0, w1],
                     "acc": accs,
                     "sig_uv": sig * 1e6})
    hdr = (f"{'paradigm':16s} {'K':>2s} {'n':>3s}  {'clean':>6s}  "
           + "  ".join(f"r={r:g}" for r in RATIOS))
    print(hdr)
    for row in rows:
        print(f"{row['paradigm']:16s} {row['k']:2d} {row['n']:3d}  "
              + "  ".join(f"{a:6.1%}" for a in row["acc"]))
    here = Path(__file__).resolve().parent / "outputs"
    here.mkdir(exist_ok=True)
    (here / "summary.json").write_text(json.dumps(
        {"ratios": RATIOS, "draws": DRAWS, "rows": rows}, indent=1))
    print(f"-> {here / 'summary.json'}")


if __name__ == "__main__":
    main()
