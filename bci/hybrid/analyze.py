"""bci/hybrid analysis: dual-stream decoding + interference.

(a) flicker frequency: per trial, Welch PSD of the concatenated
    (z-scored) channels; classify 10 vs 14 Hz by which line is
    stronger; SNR = PSD(f) / median PSD in f +/- [0.5, 3] Hz side
    bands.
(b) imagery side: spatial-rms LOO on the imagery plateau window
    (2500-9500 ms), L vs R imagery runs only.
(c) interference: SSVEP SNR distribution in hyb_n vs hyb_l/hyb_r.

Usage:
    python bci/hybrid/analyze.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
IMWIN = (2500, 9500)


def cca_max(X, f, fs, harms=(1, 2)):
    """max canonical correlation between multi-channel epoch X (ch, n)
    and a sine/cosine reference at frequency f (+ harmonics)."""
    n = X.shape[1]
    t = np.arange(n) / fs
    Y = np.concatenate([np.stack([np.sin(2 * np.pi * f * h * t),
                                  np.cos(2 * np.pi * f * h * t)])
                        for h in harms]).T
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(0)
    Cxx = Xc @ Xc.T
    Cyy = Yc.T @ Yc
    Cxy = Xc @ Yc
    M = np.linalg.pinv(Cxx) @ Cxy @ np.linalg.pinv(Cyy) @ Cxy.T
    return float(np.sqrt(max(np.linalg.eigvals(M).real.max(), 0)))


def snr_at(psd, freqs, f):
    m = freqs == freqs[np.argmin(np.abs(freqs - f))]
    lo = (freqs > f - 3.0) & (freqs < f - 0.5)
    hi = (freqs > f + 0.5) & (freqs < f + 3.0)
    return psd[m].sum() / max(psd[lo | hi].mean(), 1e-30)


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    fs = int(meta["fs"])
    conds = [f"f{f:.0f}_{im}" for f in meta["freqs"]
             for im in meta["imagery"]]
    snrs, fr_pred, fr_true, im_pred, im_true, im_cond = [], [], [], [], [], []
    Xim, yim = [], []
    for cond in conds:
        f, im = cond.split("_")
        f = float(f[1:])
        for r in range(meta["repeats"]):
            p = OUT / f"{cond}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            Z = (phi - phi.mean(0)) / (phi.std(0) + 1e-12)
            fr, psd = welch(Z.T, fs=fs, nperseg=fs * 4)
            ps = psd.mean(0)
            s10, s14 = snr_at(ps, fr, 10.0), snr_at(ps, fr, 14.0)
            snrs.append((cond, r, s10, s14))
            fr_true.append(f)
            fr_pred.append(10.0 if s10 > s14 else 14.0)
            if im != "n":
                Xim.append(np.sqrt(
                    (phi[IMWIN[0]:IMWIN[1]] ** 2).mean(axis=0)))
                im_true.append(0 if im == "l" else 1)
                im_cond.append(cond)
    fr_true = np.array(fr_true)
    fr_pred = np.array(fr_pred)
    acc_f = float((fr_pred == fr_true).mean())
    # CCA-based classification (SSVEP-branch standard): epoch vs
    # sine/cosine references at 10/14 Hz
    cca_pred, cca_true = [], []
    for cond in conds:
        f = float(cond.split("_")[0][1:])
        for r in range(meta["repeats"]):
            p = OUT / f"{cond}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            X = phi[1500:10500].T
            r10 = cca_max(X, 10.0, fs)
            r14 = cca_max(X, 14.0, fs)
            cca_pred.append(10.0 if r10 > r14 else 14.0)
            cca_true.append(f)
    acc_cca = float((np.array(cca_pred) == np.array(cca_true)).mean())
    # imagery-side decode
    Xim = np.array(Xim)
    yim = np.array(im_true)
    Zs = (Xim - Xim.mean(0)) / (Xim.std(0) + 1e-12)
    ok = 0
    for i in range(len(yim)):
        m = np.arange(len(yim)) != i
        cents = np.array([Zs[m & (yim == c)].mean(0) for c in (0, 1)])
        ok += int(np.argmin(((cents - Zs[i]) ** 2).sum(1)) == yim[i])
    acc_i = ok / len(yim)
    rng = np.random.default_rng(1)
    perm = []
    for _ in range(2000):
        yp = rng.permutation(yim)
        okp = 0
        for i in range(len(yim)):
            m = np.arange(len(yim)) != i
            cents = np.array([Zs[m & (yp == c)].mean(0) for c in (0, 1)])
            okp += int(np.argmin(((cents - Zs[i]) ** 2).sum(1)) == yp[i])
        perm.append(okp / len(yim))
    p_i = (np.sum(np.array(perm) >= acc_i) + 1) / 2001
    print(f"freq classification (10 vs 14 Hz): "
          f"PSD-peak {acc_f:.0%}, CCA {acc_cca:.0%} (chance 50%; "
          f"the visual-only circuit reaches ~100% at line ratios "
          f"~350 -- see README for the masking account)")
    print(f"imagery-side LOO: {acc_i:.0%}  perm-p={p_i:.4f} "
          f"(chance 50%)")
    # interference: SNR of the PRESENT frequency by imagery condition
    print("SSVEP SNR of the present frequency by imagery condition:")
    for im in meta["imagery"]:
        vals = []
        for (cond, r, s10, s14) in snrs:
            f, c = cond.split("_")
            if c != im:
                continue
            vals.append(s10 if f == "10" else s14)
        if vals:
            print(f"  imagery={im:1s}: median SNR "
                  f"{np.median(vals):.1f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
