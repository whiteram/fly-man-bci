"""bci/attend analysis: gain-modulation dose-response + trade-off.

Per k arm (OLR_V x {1, 4, 16}) over dual-task trials:
  (a) SSVEP line SNR of the PRESENTED frequency (Welch PSD, side-band
      normalization -- same metric as bci/hybrid);
  (b) CCA frequency classification 10 vs 14 Hz;
  (c) imagery-side spatial-rms LOO accuracy (attention cost).
Confoud check: line SNR of the nwp arm (no imagery, 0.002 working
point) vs bci/hybrid's f10_n runs (no imagery, default gains) -- how
much of the ~500x masking was the working point, not the dual task.

Usage:
    python bci/attend/analyze.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
HYB = HERE.parent / "hybrid" / "outputs"
IMWIN = (2500, 9500)


def cca_max(X, f, fs, harms=(1, 2)):
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


def line_snr(phi, f, fs):
    Z = (phi - phi.mean(0)) / (phi.std(0) + 1e-12)
    fr, psd = welch(Z.T, fs=fs, nperseg=fs * 4)
    return float(snr_at(psd.mean(0), fr, f))


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    fs = int(meta["fs"])
    rep = meta["repeats"]
    arms = {}
    for kname in meta["k_arms"]:
        rows = []
        for f in meta["freqs"]:
            for side in meta["sides"]:
                for r in range(rep):
                    p = OUT / f"f{f:.0f}_{side}_{kname}_r{r}.npy"
                    if p.exists():
                        rows.append((f, side, np.load(p) * 1e6 * 1.7))
        arms[kname] = rows
    print("== recovery dose-response (dual-task trials) ==")
    for kname, rows in arms.items():
        if not rows:
            continue
        snrs = [line_snr(phi, f, fs) for f, _, phi in rows]
        acc_cca = 0
        for f, _, phi in rows:
            X = phi[1500:10500].T
            acc_cca += (10.0 if cca_max(X, 10.0, fs) >
                        cca_max(X, 14.0, fs) else 14.0) == f
        print(f" {kname:4s}: line SNR median {np.median(snrs):8.2f}  "
              f"CCA freq acc {acc_cca}/{len(rows)}")
    print("\n== imagery cost (spatial-rms LOO, L vs R) ==")
    for kname, rows in arms.items():
        X, y = [], []
        for f, side, phi in rows:
            X.append(np.sqrt((phi[IMWIN[0]:IMWIN[1]] ** 2).mean(0)))
            y.append(0 if side == "l" else 1)
        if len(y) < 4:
            continue
        X = np.array(X)
        y = np.array(y)
        Zs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        ok = 0
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            cents = np.array([Zs[m & (y == c)].mean(0) for c in (0, 1)])
            ok += int(np.argmin(((cents - Zs[i]) ** 2).sum(1)) == y[i])
        print(f" {kname:4s}: {ok}/{len(y)}")
    print("\n== hybrid confound check (10 Hz line SNR) ==")
    nwp = [line_snr(np.load(p) * 1e6 * 1.7, 10.0, fs)
           for p in sorted(OUT.glob("f10_nwp_r*.npy"))]
    hyb_n = [line_snr(np.load(p) * 1e6 * 1.7, 10.0, fs)
             for p in sorted(HYB.glob("f10_n_r*.npy"))]
    k0 = [line_snr(phi, f, fs) for f, _, phi in arms.get("k0", [])
          if f == 10.0]
    if nwp:
        print(f" nwp (no imagery, wp 0.002): median "
              f"{np.median(nwp):.2f}  (n={len(nwp)})")
    if hyb_n:
        print(f" hyb_n (no imagery, DEFAULT gain): median "
              f"{np.median(hyb_n):.2f}  (n={len(hyb_n)})")
    if k0:
        print(f" k0 (imagery, wp 0.002): median {np.median(k0):.2f}")
    if nwp and hyb_n and k0:
        wp_share = np.median(hyb_n) / np.median(nwp)
        dual_share = np.median(nwp) / np.median(k0)
        print(f" -> working-point share x{wp_share:.0f}, dual-task "
              f"share x{dual_share:.0f}, total x"
              f"{wp_share * dual_share:.0f}")
    (OUT / "summary.json").write_text(json.dumps({
        "line_snr_median": {k: float(np.median(
            [line_snr(phi, f, fs) for f, _, phi in rows]))
            for k, rows in arms.items() if rows}}, indent=1))


if __name__ == "__main__":
    main()
