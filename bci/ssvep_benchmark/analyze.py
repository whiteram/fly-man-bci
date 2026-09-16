"""SSVEP detection analysis: canonical-CCA over the acquired trials
(the standard Wang-2016-style detector) + flicker-frequency SNR.

For each trial EEG (45 ch x T) and each candidate frequency, build the
harmonic reference Y = [sin, cos] at f, 2f, 3f and compute the maximal
canonical correlation between X and Y via SVD -- the predicted class is
the frequency with the highest rho.  Reports:
  - per-trial detected vs true frequency (confusion over the pilot set)
  - rho spectrum per trial and the SNR at each flicker frequency
    (power at f vs neighboring 1-Hz bands) on clean EEG and, under the
    replacement premise (project default), with 1.5 uV sensor noise.

Usage: python bci/ssvep_benchmark/analyze.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
rng = np.random.default_rng(7)


def cca_rho(X, Y):
    """Max canonical correlation between X (n_ch x T) and Y (n_ref x T)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Cxx = Xc @ Xc.T / X.shape[1]
    Cyy = Yc @ Yc.T / Y.shape[1]
    Cxy = Xc @ Yc.T / X.shape[1]
    inv_yy = np.linalg.pinv(Cyy + 1e-9 * np.eye(Cyy.shape[0]))
    M = np.linalg.pinv(Cxx) @ Cxy @ inv_yy @ Cxy.T
    # eigvalsh is ascending -> canonical rho is the LARGEST eigenvalue
    return float(np.sqrt(max(np.real(np.linalg.eigvalsh(M)[-1]), 0.0)))


def harmonic_ref(f, t):
    rows = []
    for h in (1, 2, 3):
        rows.append(np.sin(2 * np.pi * h * f * t))
        rows.append(np.cos(2 * np.pi * h * f * t))
    return np.array(rows)


def band_snr(eeg, f, fs):
    W = np.fft.rfft(eeg - eeg.mean(axis=1, keepdims=True), axis=1)
    P = 2 * np.abs(W) ** 2 / eeg.shape[1] ** 2
    fb = np.fft.rfftfreq(eeg.shape[1], d=1 / fs)
    Pm = P.mean(axis=0)
    i0 = np.argmin(np.abs(fb - f))
    half = max(int(round(0.5 / (fb[1] - fb[0]))), 1)
    sig = Pm[max(i0 - 1, 1):i0 + 2].max()
    neigh = (fb < f - 0.7) | (fb > f + 0.7)
    noise = np.median(Pm[neigh & (fb > 2) & (fb < 40)])
    return 10 * np.log10(sig / max(noise, 1e-30))


def main():
    meta = json.loads((OUT / "meta.json").read_text())
    freqs = meta["freqs"]
    fs = meta["fs"]
    seg = (np.arange(int(1.0 * fs), int(9.0 * fs)))   # 1-9 s analysis window
    t = seg / fs
    # _debug_phi_scalp.npy is (T, n_elec) -> transpose to (n_elec, T)
    data = {f: np.load(OUT / f"eeg_f{f:.1f}.npy").T[:, seg]
            for f in freqs}
    n_ch, T = data[freqs[0]].shape

    refs = {f: harmonic_ref(f, t) for f in freqs}
    print(f"[ssvep] {len(freqs)} trials x {n_ch} ch x {T} samples "
          f"({t[-1] - t[0]:.0f} s window)")

    for tag, add_noise in (("clean", 0.0), ("sensor-noise (k=0 premise)",
                                             1.5)):
        hit, rho_all = 0, np.zeros((len(freqs), len(freqs)))
        snrs = []
        for i, ftrue in enumerate(freqs):
            eeg = data[ftrue]
            if add_noise:
                eeg = eeg + rng.normal(0, add_noise, eeg.shape)
            rhos = np.array([cca_rho(eeg, refs[f]) for f in freqs])
            rho_all[i] = rhos
            pred = freqs[int(np.argmax(rhos))]
            hit += (abs(pred - ftrue) < 1e-6)
            snrs.append(band_snr(eeg, ftrue, fs))
        acc = hit / len(freqs)
        print(f"    {tag}: top-1 accuracy {hit}/{len(freqs)} = {acc:.0%} | "
              f"median in-band SNR {np.median(snrs):.1f} dB "
              f"(min {min(snrs):.1f}, max {max(snrs):.1f})")
        if add_noise == 0.0:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            im = ax[0].imshow(rho_all, cmap="viridis")
            ax[0].set_xticks(range(len(freqs)))
            ax[0].set_xticklabels(freqs, rotation=45, fontsize=7)
            ax[0].set_yticks(range(len(freqs)))
            ax[0].set_yticklabels(freqs, fontsize=7)
            ax[0].set_xlabel("candidate f (Hz)")
            ax[0].set_ylabel("true f (Hz)")
            ax[0].set_title("CCA rho: true vs candidate")
            plt.colorbar(im, ax=ax[0], shrink=0.8)
            e0 = data[freqs[0]]
            W = np.fft.rfft(e0 - e0.mean(axis=1, keepdims=True), axis=1)
            P = 10 * np.log10(2 * np.abs(W) ** 2 / T ** 2 + 1e-30)
            fb = np.fft.rfftfreq(T, d=1 / fs)
            ax[1].semilogx(np.maximum(fb[1:], 0.15),
                           P.mean(axis=0)[1:], lw=0.8)
            for f in freqs:
                ax[1].axvline(f, color="r", alpha=0.3, lw=0.7)
            ax[1].set_xlim(0.15, 40)
            ax[1].set_xlabel("Hz"); ax[1].set_ylabel("PSD (dB)")
            ax[1].set_title(f"trial f={freqs[0]} Hz: PSD + flicker freqs")
            fig.tight_layout()
            fig.savefig(OUT / "ssvep_detection.png", dpi=150)
            plt.close(fig)
    print(f"[ssvep] figure -> {OUT / 'ssvep_detection.png'}")


if __name__ == "__main__":
    main()
