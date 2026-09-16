"""SSVEP detection analysis: canonical-CCA over the acquired trials
(the standard Wang-2016-style detector) + flicker-frequency SNR.

Supports multi-repeat acquisition (acquire.py --repeats k): each class
has k trials with DIFFERENT RNG seeds, i.e. fresh OU-background/delay
realizations -- the trial-to-trial variability real recordings have.
Reports, per noise condition:
  - per-repeat top-1 accuracy + confusion-RATE matrix (aggregated over
    repeats);
  - mean CCA-rho matrix (near-miss structure);
  - accuracy vs number of averaged trials (response survives averaging,
    noise falls as sqrt(k)).

Usage: python bci/ssvep_benchmark/analyze.py
"""
import json
from itertools import combinations
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
    sig = Pm[max(i0 - 1, 1):i0 + 2].max()
    neigh = (fb < f - 0.7) | (fb > f + 0.7)
    noise = np.median(Pm[neigh & (fb > 2) & (fb < 40)])
    return 10 * np.log10(sig / max(noise, 1e-30))


def classify(eeg, refs, freqs):
    rhos = np.array([cca_rho(eeg, refs[f]) for f in freqs])
    return freqs[int(np.argmax(rhos))], rhos


def conf_fig(tag, freqs, rates, rho_mean, suffix, acc_str):
    n = len(freqs)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.8))
    step = 4 if n > 10 else 1
    im = ax[0].imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for a in (ax[0], ax[1]):
        a.set_xticks(range(0, n, step))
        a.set_xticklabels(freqs[::step], rotation=45, fontsize=6)
        a.set_yticks(range(0, n, step))
        a.set_yticklabels(freqs[::step], fontsize=6)
        a.set_xlabel("predicted f (Hz)", fontsize=8)
        a.set_ylabel("true f (Hz)", fontsize=8)
    ax[0].set_title(f"confusion RATE, {tag}: {acc_str}", fontsize=9)
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    im2 = ax[1].imshow(rho_mean, cmap="viridis", aspect="auto")
    ax[1].set_title("mean CCA rho per (true class, candidate f)",
                    fontsize=9)
    plt.colorbar(im2, ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_matrix_{suffix}.png", dpi=150)
    plt.close(fig)
    np.save(OUT / f"rho_matrix_{suffix}.npy", rho_mean)
    np.save(OUT / f"confusion_rates_{suffix}.npy", rates)


def main():
    meta = json.loads((OUT / "meta.json").read_text())
    freqs = meta["freqs"]
    fs = meta["fs"]
    reps = int(meta.get("repeats", 1))
    seg = (np.arange(int(1.0 * fs), int(9.0 * fs)))   # 1-9 s window
    t = seg / fs
    data = {}
    for f in freqs:
        stems = ([f"eeg_f{f:.1f}"] if reps == 1 else
                 [f"eeg_f{f:.1f}_r{r}" for r in range(reps)])
        data[f] = [np.load(OUT / f"{s}.npy").T[:, seg] for s in stems]
    n_ch, T = data[freqs[0]][0].shape

    refs = {f: harmonic_ref(f, t) for f in freqs}
    print(f"[ssvep] {len(freqs)} classes x {reps} repeats x {n_ch} ch "
          f"x {T} samples ({t[-1] - t[0]:.0f} s window)")

    fly_ac = float(np.median([
        np.sqrt(float(((e - e.mean(axis=1, keepdims=True)) ** 2).mean()))
        for v in data.values() for e in v]))
    noise_uv = 1.5 * fly_ac / 3.79          # clinical ratio, fly-scaled
    print(f"[ssvep] fly AC rms {fly_ac * 1e6:.4f} uV -> scaled "
          f"hypothetical equipment noise {noise_uv * 1e6:.4f} uV "
          f"(same sensor/signal ratio as clinical human 1.5/3.79)")

    for tag, add_noise in (
            ("clean", 0.0),
            (f"scaled sensor noise ({noise_uv * 1e6:.4f} uV)", noise_uv)):
        conf_counts = np.zeros((len(freqs), len(freqs)))
        rho_sum = np.zeros_like(conf_counts)
        hit, total, snrs = 0, 0, []
        acc_curve = {}                        # m -> [correct? per combo]
        for i, ftrue in enumerate(freqs):
            reps_eeg = []
            for e in data[ftrue]:
                if add_noise:
                    e = e + rng.normal(0, add_noise, e.shape)
                reps_eeg.append(e)
                pred, rhos = classify(e, refs, freqs)
                conf_counts[i, freqs.index(pred)] += 1
                rho_sum[i] += rhos
                hit += (pred == ftrue)
                total += 1
                snrs.append(band_snr(e, ftrue, fs))
            if reps > 1:
                for m in range(2, reps + 1):
                    for combo in combinations(range(reps), m):
                        e_avg = np.mean([reps_eeg[j] for j in combo],
                                        axis=0)
                        pred, _ = classify(e_avg, refs, freqs)
                        acc_curve.setdefault(m, []).append(pred == ftrue)
        rates = conf_counts / max(reps, 1)
        rho_mean = rho_sum / total * len(freqs)
        acc_str = f"{hit}/{total} = {hit / total:.0%}"
        print(f"    {tag}: top-1 accuracy {acc_str} | "
              f"median in-band SNR {np.median(snrs):.1f} dB "
              f"(min {min(snrs):.1f}, max {max(snrs):.1f})")
        if acc_curve:
            msg = " | ".join(
                f"k={m}: {np.mean(v):.0%}" for m, v in sorted(
                    acc_curve.items()))
            print(f"    {tag} accuracy vs averaged trials: {msg}")
        suffix = "clean" if add_noise == 0.0 else "noise"
        conf_fig(tag, freqs, rates, rho_mean, suffix, acc_str)

    print(f"[ssvep] figures -> confusion_matrix_clean.png, "
          f"confusion_matrix_noise.png (+ rho/rate .npy)")


if __name__ == "__main__":
    main()
