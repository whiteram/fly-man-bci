"""exp018: EDA of the simulated scalp EEG -- distributions, channel
differences, and whether stimulus-related information is extractable.

Input: the current public baseline export viz/data/viz_data.json
(demo_bounce video, full CNS, 45 leads).  The demo stimulus generator is
deterministic, so ball-bounce event times are recomputed exactly and used
as ERP locks.

Outputs (outputs/):
  f01_amplitude.png   pooled histogram + per-channel box plot
  f02_topography.png  RMS and PC1-PC3 head maps
  f03_structure.png   channel correlation matrix + PCA variance spectrum
  f04_spectra.png     Welch PSD (clean vs background) + spectrogram
  f05_chain.png       lagged cross-correlations stim->R->L->MID->T4/T5->EEG
  f06_erp.png         bounce-locked ERP (+ circular-shift null)
  f07_decode.png      ridge decoding CV R2 (clean / +bg / shifted null)

Run:  python experiments/exp018_eeg_eda/run.py   (conda ffbm env)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(18)

d = json.loads((ROOT / "viz" / "data" / "viz_data.json").read_text())
meta = d["meta"]
t = np.asarray(d["t_ms"], float)
X = np.asarray(d["phi_scalp_all_uV"], float)          # (n_ch, T)
BG = np.asarray(d["phi_scalp_bg_uV"], float)
names = list(meta["scalp"]["elec_names"])
dirs = np.asarray(meta["scalp"]["elec_dirs"], float)
stim = np.asarray(d["stim"], float)
rates = {k: np.asarray(v, float) for k, v in d["rates"].items()}
n_ch, n_t = X.shape
fs = 1000.0 / np.median(np.diff(t))
print(f"[load] {n_ch} channels x {n_t} samples, fs={fs:.0f} Hz "
      f"({t[-1] / 1000:.1f} s), visual input "
      f"{meta['visual_input']['id']}")
print(f"[load] background EEG params: {meta.get('bg_eeg')}")

# ---------------------------------------------------------------- 1
# amplitude distributions
rms = np.sqrt((X ** 2).mean(axis=1))
bg_rms = np.sqrt((BG ** 2).mean(axis=1))
print("\n[1] amplitude (uV): fly signal RMS per channel "
      f"min {rms.min():.4f} / median {np.median(rms):.4f} / "
      f"max {rms.max():.4f}; background RMS median "
      f"{np.median(bg_rms):.2f} -> fly/bg ratio ~"
      f"{np.median(rms) / np.median(bg_rms):.3f}")
order = np.argsort(rms)[::-1]
print("    top-5 channels:", ", ".join(
    f"{names[i]}({rms[i]:.4f})" for i in order[:5]))
print("    bottom-5:", ", ".join(
    f"{names[i]}({rms[i]:.4f})" for i in order[-5:]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(X.ravel(), bins=120)
ax[0].set_yscale("log")
ax[0].set_xlabel("phi (uV)"); ax[0].set_title("all samples, all channels")
data = [X[i] for i in order]
ax[1].boxplot(data, tick_labels=[names[i] for i in order], showfliers=False)
ax[1].set_ylabel("phi (uV)"); ax[1].set_title("per channel (sorted by RMS)")
ax[1].tick_params(axis="x", labelsize=4, rotation=90)
fig.tight_layout(); fig.savefig(OUT / "f01_amplitude.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 2
# topography: 2D embedding of electrode directions (cap is roughly
# planar -> SVD projection preserves the layout)
c = dirs - dirs.mean(axis=0)
_, _, vt = np.linalg.svd(c, full_matrices=False)
proj = c @ vt[:2].T
Xc = X - X.mean(axis=1, keepdims=True)
U, S, _ = np.linalg.svd(Xc, full_matrices=False)
lam = S ** 2
expl = lam / lam.sum()
pr = lam.sum() ** 2 / (lam ** 2).sum()          # participation ratio
print(f"\n[2] PCA: PC1 {expl[0]:.1%}, PC2 {expl[1]:.1%}, "
      f"PC3 {expl[2]:.1%}; participation rank {pr:.1f} of {n_ch}")

def headmap(ax, v, title, cmap="viridis"):
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=v, cmap=cmap, s=90,
                    edgecolors="k", linewidths=0.4)
    ax.set_title(title); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, shrink=0.8)
    for i in np.argsort(-np.abs(v))[:3]:
        ax.annotate(names[i], proj[i], fontsize=7)

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
headmap(ax[0], rms, "RMS (uV)")
for k in range(3):
    headmap(ax[k + 1], U[:, k], f"PC{k + 1} ({expl[k]:.0%})",
            cmap="RdBu_r")
fig.suptitle("head topography (SVD projection of electrode directions)")
fig.tight_layout(); fig.savefig(OUT / "f02_topography.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 3
# inter-channel structure
C = np.corrcoef(X)
off = C[np.triu_indices(n_ch, 1)]
print(f"\n[3] channel correlations: median {np.median(off):.3f}, "
      f"|r|>0.5 pairs {np.mean(np.abs(off) > 0.5):.0%}, "
      f"|r|>0.9 pairs {np.mean(np.abs(off) > 0.9):.0%}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
im = ax[0].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
ax[0].set_title("channel correlation matrix")
plt.colorbar(im, ax=ax[0], shrink=0.8)
ax[1].semilogy(np.arange(1, n_ch + 1), lam / lam.sum(), "o-")
ax[1].set_xlabel("component"); ax[1].set_ylabel("explained variance")
ax[1].set_title(f"PCA spectrum (participation rank {pr:.1f})")
fig.tight_layout(); fig.savefig(OUT / "f03_structure.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 4
# spectra
f, P = sps.welch(X, fs=fs, nperseg=2048)
_, Pb = sps.welch(BG, fs=fs, nperseg=2048)
best = order[0]
band = (f >= 0.5) & (f <= 100)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].loglog(f[band], P[best][band], label=f"fly {names[best]}")
ax[0].loglog(f[band], np.median(P, axis=0)[band], label="fly median")
ax[0].loglog(f[band], np.median(Pb, axis=0)[band], label="bg median",
             alpha=0.7)
ax[0].axvspan(8, 13, color="C1", alpha=0.15, label="alpha band (bg)")
ax[0].set_xlabel("Hz"); ax[0].set_ylabel("uV^2/Hz")
ax[0].legend(fontsize=8); ax[0].set_title("Welch PSD")
ff, tt, Sxx = sps.spectrogram(X[best], fs=fs, nperseg=256)
keep = ff <= 40
ax[1].pcolormesh(tt, ff[keep], 10 * np.log10(Sxx[keep] + 1e-12),
                 shading="auto", cmap="magma")
ax[1].set_xlabel("s"); ax[1].set_ylabel("Hz")
ax[1].set_title(f"spectrogram {names[best]} (0-40 Hz)")
fig.tight_layout(); fig.savefig(OUT / "f04_spectra.png", dpi=150)
plt.close(fig)
for lo, hi in ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 80)):
    m = (f >= lo) & (f < hi)
    print(f"    band {lo}-{hi} Hz: fly {P[:, m].sum(1).mean():.3g} "
          f"vs bg {Pb[:, m].sum(1).mean():.3g} uV^2")

# ---------------------------------------------------------------- 5
# stimulus -> cascade -> EEG lagged cross-correlations
def xcorr_lag(a, b, lags):
    """corr(a(t), b(t+l)) for each lag l (ms): b lags a."""
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    out = []
    for l in lags:
        if l >= 0:
            x, y = a[:len(a) - l], b[l:]
        else:
            x, y = a[-l:], b[:len(b) + l]
        out.append(np.corrcoef(x, y)[0, 1] if len(x) > 10 else np.nan)
    return np.array(out)

lags = np.arange(0, 401)
chain = [("stim", stim), ("R", rates["R"]), ("L", rates["L"]),
         ("MID", rates["MID"]), ("T4", rates["T4"]), ("T5", rates["T5"])]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, (nm, src) in enumerate(chain[1:], 1):
    r = xcorr_lag(chain[i - 1][1], src, lags)
    pk = np.nanargmax(np.abs(r))
    print(f"[5] lag-xcorr {chain[i - 1][0]}->{nm}: peak |r| "
          f"{np.abs(r[pk]):.3f} at {lags[pk]} ms")
    ax[0].plot(lags, r / np.abs(r[pk]), label=chain[i - 1][0] + "->" + nm)
ax[0].set_xlabel("lag (ms)"); ax[0].set_ylabel("|r| (norm)")
ax[0].legend(fontsize=7); ax[0].set_title("cascade stage-to-stage lag")

ch_r = np.zeros(n_ch); ch_lag = np.zeros(n_ch, int)
for i in range(n_ch):
    r = xcorr_lag(stim, X[i], lags)
    pk = np.nanargmax(np.abs(r)); ch_r[i] = r[pk]; ch_lag[i] = lags[pk]
print(f"    stim->EEG per channel: |r| median {np.median(np.abs(ch_r)):.3f} "
      f"max {np.abs(ch_r).max():.3f} ({names[np.argmax(np.abs(ch_r))]}); "
      f"best-lag median {np.median(ch_lag[np.abs(ch_r) > 0.2]) if (np.abs(ch_r) > 0.2).any() else float('nan'):.0f} ms")
ax[1].hist(ch_lag[np.abs(ch_r) > 0.1], bins=20)
ax[1].set_xlabel("best lag (ms)"); ax[1].set_ylabel("# channels")
ax[1].set_title("stim->EEG best lag distribution")
fig.tight_layout(); fig.savefig(OUT / "f05_chain.png", dpi=150)
plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
headmap(ax[0], ch_r, "stim->EEG peak r", cmap="RdBu_r")
headmap(ax[1], ch_lag.astype(float), "stim->EEG best lag (ms)")
fig.tight_layout(); fig.savefig(OUT / "f05_chain_map.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 6
# bounce-locked ERP + circular-shift null
FPS, HH, WW, ball_r = 30.0, 72, 96, 7.0
p, v = np.array([20.0, 40.0]), np.array([2.6, 1.9])
ev_frames = []
for fr in range(int(10.5 * FPS)):
    p = p + v
    for axi, lim in ((0, HH), (1, WW)):
        if p[axi] < ball_r or p[axi] > lim - ball_r:
            v[axi] *= -1
            p[axi] = np.clip(p[axi], ball_r, lim - ball_r)
            ev_frames.append(fr)
ev_ms = np.array(sorted(set(ev_frames))) * 1000.0 / FPS
pre, post = -300.0, 700.0
ev_ok = ev_ms[(ev_ms - pre >= 0) & (ev_ms + post < n_t)]
win = np.arange(pre, post)
E = np.stack([X[:, int(e + pre):int(e + post)] for e in ev_ok])
base_m = (win >= -300) & (win <= -50)
E = E - E[:, :, base_m].mean(axis=2, keepdims=True)
gm = E.mean(axis=0)                                    # (n_ch, T_win)
pk_win = (win >= 0) & (win <= 300)
amp = gm[:, pk_win].max(axis=1) - gm[:, pk_win].min(axis=1)
bi = np.argmax(amp)
print(f"\n[6] {len(ev_ok)} bounce events; grand-mean ERP peak-to-peak "
      f"(0-300 ms) max {amp.max():.4f} uV at {names[bi]} "
      f"(median {np.median(amp):.4f})")

def mean_curve(times):
    Ep = np.stack([X[:, int(e + pre):int(e + post)] for e in times])
    Ep = Ep - Ep[:, :, base_m].mean(axis=2, keepdims=True)
    return Ep.mean(axis=0)

null_max = np.array([np.abs(mean_curve(
    rng.uniform(400, n_t - 800, len(ev_ok)))).max() for _ in range(500)])
pv = np.mean(null_max >= amp[bi])
print(f"    circular-shift null (500 draws): observed max-channel "
      f"|ERP| z={(amp[bi] - null_max.mean()) / null_max.std():.1f}, "
      f"p={pv:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
boot = np.array([E[rng.integers(0, len(E), len(E))][:, bi].mean(0)
                 for _ in range(1000)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
ax[0].plot(win, gm[bi], lw=1.2, color="C0", label=f"{names[bi]} ERP")
ax[0].fill_between(win, lo, hi, alpha=0.25, color="C0",
                   label="95% bootstrap CI")
ax[0].axvline(0, color="k", ls=":", lw=0.8)
ax[0].axhline(0, color="gray", lw=0.5)
ax[0].set_xlabel("ms after bounce"); ax[0].set_ylabel("uV")
ax[0].legend(fontsize=8); ax[0].set_title(
    f"bounce-locked ERP, n={len(ev_ok)} events")
headmap(ax[1], amp, "ERP peak-to-peak 0-300 ms (uV)")
fig.tight_layout(); fig.savefig(OUT / "f06_erp.png", dpi=150)
plt.close(fig)

# ERP with realistic background mixed in (trial averaging)
Y = X + BG
EY = np.stack([Y[:, int(e + pre):int(e + post)] for e in ev_ok])
EY = EY - EY[:, :, base_m].mean(axis=2, keepdims=True)
snr_gain = (amp[bi] / EY.std())
print(f"    with background added: single-event ERP amplitude "
      f"{amp[bi]:.4f} uV vs single-event noise SD {EY[:, bi].std():.2f} uV "
      f"-> single-trial SNR {snr_gain:.3f}; n-avg gain -> "
      f"{snr_gain * np.sqrt(len(ev_ok)):.2f} sigma at n={len(ev_ok)}")

# ---------------------------------------------------------------- 7
# decoding: ridge from lagged EEG PCs -> target, time-series CV
def build_feats(sig2d, n_comp=8, tap_ms=150, step_ms=5, proj=None):
    Xc2 = sig2d - sig2d.mean(axis=1, keepdims=True)
    if proj is None:
        Uu, _, _ = np.linalg.svd(Xc2.T, full_matrices=False)
        comps = Uu[:, :n_comp]                          # (T, n_comp)
    else:
        # fixed model-informed spatial filters (e.g. clean-signal PCs):
        # at low SNR the internal PCA would lock onto noise modes and
        # forfeit the ~sqrt(n_ch) coherent gain against white noise
        comps = Xc2.T @ proj                            # (T, k)
    L = tap_ms // step_ms + 1
    W = np.lib.stride_tricks.sliding_window_view(comps, L, axis=0)
    W = W.reshape(W.shape[0], -1)
    return W, comps

def decode(sig2d, target, tag, proj=None):
    F, comps = build_feats(sig2d, proj=proj)
    y = (target - target.mean()) / target.std()
    y = y[-F.shape[0]:]
    # features at time t contain comps[t-L+1..t]; predict target at t
    # (EEG lags stimulus, so past window -> present target is causal)
    cv = TimeSeriesSplit(n_splits=6)
    model = RidgeCV(alphas=np.logspace(-3, 4, 15))
    sc = []
    for tr, te in cv.split(F):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        model.fit((F[tr] - mu) / sd, y[tr])
        pred = model.predict((F[te] - mu) / sd)
        sc.append(r2_score(y[te], pred))
    print(f"[7] decode {tag}: CV R2 = {np.mean(sc):.3f} "
          f"+/- {np.std(sc):.3f} (splits {np.round(sc, 3)})")
    return np.mean(sc)

r_clean = decode(X, stim, "clean -> stim")
r_bg = decode(X + BG, stim, "clean+bg -> stim")
y_shift = np.roll(stim, 3500)
r_null = decode(X, y_shift, "null (shifted stim)")
r_rate = decode(X, rates["R"], "clean -> R-rate")
# robustness: band-pass BOTH features and target (0.5-6 Hz) -- if the
# broadband R2 is only a sub-0.1 Hz clip-long co-trend, this collapses.
# filtfilt rings at the record edges for ~1/f0 seconds (0.5 Hz highpass
# pole): trim 2.5 s per side before fitting so CV scores no edge
# transient
sos_d = sps.butter(2, [0.5, 6.0], btype="band", fs=fs, output="sos")
TR = 2500
Xf = sps.sosfiltfilt(sos_d, X - X.mean(axis=1, keepdims=True),
                     axis=1)[:, TR:-TR]
r_band = decode(Xf, sps.sosfiltfilt(sos_d, stim)[TR:-TR],
                "clean 0.5-6 Hz -> stim (band-passed, edges trimmed)")
r_band_bg = decode(
    sps.sosfiltfilt(sos_d, X + BG - (X + BG).mean(
        axis=1, keepdims=True), axis=1)[:, TR:-TR],
    sps.sosfiltfilt(sos_d, stim)[TR:-TR],
    "clean+bg 0.5-6 Hz -> stim (band-passed, edges trimmed)")

fig, ax = plt.subplots(figsize=(7.5, 4))
vals = [r_clean, r_bg, r_null, r_rate, r_band, r_band_bg]
labs = ["clean->stim", "clean+bg->stim", "null->stim", "clean->R-rate",
        "clean .5-6Hz->stim", "+bg .5-6Hz->stim"]
ax.bar(range(len(vals)), vals, color=["C0", "C1", "C2", "C3", "C4", "C5"])
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labs, fontsize=8, rotation=20)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("time-series CV R^2")
ax.set_title("ridge decoding, 8 PCs x 150 ms taps")
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.3f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "f07_decode.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 8
# realistic-SNR extractability: the fly power lives at low frequency
# (bar sweep ~0.27 Hz, ball trajectories ~1 Hz), where the background
# 1/f component is strongest -- so the band choice is a trade-off.
# Evaluate candidate bands x (spatial PC1-4 projection | best single
# channel), and judge the mixed-signal correlation with a
# circular-shift permutation test (autocorrelation-safe; ms-sample
# Fisher z would be wildly liberal here).
Vsp = U[:, :4]                                  # spatial filters (clean PCs)
Ymix = X + BG
bi_ch = np.argmax(np.abs(ch_r))                 # best single channel (sec 5)
rows = []
for lo, hi in ((0.1, 6.0), (0.5, 6.0), (1.0, 6.0), (0.5, 4.0)):
    sos8 = sps.butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    p_clean = sps.sosfiltfilt(sos8, X.T @ Vsp, axis=0)
    p_mix = sps.sosfiltfilt(sos8, Ymix.T @ Vsp, axis=0)
    p_bg = sps.sosfiltfilt(sos8, BG.T @ Vsp, axis=0)
    rows.append((f"{lo}-{hi} Hz", "PC1-4", p_clean, p_mix, p_bg))
    rows.append((f"{lo}-{hi} Hz", f"1ch {names[bi_ch]}",
                 sps.sosfiltfilt(sos8, X[bi_ch][:, None], axis=0),
                 sps.sosfiltfilt(sos8, Ymix[bi_ch][:, None], axis=0),
                 sps.sosfiltfilt(sos8, BG[bi_ch][:, None], axis=0)))

def best_r(target, proj, lags):
    out = np.zeros(len(lags))
    for i, l in enumerate(lags):
        out[i] = max(abs(np.corrcoef(target[:len(target) - l],
                                     proj[l:, k])[0, 1])
                     for k in range(proj.shape[1]))
    return out


lags8 = np.arange(0, 301)
print("\n[8] band x spatial-filter sweep (single 10.5 s trial, "
      "permutation p over 200 circular shifts):")
best_row, best_val = None, -1
for band, tag, cl, mx, bg in rows:
    r_c = best_r(stim, cl, lags8)
    r_m = best_r(stim, mx, lags8)
    r_b = best_r(stim, bg, lags8)
    null = np.array([best_r(np.roll(stim, rng.integers(500, len(stim) - 500)),
                            mx, lags8).max() for _ in range(200)])
    pv = np.mean(null >= r_m.max())
    print(f"    {band:>10} {tag:>10}: clean |r| {r_c.max():.3f} @"
          f"{lags8[np.argmax(r_c)]:3d} ms | mixed {r_m.max():.3f} "
          f"(bg-control {r_b.max():.3f}) p={pv:.3f}")
    if r_m.max() - r_b.max() > best_val:
        best_val, best_row = r_m.max() - r_b.max(), (band, tag, r_c, r_m)

band, tag, r_c, r_m = best_row
print(f"    best mixed-vs-bg margin: {band} {tag} "
      f"(clean {r_c.max():.3f}, mixed {r_m.max():.3f})")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(lags8, r_c, label="clean")
ax[0].plot(lags8, r_m, label="mixed (single trial)")
ax[0].axhline(0, color="gray", lw=0.5)
ax[0].set_xlabel("lag (ms)"); ax[0].set_ylabel("|r| with stimulus")
ax[0].legend(fontsize=8)
ax[0].set_title(f"stim correlation, {band} {tag} filtering")
ax[1].plot(t / 1000, (stim - stim.mean()) / stim.std(), label="stim (z)",
           lw=0.8)
sos8 = sps.butter(2, [float(band.split("-")[0]),
                      float(band.split("-")[1][:-3])], btype="band",
                  fs=fs, output="sos")
pc1 = sps.sosfiltfilt(sos8, X.T @ Vsp[:, :1], axis=0)[:, 0]
pm1 = sps.sosfiltfilt(sos8, Ymix.T @ Vsp[:, :1], axis=0)[:, 0]
ax[1].plot(t / 1000, (pc1 - pc1.mean()) / pc1.std(), lw=0.8,
           label="clean PC1 filtered (z)")
ax[1].plot(t / 1000, (pm1 - pm1.mean()) / pm1.std(), alpha=0.5, lw=0.7,
           label="mixed PC1 filtered (z)")
ax[1].set_xlabel("s"); ax[1].legend(fontsize=8)
ax[1].set_title(f"time courses ({band})")
fig.tight_layout(); fig.savefig(OUT / "f08_realistic.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 9
# background-level sweep x band: the premise question. The stored
# background is human-literature EEG -- its alpha and 1/f components
# are NEURAL, produced by a human cortex that is still there ("fly
# implanted in an intact human head", k=1).  If the fly network
# REPLACES the brain, those vanish and only equipment noise remains
# (k=0 + white 1.5 uV sensor floor).  The fly power concentrates
# below 0.5 Hz, so the optimal band depends on the premise: white
# sensor noise is flat (favor a LOW band), human 1/f rises toward DC
# (kills the low band).  Sweep k x band.
sensor = rng.normal(0, 1.5, X.shape)          # equipment floor (white)
Vsp9 = U[:, :4]
lags9 = np.arange(0, 201, 10)                 # sparse grid for permutations


def perm_p(mix2d, band):
    sos9 = sps.butter(2, list(band), btype="band", fs=fs, output="sos")
    p_mix = sps.sosfiltfilt(sos9, mix2d.T @ Vsp9, axis=0)
    r_m = best_r(stim, p_mix, lags9)
    null = np.array([best_r(np.roll(stim, rng.integers(500, len(stim) - 500)),
                            p_mix, lags9).max() for _ in range(100)])
    return float(r_m.max()), float(np.mean(null >= r_m.max()))


print("\n[9] background-scale x band sweep (single 10.5 s trial):")
print("    k=1: intact-human-head premise | k=0: fly replaces brain "
      "(sensor noise only)")
# SNR columns are computed on PERIODOGRAM band powers (Parseval-exact,
# no filtfilt edge ringing): project DC-removed signals on the clean
# PCs, then take in-band rms.  The white sensor noise competes only
# with its IN-BAND share (1.5 uV spread over ~500 Hz).
Xc9 = Xc                                      # DC-removed clean signal

def band_rms_max(series2d, band):
    """Max over series of the in-band rms; accepts (n, T) or (T, k)."""
    s = np.atleast_2d(series2d)
    if s.shape[0] > s.shape[1]:        # time on axis 0 -> transpose
        s = s.T
    W = np.fft.rfft(s - s.mean(axis=1, keepdims=True), axis=1)
    P = 2 * np.abs(W) ** 2 / s.shape[1] ** 2
    fb = np.fft.rfftfreq(s.shape[1], d=0.001)
    P, fb = P[:, 1:], fb[1:]           # drop DC bin
    m = (fb >= band[0]) & (fb < band[1])
    return float(np.sqrt(P[:, m].sum(axis=1).max()))


rows9 = []
for k in (1.0, 0.3, 0.1, 0.03, 0.0):
    mix = X + k * BG + (sensor if k == 0.0 else 0.0)
    for band in ((0.1, 1.0), (0.5, 6.0)):
        sos9 = sps.butter(2, list(band), btype="band", fs=fs, output="sos")
        mix_f = sps.sosfiltfilt(sos9, mix, axis=1)
        stim_fb = sps.sosfiltfilt(sos9, stim)
        # fixed clean-signal spatial filters for every k (fair across
        # premises; README flags the model-informed caveat)
        r2 = decode(mix_f, stim_fb, f"k={k:g} {band[0]}-{band[1]}Hz -> stim",
                    proj=U[:, :8])
        rbest, pv = perm_p(mix, band)
        noi2d = k * BG + (sensor if k == 0.0 else 0.0)
        r_ch = band_rms_max(Xc9, band) / max(band_rms_max(noi2d, band),
                                             1e-12)
        rp = band_rms_max(Xc9.T @ U[:, :8], band) / max(
            band_rms_max(noi2d.T @ U[:, :8], band), 1e-12)
        ntri = int(np.ceil((2.0 / rp) ** 2)) if rp > 0 else 0
        rows9.append((k, band, r2, rbest, pv, r_ch, rp))
        print(f"    k={k:<5g} {band[0]:.1f}-{band[1]:.0f} Hz: "
              f"decode R2 {r2:+.3f} | best |r| {rbest:.3f} (p={pv:.2f}) "
              f"| in-band fly/noise ch {r_ch:.3f} | PC-projected {rp:.2f} "
              f"-> ~{ntri} clip-averages for SNR~2")
ok = [(k, b) for k, b, r2, r, pv, _, _ in rows9 if r2 > 0.5 and pv < 0.05]
if ok:
    print("    single-trial extraction (R2>0.5 & p<0.05) at: "
          + "; ".join(f"k={k:g} @ {b[0]:.1f}-{b[1]:.0f} Hz" for k, b in ok))
else:
    print("    single-trial extraction fails in every k x band cell")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for band, mk in (((0.1, 1.0), "o-"), ((0.5, 6.0), "s--")):
    sel = [r for r in rows9 if r[1] == band]
    ax.plot([r[0] for r in sel], [r[2] for r in sel], mk,
            label=f"{band[0]:.1f}-{band[1]:.0f} Hz")
    for r in sel:
        ax.annotate(f"p={r[4]:.2f}", (r[0], r[2]), fontsize=7,
                    textcoords="offset points", xytext=(4, 6))
ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="R2=0.5")
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("symlog", linthresh=0.03)
ax.set_xticks([r[0] for r in rows9[::2]])
ax.set_xticklabels([str(r[0]) for r in rows9[::2]])
ax.set_xlabel("background scale k (1 = full human bg, 0 = sensor only)")
ax.set_ylabel("decode CV R2")
ax.set_title("single-trial extraction vs background premise and band")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "f09_bg_sweep.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 10
# clinical-style stacked EEG traces, four conditions for visual
# comparison (what a reviewer wants to eyeball: is there anything
# EEG-like in there, and what does each premise/band look like)
def eeg_page(sig2d, title, fname, band=None, ref=None):
    t10 = t / 1000.0
    if band is not None:
        # band-pass the DC-REMOVED signal, then trim the filtfilt edge
        # transient (~1/f0 seconds per side for the low-pass pole)
        sig2d = sig2d - sig2d.mean(axis=1, keepdims=True)
        sos10 = sps.butter(2, list(band), btype="band", fs=fs,
                           output="sos")
        sig2d = sps.sosfiltfilt(sos10, sig2d, axis=1)
        lo_ms = min(1000, sig2d.shape[1] // 4)   # edge-ring trim
        sig2d = sig2d[:, lo_ms:-lo_ms]
        t10 = t[lo_ms:lo_ms + sig2d.shape[1]] / 1000.0
    n = sig2d.shape[0]
    ch_rms = np.sqrt((sig2d ** 2).mean(axis=1))
    # spacing: the noisiest channel swings ~ +-3 rms -> keep inside 0.8
    # of its slot; gain is per-panel, the printed separation is in uV
    worst = np.percentile(ch_rms, 98)
    gain = 0.8 / (3.0 * max(worst, 1e-6))
    fig, ax = plt.subplots(figsize=(14, 0.21 * n + 1.6))
    for i in range(n):
        ax.plot(t10, gain * sig2d[i] + (n - 1 - i), lw=0.55,
                color="k")
    if ref is not None:                      # stimulus brightness trace
        z = (ref - ref.mean()) / max(ref.std(), 1e-9)
        if band is None:
            ax.plot(t10, 0.4 * z + n, lw=0.8, color="C3")
        else:
            ax.plot(t10, 0.4 * z[lo_ms:lo_ms + sig2d.shape[1]] + n,
                    lw=0.8, color="C3")
        ax.axhline(n, color="gray", lw=0.4)
        ax.text(-0.4, n, "stim", ha="right", va="center", fontsize=7,
                color="C3")
    for i in range(n):
        ax.text(-0.4, n - 1 - i, names[i], ha="right", va="center",
                fontsize=6.5)
    for s in range(0, int(t10[-1]) + 1):
        ax.axvline(s, color="0.85", lw=0.4, zorder=0)
    ax.set_ylim(-0.7, n + 0.7)
    ax.set_xlim(t10[0], t10[-1])
    ax.set_yticks([])
    ax.set_xlabel("s")
    ax.set_title(title, fontsize=10)
    sep_uv = 1.0 / gain                      # uV between adjacent traces
    ax.text(0.99, 0.01,
            f"trace separation = {sep_uv:.2g} uV (scale bar = 1 slot)",
            transform=ax.transAxes, ha="right", fontsize=7, color="0.35")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.05)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print(f"[10] {fname}: sep {sep_uv:.2g} uV/trace, "
          f"median ch rms {np.median(ch_rms):.3g} uV")


sensor10 = rng.normal(0, 1.5, X.shape)       # replacement-premise floor
eeg_page(X, "clean fly signal, broadband (all 45 leads)", 
         "f10_eeg_clean_broadband.png", ref=stim)
eeg_page(X, "clean fly signal, 0.1-1 Hz (the useful band)",
         "f11_eeg_clean_band.png", band=(0.1, 1.0), ref=stim)
eeg_page(X + sensor10,
         "replacement premise: fly + 1.5 uV sensor noise, 0.1-1 Hz",
         "f12_eeg_replace.png", band=(0.1, 1.0), ref=stim)
eeg_page(X + BG,
         "counterfactual coexistence: fly + full human background, 0.1-1 Hz",
         "f13_eeg_coexist.png", band=(0.1, 1.0), ref=stim)

# ---------------------------------------------------------------- 11
# band-energy distribution vs human EEG. Full-length periodogram
# (resolution 1000/10500 = 0.095 Hz) so the sub-0.5 Hz content that
# dominates the fly signal is resolved; fractions of total power
# (0.095-500 Hz, mean over channels) per canonical band for the fly
# signal and for the literature-parameterized human background; plus
# log-log spectral slopes (aperiodic exponent).
def band_powers(sig2d):
    """Parseval-exact band powers: P = 2|X|^2/N^2, so summing all bins
    reproduces the AC variance exactly.  Also return the tonic (DC)
    per-channel offset, which real EEG separates before analysis."""
    dc = sig2d.mean(axis=1)
    Xw = np.fft.rfft(sig2d - dc[:, None], axis=1)
    n11 = sig2d.shape[1]
    f11 = np.fft.rfftfreq(n11, d=0.001)
    P = (2 * np.abs(Xw) ** 2 / n11 ** 2)[:, 1:]      # drop DC bin
    f11 = f11[1:]
    P = P[:, f11 <= 500]
    f11 = f11[f11 <= 500]
    edges = [(0, 0.1), (0.1, 0.5), (0.5, 4), (4, 8), (8, 13),
             (13, 30), (30, 80), (80, 500)]
    tot = P.sum(axis=1)
    return f11, P, tot.mean(), float(np.sqrt(np.median(dc ** 2))), [
        float(P[:, (f11 >= lo) & (f11 < hi)].sum(axis=1).mean() / tot.mean())
        for lo, hi in edges], edges


f11, Pfly, tot_f, dc_f, frac_f, edges = band_powers(X)
_, Pbg, tot_b, dc_b, frac_b, _ = band_powers(BG)
lab = ["<0.1", "0.1-0.5", "0.5-4 (delta)", "4-8 (theta)",
       "8-13 (alpha)", "13-30 (beta)", "30-80 (gamma)", "80-500"]
print("\n[11] band-energy distribution (fraction of total AC power "
      "0.095-500 Hz):")
print(f"    tonic DC offset (median): fly {dc_f:.3f} uV vs bg "
      f"{dc_b:.3f} uV -- AC rms: fly {np.sqrt(tot_f):.4f} uV vs bg "
      f"{np.sqrt(tot_b):.2f} uV")
print(f"    {'band':>14} {'fly':>8} {'human bg':>9}")
for L, a, b in zip(lab, frac_f, frac_b):
    print(f"    {L:>14} {a:8.1%} {b:9.1%}")


def slope(fv, Pv, lo, hi, drop_alpha=False):
    m = (fv >= lo) & (fv <= hi)
    if drop_alpha:
        m &= ~((fv >= 7) & (fv <= 14))
    lf, lp = np.log(fv[m]), np.log(np.median(Pv[:, m], axis=0) + 1e-30)
    return float(np.polyfit(lf, lp, 1)[0])


s_fly = slope(f11, Pfly, 0.2, 40)
s_bg = slope(f11, Pbg, 2, 40, drop_alpha=True)
print(f"    aperiodic slope (log-log): fly 0.2-40 Hz {s_fly:+.2f} | "
      f"human bg 2-40 Hz (alpha excluded) {s_bg:+.2f}")
pk_f = f11[np.argmax(np.median(Pfly, axis=0)[1:]) + 1]
al = np.median(Pbg, axis=0)[(f11 >= 8) & (f11 <= 13)]
wide = np.median(Pbg, axis=0)[(f11 >= 20) & (f11 <= 40)]
print(f"    fly dominant spectral peak: {pk_f:.2f} Hz (strongest "
      f"below 1 Hz); human bg alpha peak prominence "
      f"{al.max() / max(np.median(wide), 1e-30):.0f}x the 20-40 Hz floor")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
mf = np.median(Pfly, axis=0)
mb = np.median(Pbg, axis=0)
ax[0].loglog(f11, mf, label="fly signal")
ax[0].loglog(f11, mb, label="human bg (literature params)")
ax[0].loglog(f11, Pfly[order[0]], alpha=0.4,
             label=f"fly best ch {names[order[0]]}")
for lo, hi, c in ((0.5, 4, "C0"), (4, 8, "C1"), (8, 13, "C2"),
                  (13, 30, "C3"), (30, 80, "C4")):
    ax[0].axvspan(lo, hi, color=c, alpha=0.06)
ax[0].set_xlabel("Hz"); ax[0].set_ylabel("uV^2/Hz (median over ch)")
ax[0].legend(fontsize=8); ax[0].set_title("PSD: fly vs human")
x11 = np.arange(len(lab))
ax[1].bar(x11 - 0.2, [a * 100 for a in frac_f], 0.4, label="fly signal")
ax[1].bar(x11 + 0.2, [b * 100 for b in frac_b], 0.4, label="human bg")
ax[1].set_xticks(x11); ax[1].set_xticklabels(lab, fontsize=7,
                                              rotation=30)
ax[1].set_ylabel("% of total power"); ax[1].legend(fontsize=8)
ax[1].set_yscale("log")
ax[1].set_title("band-energy distribution")
fig.tight_layout(); fig.savefig(OUT / "f14_band_fraction.png", dpi=150)
plt.close(fig)

print("\n[done] figures + stats in", OUT)
