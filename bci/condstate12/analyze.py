"""bci/condstate12: the per-state scalp detectability account.

Zero-acquisition analysis of existing scalp recordings: the SAME A+
drive (KCg-m 1600 pA + DAN_err 800 pA) delivered in four network
states -- naive, quiet lock, latch residual, high lock.  Question: how
much does the state multiply the scalp signature of an identical
driven event, and what does that do to the trials-to-d' budget?

Metric (pitfall-#10-safe): per-channel 2-20 Hz band RMS inside the
training window (no |mean| phase cancellation), then best-electrode
(max) and mean across the 17 channels.  Trials-to-d' uses the
project's convention N = (d' * bg_rms / sig_rms)^2 with d'=2 and the
background model's 11.4 uV.

Arms (all seed 62; train windows differ per arc):
  condstate1 ctl/locked    train [2000,4000]   (t_end 6 s)
  condstate2 ctl62/lock62  train [15000,17000]
  condstate3 ctl62/res62   train [17000,19000] (latch residual)
  condstate4 ctl62/hl62    train [17000,19000] (high lock)
Baselines: the 2-s window immediately before each train window
(pre-drive, same trial, same state).

Usage (conda ffbm, repo root):
    python bci/condstate12/analyze.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BG_UV = 11.4
FS_MS = 1.0          # scalp records are 1 per ms
ARMS = {
    "naive_c1":      (ROOT / "bci/condstate1/outputs/ctl_scalp.npy",
                      2000, 4000),
    "quiet_lock_c1": (ROOT / "bci/condstate1/outputs/locked_scalp.npy",
                      2000, 4000),
    "naive_c2":      (ROOT / "bci/condstate2/outputs/ctl62_scalp.npy",
                      15000, 17000),
    "quiet_lock_c2": (ROOT / "bci/condstate2/outputs/lock62_scalp.npy",
                      15000, 17000),
    "residual_c3":   (ROOT / "bci/condstate3/outputs/res62_scalp.npy",
                      17000, 19000),
    "high_lock_c4":  (ROOT / "bci/condstate4/outputs/hl62_scalp.npy",
                      17000, 19000),
    "naive_c3":      (ROOT / "bci/condstate3/outputs/ctl62_scalp.npy",
                      17000, 19000),
    "naive_c4":      (ROOT / "bci/condstate4/outputs/ctl62_scalp.npy",
                      17000, 19000),
}


def band_rms(e, t0, t1):
    """Per-channel 2-20 Hz band RMS (uV) in [t0,t1) ms; e is
    (n_records, n_ch) in V."""
    sos = butter(4, [2.0, 20.0], btype="bandpass", fs=1000.0,
                 output="sos")
    x = e[t0:t1] * 1e6
    xf = sosfiltfilt(sos, x, axis=0)
    return np.sqrt((xf ** 2).mean(axis=0))


def main():
    res = {}
    print(f"{'arm':15s} {'win_rms(uV)':>26s} {'pre_rms':>16s} "
          f"{'trials(best)':>13s}")
    print(f"{'':15s} {'best':>12s} {'mean':>12s}")
    for tag, (p, t0, t1) in ARMS.items():
        if not p.exists():
            print(f"{tag:15s} missing")
            continue
        e = np.load(p)
        if e.ndim == 3:                 # (n_chan?, n?, ?) guard
            e = e.reshape(e.shape[-2], e.shape[-1])
        dur = e.shape[0]
        t1c, t0c = min(t1, dur), min(t0, dur)
        pre0, pre1 = max(0, t0c - (t1c - t0c)), t0c
        w = band_rms(e, t0c, t1c)
        pre = band_rms(e, pre0, pre1)
        sig_best, sig_mean = float(w.max()), float(w.mean())
        pre_best = float(pre.max())
        trials_best = (2.0 * BG_UV / sig_best) ** 2 if sig_best > 0 \
            else float("inf")
        res[tag] = {"win_best_uv": round(sig_best, 4),
                    "win_mean_uv": round(sig_mean, 4),
                    "pre_best_uv": round(pre_best, 4),
                    "trials_best": int(trials_best)}
        print(f"{tag:15s} {sig_best:12.4f} {sig_mean:12.4f} "
              f"{pre_best:10.4f} {int(trials_best):13,d}")
    # state multipliers vs the naive reference of the same arc
    print("\n== state multipliers (best-electrode band RMS, trials) ==")
    for arc, naive, state in (("c1", "naive_c1", "quiet_lock_c1"),
                              ("c2", "naive_c2", "quiet_lock_c2"),
                              ("c3", "naive_c3", "residual_c3"),
                              ("c4", "naive_c4", "high_lock_c4")):
        if naive in res and state in res:
            m_amp = res[state]["win_best_uv"] / res[naive]["win_best_uv"]
            m_tr = res[naive]["trials_best"] / res[state]["trials_best"]
            res[f"mult_{arc}"] = {"amp": round(m_amp, 2),
                                  "trials": round(m_tr, 2)}
            print(f"  {state:15s} vs {naive}: amplitude x{m_amp:.2f}, "
                  f"trials /{m_tr:.1f}")
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"\n[condstate12] summary -> "
          f"{HERE / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
