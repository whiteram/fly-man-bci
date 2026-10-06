"""bci/condstate13: SSVEP x state -- does the lock modulate the
project's best detection channel?

condstate12 showed in-state CHEM-event detection is state-neutral, but
the project's best channel is the visual SSVEP (0.25 uV, ~8.5e3
trials at 45 leads).  Question: with the SAME 14 Hz single-target
flicker (ssvep1_f14.0), does the network state multiply or suppress
the SSVEP scalp signature?  And does sustained flicker perturb the
lock itself (the visual axis was "safe" for lock FORMATION, but a
sustained oscillatory central drive is a different probe)?

Arms (t_end 10.5 s, seed 62, default 17-lead layout to chain with
condstate12's metrics, no plastic flags -- states are
plasticity-independent):
  ssvep_naive  ssvep1_f14.0
  ssvep_lock   + DA1@150 [300,600] ms (da1_dc60 entry, quiet lock)
  ssvep_hl     + DA1@1600 [300,600] ms (da1_d1600 entry, high lock)

Readout: ALPN lock check [4,10] s; SSVEP amplitude at 14 Hz (FFT,
best channel) in [4,10] s; SNR vs neighbor bins; trials-to-d' with
the in-window 12-16 Hz band RMS as noise.

Usage (conda ffbm, repo root):
    python bci/condstate13/acquire.py            # idempotent + readout
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,MBON,DAN,Kenyon_Cell"
F0 = 14.0
ARMS = {
    "ssvep_naive": {"chem": None,     "seed": 62},
    "ssvep_lock":  {"chem": "da1_dc60", "seed": 62},
    "ssvep_hl":    {"chem": "da1_d1600", "seed": 62},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists() and (out / f"{tag}_pop.npz").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS,
           "--visual-input", "ssvep1_f14.0",
           "--pop-rate", GROUPS,
           "--t-end", "10500.0", "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu"]
    if spec["chem"]:
        cmd += ["--chem-input", spec["chem"]]
    print(f"[acquire] {tag}: chem {spec['chem']}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def readout():
    print("== condstate13: SSVEP x state (canonical band SNR) ==")
    res = {}

    def band_snr(win_ch_first, f, fs=1000.0):
        W = np.fft.rfft(win_ch_first - win_ch_first.mean(
            axis=1, keepdims=True), axis=1)
        P = 2 * np.abs(W) ** 2 / win_ch_first.shape[1] ** 2
        fb = np.fft.rfftfreq(win_ch_first.shape[1], d=1 / fs)
        Pm = P.mean(axis=0)
        i0 = np.argmin(np.abs(fb - f))
        sig = Pm[max(i0 - 1, 1):i0 + 2].max()
        neigh = (fb < f - 0.7) | (fb > f + 0.7)
        noise = np.median(Pm[neigh & (fb > 2) & (fb < 40)])
        return 10 * np.log10(sig / max(noise, 1e-30)), \
            float(np.sqrt(2 * sig))

    for tag in ARMS:
        pop = HERE / "outputs" / f"{tag}_pop.npz"
        sc = HERE / "outputs" / f"{tag}_scalp.npy"
        if not (pop.exists() and sc.exists()):
            continue
        z = np.load(pop)
        row = {"alpn_4_10s": round(float(z["ALPN"][4000:10000].mean()), 2),
               "mbon_4_10s": round(float(z["MBON"][4000:10000].mean()), 2)}
        e = np.load(sc) * 1e6
        snr, amp = band_snr(e[4000:10000].T, F0)
        row.update({"snr_db": round(float(snr), 1),
                    "amp_uv": round(amp, 4)})
        res[tag] = row
        print(f"   {tag:12s} {row}")
    if "ssvep_naive" in res:
        n = res["ssvep_naive"]
        for tag in ("ssvep_lock", "ssvep_hl"):
            if tag in res:
                res[f"{tag}_dsnr_db"] = round(
                    res[tag]["snr_db"] - n["snr_db"], 1)
                res[f"{tag}_trials_x"] = round(
                    10 ** ((n["snr_db"] - res[tag]["snr_db"]) / 10), 2)
                print(f"   [mult] {tag}: dSNR "
                      f"{res[f'{tag}_dsnr_db']:+.1f} dB, trials "
                      f"x{res[f'{tag}_trials_x']}")
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[condstate13] summary -> {HERE / 'outputs' / 'summary.json'}")


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "f0": F0, "visual": "ssvep1_f14.0"}, indent=1))
    readout()


if __name__ == "__main__":
    main()
