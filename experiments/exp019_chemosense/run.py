"""exp019: chemosensory input validation -- odor/taste pulses -> EEG.

Does a CHEMOSENSORY stimulus reach the scalp EEG of the scaled fly
network, and is it distinguishable? Three 10.5 s trials via the standard
export entry (viz/export_data.py, warm caches):

  baseline       -- dark visual input, no chem drive
  odor3_block    -- 3 odor pulses into disjoint glomeruli
                    (cVA->DA1 / ester->DM2 / fatty acid->VA1v)
  taste3_block   -- 3 sugar pulses into organ-specific GRN classes
                    (labellum LgLG* / wing WG* / tarsus claw_tpGRN)

Validation questions (README carries the numbers):
  1. drive reaches the periphery  : ORN/GRN population rate follows the
     pulse train (channels fire only in their own epoch)
  2. drive reaches the brain      : CEN (antennal lobe / SEZ partners)
     rate follows with the expected ~1-step lag
  3. drive reaches the scalp      : phi_scalp band (0.1-4 Hz) envelope
     rises in chem epochs vs the same epochs in the baseline run
  4. epoch decoding               : nearest-centroid on per-channel
     log-band-power features across the 3 chem epochs (chance 33%)

Run from repository root (conda ffbm):
    python experiments/exp019_chemosense/run.py            # full
    python experiments/exp019_chemosense/run.py --smoke    # wiring only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).resolve().parent / "outputs"
# central_brain requires ol_rest upstream (exp017 wiring); VNC stays off
# (faster trials; the GRN->VNC reflex arc is exercised in the
# circuit4.py report instead)
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory,gustatory"
FS = 1000.0                       # phi sample rate (2 kHz steps, /2)

CONDITIONS = [
    ("baseline", None),
    ("odor3_block", "odor3_block"),
    ("taste3_block", "taste3_block"),
]
# small-scale fast iteration (user directive): 3 s trials, 500 ms
# pulses @ 300/1300/2300 ms, same channels and amplitudes as the full
# protocol; baseline is reused from the full run (dark visual is
# chem-independent, and the chem-run CEN_C override only matters when
# chem input drives central cells)
FAST_T_END = 3600.0
FAST_CHEM = {"odor3_block": "odor3_short", "taste3_block": "taste3_short"}


def run_condition(name, chem, fast):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    done = d / "viz_data.json"
    need_t = FAST_T_END if fast else 10500.0
    if done.exists():
        try:
            t_end = json.loads(done.read_text(
                encoding="utf-8"))["meta"]["t_end_ms"]
            if t_end >= need_t:
                print(f"[{name}] outputs exist (t_end={t_end}), skip")
                return
        except Exception:
            pass                                       # redo on doubt
    if fast:
        chem = FAST_CHEM.get(name, chem)
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS,
           "--visual-input", "dark",
           "--out", str(d)]
    if chem:
        cmd += ["--chem-input", chem]
    if fast:
        cmd += ["--t-end", str(FAST_T_END)]
    print(f"[{name}] " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def chem_epochs_from_meta(meta):
    """Epoch list [(t0, t1, label)] from an output's own meta (the
    registry entry that actually drove the trial -- full or fast)."""
    ch = (meta.get("chem_input") or {}).get("channels") or []
    return [(c["pulses"][0][0], c["pulses"][0][1],
             f"{c['pop']}[{c['group']}] n={c['n']}") for c in ch]


def band_env(phi, band=(0.1, 4.0)):
    """Per-channel band rms envelope, 50 ms hops (edge-trimmed)."""
    sos = sps.butter(2, list(band), btype="band", fs=FS, output="sos")
    x = phi.T * 1e6                        # (ch, t) uV
    x = sps.sosfiltfilt(sos, x, axis=1)
    hop = int(0.05 * FS)
    n = x.shape[1] // hop
    e = np.sqrt((x[:, :n * hop].reshape(x.shape[0], n, hop) ** 2)
                .mean(axis=2))
    return e                               # (ch, n_windows)


def analyze():
    readme = []
    readme.append("# exp019 run output\n")
    for name, _chem in CONDITIONS:
        d = OUT / name
        if not (d / "_debug_phi_scalp.npy").exists():
            continue
        meta = json.loads((d / "viz_data.json").read_text(
            encoding="utf-8"))["meta"]
        eps = chem_epochs_from_meta(meta)
        phi = np.load(d / "_debug_phi_scalp.npy")   # (t, ch) scaled
        rates = (dict(np.load(d / "_debug_extra_rates.npz"))
                 if (d / "_debug_extra_rates.npz").exists() else {})
        t_end = int(meta["t_end_ms"])
        readme.append(f"\n## {name} "
                      f"(chem={meta.get('chem_input', {}).get('id', '-')}, "
                      f"t_end={t_end} ms)")
        # 1+2: population rates per chem epoch vs the preceding gap
        for t0, t1, lab in eps:
            i0, i1 = int(t0) + 150, min(int(t1), t_end)
            g0, g1 = max(0, int(t0) - 250), max(0, int(t0) - 50)
            for pop in sorted(rates):
                tr = rates[pop]
                if tr.max() <= 0:
                    continue
                readme.append(f"- {lab}: {pop} in-epoch "
                              f"{tr[i0:i1].mean():.2f} Hz "
                              f"(pre-gap {tr[g0:g1].mean():.2f} Hz)")
        # 3: scalp band rms, chem epochs vs the pre-stimulus window
        e = band_env(phi)
        pre = e[:, :max(2, int(eps[0][0]) // 50 - 5)].mean() \
            if eps else e.mean()
        for t0, t1, lab in eps:
            w = slice(int(t0) // 50, min(int(t1), t_end) // 50)
            r_in = e[:, w].mean()
            readme.append(f"- {lab}: scalp 0.1-4 Hz rms {r_in:.4f} uV "
                          f"vs pre {pre:.4f} uV (x{r_in / pre:.2f})")
        fig(name, phi, rates, eps)
    (OUT / "summary.md").write_text("\n".join(readme),
                                    encoding="utf-8")
    print("\n".join(readme))


def fig(name, phi, rates, eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(phi.shape[0]) / FS
    nrows = 1 + len(rates) + 1
    a, axs = plt.subplots(nrows, 1, figsize=(11, 1.5 * nrows),
                          sharex=True)
    e = band_env(phi).mean(axis=0)
    tw = np.arange(e.shape[0]) * 0.05
    for t0, t1, lab in eps:
        for ax in axs:
            ax.axvspan(t0 / 1000, t1 / 1000, alpha=0.15, color="C1")
        axs[0].text((t0 + 200) / 1000, e.max() * 0.9, lab, fontsize=7,
                    rotation=90, va="top")
    axs[0].plot(tw, e, lw=0.8, color="k")
    axs[0].set_ylabel("scalp 0.1-4 Hz\nrms (uV)", fontsize=7)
    for i, (pop, tr) in enumerate(rates.items()):
        axs[1 + i].plot(t, tr, lw=0.7)
        axs[1 + i].set_ylabel(f"{pop}\nHz", fontsize=7)
    axs[-1].set_xlabel("s")
    axs[-1].set_xlim(0, t[-1])
    a.suptitle(name, fontsize=10)
    a.tight_layout()
    a.savefig(OUT / name / "traces.png", dpi=130)
    plt.close(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="3 s trials, 500 ms pulses (small-scale "
                         "iteration); skips baseline (reuses full)")
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check only (400 ms, chem never fires)")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        conds = CONDITIONS if not args.fast \
            else [(n, c) for n, c in CONDITIONS if c]
        for name, chem in conds:
            run_condition(name, chem, args.fast)
    analyze()
