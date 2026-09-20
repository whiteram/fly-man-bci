"""bci/mmdev2 DM2 dose pilot: find the matched-RESPONSE deviant amp.

The matched-intensity arm (mmn_match) showed DM2@150 pA is invisible
at the scalp (devB ~ -0.001 uV), so identity deviance was never
actually tested: intensity and RESPONSE are confounded through the
DM2 pathway's lower sensitivity.  This pilot runs SINGLE pulses
(chem entries da1_solo / dm2_solo, 300-600 ms @150 pA base) and
doses the DM2 amplitude with the global --chem-amp-scale (1x/2x/3x/
4x -- a solo entry has one channel, so the global scale IS the DM2
scale).  Readout: net = rms(300-600) - rms(100-300 baseline).

The matched-response oddball arm (chem entry mmn_matchr) then
hardcodes DM2 at 150 * s*, where s* interpolates the DM2 dose onto
the DA1@150 reference net.

Usage: python bci/mmdev2/_pilot_dm2.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
SCALES = [1.0, 2.0, 3.0, 4.0]
RUNS = 2
SEED0 = 80


def run(stem, entry, scale, seed):
    dst = OUT / f"pilot_{stem}.npy"
    if dst.exists():
        return dst
    trial = OUT / f"_trial_{stem}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", entry, "--t-end", "1200", "--seed", str(seed),
           "--chem-amp-scale", f"{scale:g}",
           "--out", str(trial), "--gpu"]
    print(f"[pilot] {stem}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"export failed for {stem}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.rmtree(trial)
    return dst


def net(p):
    phi = np.load(p) * 1e6 * 1.7
    base = np.sqrt((phi[100:300] ** 2).mean())
    return np.sqrt((phi[300:600] ** 2).mean()) - base


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    for r in range(RUNS):
        p = run(f"da1_r{r}", "da1_solo", 1.0, SEED0 + r)
        res.setdefault("da1", []).append(float(net(p)))
    for s in SCALES:
        for r in range(RUNS):
            p = run(f"dm2_s{int(s)}_r{r}", "dm2_solo", s,
                    SEED0 + 10 + int(s) * RUNS + r)
            res.setdefault(f"dm2_s{int(s)}", []).append(float(net(p)))
    summary = {k: (float(np.mean(v)), float(np.std(v)))
               for k, v in res.items()}
    (OUT / "pilot_dm2.json").write_text(json.dumps(summary, indent=1))
    ref = summary["da1"][0]
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    ys = np.array([summary[f"dm2_s{int(s)}"][0] for s in xs])
    if ys.max() >= ref:
        s_star = float(np.interp(ref, ys, xs))
        print(f"DA1 reference {ref:+.4f} uV -> matched DM2 scale "
              f"s* = {s_star:.2f} (amp {150 * s_star:.0f} pA)")
    else:
        print(f"DA1 reference {ref:+.4f} uV NOT reached by DM2 dose "
              f"(max {ys.max():+.4f} at 4x)")
    for k, (m, s) in summary.items():
        print(f"{k}: {m:+.4f} +- {s:.4f} uV")


if __name__ == "__main__":
    main()
