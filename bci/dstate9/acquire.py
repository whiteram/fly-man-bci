"""bci/dstate9: the CURRENT-era state-entry map for the sz ignition.

kcdrv3/dstate8 revealed the chem auto working point (0.002) puts the
sz ignition on a knife edge (STD-gate kernel rounding flips it) and
that gain-scale 2.0 yields a MID-LATCH where the sleep-era plateau
used to be.  This arc maps TODAY's entry thresholds systematically so
future arcs stop rediscovering them:

  gain-scale CEN_C in {1.0, 1.5, 2.0}  (abs 0.002 / 0.003 / 0.004)
  x std gate in {none, CEN_C:0.00015,5000}

30 s trials, sz_l1 (single LAL*@L 600 pA pulse 0.8-2.8 s), seed 616.
Readout: MBON/ALPN/KC in [8,28] s -> latch (MBON ~395) / mid-latch
(~76) / plateau (~3) / dark (0).

Usage (conda ffbm, repo root):
    python bci/dstate9/acquire.py
    python bci/dstate9/analyze.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
T_END = 30000.0
SEED = 616
CHEM = "sz_l1"
# tag -> (gain factor vs 0.002, std gate)
ARMS = {
    "g10_nostd": (1.0, False),
    "g15_nostd": (1.5, False),
    "g20_nostd": (2.0, False),
    "g10_std": (1.0, True),
    "g15_std": (1.5, True),
    "g20_std": (2.0, True),
}
WIN = (8000, 28000)
REFS = {"latch": 395.0, "mid_latch": 76.0, "plateau": 3.0, "dark": 0.0}


def run(tag, fac, std):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", CHEM, "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if fac != 1.0:
        cmd += ["--gain-scale", f"CEN_C={fac:g}"]
    if std:
        cmd += ["--std-gates", "CEN_C:0.00015,5000"]
    print(f"[acquire] {tag}: gain x{fac:g} std={std}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, (fac, std) in ARMS.items():
        run(tag, fac, std)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {a: {"gain_factor": ARMS[a][0], "std": ARMS[a][1]}
                  for a in ARMS},
         "groups": GROUPS, "dur_ms": T_END, "seed": SEED,
         "regions": REGIONS, "chem": CHEM, "window_ms": list(WIN),
         "refs_hz": REFS,
         "era_note": "current-era entry map; the sleep-era plateau "
                     "(MBON ~3) may no longer exist (kcdrv3)"}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
