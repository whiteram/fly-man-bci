"""bci/sleep2: what MAINTAINS the up-state? (sleep-probe follow-up)

The mechanism-14 probes left an open question: after sz_l ignition the
10.5 uV up-state plateau persists INDEPENDENT of the CEN_C STD gate --
so the maintenance current does not route through the (damped)
recurrence.  This study attributes it by ZEROING candidate pathway
groups one at a time (--gain-scale, runtime scalar, kernels cached):

  base        sz_l (LAL*@L 600 pA, 2 s) + full 0.004 recurrence
  noCEN_C     recurrence killed     (x0.02)
  noCEN_R     OLR->CEN killed       (x0.02)
  noOLR_V     visual core->OLR      (x0.02)
  noCEN_V     visual core->CEN      (x0.02)
  noORN_C     ORN->CEN              (x0.02)

Readout: plateau scalp rms over the last 10 s (the original observable)
+ class-resolved rates (--pop-rate) for attribution.  The maintainer is
whichever zeroing COLLAPSES the plateau; if none does, maintenance is
distributed/intrinsic and the question escalates.

Usage (conda ffbm, repo root):
    python bci/sleep2/acquire.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
T_END = 20000.0
SEED = 606
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
ARMS = {
    "base": None,
    "noCEN_C": "CEN_C=0.02",
    "noCEN_R": "CEN_R=0.02",
    "noOLR_V": "OLR_V=0.02",
    "noCEN_V": "CEN_V=0.02",
    "noORN_C": "ORN_C=0.02",
}


def main():
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for arm, gs in ARMS.items():
        dst = out / f"{arm}_scalp.npy"
        if dst.exists():
            continue
        trial = out / f"_trial_{arm}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "dark",
               "--chem-input", "sz_l", "--pop-rate", GROUPS,
               "--t-end", str(T_END), "--seed", str(SEED),
               "--out", str(trial), "--gpu"]
        if gs:
            cmd += ["--gain-scale", gs]
        print(f"[acquire] {arm}: gain-scale {gs}", flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed for {arm}")
        shutil.copy(trial / "_debug_phi_scalp.npy", dst)
        shutil.copy(trial / "_pop_rate.npz", out / f"{arm}_pop.npz")
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "dur_ms": T_END, "seed": SEED, "chem": "sz_l",
         "groups": GROUPS, "regions": REGIONS,
         "plateau_ms": [10000, 20000]}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
