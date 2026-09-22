"""bci/sign2: sign-on model -- WHERE does the odor signal die, and can
the loop be recalibrated?  (sign1 Phase C)

C1 finding (default gain, mmn_match A1, seed 616): the collapse is in
the AL ENGINE -- sign leaves ORN bit-identical but takes ALPN
11.3 -> 0.14 Hz (82x) and ALON 28 -> 1.2 Hz (23x); scalp 0.37 -> 0.10
uV.  KC/MBON take no part in the odor signal even in base (the PN->KC
leg is working-point-suppressed, condit3) -- the odor response is an
AL-dipole + recurrent-current phenomenon, and sign inverts the PN<->LN
loop's regenerative gain (LN inhibition now actually inhibits).

Arms (ARMS table; default gain unless stated):
  base / sign            sign off vs on (default gain) -- C1 pair
  halfbase / halfsign    the same at CEN_C=0.5 (the conditioning
                         working point; AL barely responds in base
                         either -- odor protocols live at full gain)
  periph                 --use-sign-groups CEN_R,OLR_V: peripheral
                         recurrent groups get true inhibition, the
                         CEN_C amplifier (with the AL-internal rows)
                         stays all-excitatory -- the intermediate-
                         fidelity candidate (C3)
  ampsign_F              --chem-amp-scale F under sign: is the loss
                         linear-compensable? (C2)

Usage (conda ffbm, repo root):
    python bci/sign2/acquire.py --arms base,sign,periph
    python bci/sign2/acquire.py --arms ampsign_4,ampsign_16
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
CHEM = "mmn_match"
GROUPS = "ORN,ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 3600.0
SEED = 616
ARMS = {
    "base": {"sign": False, "amp": 1.0, "gain": None},
    "sign": {"sign": True, "amp": 1.0, "gain": None},
    "halfbase": {"sign": False, "amp": 1.0, "gain": "CEN_C=0.5"},
    "halfsign": {"sign": True, "amp": 1.0, "gain": "CEN_C=0.5"},
    "periph": {"sign": False, "amp": 1.0, "gain": None,
               "groups": "CEN_R,OLR_V"},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", CHEM,
           "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if spec.get("gain"):
        cmd += ["--gain-scale", spec["gain"]]
    if spec.get("sign"):
        cmd += ["--use-sign-all"]
    if spec.get("groups"):
        cmd += ["--use-sign-groups", spec["groups"]]
    if spec["amp"] != 1.0:
        cmd += ["--chem-amp-scale", f"{spec['amp']:g}"]
    print(f"[acquire] {tag}: " + " ".join(cmd[1:]), flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,sign,periph",
                    help="comma list of ARMS keys; ampsign_F adds a "
                         "sign+amp-F arm on the fly")
    args = ap.parse_args()
    for a in args.arms.split(","):
        if a.startswith("ampsign_"):
            ARMS[a] = {"sign": True, "amp": float(a.split("_")[1]),
                       "gain": None}
        elif a not in ARMS:
            raise SystemExit(f"unknown arm {a}")
    for a in args.arms.split(","):
        run(a, ARMS[a])
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {k: ARMS[k] for k in args.arms.split(",")},
         "chem": CHEM, "groups": GROUPS,
         "dur_ms": T_END, "seed": SEED, "regions": REGIONS,
         "windows_ms": {"A1": [300, 600], "dev_DM2": [2300, 2600],
                        "return": [2700, 3000],
                        "baseline": [0, 250]}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
