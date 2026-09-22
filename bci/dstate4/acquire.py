"""dstate4: entry x gain x STD matching -- the real unification test.

dstate1 refuted the odor-lock == sz-plateau unification, but the
comparison confounded three variables at once: entry (odor vs
ignition), CEN_C gain (0.002 vs 0.004), and the slow-STD gate
(odor runs had none; the sz plateau needed it to collapse).  This
study un-confounds with a matching matrix (all trials t_end 60 s):

  odor_g002     dstate1's s62 run (reused)      -- the quiet lock
  odor_g004     da1_dc60 @ CEN_C x2 (0.004)     -- full-gain odor
  ign_g002      sz_l1 @ CEN_C x0.5 (0.002)      -- ignition at the
                                                  working point
  ign_g004_std  sleep4's sz_l1 U=1.5e-4/tau=10s run, first 60 s
                (deterministic prefix, reused)  -- the 38 Hz plateau
  odor_g002_std da1_dc60 + the same STD gate     -- the lock under
                                                  the slow STD

Readout per arm: ALPN/ALLN/ALON/MBON plateau rates (10-60 s),
|mean|/std, and a state label (dark / quiet lock / attractor /
latch / post-collapse plateau).  The question: at matched (gain,
STD) do the two entries land in the same state?

Usage (conda ffbm, repo root):
    python bci/dstate4/acquire.py
    python bci/dstate4/analyze.py
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
GROUPS = "ORN,MBON,ALPN,Kenyon_Cell,DAN,ALLN,ALIN,ALON"
STD = "CEN_C:0.00015,10000"
T_END = 60000.0
NEW = {
    "odor_g004": {"chem": "da1_dc60", "seed": 62, "gain": 2.0,
                  "std": False},
    "ign_g002": {"chem": "sz_l1", "seed": 616, "gain": 0.5,
                 "std": False},
    "odor_g002_std": {"chem": "da1_dc60", "seed": 62, "gain": 1.0,
                      "std": True},
}
REUSE = {
    "odor_g002": str(ROOT / "bci" / "dstate1" / "outputs"
                     / "da1_dc60_s62_scalp.npy"),
    "ign_g004_std": str(ROOT / "bci" / "sleep4" / "outputs"
                        / "sz_l1_u0.00015_t10000_s616_scalp.npy"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(list(NEW) + list(REUSE)))
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    for arm in arms:
        dst = out / f"{arm}_scalp.npy"
        if arm in REUSE:
            if not dst.exists():
                shutil.copy(REUSE[arm], dst)
                print(f"[acquire] {arm}: reused prefix", flush=True)
            continue
        if dst.exists():
            print(f"[acquire] {arm}: exists, skip", flush=True)
            continue
        a = NEW[arm]
        trial = out / f"_trial_{arm}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "dark",
               "--chem-input", a["chem"], "--pop-rate", GROUPS,
               "--t-end", str(T_END), "--seed", str(a["seed"]),
               "--out", str(trial), "--gpu"]
        if a["gain"] != 1.0:
            cmd += ["--gain-scale", f"CEN_C={a['gain']:g}"]
        if a["std"]:
            cmd += ["--std-gates", STD]
        print(f"[acquire] {arm}", flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed for {arm}")
        shutil.copy(trial / "_debug_phi_scalp.npy", dst)
        shutil.copy(trial / "_pop_rate.npz", out / f"{arm}_pop.npz")
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"arms": {**{k: dict(v) for k, v in NEW.items()},
                  **{k: {"reuse": True} for k in REUSE}},
         "std_gate": STD, "t_end_ms": T_END, "groups": GROUPS,
         "regions": REGIONS,
         "references": {
             "quiet_lock": "dstate1: ALPN 19.9, std 0.0145, MBON 0",
             "sz_plateau": "sleep4: ALPN 38, std 0.18, MBON 3-5 "
                           "(the 38 Hz figure is post-collapse)"}},
        indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
