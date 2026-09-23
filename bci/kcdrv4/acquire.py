"""bci/kcdrv4: the USxUS pairing window -- can the DAN input convert a
SUBTHRESHOLD KC drive into a suprathreshold one?

kcdrv3 closed the direct-injection bracket (DAN->KC dead in every
reproducible state) and left one route open: the w6.4 channel might
act PERMISSIVELY -- not firing KC on its own, but pushing a
subthreshold KC drive over threshold when paired with it (classic
pairing structure, and the conditioning arcs' US arrives exactly this
way next to a strong direct KC drive).

Arms (naive dark, seed 616, 5 s, no plasticity):
  kc_w400   KCab-m @400 pA only  (the conditioning drive is 1600;
            400 is deliberately weak -- where does the KC input-output
            curve sit?)
  us_ppl103 PPL103* @800 only    (kcdrv2: dead)
  kc_p400   both together        (paired; verdict = KC response vs
            kc_w400: >> means DAN converted sub->supra; ~= means the
            channel adds nothing even permissively)

Usage (conda ffbm, repo root):
    python bci/kcdrv4/acquire.py
    python bci/kcdrv4/analyze.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "Kenyon_Cell,MBON,DAN"
T_END = 5000.0
SEED = 616
ARMS = {"kc_w400": "kc_w400", "us_ppl103": "us_ppl103",
        "kc_p400": "kc_p400"}
WIN = (600, 2400)
BASE = (0, 400)


def run(tag, chem):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem, "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {chem}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, chem in ARMS.items():
        run(tag, chem)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "groups": GROUPS, "dur_ms": T_END,
         "seed": SEED, "regions": REGIONS,
         "windows_ms": {"base": list(BASE), "us": list(WIN)},
         "ref_cond_drive_pa": 1600}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
