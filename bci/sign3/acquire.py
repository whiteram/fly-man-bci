"""bci/sign3: can ALL-group gain restore the AL odor response under
sign?  (sign2 Phase C step 2 -- the cheap discriminator)

sign2 localized the sign odor collapse to the AL engine: the AL-local
rows (ALL group, split of CEN_C at --al-local-gain) carry the PN<->LN
loop, and flipping 40% inhibitory weight to true inhibition inverts
the loop's regenerative gain (ALPN A1 11.3 -> 0.14 Hz).  If scaling
the WHOLE ALL group back up restores the response, recalibration is a
gain knob; if not, it needs a per-leg redesign (excitatory legs only),
i.e. the real 'big' program.

Arms (all --use-sign-all, mmn_match A1, seed 616, chem wp 0.002):
  g002  ALL gain 0.002 (neutral; should reproduce sign2's sign arm)
  g004 / g008 / g016   2x / 4x / 8x the ALL group

Reference: base (no sign) ALPN A1 +11.3 Hz, ALON +27.6.

Usage (conda ffbm, repo root):
    python bci/sign3/acquire.py
    python bci/sign3/analyze.py
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
GROUPS = "ORN,ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 3600.0
SEED = 616
ARMS = {"g002": 0.002, "g004": 0.004, "g005": 0.005, "g006": 0.006,
        "g007": 0.007, "g008": 0.008, "g016": 0.016}


def run(tag, g):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", "mmn_match", "--use-sign-all",
           "--al-local-gain", f"{g:g}", "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: ALL gain {g:g}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    for a in args.arms.split(","):
        run(a, ARMS[a])
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {a: ARMS[a] for a in args.arms.split(",")},
         "sign": True, "chem": "mmn_match", "groups": GROUPS,
         "dur_ms": T_END, "seed": SEED, "regions": REGIONS,
         "ref_base": {"ALPN_A1": 11.27, "ALON_A1": 27.62}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
