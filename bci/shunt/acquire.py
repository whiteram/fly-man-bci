"""bci/shunt: pure-shunting vs subtractive APL feedback (APL arc endgame).

The APL arc (condit3 follow-ups) concluded that ADDITIVE conductance
feedback cannot rebuild divisive normalization of the KC code.  But all
those measurements ran with the latent-sign gap -- MBM was effectively
all-excitatory, and E_rev_inh sat at -0.080 V (hyperpolarizing =
subtractive-dominated).  Two changes reopen the question honestly:

  1. MBM now uses the sign column (GABA rows inhibit)
  2. --mb-mod-erev moves the inhibitory reversal to REST (-0.060 V):
     zero current at rest, pure conductance that DIVIDES KC excitation
     -- textbook shunting inhibition

Paradigm: true odor (odor_da1_short) -> ORN ->(ORN_C) PN ->(ALK x0.01)
KC, x2 intensities; arms = none / sub (-0.080) / shunt (-0.060) at a
matched MBM gain.  Readout (--pop-rate-neurons): KC mean evoked rate
AND responsive fraction (sparse code), MBON rate.  Literature anchor:
~5-10% of KCs respond per odor at few-Hz rates (Honegger 2011);
divisive (shunt) should suppress the responsive tail at lower cost in
mean-rate distortion than subtractive.

Usage (conda ffbm, repo root):
    python bci/shunt/acquire.py [--gain 0.5]
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
CHEM = "odor_da1_short"
AL_GAIN = "0.01"
T_END = 3600.0
SEED = 313
INTENSITIES = [1.0, 4.0]
EREVS = {"none": None, "sub": "-0.080", "shunt": "-0.060"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", default=0.5, type=float,
                    help="MBM g_unit for sub/shunt arms")
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for inten in INTENSITIES:
        for arm, erev in EREVS.items():
            tag = f"{arm}_g{args.gain:g}_x{inten:g}"
            dst = out / f"{tag}_pop.npz"
            if dst.exists():
                continue
            trial = out / f"_trial_{tag}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", CHEM,
                   "--chem-amp-scale", f"{inten:g}",
                   "--al-gain", AL_GAIN,
                   "--pop-rate", "Kenyon_Cell,MBON,ALPN",
                   "--pop-rate-neurons",
                   "--t-end", str(T_END), "--seed", str(SEED),
                   "--out", str(trial), "--gpu"]
            if arm != "none":
                cmd += ["--mb-mod-gain", f"{args.gain:g}",
                        "--mb-mod-erev", erev]
            print(f"[acquire] {tag}", flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {tag}")
            shutil.copy(trial / "_pop_rate.npz", dst)
            shutil.copy(trial / "_pop_rate_neurons.npz",
                        out / f"{tag}_neurons.npz")
            shutil.rmtree(trial)
    (HERE / "outputs" / f"meta_g{args.gain:g}.json").write_text(json.dumps(
        {"chem": CHEM, "al_gain": AL_GAIN, "mbm_gain": args.gain,
         "erevs": EREVS, "intensities": INTENSITIES, "seed": SEED,
         "dur_ms": T_END, "regions": REGIONS,
         "odor_ms": [300, 1200]}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
