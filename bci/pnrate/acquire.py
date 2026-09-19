"""bci/pnrate: PN-layer weak-signal amplification (VALIDATION row 5).

Bhandawat 2007: weak ORN input is AMPLIFIED at the PN layer, strong
input is not (expansive nonlinearity).  The model side was never
instrumented (PNs sit inside the 33k CEN pop).  With --pop-rate the
ALPN subpopulation rate is now directly measurable, and --chem-amp-
scale sweeps odor intensity with a single entry (odor_da1_short:
ORN_DA1 150 pA, pulse 300-1200 ms).

Dose-response: F in {.25, .5, 1, 2, 4} x 2 seeds, recording ORN
(the driven receptor population), ALPN, Kenyon_Cell, MBON.

Usage (conda ffbm, repo root):
    python bci/pnrate/acquire.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
CHEM = "odor_da1_short"
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
SEEDS = [171, 271]
T_END = 3600.0
GROUPS = "ORN,ALPN,ALLN,Kenyon_Cell,MBON"


def main():
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for f in SCALES:
        for s in SEEDS:
            tag = f"f{f:g}_s{s}"
            dst = out / f"{tag}_pop.npz"
            if dst.exists():
                continue
            trial = out / f"_trial_{tag}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", CHEM,
                   "--chem-amp-scale", f"{f:g}",
                   "--pop-rate", GROUPS,
                   "--t-end", str(T_END), "--seed", str(s),
                   "--out", str(trial), "--gpu"]
            print(f"[acquire] {tag}: x{f:g} seed {s}", flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {tag}")
            shutil.copy(trial / "_pop_rate.npz", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"chem": CHEM, "scales": SCALES, "seeds": SEEDS,
         "dur_ms": T_END, "groups": GROUPS, "regions": REGIONS,
         "odor_ms": [300, 1200], "base_pa": 150.0}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
