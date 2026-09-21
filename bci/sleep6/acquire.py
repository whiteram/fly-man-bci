"""bci/sleep6: what IS the re-arm ceiling made of?  d(t) vs R(t).

sleep5 measured the recovery curve R(t) (p2 response vs delay) and
found it caps at ~42-45% where the single-tau resource model predicts
~100% recovery -- interpreted as "the plateau state caps the response,
not the resource".  This study measures the RESOURCE directly: the
std-debug dump (added with the sleep3 recovery fix) saves the final
per-edge depletion state d of the std-gated CEN_C pool at trial end.

Five single-ignition trials (chem sz_l1, U=1.5e-4, tau=10 s, seed
616 -- the sleep5 dose) with t_end = 10/20/35/50/80 s: each trial
ends exactly when sleep5's second pulse would ARRIVE, so the final d
is the resource state at the R(t) measurement moment (deterministic
prefix: identical dynamics to sleep5's runs).  Recovery prediction
for an edge last active around the collapse (~8 s):
d(t) = 1 - exp(-(t-8)/10) ~ 0.18/0.70/0.93/0.98/1.00.

Verdict logic: d recovers while R caps -> the ceiling is the plateau
STATE (sleep5 confirmed quantitatively); d tracks R -> the resource
itself never recovers (sleep5's interpretation refuted).  Ever-
depleted edges are identified as d < 0.9999 (at 80 s recovery leaves
d = 0.9993, well above the f32 epsilon).

Usage (conda ffbm, repo root):
    python bci/sleep6/acquire.py
    python bci/sleep6/analyze.py
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
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
CHEM = "sz_l1"
STD = "CEN_C:0.00015,10000"
SEED = 616
T_ENDS = [10000.0, 20000.0, 35000.0, 50000.0, 80000.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ends", default=",".join(f"{t:g}" for t in T_ENDS))
    args = ap.parse_args()
    ends = [float(v) for v in args.ends.split(",")]
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for t_end in ends:
        tag = f"szl1_e{t_end / 1000:g}s_s{SEED}"
        dst_d = out / f"{tag}_stddebug.npz"
        dst_s = out / f"{tag}_scalp.npy"
        if dst_d.exists() and dst_s.exists():
            print(f"[acquire] {tag}: exists, skip", flush=True)
            continue
        trial = out / f"_trial_{tag}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "dark",
               "--chem-input", CHEM, "--pop-rate", GROUPS,
               "--std-gates", STD,
               "--t-end", str(t_end), "--seed", str(SEED),
               "--out", str(trial), "--gpu"]
        print(f"[acquire] {tag}: t_end {t_end:g} ms", flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed for {tag}")
        shutil.copy(trial / "_std_debug_CEN_C.npz", dst_d)
        shutil.copy(trial / "_debug_phi_scalp.npy", dst_s)
        shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"chem": CHEM, "std": STD, "seed": SEED, "t_ends_ms": ends,
         "regions": REGIONS, "note": "final d = resource state at the "
         "sleep5 R(t) measurement moments (deterministic prefix)"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
