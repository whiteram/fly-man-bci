"""bci/condstate8: micro-pool close-up + forward closure.

condstate7 localized the quiet lock's rent carrier to the GABA-subset
micro-pools (MBON09/30/11/03/31, saturated at 100-240 Hz).  Anatomy
(zen_typed): their dominant non-KC afferents are the DAN->MBON trunk
(PAM06/12/13/14, PPL101/103), DPM, and CRE067 -- but the DANs
themselves are V-clamped, so the lock's drive onto the micro-pools
could be direct disinhibition, the DAN trunk, or DPM.  This arc
records ALL relevant types at once (KC families for the R_kcdan
forward term, all MBON types, DAN subtypes, and the named-candidate
pools) in a naive-vs-quiet-lock pair, then:

  1. micro-pool drive decomposition: for each micro-pool M,
     dI_M = sum_T sign*W_{T->M} * (r_T^lock - r_T^naive) -- who adds
     the input that pushes MBON09 from 157 to 235 Hz?
  2. forward closure: r_pred = R_us + b*R_kcdan - c*V with the KC
     term now at type level; fit (b, c) on us62+ctl62t, test on
     lock/res/hl/naive2r traces from condstate7.

Arms (seed 62): ctl8 (cs2ctl), lock8 (cs2lock), t_end 20 s.

Usage (conda ffbm, repo root):
    python bci/condstate8/acquire.py
    python bci/condstate8/analyze.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,MBON,DAN,Kenyon_Cell"
TYPES = (
    # all KC families (R_kcdan forward term)
    "KCg-m,KCg-d,KCg-s1,KCg-s2,KCg-s3,KCab-c,KCab-m,KCab-p,KCab-s,"
    "KCa'b'-ap1,KCa'b'-ap2,KCa'b'-m,"
    # all MBON types (the V subset + the rest)
    "MBON01,MBON02,MBON03,MBON04,MBON05,MBON06,MBON07,MBON09,MBON10,"
    "MBON11,MBON20,MBON25,MBON30,MBON31,MBON32,MBON34,MBON15-like,"
    "MBON25-like,"
    # DAN subtypes (the ->MBON trunk candidates)
    "PAM01,PAM02,PAM04,PAM05,PAM06,PAM07,PAM08,PAM09,PAM10,PAM11,"
    "PAM12,PAM13,PAM14,PAM15,PPL101,PPL102,PPL103,PPL104,PPL105,"
    "PPL106,PPL107,PPL108,PPM1201,PPL201,PPL203,"
    # named candidate pools from the micro-pool afferent table
    "DPM,APL,LHMB1,CRE067,LAL155,LHPV9b1,FR1,GNG321,SMP146,SMP012,"
    "SMP011_b,LAL171,LAL172,AVLP749m,SMP207,CRE108,SMP176,SMP742,"
    "LHPV7c1,LHPD5a1,SMP115,SMP165,OA-VPM3")
ARMS = {
    "ctl8":  {"chem": "cs2ctl",  "seed": 62, "t_end": 20000.0},
    "lock8": {"chem": "cs2lock", "seed": 62, "t_end": 20000.0},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_pop.npz"
    if dst.exists() and (out / f"{tag}_dan.npy").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"],
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", "0.03",
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(spec["t_end"]), "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']}",
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", dst)
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = HERE / "outputs" / "meta.json"
    meta_p.write_text(json.dumps(
        {"arms": ARMS, "types": TYPES, "groups": GROUPS}, indent=1))
    print("[condstate8] acquire done; run analyze.py")


if __name__ == "__main__":
    main()
