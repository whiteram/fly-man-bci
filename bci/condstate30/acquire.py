"""bci/condstate30: hlock 300 s 寿命的 seed-63 复制（τ_sig 承重数的双种子化）。

cs15 的 τ_sig(hlock)=336 s 是唯一"实测上限"的寿命（qlock/resid 为
外推下限），单 trial 单种子。本弧复制它：cs4hlnt（DA1@1600 单脉冲
→高锁无训练）seed **63**，t_end 300 s，配方逐字同 condstate4/
condstate15。收敛证书（行 97）列出的高成本可选项之一。

问题三连（stability63.py）：
  1. 确定性复证: 前 22 s vs condstate4/hlnt63（同 chem 同种子）;
  2. τ_sig(63) = 3σ/联合漂移速度——与 62 的 336 s 对照;
  3. 机制: w_A 漂移 + DAN 调制积分（62: +0.0109 / 5110 ms）。

用法 (conda ffbm, repo root):
    python bci/condstate30/acquire.py     # ~52 min GPU, 幂等
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN"
T_END = 300000.0
ARMS = {"hll63": {"chem": "cs4hlnt", "seed": 63}}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists() and (out / "states" / f"{tag}.npz").exists() \
            and (out / f"{tag}_pop.npz").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"], "--pop-rate", GROUPS,
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", "0.03",
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(T_END), "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(out / "states" / f"{tag}.npz")]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']} "
          f"t_end {T_END/1000:.0f} s", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)
    print(f"[acquire] {tag}: done in {(time.time()-t0)/60:.1f} min",
          flush=True)


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = HERE / "outputs" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": {**meta.get("arms", {}),
                          **{t: ARMS[t] for t in ARMS}},
                 "groups": GROUPS, "dur_ms": T_END,
                 "regions": REGIONS, "chem": "cs4hlnt",
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)",
                 "question": "τ_sig(hlock) 的 seed-63 复制——唯一实测"
                             "寿命的双种子化"})
    meta_p.write_text(json.dumps(meta, indent=1))
    pop = np.load(HERE / "outputs" / "hll63_pop.npz")
    for g in GROUPS.split(","):
        r = pop[g]
        print(g, [round(float(r[a:b].mean()), 2)
                  for a, b in ((2000, 10000), (100000, 110000),
                               (280000, 290000))])


if __name__ == "__main__":
    main()
