"""bci/condstate20: 残留态 300 s 漂移率——总账最后一个未测量的关闭弧。

condstate16 的 τ_sig=336 s 来自 hlock 单 trial 外推，resid（最强
态, σ*=0.78 µV）的漂移速率未测（README 标注"方向有利"）。本弧
补测：cs3resnt（LAL 点火 + 6 s ×0.5 残留态，无训练纯漂移探针）
+ --gain-schedule 6000:CEN_C=0.5，t_end 外推 300 s（配方逐字同
condstate3 acquire，pitfall #15：调度在 CLI 不在化学条目）。

问题三连（stability_resid.py）：
  1. 残留态签名 300 s 平稳吗（池率 + 头皮滑窗特征漂移速度）？
  2. τ_sig(resid) = 3σ/联合漂移速度 → σ*(resid) 修正多少？
  3. 漂移机制同 hlock 吗（w_A 慢写 + DAN 积分）？——残留态
     DAN 门更关（mod_ms 2.2-2.5 vs hlock 35.9），预测漂移更慢。

用法 (conda ffbm, repo root):
    python bci/condstate20/acquire.py     # 幂等, ~52 min GPU
数据: outputs/resl62_scalp.npy (300k x 17), resl62_pop.npz,
      states/resl62.npz, meta.json
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
SCHED = "6000:CEN_C=0.5"
ARMS = {"resl62": {"chem": "cs3resnt", "seed": 62}}


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
    cmd += ["--gain-schedule", SCHED]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']} "
          f"sched {SCHED} t_end {T_END/1000:.0f} s", flush=True)
    t0 = time.time()
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
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
                 "regions": REGIONS,
                 "chem": "cs3resnt (LAL ignition, no training)",
                 "sched": SCHED,
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)",
                 "question": "残留态签名 300 s 漂移率 + "
                             "σ*(resid) 的 τ 修正"})
    meta_p.write_text(json.dumps(meta, indent=1))
    pop = np.load(HERE / "outputs" / "resl62_pop.npz")
    for g in GROUPS.split(","):
        r = pop[g]
        segs = [(2000, 10000), (100000, 110000), (280000, 290000)]
        print(g, [round(float(r[a:b].mean()), 2) for a, b in segs])


if __name__ == "__main__":
    main()
