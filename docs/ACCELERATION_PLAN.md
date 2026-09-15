# 计算加速路线图（下一阶段，暂缓执行）

状态（2026-09-15 更新）：**P0 + P2（第一、二阶段）已实施并验证**——
numba 0.67 + CuPy 14.2 已装入 conda `ffbm` 环境（GPU 实测可算）；
七个 JIT 内核（分级突触 step / 电导 drive / edge_currents / 指数突触
衰减+投递（延迟与非延迟）/ OU 噪声 / LIF 显式与半隐式）以**逐位一致**
落地（单元对拍 + smoke 端到端 A/B，tests/test_numba_parity.py），
生物学循环实测 **2.38×**（600 ms 全区域单进程双轮基准：75.7 s →
31.8 s，全量估 22.1 → 9.3 min）。P3（CuPy）待续。前置工作（路线 A）
见 PERFORMANCE.md §7。

---

## 0. 基线事实（2026-09 实测）

全量导出（45 导 / 10.5 s 生物学时间 / 全 CNS 5 区域 / 123 万投影
偶极）当前总耗时 **~55 min**，构成：

| 构成 | 占比 | 实测热点（scripts/profile_export_loop.py） |
|---|---|---|
| 每步生物学（21,000 步） | ~70% | `to_neuron_drive` 的 scipy CSR matvec（每步 ~38 次小矩阵乘，32%）+ 指数突触衰减（15%，gather + 4 遍整数组临时对象） |
| 每毫秒前向记录（10,500 次） | ~28% | `edge_currents` 对 123 万边算电流；v_post 为 f64 会把 f32 状态**隐式升精度**，流量翻倍 |
| 其他（延迟投递/OU/LIF） | 散布 | — |

每步 ~80-120 次 numpy 调用、每次分配整数组临时对象。

## 1. 执行环境

**一切跑数用 conda `ffbm` 环境**
（`C:\Software\Devel\Anaconda3\envs\ffbm\python.exe`，
Python 3.12.14 + numpy 2.5.3 + scipy 1.18.1）。已认证：该环境跑
smoke 的 phi_scalp 校验和与系统 Python（numpy 2.4.4）逐位一致
（L2=4.798226e-05）。环境内**尚无** numba/cupy/torch，P0 安装。

## 2. 分阶段计划

### P0 环境准备（半小时）
```bash
C:\Software\Devel\Anaconda3\envs\ffbm\python.exe -m pip install "numba>=0.65" cupy-cuda12x
```
- numba ≥0.65 才支持 numpy 2.5（0.64 支持 2.4）；
- cupy-cuda12x 匹配驱动 CUDA 12.x（驱动 591.86 ✓，RTX 4060 Ti 16GB）；
- 装完各跑一次 smoke 认证校验和不变。

### P1 CPU 微优化（半天，低风险）
- `edge_currents`：v_post 预转 f32（消隐式升精度）、预分配缓冲消临时对象；
- `to_neuron_drive` 的两次 CSR matvec 合并为一次遍历（手写 CSR 循环）；
- 预期 55 → ~40 min；逐位对拍验证。

### P2 Numba JIT（第一、二阶段均已实施，2026-09-15）
- **已落地（7 内核）**：`_graded_step`（释放率平滑+权重，原地写）、
  `_drive_cond`（电导 drive 双 CSR matvec 融合为单遍 f32 累加，逐行
  加法次序与 scipy csr_matvec 相同）、`_edge_currents`（f64 输出，
  保持下游混合 dtype 点积走原 BLAS 路径）、`_exp_step` /
  `_exp_step_delayed`（衰减+投递融合；非延迟=缓冲式花式加法语义
  （重复目标 last-wins），延迟=按 bin 升序、bin0 缓冲式、bin≥1
  add.at 累加——逐条复刻 numpy 语义）、`_ou_step`（噪声仍由 numpy
  rng 抽取）、`_lif_step` / `_lif_step_g32`（显式 + 半隐式 Euler；
  g_tot f32 变体把分母全程按 f32 舍入，复刻 weak-scalar 提升）；
- **RNG 全程 numpy**：随机序列不变；逐位一致已验证（单元对拍 +
  smoke 端到端 FFBM_NUMBA=0/1 A/B + 首步逐状态探针）；
- **实测**：生物学循环 **2.38×**（31.8 s vs 75.7 s @600 ms 全区域，
  scripts/profile_export_loop.py 单进程双轮）；全量导出估
  22.1 → 9.3 min（不含核构建 68 s 与装配）；
- **dtype 语义陷阱记录**（复刻 numpy 必须）：① GradedSynapsePool 无
  sign 时 e_rev 曾是标量、ExponentialSynapses 无 sign 时 np.where 曾
  给 0 维数组——均已改逐边 1-D 数组；② weak-scalar 提升使
  `g_unit * y` 保持 f32、`y *= np.float64 衰减` 为 f64 乘后 f32 存
  （已全 tau 扫描验证）；③ **L/MID 的 g_tot 是 f32**（直接传 drive
  数组）而 T45/extra 是 f64 累加——半隐式分母的舍入路径不同，内核
  按 dtype 分派；④ 投递空选集必须传空数组而非 None（numba 无法
  类型化 None）；⑤ 花式加法 `y[t] += k[t]` 是缓冲式（last-wins），
  与 add.at（累加）语义不同，两条投递路径各用各的；
- **剩余候选**：释放率 sigmoid（r/l_release）、PhotoCascade、
  spikes 掩码开销——占比已小，进一步收益有限；直接进 P3。

### P3 CuPy 全 GPU（2-4 天，二阶段）
- 架构（FastFly + GeNN/Brian2GeNN 双重先例）：**全部状态 f32 常驻显存，
  整场仿真零回传**；~15 种逐元素操作用 `cupy.fuse`/`RawKernel`（NVRTC
  运行时编译，**不需要 Visual Studio/nvcc**）；整步用
  `cupy.cuda.Graph` 流捕获 + replay 21,000 次；Y 缓冲与 45×N GEMM
  留 GPU（cublas），最后只传回 45×10,500 头皮电位；
- 可选借鉴 FastFly：尖峰驱动组（MT_*/V2C/CEN_*）的事件驱动 push；
  FP16 权重（远期，需单独校准验证）；
- 预期 **全导出 <2 min**（步进工作内存受限 ~5-10 s 量级）；
- **验证判据改为统计性**（见 §3 RNG 决策）；
- **P3-1 PoC 已通过（2026-09-16，scripts/poc_cupy_parity.py，
  16/16 逐位一致）**：LIF 三变体 / OU / graded / 投递 scatter 的
  RawKernel 与 numba 参考逐位相等，关键结论——
  ① **NVRTC 必须 `-fmad=false`**（默认 fma 收缩使 OU 偏差 ~1e5，
  实测复现）；
  ② **drive 改突触后主序 CSR**：numba 按全局边序累加到 out[post]，
  GPU 原子加会打乱次序；稳定 argsort(post) 的 CSR 每线程串行累加
  与之逐位相同（构建期一次性，如路线 A）；
  ③ graded 的 numba 语义：clip 字面量 0.0/1.0 把 r 统一成 **f64**，
  `(r - s)` 是 f64 精确减法（CUDA 里按 double 复刻即逐位一致）；
  ④ 投递目标唯一（pre 行是不相交边区间）→ gather-add-scatter
  无竞争，免原子、逐位确定；
  ⑤ NVRTC 无系统头文件（stdint.h 不可用，用 `long long`）；
  CuPy 14 RawKernel 调用签名为 `kernel(grid, block, args元组)`。
  潜在注意：合成数据下 graded 的 numba 与纯 numpy 路径存在
  ~1e-7 级差异（numba 的 k 为 f64、numpy weak-scalar 全 f32），
  生产 r_pre∈[0,1] 未触发；GPU 对拍基准 = numba（生产路径）。
- **P3-2 PoC 实测（2026-09-16，scripts/poc_cupy_graph.py）**：
  ① **CUDA Graph 无收益**（eager 与 graph 每步差 <2%，启动开销
  在真实内核耗时下可忽略）→ 放弃 Graph，工程更简单；
  ② 合成随机布局下 per-post gather 内核 61.6 ms/步（随机 gather
  是延迟硬地板：64M 次随机读 ~29 ms，unroll/int32 都救不了）；
  ③ **生产布局无此问题**：y 本就按 `lexsort((pre, post))` post 主序
  连续存储，drive = 流式段求和（实测 64M 边 6.8 ms，逐位次序与
  numba 全局边序相同），无需 permute/gather；
  ④ edge_currents 真实规模是 123 万投影边（PoC 误用 64M）；
  ⑤ 生产布局全步估算 ~16 ms/步（decay 2.1 + ringadd 3.2 + 稀疏
  deliver ~3 + drive 流式 6.8 + ecur@1.23M ~0.5）→ 21k 步 ≈
  5.6 min；decay+ring+clear 可融合（traffic 减半）→ ~10-12 ms/步
  ≈ 4 min。结论：先按生产布局落地 P3-3 实测，融合优化视实测再做。
- **P3-3 引擎已落地（2026-09-16，src/ffbm/gpu.py）**：
  `GPUTrial` 全状态常驻 GPU 复刻 mech 分支循环——RNG 方案 a 实装
  （噪声按每步 R→L→MID→T45→extras 交错顺序在 CPU 预算、分批
  512 步上传，numpy 流逐位不变）；drive 走生产 post-sorted 布局的
  流式段求和；投递 = 每 pre 行一线程（不相交区间免原子），ring
  指针先推进再投递（复刻 numba 次序，bin b 落 (ptr+b-1)%len 行）。
  **验证**：① 合成小电路（含 extra 区+双 pre 源组）60/60 采样步
  全状态逐位一致（scripts/gpu_loop_parity.py，v/refrac/s/y/
  buffer/ptr/OU-x）；② 真实 visual_bilateral 电路 200 步逐位一致
  （scripts/gpu_real_parity.py），GPU **2.36 ms/步**（视觉规模，
  含记录钩子；全区域待 P3-4 实测）。曾修 1 个关键 bug：LIF 的
  i_ext 必须用 OU 更新后的状态 x，不是原始噪声 w。

### P4 多 trial 并行（零风险，随时可加）
- SNR 场景 d′=2 需 ~1,630 trials → **吞吐比延迟重要**；
- multiprocessing 16 seed 并行，代码零改动，与 P2/P3 叠乘（16 核）。

## 3. ⚠️ B3 前必须决策：RNG 流一致性（外部审查发现的暗礁）

红线原文要求"GPU 与 CPU 轨迹逐点对拍"，但 CPU 仿真的 OU 噪声/投递
抖动来自固定种子的 numpy 随机流。GPU 化后二选一：

- **方案 a（推荐）**：噪声仍在 CPU numpy 生成，分批（如每 100 步一批）
  传 GPU（pin memory，21k 步 × 每批 ~4 个小数组 ≈ 总计 ~百 MB 传输，
  ~秒级）→ GPU 轨迹与 CPU **逐位一致**，红线不变；
- **方案 b**：噪声用 CuPy 计数器型 RNG（philox）在 GPU 生成 → 序列与
  numpy 不同 → 轨迹统计去相关（与"漏 --regions 事故"同款现象），
  **必须把红线改为统计判据**（带限功率谱/相关系数/rate 轨迹 + SNR
  量级复算），并记录在案。

默认走方案 a；只有方案 a 的传输成为新瓶颈时才考虑 b。

## 4. 内存备忘

- 默认精确路径：45 × 123 万 float64 系数 ≈ **443 MB**（已按组分批
  构建；非 pool 模式下 f64 副本用后即释，仅保留 f32 应用副本）；
- 将来 EGI 256 导全量精确波形：系数内存 ×5（~2.2 GB f64）——
  f32 化 + 分组批放可解，但不耐free运行内存；届时先跑一次实测。

## 5. 明确不做 / 已否决

- **B1 GPU 核构建**：线程池已拿到 68 s（15×），GPU 边际收益 ~1 min，
  不值得；
- **迁移 Brian2/Brian2GeNN**：现成 GPU 引擎，但需重写 PhotoCascade、
  分级突触、延迟环等自定义模型，漂移风险远大于自移植 ~15 种算子；
- **kernel pooling（A1）**：曾有 24% 精度问题，维持 `--pool` 显式
  opt-in + in-run 校验的现状。

## 6. 恢复工作的第一步

1. `git log --oneline -5` 确认在含路线 A 的版本之后；
2. 执行 P0 装包 + smoke 认证；
3. P1 改 `src/ffbm/simulation.py`（edge_currents / to_neuron_drive），
   每步跑 `scripts/profile_export_loop.py 600` 看收益；
4. P2 逐函数 JIT，每个函数跑 tests/ + smoke 对拍后再做下一个；
5. 全量验证命令（后台）：
   `python viz/export_data.py --visual-input demo_bounce --elec-layout viz/data/elec_layout_1010.json --regions visual_bilateral,vpn_central,ol_rest,central_brain,vnc`
   （勿漏 --regions！）

## 7. 参考资料

- FastFly（eonfathom）：果蝇全脑 connectome 单卡实时仿真，
  Python/CuPy + NVRTC 运行时编译kernel 的完整先例
  <https://github.com/eonfathom/FastFly>
- Brian2GeNN（状态常驻架构，35-400×）：
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6962409/>
- Brian2CUDA：<https://www.frontiersin.org/articles/10.3389/fninf.2022.883700/full>
- cupy.cuda.Graph：<https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.Graph.html>
- CuPy 自定义 kernel（fuse/RawKernel）：<https://docs.cupy.dev/en/latest/user_guide/kernel.html>
- Numpy 2.5 需 numba ≥0.65：<https://github.com/numba/numba/releases>
- NVIDIA CUDA Graphs：<https://developer.nvidia.com/blog/cuda-graphs/>
