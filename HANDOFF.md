# HANDOFF — 项目交接文档

> 写于 2026-09-13。用途：把本项目迁移到另一台电脑 / 另一个 AI 会话继续工作。
> 本文档包含新会话所需的**全部**上下文：科学背景、数据事实、代码 API、
> 实验记录（含修过的 bug）、下一步计划、领域调研结论。读完本文即可无缝继续。

---

## 1. 这个项目是什么

**一句话**：基于官方 MaleCNS v1.0 雄性果蝇全 CNS 连接组（166,700 神经元 /
1.25 亿突触，HHMI Janelia FlyEM，CC-BY 4.0），做**视觉系统群体放电活动 →
细胞外场电位的前向生成建模**——即"假设存在头皮外脑电，如何从神经元群体
活动仿真生成它"的物理管线（以 ERG 眼表面电位为现实基准）。

**科学定位**：连接组只有结构（谁连谁、突触数、递质类型、坐标），没有动力学
和功能。本项目用 LIF 点神经元 + 真实连接结构 + 准静电前向核，把"结构"变成
"可测信号"，以经典电生理波形（ERG/LFP）为验证基准。

**已回答的可行性结论**（前几次会话的调研结果，详见 §9）：
- 视觉子网络 105,265 神经元 / 1,249 万边，细胞类型注释齐全
- 抽出的子网络完整复现教科书回路（R1-R6→L1/L2/L3；T4 输入=Mi1/Tm3/Mi9/Mi4；
  T5 输入=Tm9/Tm2/Tm1/Tm4），数据质量足够建模
- "苍蝇头皮 EEG"物理上不存在（闭场构型，nV~pV 级），正确题目是 ERG/LFP 前向

---

## 2. 新机器恢复步骤

### 2.1 环境

- **专用 conda 环境 `ffbm`**（2026-09-13 在新机器上创建并验证）：
  conda 在 `C:\Software\Devel\Anaconda3`（conda 22.9，不在 Git Bash PATH 里，
  直接用绝对路径调用）
  ```bash
  # 创建（已建好，无需重跑）
  C:/Software/Devel/Anaconda3/Scripts/conda.exe create -n ffbm python=3.12 -y
  # 本项目的 python（Git Bash 里直接用这个路径，不必 activate）
  C:/Software/Devel/Anaconda3/envs/ffbm/python.exe
  ```
  环境 Python 3.12.14，已 `pip install -e . pytest`；系统全局 Python
  （C:\Python312）保持干净，勿再往里装项目依赖
- 依赖：numpy / pandas / pyarrow / scipy / matplotlib / pytest
- 内存建议 **≥16 GB**（neuron_sites 聚合时内存映射 12.7GB 文件，本机 32GB）
- 磁盘：约 20 GB（数据 15.7GB + 余量）

### 2.2 恢复命令（在项目根目录）

```bash
# 1. 解包代码（二选一）
#    a) 压缩包方式（含 .git 历史）：
tar -xzf male-fruit-fly-brain-map-20260913.tar.gz
#    b) git bundle 方式（只有历史，无未跟踪文件）：
git clone male-fruit-fly-brain-map.bundle male-fruit-fly-brain-map

# 2. 安装基础设施包（可编辑模式；用 ffbm 环境的 python，见 §2.1）
cd male-fruit-fly-brain-map
pip install -e .

# 3. 下载数据（~13.8 GB，公开直链无需 token；syn-points 12.7GB 约 25 分钟）
python scripts/download_data.py

# 4. 重建派生数据
python scripts/build_visual_subnetwork.py   # ~1 min
python scripts/build_neuron_sites.py        # ~1-2 min（内存映射 12.7GB 文件）

# 5. 验证
python -m pytest tests/ -q                  # 应 3 passed
python experiments/exp001_erg_forward/run.py  # 应重现 ERG 三段波形
```

**不想重下数据？** 直接把旧机器的 `data/` 整个目录拷过来（raw 14GB +
derived 1.7GB），跳过步骤 3-4。

### 2.3 git 身份

本仓库 user 配置是仓库级的（`xucon <xucon@local>`），迁移后如需改：
`git config user.name "你的名字" && git config user.email "你的邮箱"`。

---

## 3. 仓库结构

**核心技术文档：`docs/TECHNICAL.md`**——非可视化部分的完整技术
说明（电路推断/仿真机制/校准方法学/前向核与验证/思想实验几何/
背景 EEG 与 SNR/已知局限/后续改进清单），改机制时同步更新它。

```
├── HANDOFF.md              ← 本文档
├── README.md               ← 项目简介（中文）
├── pyproject.toml          ← ffbm 包定义（setuptools, src 布局）
├── .gitignore              ← data/ 全部忽略（可重建）
├── data/                   ← 不进 git
│   ├── raw/                ←   官方 4 个文件（见 §4）
│   └── derived/            ←   visual_nodes/edges/neuron_sites.parquet
├── docs/data.md            ← 数据来源说明
├── scripts/
│   ├── download_data.py    ← GCS 直链下载（含 syn-points）
│   ├── build_visual_subnetwork.py  ← 全量边表 → 视觉子网络
│   └── build_neuron_sites.py       ← 3.57 亿突触位点 → 每神经元均值坐标
├── src/ffbm/               ← 基础设施包（pip install -e .）
│   ├── data.py             ← 加载器 + URL + 常量
│   ├── simulation.py       ← LIFPopulation / ExponentialSynapses
│   └── forward.py          ← StaticPairField（准静态点源-汇对核）
├── tests/test_ffbm.py      ← 3 个冒烟测试
└── experiments/
    └── exp001_erg_forward/ ← 第一个实验（完成，见 §6）
        ├── README.md       ←   实验设计 + 结果记录
        ├── run.py          ←   主脚本
        └── outputs/        ←   fig1_rates.png / fig2_erg.png / summary.json
```

实验目录规范：`experiments/expNNN_主题名/`，内含 README.md（问题、设计、
结果、局限）、run.py（可独立运行）、outputs/（小体积结果进 git）。

---

## 4. 数据：全部事实

### 4.1 原始文件（data/raw/，官方公开 GCS，CC-BY 4.0）

URL 前缀：`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

| 本地名 | 官方名 | 大小 | 内容 |
|---|---|---|---|
| body-annotations.feather | body-annotations-male-cns-v1.0-minconf-0.5.feather | 14 MB | 211,577 body × 36 列注释 |
| body-neurotransmitters.feather | body-neurotransmitters-male-cns-v1.0.feather | 43 MB | 每神经元递质预测 |
| connectome-weights.feather | connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1.05 GB | 151,856,684 条边 (body_pre, body_post, weight=突触数) |
| syn-points.feather | syn-points-male-cns-v1.0-minconf-0.5.feather | 12.7 GB | 357,489,383 个突触位点 |

**syn-points 关键列**：`x/y/z`（8nm 体素整数坐标）、`kind`（PreSyn/PostSyn）、
`body`、`conf`；还带视叶柱/层注释（medulla/lobula/lobula_plate 的 column/layer，
L/R 各一套）——**以后做视网膜拓扑/分层建模直接可用，不必另下数据**。

### 4.2 派生文件（data/derived/，脚本可重建）

| 文件 | 内容 |
|---|---|
| visual_nodes.parquet | 105,265 视觉神经元 + type/superclass/NT/dimorphism |
| visual_edges.parquet | 12,490,360 条视觉内部边 + nt_pre/sign |
| neuron_sites.parquet | 89,411,645 行 (body, kind) → 均值坐标(μm) + 位点数 |

### 4.3 测得的关键统计（不必重测）

- 完整注释神经元 = 166,700（`type` 非空；其余是片段/Orphan/Glia）
- 视觉系统（superclass ∈ ol_intrinsic 89,403 / visual_projection 9,201 /
  ol_sensory 6,098 / visual_centrifugal 563）= **105,265 神经元**，内部边
  12,490,360（全 CNS 的 8.2%），突触 4,978 万
- **R1-R6 → L1/L2/L3 真实回路**：3,377 个光感受器、5,327 个板层神经元、
  **9,636 条边、234,561 个突触**（每 R 约 2.85 个 L 伙伴、每边均值 ~24 突触
  ——四联体结构，巨型多突触 bouton）
- **坐标覆盖率**：`somaLocation` 在视网膜区几乎为零（R: 13/3,377，
  L: 797/5,327）→ **必须用 syn-points 位点坐标**（覆盖率 100%）
- `assignedOlHex1/2`（视叶六角柱坐标）：23,720 个神经元有值
- 递质：ACh 兴奋（104k）> 谷氨酸 > GABA > 组胺（R1-R6 是组胺能）；
  85,484 个有 ground_truth
- 边权重长尾：中位数 1、62% 是单突触边、max 2,591
- **左右不对称（样本真实特性）**：左眼 2,265 个 R 末梢 vs 右眼 1,112；
  左视网膜更厚（沿视轴投影 p5-p100 跨度 47 vs 21 μm）
- 眼轴标定（12 个有胞体的 R 实测）：u_out ≈ (-0.846, -0.162, -0.509)，
  末梢→胞体深度 23.5 μm

### 4.4 陷阱记录（数据相关，避免重踩）

1. **边表里大量边指向未注释碎片**（bodyId 上亿的那些）——过滤视觉子网后
   自动消失，属正常
2. **pandas 3.0 的 map/dtype 更严格**——ID 匹配时 int32/int64 混用虽然 map
   能工作，但最稳妥统一 astype(int)
3. syn-points 的 Feather 单文件 12.7GB，`ft.read_table(..., memory_map=True)`
   + Arrow 原生 `group_by` 聚合（不落 pandas）是 32GB 内存下的可行路线
4. feather 文件下载未完成时 pyarrow 报 "Not an Arrow file"——校验字节大小
   （官方大小见 §4.1）再解析

---

## 5. ffbm 基础设施 API 参考

### 5.1 单位约定（全包一致，改任何一处都要同步）

**电压 mV、电流 pA、时间 ms、电阻 GΩ**（GΩ × pA = mV 自洽）；
forward 模块：位置 μm、电导 S/m、输出 volts。

### 5.2 ffbm.data

```python
load_annotations() -> DataFrame      # 36 列注释
load_visual_nodes() / load_visual_edges()
load_neuron_sites() -> DataFrame     # body, kind, x/y/z_um, n_sites
site_positions(sites, kind) -> dict[bodyId, ndarray(3)]   # 'PreSyn'/'PostSyn'
neuron_positions(ann) -> dict[bodyId, ndarray(3)]  # somaLocation，μm（视网膜区覆盖差）
RAW_URLS / URL_PREFIX                # 官方直链
VISUAL_SUPERCLASSES / VOXEL_UM=0.008
```

### 5.3 ffbm.simulation

```python
LIFPopulation(n, dt, tau_m=20, v_rest=-70, v_th=-50, v_reset=-70,
              t_refrac=2, R_m=0.1)   # Euler；step(i_ext_pA) -> spike mask

ExponentialSynapses(pre, post, weight, post_index, dt, gain=1.0,
                    tau_s=5.0, n_post=None)
# 关键实现细节（曾出 bug，见 §6.3）：
#   - 内部按 (post, pre) lexsort 存储 y（CSR matvec 需要）
#   - deliver() 用预计算的 flat_idx（按 pre 的 argsort）投递，O(spikes×度)
#   - n_post 必须传全群体大小，否则"无入边神经元"缺行（曾出 bug）
# API: step(spiked_pre_ids) -> y(每边电流); to_neuron_current() -> 每神经元电流
```

### 5.4 ffbm.forward

```python
StaticPairField(pre_pos, post_pos, electrodes, sigma=0.33)
# phi(r) = (1/4πσ) Σ y_e (1/|r-pre| - 1/|r-post|)
# 符号约定（曾出 bug）：y>0 = 电流在 post 流入膜 = 介质汇 = 负电位贡献
# field(y_pA) -> volts;  field_timeseries(Y) 可批量
```

---

## 6. 实验记录：exp001（完成）

**问题**：R1-R6 + L1/L2/L3 群体放电能否前向生成 ERG 样眼表面电位？

**方法**（`experiments/exp001_erg_forward/run.py`，SEED=42）：
- 3,377 R + 5,327 L，9,636 边；R 用 PreSyn 均值坐标、L 用 PostSyn 均值坐标
- 闪光 500-1500ms；R: I=150+350pA、τm=10ms；L: τm=20ms；
  突触 gain=12pA/突触、τs=5ms
- 眼轴由 12 个有胞体 R 标定；右眼轴 = 左眼轴 x 分量镜像
- 电极放视网膜云外侧（云最远投影 +20μm，且最小源距 ≥30μm 双眼一致）
- 前向 = 突触偶极（R 末梢→L PSD）+ 光电偶极（末梢→末梢+23.5μm 视轴外推的
  感光部位，汇在远端）

**结果**（`outputs/summary.json`、两张图）：
- 放电：R 暗态 0 → 闪光 130Hz；L1 68 / L2 69 / L3 14 Hz
- **波形：双眼均为 负On瞬变 → 持续负平台 → 正Off瞬变，极性与经典果蝇
  ERG 一致**（R 电极平台 ~-0.7 mV 量级——量级意义，绝对值不可当真）
- **发现（左右不对称）**：左眼平台 -119μV vs 右眼 -675μV。机制：
  (a) 样本左右重建规模差 2 倍；(b) 突触偶极贡献正电位、部分抵消光电偶极
  负平台，左眼抵消更强。这是"ERG 发生器层竞争"的真实物理，可定量利用

### 6.3 本项目修过的 bug（新会话改代码前必读）

1. **前向核符号反了**（冒烟测试抓到）：汇应在 post 侧给负贡献
2. **突触投递索引错位**：y 按 post 排序存储但投递切片按 pre 排序假设算的
   → 重写为 flat_idx（argsort 索引）方案
3. **n_post 缺行**：36 个无 R 输入的 L 神经元被 CSR 丢掉 → 加 n_post 参数
4. **电极近场伪影**：电极放云内均值+25μm 处，1/d 爆炸到 mV 级、左右乱序
   → 电极必须放云外
5. **右眼轴向未镜像**：标定轴只适用左眼 → axis_side 字典按侧镜像 x 分量
6. **R 切片取前 5 个恰好都没胞体** → eye_axis 校准函数内部自行过滤

---

## 7. 已知局限（exp001 的诚实清单）

- 突触电流位置用每神经元均值（未逐突触）；光感受器"感光部位"用末梢+
  23.5μm 外推近似
- 均匀无限介质（无头壳分层电导）；LIF 点神经元（R 实际是分级电位神经元，
  不放电——LIF 是速率码近似）；矩形光电流（无光转导动力学）
- 电极位置由点云几何外推（非真实角膜曲面）
- 幅值数量级有意义，绝对值不可当真

---

## 8. 下一步实验设计（已想好技术方案，未实现）

### exp002：逐突触几何 + 光转导动力学 + 强度系列 —— **已完成（2026-09-13，见 experiments/exp002_per_synapse_erg/）**

核心结果：几何匹配恢复真实四联体接触（中位 0.159 μm）；突触成分的
细胞外可见性由回流位置决定（t-bar 邻近回流 0.4% vs 神经突跨区回流
163–193%，以均值位置核为 1）；exp001 的正 Off 瞬变确认为均值核伪影；
强度系列给出压缩式 V–log I 与强度依赖的适应下垂。原方案存档如下：

1. **逐突触偶极**：对每条边 (R_i, L_j, w)，取 R_i 的全部 PreSyn 位点 P_i 和
   L_j 的全部 PostSyn 位点 Q_j，按距离贪心匹配出 w 对（PSD 每边内唯一即可，
   跨边唯一性做不了——精确配对在 syn-partners.feather 6.8GB 里，可选下载）。
   每边有效系数 = w 对点源-汇系数的均值（推导：Σ_pairs c·y/w = y·mean(c)，
   零额外运行时开销）。注意：R 的 t-bar 也服务于 L4/无长突细胞等不在边表里
   的伙伴，几何匹配自动只用属于 L_j 的 PSD，方向正确
2. **光转导**：矩形光电流 → 双级一阶低通（τ1=τ2≈10ms，真实光转导延迟
   ~20ms）+ 慢适应（τ_adapt≈800ms，平台 ~30% 下垂，复现真实 ERG 适应现象）
3. **强度系列**：4 档强度 ×3s 闪光，输出"平台幅值 & On/Off 瞬变 vs 强度"
   曲线（真实 ERG 的经典定量关系）

exp002 衍生的下一步（按优先级）：正 Off 瞬变的再生需要 (a) 非对称光转导
（τ_off < τ_on，Hardie & Raghu 数值）或 (b) 扩电路到含 Off 通路（L2→Mi/Tm）；
回流位置假设族的实验约束（哪条核的波形能同时匹配 ERG 的平台/瞬变比）；
跨边精确配对（syn-partners.feather）。

### exp005：髓质级联 T4/T5 方向选择性 —— **已完成（2026-09-13，见 experiments/exp005_medulla_ds/）**

主产出：**(a) 2D 结构蓝图**——T4/T5 柱位由主输入伙伴票推断（Tm3/Tm4 柱位
由 L1/L2 票补齐；hex1/hex2 是六角网格两轴而非左右叶！两叶 u 轴物理镜像）；
输入偏移向量亚型特异、大多偏离 u 轴、**抑制输入偏移最大**（Mi4→T4c 1.28、
Mi9→T4d 1.63 hex 单位）；递质符号数据驱动（Mi4=GABA、Mi9=Glu，其余 ACh）。
**(b) u 轴负结果**——左叶全级联（~1.9 万神经元/167 万突触）光栅仿真，
均匀与差异化动力学（Δτ≤17ms）下 T4/T5 DSI 均 ≤0.05（差异化使响应幅值
翻倍但无方向不对称）——定量排除大 u 轴分量 DS；负结果主要是轴伪影
（偏移向量大多不在 u 轴）。exp006 设计由此导出：六方向 × 波长匹配。

### exp007：fly EEG 旗舰演示 — **已完成（2026-09-13，见 experiments/exp007_fly_eeg/）**

主线"从神经脉冲生成脑电的过程"的端到端交付：自然刺激（黑暗/1/f 闪烁/
±方向漂移纹理/DS 尺度带通纹理，时间反演对照）驱动左叶全级联（1.9 万
神经元/167 万突触），眼表面/板层/髓质三通道场电位由瞬时突触电流前向
生成（极性与幅值结构通道间分化，髓质通道 −2mV 级）。诚实发现：时间
反演对照下输入层 R 自身响应比 1.22–1.49（光适应 × 1/f 长相关的真实
非线性效应，非 DS）——自然刺激下干净 DS 需相干光栅（exp006）；
反演对照必须以输入层归一化（方法论教训）。

### exp009：数据驱动递质符号 → 板层 ON/OFF 架构 + 正 Off 瞬变 — **已完成（2026-09-13，见 experiments/exp009_lamina_polarity/）**

数据 sign 列推翻了 exp001–008 的板层极性：R→L 组胺抑制、L1→Mi1/Tm3
谷氨酸抑制、L2/L3→Tm ACh 兴奋。配 LMC 暗态基流后双重反转自动给出
ON/OFF 通路（L 光中超极化、Mi1/Tm3 光中去抑制=T4 ON；Tm1/2/4 光中被
抑制=T5 OFF），且眼电极出现**正 Off 过冲（+188μV）**——exp003 悬置的
Off 瞬变来源问题闭合（暗态活跃细胞群的主动恢复）。遗留：工作点未精调
（T5 暗态 133Hz 偏热）、LMC 反弹爆发缺失。

### exp010：人脑尺度思想实验 — **已完成（2026-09-13，见 experiments/exp010_human_scale/）**

新增 `ffbm.forward.ScalpPairField`（人脑 EEG 经典三球模型，电极在头皮层，
远极限收敛 8e-7）。把果蝇级联放进人脑壳扫尺度：S=1 → 0.6–3 nV（不可见，
即使换人脑壳"果蝇头皮 EEG"也不存在）；S=400 纯几何放大 → 暗 0.95μV
（检测极限边缘）/光响应 0.24μV（低于底）；×10³ 独立拷贝 → 7.7μV =
人类视觉诱发电位量级（与"EEG 源需 10⁶–10⁸ 同步神经元"自洽）；净偶极矩
0.05–0.19 nA·m vs 人脑源 10–100 nA·m 差距=拷贝数差距。涌现平行：光对
头皮信号是负向调制（暗态活动被抑制）= 人脑 α 阻断/ERD 形态。

### exp011：单份放大网络的检测极限 — **已完成（2026-09-13，见 experiments/exp011_detection/）**

17s 闪光串（12 循环）× S=400 头皮信号：闪光对均值的调制 −2.96μV（暗态
活动被抑制），epoch 自噪声 0.021μV（窗口平均压两个数量级）；临床级
1μV 白噪声下单次试验 d'≈28（阈值 2）——**单份放大网络 + 标准采集工程
（带宽/低阻抗/试验平均）即可成立脑电采集**；热噪声地板 0.03μV ≪ 信号。
与绝缘壳外的区别：那是严格零（物理禁止），这是弱而非零（工程可解）。
注意 exp010 的 0.24μV 是波动幅度而非均值调制。

**主线状态（exp001–014 闭环）**：结构→放电→细胞外场的完整管线已建成
并在功能层验证。2026-09-13 第三轮（扩展计划执行）：**exp013 机制板层**
（文献核实纠正了 inverted-Cl 记忆：L1/2 的 E_Cl≈−70mV 常规极性 +
去极化漏 v_K≈−25；分级释放 + HisCl 电导复现暗 −37mV/光超极化；
**off 锚点首次可达**，v1 中 neurite 回流几何独享通过——判别指向
neurite 但参数敏感，待二维扫描闭合）；**exp014 薄板扩展**
（阶段 1 盘点 + 阶段 2 T4/T5→VS 仿真：**切向偶极增强假设被定量否定**
（VS 贡献 <1e-4 μV，信号由突触数量主导）；意外收获：**正/反漂移的
头皮幅度差 15×**——场水平方向选择性成立；VS 不入主管线）。
GradedSynapsePool 已入库（tests 27 项全过）。剩余：exp013 二维参数
扫描闭合回流判别 → 分级 L1/L2 进主管线 + 末轮校准（VS/LPTC 作验收
场景）；回流判别闭合前绝对幅值保持"量级可信"。

### exp006：2D 方向扫描 — **方向选择性涌现（2026-09-13，见 experiments/exp006_ds_directions/）**

波长匹配（λ=6 hex 单位=22μm）+ 六方向扫描后 T4/T5 方向选择性自发涌现：
实测偏好方向与结构蓝图 HR 线性化预测定量一致（T4a/b/c、T5d 偏差 ≤5°，
最强 T4b DSI=0.34；T5c 预测反方向=线性化失败案例；全部聚集 −135°=
蓝图主导偏移方向）。**髓质场电位本身方向选择性（DS 向量 0.166）**。
exp005 的 u 轴负结果确认是波长失配（u 轴 DSI 0→−0.086）。主线
"结构→放电→细胞外场"在功能层面闭环。下一步 exp007：自然刺激全级联
多电极"fly EEG"演示；之后头壳体积导体。

### exp004：视网膜拓扑运动光栅 → 时间频率调谐 + 方向对称性 —— **已完成（2026-09-13，见 experiments/exp004_motion_grating/）**

核心结果：板层水平 DSI ≤ 0.02（无方向选择性，文献预期证实）；phi 频率
响应逐点复现光转导级联传递函数 1/(1+(2πfτ)²)；相位群延迟 18.8→10.2ms
吻合解析值 2τ/(1+(2πfτ)²)。衍生下一步：空间频率扫描（改 λ 即可）、
扩电路到 L1/L2→Mi/Tm→T4/T5 做方向选择性涌现（flyvis 范式，边表已有）。
原方案存档：

> 注：实际执行时 exp003 编号让位给了"非对称光转导 → Off 瞬变"追问
> （experiments/exp003_asymmetric_off/，2026-09-13 完成）。exp003 结论：
> 正 Off 瞬变需要时间尺度分离（非对称光转导）与突触场符号为正两个
> 条件同时成立；真实 Off 瞬变的发生器大概率是 Off 通路的活动性关断
> 响应（L2 是 Off 细胞）。

- R 末梢位置投影到视网膜平面（即视叶柱坐标的几何代理；或用 assignedOlHex
  ——但需先查 R 细胞的 hex 覆盖率）
- 刺激：漂移光栅 I(x,t) = I0·(0.5+0.5cos(k·x−ωt))，ω ∈ {1,2,5,10,20Hz}、
  方向 ±
- 预期（可检验的文献事实）：板层水平**无方向选择性**（方向 ± 响应对称，
  方向选择性在髓质 T4/T5 才出现）；调制幅度随时间频率衰减——闪烁融合
- 若要方向选择性：扩电路到 L1/L2→(Mi/Tm)→T4/T5（边表已有，见 §4.3 回路
  验证），即 flyvis 范式的结构基础

### exp004+：头壳分层电导（用 NeuroMechFly micro-CT 头部几何）；medulla LFP

---

## 8.5 3D 可视化（viz/）

`viz/index.html` — Three.js 实时回放页（数据 viz/data/viz_data.json 来自
exp009 工作点的一次自然刺激仿真：黑暗→1/f 闪烁→正/反向漂移→DS 带通漂移，
共 10.5s；export_data.py 可再生）：五层真实坐标点云按放电率脉动、三级连接流、
三电极搏动、头壳线框、右侧四通道电位（眼表面/板层/髓质 + **人脑头皮×400·眼轴**
粉色通道）+ **17 通道头皮电极阵列热图**（行按与眼轴夹角排序）+ 五层放电率
实时图表 + 播放控制。右上按钮 **🧠 人脑壳视图**把网络 ×400 缩放进**四层
同心球壳**（脑/脑脊液/颅骨/头皮，FourSpherePairField；CSF 层是 2026-09-13
精度升级，交叉验证：σ_CSF=σ_脑 时与三球壳解一致到 1e-8）并**移到枕极**
（T4/T5 端贴颅骨内壁，偏移 31mm，像真实视觉皮层；最强信号随之移到高夹角
电极 141-162°，与真实视觉诱发电位拓扑一致）：17 个电极均匀贴头皮（0 号在
眼轴），头皮球面按电位插值**实时着色**（topomap，蓝负红正）。对极反相
（0° vs 162° corr ≈ −0.98，偶极拓扑）。本地服务器打开
（python -m http.server），vendor/ 内置 Three.js r128 离线可用。
正确性审计脚本 scripts/audit_scalp_forward.py（双解交叉验证/反对称/
级数收敛/颅骨衰减对文献）。

## 9. 领域调研结论（前几次会话积累，新会话直接可用）

### 9.1 数据集版图

- **MaleCNS v1.0**（本项目数据）：雄性全 CNS（脑+视叶+腹索），166,700 神经元，
  Cell 2026-09-03，https://male-cns.janelia.org/ ，CC-BY；官网有 neuPrint
  （需注册 token）、Cell Type Explorer、Dimorphism Explorer；R 包 malecns
- **FlyWire/fafb**：雌性全脑（无 VNC），139,255 神经元，Nature 2024，
  flywire.ai，CAVEclient/Codex 访问；philshiu/Drosophila_brain_model 直接吃
  它的 v630/v783 文件
- 两者精度同级（突触级+人工校对），MaleCNS 更新（2026-06 定版）更全（含 VNC）

### 9.2 建模范式（本项目对标的学术线）

- **flyvis**（TuragaLab/flyvis，Lappalainen et al. Nature 2024）：连接组约束
  深度网络。视叶六角格化，类型间连接模式→卷积核，任务优化（光流），
  集成训练，与 26 项电生理研究对比验证；T4/T5 方向选择性自发涌现。
  有预训练模型 + 7 个 Colab 教程。**研究视觉编码的正统范式**
- **NeuroMechFly v2**（NeLy-EPFL/NeuroMechFly，Wang-Chen et al. Nat Methods
  2024）：具身仿真。micro-CT 身体 + MuJoCo 物理 + 视觉/嗅觉/本体感觉 +
  CPG/RL 控制器
- **Shiu et al. Nature 2024**（philshiu/Drosophila_brain_model，MIT）：全脑
  LIF（Brian2），糖→MN9 78Hz 验证，是 chetaslua/DOOMFLY 一类 demo 的源头
- **DOOMFLY**（nftechie/doomfly）等猎奇 demo：全图 166,700 神经元固定连接 +
  仅 4,184 个 KC→MBON11 突触可训练，多巴胺门控可塑性（非标准 RL），
  作者自述"not demonstrated learned survival"。工程管线可参考，科学上
  不可对标

### 9.3 有用的文献

- Berg et al., Cell 2026 — MaleCNS 数据集论文（引用必备）
- Nern et al., Nature 2025 — 视叶专项清单（optic-lobe:v1.0.1）
- Lappalainen et al., Nature 2024 — 连接组约束视觉网络
- Shiu et al., Nature 2024 — 全脑 LIF
- Wang-Chen et al., Nat Methods 2024 — NeuroMechFly v2
- Hardie & Raghu 2001 Nature — 果蝇光转导（exp002 参数依据）
- Lindén et al. 2014 Front Neuroinform — LFPy 前向建模方法学

---

## 10. 迁移后验证清单

- [ ] `python -m pytest tests/ -q` → 3 passed
- [ ] `python scripts/build_visual_subnetwork.py` 末尾打出
      "R->lamina …"（若重跑）且 T4/T5 输入类型命中文献组合
- [ ] `python experiments/exp001_erg_forward/run.py` → summary.json 里
      phi_nV 双眼均为"负 on / 负平台 / 正 off"结构
- [ ] `git log --oneline` 至少 3 个 commit（含 HANDOFF 提交）
- [ ] 新实验目录从 `experiments/exp002_*` 开始编号

---

*交接人：ZCode 会话（2026-09-13）。核心数据事实均在本机实测过，非转抄。*
