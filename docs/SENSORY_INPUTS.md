# 感觉输入映射文档（感官 → 神经输入 → 代码）

本文档回答一个问题：**每一种感觉模态，对应到 MaleCNS v1.0 连接组里的
哪些神经元、以什么方式注入、由哪段代码实现**。每个已实现的模态都给出
数据侧证据（本地 `data/raw/body-annotations.feather` 可复核的标注）、
注入机制、刺激注册表和代码入口。

范围决定：按项目方向，我们做 **单向读出**（果蝇网络 → 头皮 EEG），
不做反向/双向接口（不把外部信号注入后"驱动"果蝇做闭环控制——那是
awesome-fly 社区 closed-loop-fly/webgpu-fly 一类项目的做法）。

## 总览

| 模态 | 神经输入（数据标注） | 数量 | 注入方式 | 代码入口 | 状态 |
|---|---|---|---|---|---|
| 视觉 | R1–R6 光感受器（`superclass=ol_sensory`, type R1-R6） | 3,377 | 亮度→光电流（PhotoCascade 转导级联） | `--visual-input` | ✅ 已有（exp001–015） |
| 嗅觉 | 嗅觉受体神经元（`class=olfactory`, type ORN_*） | 2,639 | 气味→肾小球通道电流 `chem_fn` | `--chem-input` | ✅ exp019 |
| 味觉 | 味觉受体神经元（`class=gustatory`） | 1,428 | 糖脉冲→器官通道电流 `chem_fn` | `--chem-input` | ✅ exp019 |
| 本体感受/运动 | 腿本体感受 SNpp/SApp、弦音器官、钟形感器、板；读出 = 下行神经元 DN/腿运动 MN | ~1,454 + ~1,457 感觉；1,342 DN | （规划）关节状态反馈注入 + DN/MN 群读出 | exp020（待做） | 📋 规划 |
| 听觉 | Johnston 器 JO-A/JO-B + AMMC 通路 | 115 + 208 | （规划）近场声（150–500 Hz 求偶歌段）→ JON 电流 | 未排期 | 📋 空白（社区也无人做） |

为什么听觉也在清单里：果蝇**有**听觉——触角第二节梗部的 Johnston
器官（弦音器官的一种）感受近场声，上行到触角机械感觉运动中枢 AMMC。
本地数据可验证：`class/subclass/type` 含 JO-A1/A2/B1/B3/CA2 共 115 个
（auditory），AMMC 相关 208 个。社区（awesome-fly 全部项目）无人实现
声音输入，属于空白。

---

## 1. 视觉（已实现，exp001–015）

**数据侧**：R1–R6 光感受器 3,377 个（左眼 2,265 / 右眼 1,112），由
exp015 的 bilateral 构建器选入核心级联。

**映射**：外部世界 → 视网膜采样 → 光转导 → 级联

```
视频帧/1/f 纹理 → 每只眼各自完整采样一份屏幕图像（中矢状面镜像,
对齐每只眼 1–99 百分位视野足迹）→ 每个 R 的亮度 b(t) ∈ [0.05, 3]
→ PhotoCascade（10 ms 滤波+适应+sag）→ I_R_BASE(150 pA) + i_lum(150 pA)×(b−1)
→ R(LIF) → 组胺 → L1/L2 → Mi/Tm → T4/T5 → (可选) VPN/中央脑
```

**代码**：
- 刺激注册表 `viz/data/visual_inputs.json`（`natural_1d` 自然协议 /
  `demo_bounce` 演示 / `ssvep1_f*` 单目标频闪 / `dark` 全暗基线）
- 注入 `viz/export_data.py`（`--visual-input`；`luminance(t)` 闭包 →
  `fp.simulate(..., lum_inc_fn)`）
- 级联与群体 `src/ffbm/pipeline.py`（R/L/MID/T45 + extra 区域机制）

## 2. 嗅觉（exp019 实现）

**数据侧证据**（可用下方一行代码复核）：

- `class == "olfactory"`：**2,639 个 ORN**（触角神经 AN 入 2,455 +
  下颚须神经 MxLbN 入 184），类型按**肾小球**命名（53 类）：
  ORN_DA1 204、ORN_VA1d 132、ORN_VA1v 130、ORN_DL3 103 …
- 下游（connectome-weights, w≥5）：→ ALLN（触角叶本地神经元，40k 突触）
  + ALPN（投射神经元，16k 突触）+ ORN↔ORN 本地 9.7k。**ALLN/ALPN 全部
  是 `cb_intrinsic`，即已在 central_brain 区域内被仿真**——嗅觉区域
  只需加 ORN 群体和连边，触角叶回路"免费"复用。
- 连边组：`ORN_C`（ORN→中央脑）57,489 对 / 954,464 突触；`ORN_R`
  （ORN↔ORN）9,721 对。
- 位置：ORN 胞体在触角里，MaleCNS **没有**它们的 somaLocation；改用
  **平均突触前位点**（中枢端末梢，在触角叶神经毡内）——这正是
  ORN→ALPN 突触电流偶极子的源头。所有 2,639 个都有。
- 前向：`ORN_C` forward=True（触角叶偶极进头皮核）；`ORN_R`
  forward=False（外周轴突侧，与 exp017 对 VNC 的处理同理）。

**映射**：气味分子 → 受体 → 肾小球组合编码 → 电流注入

```
气味 X(浓度) → 按"气味→受体→肾小球"文献映射选出 ORN 子群
→ 转导级联（10 ms 低通×2 + 800 ms 适应至 30% 平台，与视觉同一
  PhotoCascade 动力学——真实 ORN 是相张性响应）
→ I_ORN_BASE(60 pA) + amp_pa × 级联输出
→ ORN(LIF) → ORN_C 电导连边 → 触角叶(ALLN/ALPN, 在 CEN 内)
→ CEN_C → 中央脑 → 头皮 EEG 前向
```

气味→肾小球映射依据（chem_inputs.json `note` 字段同步）：
- **cVA**（cis-vaccenyl 乙酸酯，雄性信息素）→ Or67d → **DA1**
  （Kurtovic et al. 2007）——雄性连接组里有完整的信息素通道；
- **果醋酯类**（乙酸乙酯/丁酸乙酯）→ Or22a → **DM2**
  （Hallem & Carlson 2006 强响应；Fishilevich & Vosshall 2005 图谱）；
- **脂肪酸蒸汽** → Or47b → **VA1v**（Dweck et al. 2015）。

**代码**：
- 构建器 `experiments/exp019_chemosense/circuit4.py::build_olfactory`
  （注册于 `src/ffbm/regions.py` 区域名 `olfactory`；要求
  central_brain 同时开启）
- 刺激注册表 `viz/data/chem_inputs.json` 条目 `odor3_block`
  （3 气味 × 顺序 2 s 脉冲块；幅度 150 pA = 与视觉 i_lum 同为满量程）
- 注入 `viz/export_data.py --chem-input` → `chem_fn(t)` →
  `ffbm.pipeline.simulate(..., chem_fn=...)`（GPU 路径
  `ffbm.gpu` 内核 `k_chem_add`）
- 验证实验 `experiments/exp019_chemosense/run.py`

**文档化为假设**：ORN/GRN 与视觉共享转导级联动力学（10 ms 低通 +
800 ms 适应、30% 平台；真实 ORN 相张性响应的近似，参数未独立标定）；
浓度→电流线性；气味 = 若干肾小球的组合脉冲（真实气味激活几十类
受体，此处取文献最强通道）。

**标定发现（exp019，重要模型性质）**：exp017 的中央脑在默认递归
增益（CEN_C，1.7M 对/29M 突触，`G_UNIT_CX`）下是一个**点火吸引子**
——任何阈上化学输入都会把整个 CEN 锁进 ~82 Hz 自持态且永不衰减
（阈值极窄：驱动 80→150 pA 或增益 0.0015→0.0018 之间从静默直接
跳到点火）。**chem 运行的工作点**：把 CEN_C（纯内部递归边组，
前向通路不经过它）的增益覆写为 0.002（`export_data.py` chem 块），
响应变为触角叶特异（ALPN ~30 Hz / ALLN ~50 Hz，其余 CEN ~0.1 Hz）
且脉冲结束后 ~1–2 s 衰减。视觉实验保持 exp017 行为不受影响。
标定过程与数据见 exp019 README。

## 3. 味觉（exp019 实现）

**数据侧证据**：

- `class == "gustatory"`：**1,428 个 GRN**。类型标注携带**器官**而
  非品质：唇瓣 LgLG1a–8（667）、翅 WG1–4（385）、跗节 claw_tpGRN（50）、
  咽 PhG1–16、腿 SNpp 味觉亚类等——**甜/苦/水通道在 MaleCNS 里没有
  标注**（文档化为假设：按器官分通道，品质不区分）。
- 下游（w≥5）：→ SEZPN + cb_*（咽下神经节回路，`GRN_C` 8,381 对 /
  150,217 突触）+ VNC 上升支（`GRN_V` 16,210 对，forward=False）
  + GRN↔GRN 本地 12,352 对。
- 位置/前向：同 ORN（平均突触前位点；GRN_C forward=True——咽下
  神经节在头壳内）。

**映射**：糖液接触器官 → 该器官 GRN 群脉冲（经同一转导级联）→ SEZ/中央脑

```
糖(t) → 转导级联 → I_GRN_BASE(60 pA) + amp_pa × 级联输出 → GRN(LIF)
→ GRN_C → SEZ(cb_intrinsic, 在 CEN 内) → 中央脑 → 头皮 EEG
```

**代码**：同嗅觉，条目 `taste3_block`（唇瓣/翅/跗节三通道顺序脉冲；
构建器 `build_gustatory`，区域名 `gustatory`）。GRN_C 连边强度只有
ORN_C 的约 1/6（150k vs 954k 突触），同一标定工作点下的响应预期更弱。

**验证结果（exp019，3.6 s 快速协议，900 ms 脉冲）**：ORN 三通道
各自特异（脉冲内 0.65/0.18/0.42 Hz 群体均值 ≈7 Hz/神经元，间隙
严格 0）；CEN 募集 ~1 Hz（≈ALPN 20+ Hz）；**头皮 0.1–4 Hz rms
0.62→0.87 µV（×1.4），首脉冲点火瞬态带宽内 ×1.65**。味觉：GRN
器官特异发放干净（27.6/16.5/2.0 Hz @400 pA），但 SEZ 不募集
（800 pA 阴性）→ 头皮签名 ×1.04——味觉连边大头在 VNC 反射弧
（forward=False），头壳内只有 SEZ 一支，签名弱是解剖性的。
数据与解读详见 exp019 README。

## 4. 本体感受 / 运动感觉（exp020 实现，运动想象范式）

**数据侧证据**：

- 本体感受器：`mechanosensory_proprioceptive` 1,454（腿 SNpp 系列、
  翅 SApp 系列；弦音器官 425、钟形感器 426、毛板 113 均为其子集），
  侧别在 `rootSide`（L 735 / R 718）——**左/右就是运动指令的分类轴**。
- 下游（w≥5）：VNC 反射弧主通路（vnc_intrinsic 36k + **vnc_motor
  3,062+** 突触，`PRO_V` 47,888 对，forward=False）+ 上升支/中央
  （`PRO_C` 4,847 对 / 56,417 突触，forward=True）+ PRO 本地。
- 运动输出侧：1,342 个下行神经元 DN（胞体在中央脑，**forward=True**
  ——运动计划的头皮源）+ 311 个腿运动神经元（VNC 内，执行层）。

**运动想象范式映射**（`mi_run.py`）：

```
指令窗(300-1200 ms)：一侧腿本体感受器注入 400 pA 运动反馈模式
                    （= 意向运动的感知后果/感知副本）
延迟窗(1200-3600 ms)：无驱动——若头皮上仍有类别信息，说明它被
                    中央（DN 层）持续态保持，即"无执行的运动计划"
解码：17 通道 0.1-4 Hz 带内 rms 特征，最近质心分类器，
    cue 窗 vs 三个延迟子窗分别报告，置换检验给 chance
```

**状态**：区域已实现（`proprioception`，构建器
`experiments/exp020_proprioception/circuit5.py`，I_PRO_BASE 60 pA，
chem_groups 按 `类型@侧别` 建组如 "SNpp*@L"）。

**运动意图解码结果（12 试次，诚实版）**：指令窗 L/R 可解码
（最近质心 99–100%，效应 +0.0023 µV、p=0.002）——但信号完全来自
PRO_C 末梢偶极子（指令窗 PRO 25 Hz、CEN 0.005 Hz），幅度只有背景的
0.3%、比缩放前提的假想设备噪声低 ~30×；**延迟窗（驱动停止后）
无任何类别信息**（PRO/CEN 严格 0 Hz，原始 rms 差 ≈0，p=0.09–0.90）
——第一版带通包络分析报出的 78–89% 是 filtfilt 滤波器长尾伪影
（exp018 已记录的坑）。结构性结论：PRO_C 与 GRN_C 同为 ≤5k 对量级
输入边组，在 CEN_C=0.002 工作点点不亮中央；嗅觉"可见"靠的是
ORN_C 57k 对+触角叶回路放大。真正的"运动想象保持"需要 DN 层被
招募（更大本体→中央连边，或对 DN 直接注入感知副本指令）——后续方向。

## 5. 听觉（空白，未排期）

数据侧：JO-A1/A2/B1/B3/CA2（Johnston 器听觉神经元）115 + AMMC 208。
近场声 150–500 Hz（求偶歌脉冲/正弦段）→ JON 电流 → AMMC → 中央脑。
实现路径与嗅觉完全同构（一个新区域构建器 + chem_fn 式驱动），
awesome-fly 社区无先例，适合作为差异化扩展。

---

## 实现位置说明（为什么这样放）

- **区域构建器与实验同目录**：`experiments/expNNN_*/circuit*.py` 是
  本仓库的既定模式（exp015 bilateral / exp016 vpn / exp017 full-CNS），
  `src/ffbm/regions.py` 是唯一知道如何组装它们的注册表；exp019 沿用。
- **刺激是数据不是代码**：刺激定义放在注册表 JSON
  （`visual_inputs.json` / `chem_inputs.json`），代码只提供解释器；
  换刺激不重建任何缓存（与换视频同一机制）。
- **缓存**：circuit_key 哈希 `regions + 数据文件 + 构建器代码清单`
  （cache.py `_CIRCUIT_CODE` 含 circuit4.py）；chem 驱动是运行时输入，
  不参与任何 cache key（种子同理）。新区域集首次运行构建一次
  circuit+kernel 缓存，之后命中。
- **不做反向接口**：驱动只进感觉神经元（R/ORN/GRN），不消费模拟
  输出去控制外部身体；无闭环。

## 复核与复现

```bash
# 数据侧验证（本文档的标注数字）
conda activate ffbm
python -c "import pandas as pd; a=pd.read_feather('data/raw/body-annotations.feather'); \
print(a['class'].value_counts().head(12))"

# 电路装配报告（区域细胞数/连边组/chem_groups）
python experiments/exp019_chemosense/circuit4.py

# 化学感觉验证实验（3 试次：baseline/odor3_block/taste3_block）
python experiments/exp019_chemosense/run.py            # 全量 10.5 s
python experiments/exp019_chemosense/run.py --smoke    # 400 ms 接线检查

# 单试次（标准导出入口，可 --gpu）
python viz/export_data.py --regions visual_bilateral,ol_rest,central_brain,olfactory,gustatory \
    --visual-input dark --chem-input odor3_block --out <dir> [--seed N]
```
