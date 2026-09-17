# exp019: 化学感觉输入（嗅觉 + 味觉）

把连接组的另外两条感觉输入通道接进"蝇人"仿真：气味 → ORN → 触角叶、
糖 → GRN → 咽下神经节（SEZ），并验证它们能否在头皮 EEG 里看到。
模态→神经输入的完整映射文档见 [docs/SENSORY_INPUTS.md]
(../../docs/SENSORY_INPUTS.md)。

## 电路（circuit4.py，注册为 ffbm.regions 区域 `olfactory` / `gustatory`）

| 区域 | 群体 | 连边组（w≥5, 共识递质符号） |
|---|---|---|
| olfactory | ORN 2,639（class=olfactory；触角神经 2,455 + 下颚须 184；53 个肾小球类） | ORN_R 9,721 对 / 77,895 突触（本地，forward=False）；ORN_C **57,489 对 / 954,464 突触** → 触角叶（ALLN+ALPN 等 cb_intrinsic，已在 central_brain 区域内），forward=True |
| gustatory | GRN 1,428（class=gustatory；类型=器官：唇瓣 LgLG* 667、翅 WG* 385、跗节 claw_tpGRN 50、咽 PhG* 等） | GRN_R 12,352 对；GRN_C **8,381 对 / 150,217 突触** → SEZ（forward=True）；GRN_V 16,210 对 → VNC（forward=False，仅当 vnc 区域开） |

位置：ORN/GRN 没有 somaLocation（外周胞体），用**平均突触前位点**
（中枢端末梢）——即 ORN→ALPN / GRN→SEZ 突触电流偶极子的源头，
2,639 + 1,428 个全部有。`chem_groups`（标注类型 → 紧凑索引）存进
circuit 供 `--chem-input` 解释刺激规格。

## 转导与标定（重要模型发现）

- **转导级联**：ORN/GRN 与视觉共用 PhotoCascade 动力学（10 ms 低通×2
  + 800 ms 适应到 30% 平台，`pipeline.simulate(chem_fn=...)`，GPU 走
  `k_chem_add`）。真实 ORN 是相张性响应；级联的适应让驱动在 ~1 s 内
  衰减到平台。
- **中央脑点火吸引子**：exp017 的中央脑在其默认递归增益（CEN_C
  1.7M 对 / 29M 突触，G_UNIT_CX）下是激发锁定网络——**任何阈上化学
  输入都把整个 CEN 锁进 ~82 Hz 自持态且永不衰减**。分岔极窄：驱动
  80→150 pA（或增益 0.0015→0.0018）之间从静默直跳点火；400 ms 短脉冲、
  降 CEN 兴奋性（icen 60→20 pA）都压不住。
  **chem 运行工作点**：CEN_C（纯内部递归边组，前向不经过）增益覆写
  为 **0.002**（export_data.py chem 块，仅 --chem-input 时生效）→
  响应变为触角叶特异（ALPN ~30 Hz / ALLN ~50 Hz，其余 CEN ~0.1 Hz）
  且脉冲结束后 ~1–2 s 衰减。视觉实验行为不变。
- **脉冲长度门槛**：AL 募集需要 ≥~600 ms 的全幅驱动（900 ms 脉冲
  @150 pA 可靠点燃；500 ms + 100 ms 平滑沿→ORN 只发 ~7 Hz/个→
  达不到募集阈值）。全量协议 2 s 脉冲安全。

## 验证结果（快速迭代协议：3.6 s 试次，900 ms 脉冲 @ 300/1500/2700 ms）

三条件：baseline（全暗视觉，无化学驱动，10.5 s） /
odor3（cVA→DA1 204 个、果醋酯→DM2 54 个、脂肪酸→VA1v 130 个，各
150 pA）/ taste3（唇瓣/翅/跗节糖脉冲，400 pA）。

| 检验 | 嗅觉 | 味觉 |
|---|---|---|
| 驱动到达外周 | ORN 三通道各自特异：脉冲内 0.65 / 0.18 / 0.42 Hz 群体均值（≈7 Hz/神经元），**脉冲间隙严格 0.00** | GRN 器官特异：27.6 / 16.5 / 2.0 Hz（群体均值，通道大小 667/385/50） |
| 驱动到达中央 | CEN 募集 ~1 Hz 群体均值（≈ALPN 20+ Hz），通道持续态在间隙不衰减 | SEZ 不募集（0.0 Hz）：GRN_C 突触量只有 ORN_C 的 1/6 |
| 头皮可见 | **0.1–4 Hz rms 0.62→0.87 µV（×1.4）**；首脉冲点火瞬态带宽内 ×1.65 | ×1.04（无中央放大，GRN_C 偶极太弱） |

味觉的解剖学解释：GRN 下游大头在 VNC 反射弧（16,210 对，forward=
False 不投影），头壳内只有 SEZ 一支（8,381 对）——**味觉的头皮
EEG 签名天然弱**，这是解剖结构不是 bug。要做出可检测的味觉 EEG 需
(i) 更长/更强驱动 recruit SEZ 或 (ii) 多试次平均。嗅觉的信号放大
来自触角叶回路（ALPN/ALLN 胞体电流，CEN_C/CEN_R 前向核）。
阴性对照：幅度 800 pA（GRN 打到峰值 74 Hz）时 SEZ 依然不募集——
中央限制不是驱动强度问题，是 SEZ 伙伴的递归连接规模问题。

## 工程修复（本次迭代中发现）

- pipeline.simulate 的 chem 级联最初为**每个** extra 群体每步分配
  零向量（OLR 62k + CEN 33k），~0.8 MB/步的分配churn 使 RSS 气球化
  到 14 GB → 换页 → 322 ms/步（1932 s/试次）。修复（预分配复用 +
  只步进有驱动/有残留状态的级联）后回到 **208 s/试次**（58 ms/步）。
- export_data.py 新增 `--t-end`（短试次）与 `_debug_extra_rates.npz`
  （每个 extra 群体的 1 kHz 群体发放率，CPU 路径）。

## 复现

```bash
conda activate ffbm
# 电路装配报告
python experiments/exp019_chemosense/circuit4.py
# 快速验证（~3.5 min/试次；baseline 复用 10.5 s 全量）
python experiments/exp019_chemosense/run.py --fast
python experiments/exp019_chemosense/run.py --analyze-only
# 全量 10.5 s 三条件
python experiments/exp019_chemosense/run.py
```

状态：嗅觉验证通过（外周特异 → AL 募集 → 头皮 ×1.4–1.65）；味觉
外周验证通过、中央/头皮受解剖限制（见上）。后续 BCI 范式（嗅觉
N 类分类解码）建议以 ORN_C/触角叶持续态 + 全量 2 s 脉冲协议为基础，
走 bci/olfactory/ 分支。
