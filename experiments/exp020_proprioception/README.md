# exp020: 本体感觉输入（运动感觉）+ 运动意图解码

把身体的"运动感觉"通道接进仿真，并做第一个运动想象式 BCI 实验：
从头皮 EEG 解码"哪个运动被指令了"。模态映射文档见
[docs/SENSORY_INPUTS.md](../../docs/SENSORY_INPUTS.md) §4。

## 电路（circuit5.py，区域名 `proprioception`）

- **PRO 群体 1,454**（class `mechanosensory_proprioceptive`；腿 SNpp +
  翅 SApp；弦音器官/钟形感器/毛板为其子集）。位置=平均突触前末梢
  （全部有）；侧别来自 `rootSide`（**L 735 / R 718**，somaSide 为空）。
- 连边（w≥5，共识递质符号）：`PRO_V` → VNC 反射弧 **47,888 对 /
  687,803 突触**（vnc_intrinsic + vnc_motor，forward=False）；
  `PRO_C` → 中央上升目标 **4,847 对 / 56,417 突触**（forward=True）；
  `PRO_R` 本地 1,325 对。
- 运动输出层：DN 1,342（中央脑内，forward=True）+ 腿 MN 311
  （VNC 内，vnc_motor）。`chem_groups` 按 `类型@侧别` 建组
  （如 `SNpp*@L`），驱动复用 chem_fn + 转导级联，零新管线代码。
- 工作点：I_PRO_BASE 60 pA；chem 运行自动套用 CEN_C→0.002（exp019）。

## 运动想象范式（mi_run.py，12 试次 = 左/右 × 6 种子）

每试次 3.6 s：300–1200 ms 指令窗——向一侧腿本体感受器注入 400 pA
运动反馈模式（= 意向动作的感知副本）；其后 2.4 s 无驱动
延迟窗——检验类别信息是否被中枢持续态保持（"无执行的运动计划"）。
区域集：visual_bilateral,ol_rest,central_brain,**vnc**,proprioception。

## 结果（诚实版）

| 窗口 | L rms (µV) | R rms (µV) | 差值 | 置换 p |
|---|---|---|---|---|
| cue 300–1200 | 0.6968 | 0.6945 | **+0.0023** | **0.002** |
| delay 1300–2000 | 0.7328 | 0.7329 | −0.0001 | 0.512 |
| delay 2000–2700 | 0.7325 | 0.7328 | −0.0003 | 0.089 |
| delay 2700–3500 | 0.7329 | 0.7329 | +0.0000 | 0.904 |

1. **指令窗可解码**：L/R 最近质心分类 99–100%（干净信号）。但效应
   只有背景的 **0.3%**（0.0023/0.695 µV）——信号完全骑在 PRO_C 末梢
   偶极子上（指令窗 PRO 25 Hz/个、CEN 仅 0.005 Hz，中央没被点亮），
   且比 exp018 缩放前提下的假想设备噪声（0.064 µV 带内）还低 ~30×
   ——干净设定下可分，真实采集需要大量平均。
2. **延迟窗无信息 = 没有中枢"运动计划保持"**：驱动结束后 PRO 与
   CEN 严格 0.0 Hz，原始 rms 差值 ≈0（p=0.09–0.90）。**教训**：第一版
   分析用 0.1–4 Hz 带通包络曾报出延迟窗 78–89% "解码"，那是滤波器
   冲激响应把指令窗响应拖进延迟窗的伪影（exp018 已记录的 filtfilt
   长尾坑）；本解码器已改为无记忆的原始 rms + 直接置换检验。
3. **结构性结论**：PRO_C 与 GRN_C 一样（≤5k 对量级的输入边组）在
   CEN_C=0.002 工作点点不亮中央回路——嗅觉之所以"看得见"，是因为
   ORN_C 有 57k 对 + 触角叶回路的放大。要让"运动想象"真正可在延迟
   窗解码，需要 (i) 更大的本体→中央连边驱动 DN 持续态（生物学上
   对应实际运动后的感知回馈再入），或 (ii) 直接对 DN 群注入指令
   （corollary discharge,EOF 有解剖数据支持）——列为后续方向。

## 复现

```bash
conda activate ffbm
python experiments/exp020_proprioception/circuit5.py        # 装配报告
python experiments/exp020_proprioception/mi_run.py          # 12 试次采集+解码
python experiments/exp020_proprioception/mi_run.py --decode # 仅解码
```

（首试次含新区集 circuit/内核构建；稳态 ~190 s/试次。）
