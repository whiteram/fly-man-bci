# bci/sign2：sign 模型的气味环路——塌缩级位定位 + 重标定路线判决

sign1 Phase C 的三条后续（幅度扫描 / 腿重标定 / 按-组启用）的第一轮
判决性实验。**新旗标 `--use-sign-groups G1,G2`**（按组启用真抑制，
ckey 盐含组名集合；`CEN_R,OLR_V` 即「除 CEN_C 外全开」）。

协议：mmn_match A1（首个 DA1 标准脉冲 [300,600] ms，种子 616，
chem 运行自动工作点：CEN_C recurrence gain → 0.002，exp019 校准；
half 臂 = 0.001（工作点之半）下 base 的 AL 响应只剩 1/50）。

## C1：塌缩定位——AL 引擎本身

| A1 增量（Hz） | base | sign（全开） | 比 | periph（只开 CEN_R,OLR_V） |
|---|---|---|---|---|
| ORN | 0.95 | 0.95 | **1.000** | 0.95 |
| ALPN | 11.27 | 0.14 | **0.012（82×塌缩）** | **11.27（逐位一致）** |
| ALON | 27.6 | 1.19 | 0.043 | **27.62（逐位一致）** |
| ALLN | 28.0 | 0.18 | 0.006 | **28.00（逐位一致）** |
| KC / MBON | 0 / 0 | 0 / 0 | — | 0 / 0 |
| 头皮 dRMS（µV） | +0.104 | −0.066 | — | +0.100 |

三个定位：

1. **塌缩在 AL 局部引擎**：sign 下 ORN 逐位不变、ALPN 响应 82×
   塌缩——AL-内部行（PN↔LN）就在 CEN_C 里，40% 抑制权重上线后
   LN 的抑制真的抑制，PN↔LN 环路的再生增益被倒转。sign1 的
   「34×」是头皮的混合视角；气味信号的本体是 AL 偶极子 + 复发
   电流，不是中央放大器。
2. **KC/MBON 从来不在气味信号链上**：base 下 KC=MBON=0（condit3
   已知 PN→KC 腿被 CEN_C 工作点压制；条件化弧线的 KC 都是化学
   直驱）——sign 的气味重标定与 KC 读出腿无关。
3. **0.001 增益不支撑气味协议**：halfbase（工作点之半）的 ALPN
   只有 +0.22 Hz（工作点的 1/50）——气味环路贴着成锁阈值运行，
   增益减半即塌；chem 运行自动校准的 0.002 就是气味协议的实际
   工作点（mmdev2 口径）。

## C2：幅度补偿——定量衰减，但代价 ~30×

sign 下 ORN 驱动放大 F 倍的 A1 响应：F=1 → ALPN 0.14 Hz；
F=4 → 1.85；F=16 → 5.67（仍只有 base 的 50%）。曲线近似幂律、
无饱和——**衰减是定量的、幅度可补偿，但需要 ~30× 幅度才回到
base 水平**（ORN 150 pA → 4500 pA，生理上不可理）。
结论：chem 幅度不是重标定旋钮，只是机制注脚。

## C3：按-组启用——气味保真免费，癫痫代价照旧

**periph（`--use-sign-groups CEN_R,OLR_V`）的 A1 气味响应与 base
逐位一致**（AL 引擎自持于 CEN_C 内，外周组不反馈进 AL）；后段
窗口速率差 ~10%（ALON 45→51 Hz）、头皮差 1-2%——外周复发组
拿到真抑制，气味通路零代价。

代价验证（sz_l2 60 s 探针）：periph 与 base 同样闩锁——脉冲后
MBON 395 Hz、头皮 10.4 µV 平台（= sleep3 ctl60 基线逐位同源）。
**periph = 气味保真 + base 的癫痫易感性；full sign = 癫痫抑制 +
气味环路死亡**。两者是端点，中间无免费午餐。

## 判决（sign1 Phase C 第一轮）

- **气味环路重标定 = AL 引擎再设计**（LN 抑制权重的幅度/符号
  结构重调，如 --al-local-gain 类手术的 sign 版），不是幅度或
  增益旋钮能解决的——sign1 的判断在级位上收窄并确认。
- **`--use-sign-groups CEN_R,OLR_V` 是可用的中间保真变体**：
  外周保真免费、气味全保真，保留 epilepsy 表型——适合「要
  sign 的结构修正但不要 AL 重设计」的实验。
- 若要 full sign 下的气味功能：AL 引擎 sign 版重标定是独立的
  多轮程序（LN 权重扫描 → ALPN 自持点重标定 → 气味响应验证），
  价值取决于是否需要「真抑制的中央脑」这一保真度。

诚实边界：单种子；KC/MBON=0 使「气味→KC 读出」的 sign 效应
在本协议不可测（该腿本来就不通）；periph 的后段窗口 ~10% 差异
来自外周组对中央复发的贡献改变（未逐群分解）；sz_l2 探针无 STD。

## 用法

```bash
conda activate ffbm
python bci/sign2/acquire.py --arms base,sign,periph
python bci/sign2/acquire.py --arms ampsign_4,ampsign_16
python bci/sign2/analyze.py
```

汇总：outputs/summary.json。
