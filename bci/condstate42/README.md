# bci/condstate42：静默锁 seed-63 点火失败——最长寿命态恰是诱导最脆弱的态

## 问题

cs21 的 τ_sig(qlock)≥2988 s 是关闭表最后一个单种子寿命。本弧
逐字复制其配方（cs21locknt = DA1@150 pA 单脉冲无训练，dangate2
全套旗标，t_end 300 s，仅 seed 62→63；57.6 min GPU）。

## 主结果：不点燃

**seed 63 在 62 的校准剂量下根本进不了静默锁**：

| 量 | seed 62（cs21）| **seed 63（本弧）** |
|---|---|---|
| 脉冲窗 ALPN 峰 | 29.4 Hz | **1.47 Hz（20× 弱）** |
| 脉冲后 299.4 s ALPN max | 47.1 Hz（自持引擎）| **0.000 Hz** |
| ALPN 300 s 均值 | 19.7 Hz | **0.00012 Hz** |
| 监测器（62 折→63 评）| qlock 保持 74.7% | **99.7% 判 naive** |
| w_A 漂移 / DAN 积分 | −0.0 / 0.9 ms | −0.0 / 0.1 ms |

未点火轨迹与幼稚静息**不可分**——不是"锁得弱"，是压根没有锁。

## 确定性诊断（7e-7 V 差 ≠ 确定性破坏）

首跑后对照 condstate2/lock63 发现差 7e-7 V（cs21 的 62 对照是
逐位 0）。排查：**condstate2 的 seed-63 lock 臂用的是
cs2lock300 = DA1@300 pA 双倍剂量**（其 README 第 13 行明确记载
"150 pA（种子 62）/ 300 pA（种子 63）成锁"）。本弧数据前
**300 ms 逐位相同（0.00 V）**，分歧起点 **332 ms = 脉冲上升沿
内**——差值纯粹是剂量差，确定性框架无损；同时反证 condstate2
当年就发现 150 pA 点不着 63，靠加倍剂量才成锁。

## 语义

1. **寿命谱的长尾站在脆弱的诱导上**：τ 谱（hlock 284/336 ≪
   qlock ≥2988 ≈ resid 741~≥2948 s）里最长的 qlock 同时是唯一
   在半数种子上**无法诱导**的态——62 的 2988 s 地平线之外还有
   一道"能否进入此态"的种子门槛；
2. 对主论点（可测性边界）：最强可塑态的到达本身就是种子敏感
   的——BCI 应用视角下，"哪个种子/个体"不仅影响签名寿命，还
   决定状态能否存在；
3. 次级观察：null（naive 样）轨迹的漂移 τ_null=1399 s
   （0.00214 SD/s）——与 cs15 的 naive ≥1985 s（62）同量级，
   naive 态的短漂移跨种子成立。

## 后续（本弧启动）

τ 真复制改用 63 的成锁剂量：新增 chem 条目 **cs42locknt300**
（cs2lock300 的无训练版），condstate43 采集中（300 s，seed 63）
——数据落地后补 τ_sig(qlock,63@300pA) 与剂量-响应第四点。

## 诚实边界

- 单 trial/种子；点火阈值未扫（只有 150 不燃 / 300 燃两点，
  阈值在 (150,300] pA 内未定位）；
- 150 pA 在 63 上的 1.47 Hz 瞬态说明 ORN→PN 通路有响应但低于
  引擎自持的复发环路阈值——机制层面定位（LNL 环增益 vs
  PN 输入幅度）未做。

## 复现

```bash
# conda ffbm, repo root
python viz/export_data.py --regions visual_bilateral,ol_rest,central_brain,olfactory \
  --visual-input dark --chem-input cs21locknt --pop-rate ALPN,Kenyon_Cell,MBON,DAN \
  --plastic-mb --plastic-lr=-2e-7 --plastic-w0 0.03 --plastic-kcm-gain 0.015 \
  --mbon-dan-gain 178 --plastic-dan-gate 150,0.02,400 \
  --t-end 300000 --seed 63 --out <trial> --gpu --plastic-state-out <states>/qll63.npz
python bci/condstate42/stability_qlock63.py
```
