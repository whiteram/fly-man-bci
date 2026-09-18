# bci/gustatory：味觉器官身份解码（3 类）

每试次 3.6 s，单器官糖脉冲 900 ms（@300 ms），类别 = 器官：唇瓣
LgLG（667 细胞，400 pA）/ 翅 WG（385，693 pA）/ 跗节 claw_tpGRN
（50，5336 pA）——幅度按群大小均衡总驱动（嗅觉范式的同款修正）。
电路复用 exp019（`gustatory` 区域，GRN_C 8,381 对 / 150,217 突触）。

## 结果（18 试次，17 通道空间模式 LOO + 置换）

| 解码 | 准确率 | perm-p | chance |
|---|---|---|---|
| 3 类器官身份 | 39% | 0.654 | 33%（perm95 56%）|

三类 rms 几乎相同（0.641 / 0.641 / 0.657 µV）——**阴性结果，如实
报告**：器官均衡驱动下头皮上没有器官身份信息。与 exp019 的结构性
结论一致：GRN_C 是全部感觉输入边组里最小的（8.4k 对，约为
ORN_C 的 1/7、TC_C 的一半），GRN 输出大头走向 VNC 反射弧
（forward=False），器官身份到不了头皮。味觉通道在本模型里是
"结构上不可见"的模态——除非未来上调 GRN→SEZ 的增益或复归招募。

## 用法

```bash
conda activate ffbm
python bci/gustatory/acquire.py      # 3 类 × 6（与 touch 共区域集）
python bci/gustatory/analyze.py
```
