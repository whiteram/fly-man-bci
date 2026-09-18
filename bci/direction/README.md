# bci/direction：运动方向解码（4 类）

漂移光栅方向范式：0°/90°/180°/270° 全场正弦光栅（时间频率 4 Hz、
空间周期 24 px、3 s @ 60 fps）。T4/T5 方向细胞（果蝇方向视觉的
核心，exp003/004 已在视网膜电流代理下验证调谐）经标准视频输入
级联驱动；区域集 visual_bilateral,ol_rest,central_brain。

## 结果（24 试次，17 通道，LOO + 置换）

| 解码 | 准确率 | perm-p | chance |
|---|---|---|---|
| 空间 rms 模式 | 42% | 0.062 | 25%（perm95 42%）|
| **时间形状模板**（c-VEP 式） | **46%** | **0.022** | 25%（perm95 42%）|

**诚实解读**：方向信息在头皮上**弱但真实**（时间形状解码显著，
p=0.022）——漂移相位进入响应形状而非幅度，所以模板匹配优于空间
rms。这与 looming 的结论一致：运动"发生"强烈外显（threat 范式
100%），运动"方向"只余边际信息——T4/T5 对方向的对立配对
（0°/180° 等）在 17 导 rms/模板读出下大部分被抵消。人类 EEG 运动
方向解码文献同样在 30–50% 区间（chance 25%），量级相当。

## 用法

```bash
conda activate ffbm
python bci/direction/make_stimulus.py --all
python bci/direction/acquire.py      # 4 方向 × 6（~70 s/试次）
python bci/direction/analyze.py
```
