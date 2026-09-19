# bci/thermal：温度与湿度辨别——新模态区域 thermal（exp023）

新区域 `thermal`：TH 25 个温度受体（TRN_VP1m 热通路 11 / VP2+VP3*
冷通路 14）+ HY 66 个湿度受体（VP4 干 28 / VP1*+VP5 湿 46），
各自 ~45k 突触的 CEN 前馈偶极子（TH_C 1,257 对/44,989 突；
HY_C 2,524 对/45,968 突）。3s 慢斜坡（300ms 沿）——TRN/HRN 紧张
性发放的自然包络。

## 结果（每类 6 runs = 24 试次，3.6s）

| 块 | 类 | 空间 rms LOO | 置换 p | 整体 rms |
|---|---|---|---|---|
| 温度 | warm vs cool | **92%** | **0.0170** | 平坦（1.392→1.392）|
| 湿度 | dry vs moist | 58%（n.s.）| 0.4747 | 平坦 |

- **温度 2 类 92% 阳性**：整体 rms 几乎不动（第 3 位小数内）而
  空间模式可分——又一次呼应"空间模式是最稳健读出轴"的主线
  结论（25 个受体的末梢分布差异被 17 电极分辨）；
- **湿度诚实阴性**：HRN_C 投射与温度同量级（46k 突），干/湿类
  却不可分（58%）——干/湿受体的中央靶点重叠度高于热/冷通路
  （结构假设；与味觉 GRN 阴性同类的"通路几何决定可见性"）。

## 用法

```bash
conda activate ffbm
python experiments/exp023_thermal/circuit8.py   # 区域报告
python bci/thermal/acquire.py                   # 24 试次（~25 min）
python bci/thermal/analyze.py                   # 双块 LOO + 置换
```
