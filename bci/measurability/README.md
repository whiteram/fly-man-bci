# bci/measurability — 可测性边界总账（六弧合成图）

零采集总结弧：把 condstate12/13/14/15/16/17/18 六层可测性
闭环的全部头条数字合成**一张总图 + 一篇总账文档**。

**2026-11-15 同步修正**：总图并入 cs20/21 的实测寿命谱与修正后
σ*（qlock 0.21 / hlock 0.233 / resid **2.30 µV**）；面板 B 改为
各态寿命线 + 曲线∩寿命的 ★ 点（= 修正 σ*）；幅值阶梯差距表述
更新为 5-3600×。

- 图：`outputs/measurability_summary.png`（幅值阶梯 / 积分-漂移
  不相容（各态寿命线）/ 空间膝点 / 信道容量，四面板）
- 文档：`docs/MEASURABILITY.md`（连贯叙事 + 数字速查表 +
  三重否定论证 + 解除条件 + 复现命令）
- 数据源：condstate16/17/18/20/21 的 `outputs/*.json`（不重算；
  cs16 的 json 保留历史值，修正值在脚本内合并并标注来源）

## 复现

```bash
# conda ffbm, repo root
python bci/measurability/summary_fig.py
```
