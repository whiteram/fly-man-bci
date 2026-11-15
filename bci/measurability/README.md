# bci/measurability — 可测性边界总账（六弧合成图）

零采集总结弧：把 condstate12/13/14/15/16/17/18 六层可测性
闭环的全部头条数字合成**一张总图 + 一篇总账文档**。

- 图：`outputs/measurability_summary.png`（幅值阶梯 / 积分-漂移
  不相容 / 空间膝点 / 信道容量，四面板）
- 文档：`docs/MEASURABILITY.md`（连贯叙事 + 数字速查表 +
  三重否定论证 + 复现命令）
- 数据源：condstate16/17/18 的 `outputs/*.json`（不重算）

## 复现

```bash
# conda ffbm, repo root
python bci/measurability/summary_fig.py
```
