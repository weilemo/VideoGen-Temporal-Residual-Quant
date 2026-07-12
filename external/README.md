# External References

这个目录存放从其他实验仓或参考实现拷贝来的代码与分析结果。

## 目录约定

- `focused-forcing-code/`：Focused-Forcing 相关代码副本，以及已分析好的 `dm_loss.json`。

## 使用原则

- `temporalresidualkvquant/` 是当前主方法代码库，后续新方法和实验入口优先放在那里。
- `external/` 下的内容作为参考或数据源使用，尽量不直接作为主线开发位置。
- 如果从 `external/` 中提取结论、配置或 policy，建议同步记录到 `STATUS.md`、`DECISIONS.md` 或 `logs/`。

## 已知可用数据

Focused-Forcing 的 DMD loss 分析结果位于：

- `focused-forcing-code/focusedforcing_sf/dm_loss.json`
- `focused-forcing-code/focusedforcing_cf/dm_loss.json`
- `focused-forcing-code/focusedforcing_rf/dm_loss.json`
- `focused-forcing-code/focusedforcing_longlive/dm_loss.json`

这几份文件当前内容相同，包含 360 个 global head 的 DMD loss 分数，可用于生成 `temporalresidualkvquant` 的 top-k head policy。
