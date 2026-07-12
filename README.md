# videoquant-trq 项目协作记录

`videoquant` 聚焦长视频自回归 diffusion 生成中的 KV cache quantization，目标是在显著降低 KV 显存占用的同时，尽量不破坏长时视频质量，尤其是 identity consistency、scene consistency 和 motion continuity。

这套记录文件服务统一后的 TRQ 研究工作区，供开发、训练、调试、评测与作业排查使用。

## 给协作者的代码入口

实际方法代码在 `temporalresidualkvquant/`。第一次使用请按下面顺序阅读：

1. [temporalresidualkvquant/README.md](temporalresidualkvquant/README.md)：当前框架、稳定边界和快速命令。
2. [temporalresidualkvquant/docs/getting_started.md](temporalresidualkvquant/docs/getting_started.md)：环境、生成实验、诊断、评估和排错。
3. [temporalresidualkvquant/docs/trq_unification.md](temporalresidualkvquant/docs/trq_unification.md)：TRQ codec contract 与历史兼容关系。

最小验证：

```bash
cd temporalresidualkvquant
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 当前本机布局（这部分不用看）

```text
/Users/moweile/Code/LAB/
  videoquant-trq/      # 当前统一后的主仓库
    temporalresidualkvquant/   # 方法、策略、codec 与 Self-Forcing 集成
    Quant-VideoGen/    # 原始 QVG 参考实现
  qvg/                 # 学弟仓库，保留作算法参考
```

进入主仓库：

```bash
cd /Users/moweile/Code/LAB/videoquant-trq
```

查看当前状态：

```bash
git status --short --branch
```

五月份服务器上的多 worktree 布局属于历史执行环境，保留在 `DECISIONS.md`
和旧日志中，不再作为本机当前入口。

## 记录文件

- `README.md`：说明如何使用这套记录。
- `MEMORY.md`：长期稳定事实。
- `STATUS.md`：当前状态。
- `HANDOFF.md`：agent 交接页。
- `DECISIONS.md`：关键决策。
- `TASK_TEMPLATE.md`：单次任务日志模板。
- `logs/`：单次任务日志目录。

## 每次开始任务先看什么

1. 先进入 `/Users/moweile/Code/LAB/videoquant-trq`。
2. `STATUS.md`
3. `HANDOFF.md`
4. `MEMORY.md`
5. `DECISIONS.md`
6. 需要上下文细节时，再看 `logs/` 最近几条。

## 维护规则

- 长期稳定信息写 `MEMORY.md`，不要写临时状态。
- 当前任务进展写 `STATUS.md`，保持短。
- agent 切换前更新 `HANDOFF.md`，只保留下一位立刻需要的信息。
- 单次训练、调试、环境排查、Slurm 作业提交，统一写到 `logs/`。
- 服务器资源使用遵守 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)。
