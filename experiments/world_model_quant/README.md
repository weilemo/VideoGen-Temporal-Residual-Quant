# Causal Forcing、LongCat 与 HY-WorldPlay 的 TRQ 实验计划

本目录只负责三条基线的实验编排。模型参数映射继续由 `integrations/`
维护，TRQ codec 继续由 `temporalresidualkvquant/src/trq/` 唯一维护。

## 研究问题与预注册假设

三条基线统一使用 `bf16`、`trq_int4`、`trq_int2`、`naive_int4` 和
`naive_int2` 五档精度。主要比较是同 bit 下 TRQ 与 packed-naive 的差异。

- H1：TRQ INT4 的质量接近 BF16，并优于 packed-naive INT4。
- H2：INT2 允许出现更明显退化，但 TRQ INT2 应优于 packed-naive INT2。
- H3：随生成长度增加，TRQ 的首次量化边界跳变和后期漂移低于 naive。
- H4：HY-WorldPlay 中，TRQ 应保持 BF16 的动作方向、响应时机和反事实分离。

| 基线 | 数据 | 正式矩阵 | 首要问题 |
| --- | --- | --- | --- |
| Causal Forcing | MovieGen10 | 10 prompts x 5 modes = 50 videos | 因果边界跳变和长期漂移 |
| LongCat | MovieGen10 | 10 个固定 BF16 prefix + 10 continuations x 5 modes | 续写阶段的量化退化 |
| HY-WorldPlay dev | 官方 test cases 1-5 | 5 scenes x 4 actions x 5 modes = 100 videos | 动作与反事实可控性 |
| HY-WorldPlay holdout | 官方 test cases 6-10 | dev 通过后追加 100 videos | 独立场景确认 |

不能横向比较三种模型的绝对 VBench。报告各模型相对自身 BF16 的下降量。
PSNR、SSIM 和 LPIPS 是同输入、同 seed 下的轨迹一致性指标，不是绝对视频质量。

## 执行阶段与准入门

1. Smoke（已完成，2026-07-24）：每条基线用一个输入跑完五档精度，共 15 个
   对比视频。三条 backend 均产生真实 packed cache 和非空视频；人工中点帧检查发现
   Causal packed-naive INT2 已出现结构崩坏，而 TRQ INT2 保留主体和街景。HY 与
   LongCat 未见中点帧灾难，但尚未建立动作可控性或长时质量结论。
2. Causal length pilot：用三个 prompt 跑 21/42/84 帧和五档精度。选择最长的稳定
   长度作为正式矩阵长度；若 BF16 自身失败，则该长度无效。
3. Expansion A：Causal 和 LongCat 跑 MovieGen10；HY 跑官方 test cases 1-5 的
   四动作五精度矩阵。生成后立即计算配对指标并人工审阅。
4. Expansion B：只有 HY dev 没有新增 TRQ-only 动作反转或灾难时，才在官方
   test cases 6-10 上复现同一矩阵。holdout 不用于调参。
5. Review：逐样本盲审 identity switch、background jump、motion freeze、
   color drift、action reversal、black/NaN 等灾难现象。

同 bit 下，只有当 TRQ 的配对指标优于 naive，并且没有新增 TRQ-only catastrophe
时，才认为方法通过。VBench 数值只做描述，不单独充当硬门槛。10 个 prompt 只报告
逐样本结果、中位数和 bootstrap 区间，不做过强的显著性结论。

LongCat 的配对指标跳过前 13 个共享 conditioning frames。HY 的 optical-flow
方向和反事实分离是代理指标，必须与人工 action review 一起解释。

## 远端同步

远端只从 GitHub 获取代码。本地推送完成后，在 code-server 手工执行一次：

```bash
cd /path/to/VideoGen-Temporal-Residual-Quant
bash experiments/world_model_quant/pull_remote_once.sh codex/trq-online-causal-gates
```

该脚本只允许 clean checkout 上的 fast-forward pull，没有轮询逻辑。权重、上游运行
仓库、视频、cache 和日志继续保留在远端并由 Git 忽略。

## Smoke 与长度 Pilot

```bash
GPU=2 bash experiments/world_model_quant/run_generation.sh longcat smoke
GPU=4 bash experiments/world_model_quant/run_generation.sh causal_forcing smoke
GPU=4 bash experiments/world_model_quant/run_generation.sh hy_worldplay smoke

GPU=4 CAUSAL_LENGTHS="21 42 84" \
  bash experiments/world_model_quant/run_causal_length_pilot.sh
```

Causal length pilot 必须先确定后端允许且稳定的长度，再通过 `CAUSAL_FRAMES` 改变
正式实验长度。不能仅因 21 帧 smoke 成功就宣称长时量化有效。

## HY 场景清单

`hy_scenes.example.json` 只包含 Quant-VideoGen 上游 demo，用于 smoke，不能把同一张
图复制五次冒充扩大数据集。正式场景来自 HY-WorldPlay 官方仓库的
[`assets/test_case.csv`](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/assets/test_case.csv)
和对应的 `assets/img/1.png` 至 `assets/img/10.png`。官方清单覆盖游戏、城市、海岸、
森林、麦田、外星地貌、雪景和城堡等不同场景。

- dev：固定使用官方 1-5，允许依据其结果检查实现和协议；
- holdout：固定使用官方 6-10，只在 dev 通过后运行，不据其结果回改参数；
- 图片和远端 manifest 属于运行资产，不提交模型权重或生成视频到 Git。

```json
{
  "schema_version": 1,
  "scenes": [
    {
      "id": "scene_id",
      "image_path": "assets/worldplay/scene.png",
      "prompt": "First-person view ..."
    }
  ]
}
```

每个场景固定 prompt、conditioning image 和 seed，分别执行 `turn_left`、
`turn_right`、`forward` 和 `backward`。左右与前后分别构成两组反事实对。

## GPU 2/4 扩展队列（待批准，尚未启动）

GPU 0 已关闭。计划先让 GPU 2 执行 LongCat 前半 shard，同时让 GPU 4 顺序执行
Causal length pilot、Causal MovieGen10 和 HY dev。GPU 4 完成上述队列后，再接手
LongCat 后半 shard。LongCat smoke 实测单视频约 18 分钟，分 shard 能显著缩短墙钟
时间；两个 shard 的 prompt index 和输出文件名必须互斥。

| 顺序 | GPU 2 | GPU 4 | 进入下一步条件 |
| --- | --- | --- | --- |
| 0 | 不启动 | 不启动 | 用户批准本计划 |
| 1 | LongCat prompts 0-4 | Causal 21/42/84 length pilot | BF16 和 TRQ 输出非空、有限 |
| 2 | 继续 LongCat 0-4 | Causal MovieGen10 x 5 modes | 选定并冻结正式帧数 |
| 3 | 等待或做 CPU 评测 | HY dev 5 scenes x 4 actions x 5 modes | dev 人工 action review 完成 |
| 4 | 空闲 | LongCat prompts 5-9 | 前后 shard 无索引冲突 |
| 5 | 配对评测 | 配对评测；可选 HY holdout | 用户批准 holdout |

自动编排由 `run_two_gpu_matrix.sh` 实现，正式启动前必须再次做 dry-run 和 shell 检查：

1. secondary 默认使用 GPU 4，LongCat 默认使用 GPU 2；
2. LongCat 通过 `START_INDEX`/`LIMIT` 拆成互斥 shard，禁止两个 GPU 写同一索引；
3. `prepare_hy_official_scenes.py` 下载官方 test cases，并生成固定的 dev/holdout
   manifest；
4. 每次输出带独立 `RUN_ID`，不会覆盖 smoke；LongCat prompt 0 和 Causal pilot 的
   可兼容 smoke 结果通过符号链接复用；
5. Causal pilot 自动选择通过数量、非空和可解码工程门的最长帧数。人工质量审阅仍在
   生成完成后进行。

Expansion A 的计划生成量为 210 个视频：Causal 50、LongCat 10 个 BF16 prefix 加
50 个 continuation、HY dev 100。Causal length pilot 另计 45 个视频。HY holdout
如获批准再追加 100 个视频。

停止条件：BF16 失败、NaN/黑帧、输出缺失、量化静默回退或新增 TRQ-only catastrophe
时停止对应队列并保留日志。packed-naive 的预期低精度崩坏应记录，但不应阻止同一
输入上的 BF16/TRQ 对照完成。HY dev 若出现重复的 TRQ-only 动作反转，则不进入
holdout。

生成完成后执行：

```bash
bash experiments/world_model_quant/run_evaluation.sh causal_forcing full
bash experiments/world_model_quant/run_evaluation.sh longcat full
bash experiments/world_model_quant/run_evaluation.sh hy_worldplay full
```

主要生成结果位于 `results/world_model_quant/<baseline>/`。配对指标写入对应实验根目录的
`paired_metrics/`，VBench 写入 `results/world_model_quant/vbench/`，HY 动作结果写入
`action_metrics/`。这些运行产物不提交 GitHub；提交的是可重放脚本、配置和结果摘要。
