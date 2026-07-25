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

## GPU 2/4 扩展队列与 2026-07-24 恢复计划

GPU 0 已关闭。GPU 2 执行 LongCat 前半 shard，同时让 GPU 4 顺序执行
Causal length pilot、Causal MovieGen10 和 HY dev。GPU 4 完成上述队列后，再接手
LongCat 后半 shard。LongCat smoke 实测单视频约 18 分钟，分 shard 能显著缩短墙钟
时间；两个 shard 的 prompt index 和输出文件名必须互斥。

| 顺序 | GPU 2 | GPU 4 | 进入下一步条件 |
| --- | --- | --- | --- |
| 0 | 保留现有可解码输出 | 保留现有可解码输出 | 同一 `RUN_ID` 恢复，不覆盖结果 |
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

首次 `expansion_a_20260724` 在 42 帧 BF16 写 cache 时暴露固定容量错误：官方适配
沿用了 `32760 = 21 x 1560` token 容量，导致 42 帧写入区间超过 tensor。修复协议为：

1. `kv_cache_capacity_frames` 默认等于 `num_output_frames`，BF16、TRQ 和 naive 使用
   完全相同的逻辑容量；不通过启用 rolling window 改变实验变量；
2. 先跑 42 帧 BF16 单 prompt，再跑五档单 prompt；随后对 84 帧重复同一 gate；
3. BF16 84 帧必须记录峰值显存，超过 A100 80GB 时将 42 帧冻结为正式长度，不把
   OOM 解释为量化质量失败；
4. Causal、HY 和 LongCat 阶段各自记录 `running/done/failed`，一个 baseline 失败不再
   终止其他 baseline；所有恢复都以 `ffprobe` 可解码为完成条件；
5. 已完成的 Causal 21 帧 `3 prompts x 5 modes`、LongCat smoke 和扩展样本全部复用。

当前实现采用 full-history cache 扩容，因为研究问题是完整历史 KV 的量化。固定窗口
eviction 属于另一项消融，不能混入本轮 TRQ 与 naive 的主比较。

2026-07-24 修复后单 prompt gate 实测：

| 长度 | 模式 | 原生等价 cache | packed cache | 峰值 CUDA | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| 42 | BF16 | - | - | 未记录 | 通过、可解码 |
| 42 | TRQ INT4 / INT2 | 12.08 GB | 3.27 / 2.26 GB | 未记录 | 均通过 |
| 42 | naive INT4 / INT2 | 12.08 GB | 3.40 / 1.89 GB | 未记录 | 均通过 |
| 84 | BF16 | - | - | 首次 gate 未记录 | 通过、可解码 |
| 84 | TRQ INT4 / INT2 | 24.15 GB | 6.54 / 4.53 GB | 26.25 / 24.25 GB | 均通过 |
| 84 | naive INT4 / INT2 | 24.15 GB | 6.79 / 3.77 GB | 26.46 / 23.44 GB | 均通过 |

BF16 的 `native_cache_bytes` 为 0 是统计接口只汇总 packed cache，并不代表没有分配
BF16 tensor；两次 BF16 gate 均完成且未 OOM。峰值字段是在首次 BF16 gate 后加入，
正式矩阵剩余 BF16 prompt 会记录精确值。84 帧五档通过后，恢复队列已自动放行。

### 2026-07-25 HY dev 恢复计划

`expansion_a_20260724` 的 Causal 正式矩阵已完成五档各 10 条；LongCat 已完成
BF16 prefix、BF16/TRQ INT4/TRQ INT2/naive INT4 各 10 条，naive INT2 在
2026-07-25 13:05 已完成 7/10 并继续运行。HY dev 在第一条生成前退出，产出为
0/100。错误不是权重、数据或显存不足，而是 `runner.py` 对官方场景的长文本 prompt
调用 `Path.exists()`，把文本误判为文件名并触发 `OSError: File name too long`。

恢复按以下顺序执行：

1. prompt 仅在确实对应现有文件时解析为绝对路径；任意长度的字面文本保持原样传给
   HY-WorldPlay 的 `--input`；
2. CPU 回归同时覆盖长字面 prompt 和真实 prompt 文件，现有动作、精度和受控变量
   测试必须全部通过；
3. 本地提交并推送后，远端只做一次 fast-forward 更新，不复制权重、结果或日志；
4. GPU 2 空闲时只启动 `hy_worldplay full` recovery，继续使用原
   `RUN_ID=expansion_a_20260724`。Causal 与 LongCat 的可解码输出不得重跑；
5. recovery 必须独立记录 `running/done/failed` 和日志。只有 5 scenes x 4 actions x
   5 modes 共 100 个视频全部存在且可由 `ffprobe` 解码，才能把 HY 工程状态改为
`done`；动作可控性仍需代理指标和人工审阅，不能由数量门代替。

第一阶段 prompt 修复只改变输入分派；后续 action-horizon 纠错只把四个动作补齐到
既定生成长度。prompt、conditioning image、动作类别、seed、生成长度和量化参数仍在
五档精度间固定，因此 BF16/TRQ/naive 的受控比较成立。HY recovery 完成前，
Expansion A 只能标记为 partial，不能进入 holdout。

首次按上述方案恢复后又暴露出独立的 action-horizon 错误：四个正式动作原为
`3 x 8 = 24` latent steps，而 `num_chunks=12` 的 WAN pipeline 每个 chunk 消耗 4 个
pose latents，共需要 48 steps。上游在后半程得到空的 `curr_viewmats`，写入
`err.txt` 后仍以退出码 0 返回，导致旧 runner 继续下一个组合。2026-07-25 14:22
检查时已有 24 个 `err.txt`、0 个 MP4；该无效 recovery 已按“输出缺失即停止”门终止，
孤立 torchrun 也已清理，GPU 2 显存归零。

因此恢复计划追加以下硬门：

1. 四个反事实动作统一覆盖 48 steps：转向使用 `w-16,a/d-16,w-16`，纵向使用
   `w/s-16,w/s-16,w/s-16`；动作语义、总长度和各阶段时长在五档精度间固定；
2. runner 在启动 GPU 前计算 pose latent steps，不足 `num_chunks x 4` 直接失败；
3. 上游返回后必须重新扫描输出目录，至少存在一个可解码 MP4，否则即使退出码为 0
   也按失败处理并指向 `err.txt`；
4. 正式恢复前先跑 official scene 1、`turn_left`、BF16 单视频 gate。只有该视频可解码
   且没有 `err.txt`，才继续同一 `RUN_ID` 的剩余矩阵。

动作从 24 steps 修正为 48 steps 是使动作条件覆盖既定生成 horizon 的协议纠错。旧
24-step 运行没有视频，不进入对照数据，也不能与修复后的结果混合。

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
