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
| HY-WorldPlay | 5 个 conditioning scenes | 5 scenes x 4 actions x 5 modes = 100 videos | 动作与反事实可控性 |

不能横向比较三种模型的绝对 VBench。报告各模型相对自身 BF16 的下降量。
PSNR、SSIM 和 LPIPS 是同输入、同 seed 下的轨迹一致性指标，不是绝对视频质量。

## 执行阶段与准入门

1. Smoke：每条基线用一个输入跑完五档精度。检查视频非空、没有静默回退
   BF16，并在后端支持时检查 packed cache、实际字节数和有限的显存统计。
2. Pilot：Causal Forcing 用三个 prompt 测试递增长度；人工检查首次量化边界和
   LongCat continuation 接缝。
3. Full：运行上表矩阵，再计算配对指标、VBench 和动作代理指标。
4. Review：逐样本盲审 identity switch、background jump、motion freeze、
   color drift、black/NaN 等灾难现象。

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
GPU=3 bash experiments/world_model_quant/run_generation.sh causal_forcing smoke
GPU=3 bash experiments/world_model_quant/run_generation.sh hy_worldplay smoke

GPU=3 CAUSAL_LENGTHS="21 42 84" \
  bash experiments/world_model_quant/run_causal_length_pilot.sh
```

Causal length pilot 必须先确定后端允许且稳定的长度，再通过 `CAUSAL_FRAMES` 改变
正式实验长度。不能仅因 21 帧 smoke 成功就宣称长时量化有效。

## HY 场景清单

`hy_scenes.example.json` 只包含上游 demo，用于 smoke。正式矩阵需要在远端准备至少
5 个代表性 conditioning scenes。图片路径相对 `HY_WORLDPLAY_SOURCE` 解析，图片本身
不进入 Git。

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

## 两张 A100 的正式队列

LongCat 独占一张 A100；Causal Forcing 和 HY-WorldPlay 在另一张卡上顺序执行。
脚本只等待自己启动的进程，不轮询 GitHub 或远端目录。

```bash
HY_SCENES=/remote/path/hy_scenes.json \
LONGCAT_GPU=2 SECONDARY_GPU=3 \
  bash experiments/world_model_quant/run_two_gpu_matrix.sh
```

生成完成后执行：

```bash
bash experiments/world_model_quant/run_evaluation.sh causal_forcing full
bash experiments/world_model_quant/run_evaluation.sh longcat full
bash experiments/world_model_quant/run_evaluation.sh hy_worldplay full
```

主要生成结果位于 `results/world_model_quant/<baseline>/`。配对指标写入对应实验根目录的
`paired_metrics/`，VBench 写入 `results/world_model_quant/vbench/`，HY 动作结果写入
`action_metrics/`。这些运行产物不提交 GitHub；提交的是可重放脚本、配置和结果摘要。
