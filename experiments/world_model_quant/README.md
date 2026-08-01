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

当前结果支持把 H2 作为下一阶段的确认性假设：三条基线上 TRQ INT2 的
PSNR/SSIM/LPIPS 均优于 naive INT2，HY 的 BF16 归一化反事实分离保留率为
`0.886`，naive INT2 为 `0.510`。H1 仍是探索性假设：Causal 与 LongCat 的 INT4
指标互有胜负，HY INT4 略偏向 naive，不能依据 10 个 prompt 或 5 个场景下结论。

不能横向比较三种模型的绝对 VBench。报告各模型相对自身 BF16 的下降量。
PSNR、SSIM 和 LPIPS 是同输入、同 seed 下的轨迹一致性指标，不是绝对视频质量。
当前八维 VBench 聚合是仓库内归一化描述指标，不是官方 VBench Total Score。

## 2026-07-28 实验进展

MovieGen10 / HY dev 的自动生成与统一评测已经完成；下面的数值均以同输入、同 seed 的
BF16 为参考。Expansion B1 的扩展视频也已全部生成并通过可解码检查，但扩展集统一评测
和人工 catastrophe/action review 尚未完成，因此科学状态仍是
`B0 automatic done; B1 generation done, evaluation pending; manual pending`。

| 基线与阶段 | 已完成生成 | 完整性状态 | 下一步 |
| --- | ---: | --- | --- |
| Causal B1 | indices 10-31，五档各 22 条，共 110 条 | 生成 exit 0；旧 10 条与新增 22 条分处不同结果根 | 建立 32 条统一 manifest 后评测 |
| LongCat B1 | 22 个 BF16 prefix；五档 continuation 各 22 条，共 132 条 | 六个目录均 `22/22, bad=0`；双 GPU lane 均 exit 0 | 评测 32 条超集并做分段漂移分析 |
| HY-WAN holdout | cases 6-10，四动作五档，共 100 条 | `done`，100 条可解码 | action proxy、感知指标和人工盲审 |

Causal 编排状态曾显示 `failed:validation`，原因是验证器只扫描当前 checkout，而旧
MovieGen10 结果位于另一个运行 worktree。该状态是结果索引缺陷，不是模型生成失败；
禁止通过重跑 22 条扩展视频来修复。

| 基线 | 模式 | PSNR | SSIM | LPIPS | 补充结果 |
| --- | --- | ---: | ---: | ---: | --- |
| Causal | TRQ INT4 / naive INT4 | 11.667 / 11.823 | 0.494 / 0.501 | 0.430 / 0.437 | INT4 混合 |
| Causal | TRQ INT2 / naive INT2 | 10.438 / 9.372 | 0.440 / 0.376 | 0.555 / 0.674 | TRQ INT2 三项更优 |
| LongCat | TRQ INT4 / naive INT4 | 30.248 / 30.303 | 0.929 / 0.924 | 0.026 / 0.031 | INT4 混合 |
| LongCat | TRQ INT2 / naive INT2 | 23.333 / 19.904 | 0.827 / 0.727 | 0.081 / 0.170 | TRQ INT2 三项更优 |
| HY-WAN | TRQ INT4 / naive INT4 | 20.585 / 20.715 | 0.721 / 0.724 | 0.134 / 0.132 | INT4 略偏 naive |
| HY-WAN | TRQ INT2 / naive INT2 | 17.787 / 16.141 | 0.604 / 0.543 | 0.243 / 0.376 | TRQ INT2 三项更优 |

HY 四种量化模式的 optical-flow 方向符号保持率均为 100%，该指标已经饱和。后续以
反事实分离保留率、动作切换响应延迟和人工动作判断为主，方向符号只作为最低工程门。

## 下一阶段总体矩阵

### Causal Forcing 与 LongCat：MovieGen 嵌套扩样

保持当前 prompt 顺序、seed、分辨率、生成长度和五档精度不变。每一级数据集包含前一级，
只生成缺失索引，不重复 MovieGen10。B1 使用
`temporalresidualkvquant/assets/moviegenbench_resume_32.txt`：前 10 条逐字保留已经生成
的 `integrations/evaluation/moviegen10.txt`，后 22 条取自原 MovieGen32。原
`moviegenbench_32.txt` 的弯引号与已运行列表不完全一致，不能直接用于文件级续跑。B2
启动前按相同规则冻结 128 条超集，不能临时改 prompt 文本。

| 阶段 | 数据集 | 新增 Causal | 新增 LongCat | 目的 |
| --- | --- | ---: | ---: | --- |
| B0 | MovieGen10 | 0 | 0 | 补人工盲审和逐 prompt 统计 |
| B1 | MovieGen32 | 22 x 5 = 110 | 22 prefixes + 22 x 5 continuations | 确认效应方向并估计方差 |
| B2 | MovieGen128 | 96 x 5 = 480 | 96 prefixes + 96 x 5 continuations | 形成论文主结果 |
| B3 | MovieGen1003 | 暂不执行 | 暂不执行 | 只有最终论文需要全量时再批准 |

B1 两条基线合计新增 220 个对比视频，另有 22 个 LongCat BF16 prefix；B2 在 B1
基础上合计新增 960 个对比视频，另有 96 个 prefix。B1 通过人工灾难门后再启动 B2，
不把 MovieGen1003 作为默认队列。

每个阶段同时报告全部样本和 MovieGen motion/concept tag 子组。主统计单位是 prompt，
使用 paired bootstrap 95% CI、配对胜率、中位数和 IQR；不能把视频帧当作独立样本来
虚增显著性。INT2 是确认性比较，INT4 保持探索性并完整报告，不依据单一指标挑选结果。

### 长时误差轴

prompt 扩样不能替代 H3。B1 完成后，在冻结的 MovieGen32 子集上增加分段分析：

- Causal 固定已通过工程门的 84 帧协议，分别统计首个量化边界、前段、中段和末段；
- LongCat 固定 73 帧 conditioning context，并统计第 1、3、5、10 个 20-frame
  continuation segment；
- 每个检查点输出 PSNR/SSIM/LPIPS、boundary jump、累计漂移和 catastrophe tags；
- LongCat 的配对指标继续跳过前 13 个共享 conditioning frames，不能与后续生成帧混算。

如果当前 LongCat runtime 与 `73 context + 20-frame segment` 不一致，先记录当前协议，
再新建 QVG-aligned 长度实验；禁止把不同协议的视频并入同一统计表。

## HY-WorldPlay 双轨计划

### Track H1：现有 HY-WAN 适配实验

官方 test cases 1-5 与 holdout cases 6-10 均已生成，不重跑。2026-07-28 已补齐：

| 数据 | 动作 | 模式 | 新增量 |
| --- | --- | --- | ---: |
| HY 官方 cases 6-10 | `turn_left`、`turn_right`、`forward`、`backward` | 五档精度 | 100 videos |

该轨用于验证现有 TRQ adapter 的跨场景稳定性，结果统一命名为 `HY-WAN adaptation`。
WAN pipeline 是 HY-WorldPlay 的轻量路线，不能将其结果写成 QVG 论文的
`HY-WorldPlay-8B` 复现。holdout 不用于改 prompt、动作时长、量化 block 或阈值。

### Track H2：QVG-aligned HY-WorldPlay-8B

[Quant VideoGen](https://arxiv.org/abs/2602.02958) 在 480p 下评估
LongCat-Video-13B、HY-WorldPlay-8B 和 Self-Forcing-Wan-1.3B。HY 使用全历史条件和
12-frame chunks；论文 Figure 1 展示 285 帧的 `right -> forward -> left` 轨迹。
论文只说明使用 MovieGen prompt suite 并沿用 Self-Forcing prompt 设置，没有公开 HY
输入图像的完整构造规则和确切样本数。因此本项目采用官方 HY 场景形成可重放协议，标记为
`QVG-aligned`，不声称是完全复现。

QVG-aligned 的顺序是：

1. 新建并 smoke `HY-WorldPlay-8B/HunyuanVideo` backend；现有 WAN 结果不能复用为 8B
   结果，但 adapter、量化 codec 和评测代码可以复用；
2. 固定 480p、12-frame chunks、约 24 chunks/285 帧；在运行前验证真实输出帧数和
   pose latent horizon 完全覆盖生成长度；
3. 使用官方 10 个场景，执行 `right -> forward -> left` 和严格反事实
   `left -> forward -> right` 两条等长轨迹；
4. 先跑一个场景的 BF16 和五档单场景 gate，再运行 `10 scenes x 2 trajectories x
   5 modes = 100` 条长视频；
5. 所有模式使用相同初始图、prompt、seed、轨迹、chunk 数和 history policy。

QVG 论文的主要基线是 RTN、KIVI 和 QuaRot，公平设置使用 block size 16，QuaRot 只量化
KV cache。当前 packed-naive 继续作为历史对照；另在 5 场景 calibration 子集增加
`RTN-B16`，先确认其与当前 naive 的差异。如果后续需要与 QVG 表格直接比较，再单独接入
KIVI、QuaRot 和 QVG/QVG-Pro，不能把当前 naive 直接改名为论文 RTN。

### HY 评价指标

- BF16 轨迹一致性：PSNR、SSIM、LPIPS；
- QVG 感知质量：Background Consistency、Imaging Quality、Subject Consistency、
  Aesthetic Quality；
- action：方向最低门、反事实分离保留率、动作切换响应延迟、每个 action segment 的
  optical-flow 幅度；
- 系统：实际 packed KV bytes、元数据开销、压缩率、peak CUDA memory 和端到端延迟；
- 人工盲审：动作执行、切换时机、identity/background catastrophe、freeze、black/NaN。

只有 BF16 本身具有可辨认的反事实分支，量化 action 指标才有效。TRQ 的预注册目标是
保留至少 80% 的 BF16 反事实分离幅度、无方向反转，并在同 bit/相近实际压缩率下优于
naive 或 RTN。代理指标必须与人工 action review 一起解释。

## 执行顺序与准入门

1. `[pending] Review A`：完成 MovieGen10 与 HY cases 1-5 的盲审、failure tags 和逐样本表；
2. `[done] Expansion B1 generation`：Causal 与 LongCat 已补 MovieGen indices 10-31；
3. `[done] HY-WAN holdout generation`：官方 cases 6-10 已补齐，未据 holdout 回改参数；
4. `[next] B1 unified evaluation`：统一 Causal/LongCat 的 32 条索引，运行配对指标、
   VBench-derived、prompt-level bootstrap CI、分段漂移与人工 failure tags；
5. `[blocked] Expansion B2`：Review A 无 TRQ-only catastrophe，且 B1 没有出现 INT2 效应方向
   反转时，只补 MovieGen indices 32-127；
6. `HY-8B setup`：独立完成 backend、单场景五档 gate 和长轨迹 horizon gate；
7. `HY-8B QVG-aligned`：通过 gate 后运行 10 场景、两条反事实长轨迹；
8. `Final review`：按 baseline 报告 BF16 delta、统计区间、长时曲线、action 与系统指标。

### B1 统一索引与自动评测

旧 MovieGen10 与新增 22 条结果允许位于不同结果根，但评测前必须先生成规范化的只读
符号链接视图。索引器会逐 prompt、逐精度检查缺失、跨根重复和 `ffprobe` 可解码性；
任一检查失败时不会启动 GPU 指标：

```bash
python experiments/world_model_quant/prepare_moviegen32_manifest.py \
  --baseline causal_forcing \
  --prompts temporalresidualkvquant/assets/moviegenbench_resume_32.txt \
  --source-root legacy=/path/to/causal/moviegen10 \
  --source-root b1=/path/to/causal/expansion_b1 \
  --output-root results/world_model_quant/indexes/b1_moviegen32/causal_forcing
```

两张卡并行执行，GPU 6 负责 Causal 后接 HY holdout，GPU 7 负责 LongCat；每张卡内部
严格串行。脚本先建 Causal/LongCat manifest，再运行 BF16-reference
PSNR/SSIM/LPIPS、TRQ-vs-naive prompt-level bootstrap 95% CI 和八维 VBench-derived：

```bash
CAUSAL_GPU=6 LONGCAT_GPU=7 \
CAUSAL_LEGACY_ROOT=/path/to/causal/moviegen10 \
CAUSAL_B1_ROOT=/path/to/causal/expansion_b1 \
LONGCAT_LEGACY_ROOT=/path/to/longcat/moviegen10 \
LONGCAT_B1_ROOT=/path/to/longcat/expansion_b1 \
HY_ROOT=/path/to/hy/action_control_holdout \
RUN_ID=b1_moviegen32_20260728 \
  bash experiments/world_model_quant/run_b1_evaluation.sh
```

`paired_metrics/summary.json` 中的 `paired_trq_vs_naive` 以正值统一表示 TRQ 比 naive
更接近同 prompt、同 seed 的 BF16；它仍是轨迹相似度，不是绝对视频质量。脚本的
`.done` 标记只证明对应自动阶段完成，不能替代人工 catastrophe/action review。

### B1 匿名人工审阅包

人工审阅使用独立的静态网页。公开目录只包含匿名 A--E 面板和媒体链接；方法、bit、原始
路径与 A--E 对应关系只写入公开目录上一级的 `private_mapping.json`，不能把整个输出根
作为网页根目录。Schema v2 将同一 prompt 的 BF16 和四个量化候选五路合并：MovieGen32
共 `2 x 32 = 64` 个 catastrophe 面板；HY holdout 将每个场景的五种精度和四个动作合并，
5 个场景共 5 个 action 面板。视频覆盖不变，但不再重复展示 BF16 或填写重复表单。

```bash
python experiments/world_model_quant/prepare_b1_review.py \
  --causal-manifest /path/to/unified/causal_forcing/manifest.json \
  --longcat-manifest /path/to/unified/longcat/manifest.json \
  --hy-root /path/to/hy/action_control_holdout \
  --output-root results/world_model_quant/review/b1_moviegen32_20260729

cd results/world_model_quant/review/b1_moviegen32_20260729/public
python -m http.server 8766
```

网页使用浏览器本地存储自动保存，并导出带 manifest SHA-256 的 JSON。它支持五路同步
播放、统一倍速、动作标签页和快速通过；按 `Space` 播放/暂停、`1`--`5` 标记可疑视频、
`N` 快速通过、方向键导航。只有可疑视频需要展开 catastrophe 细节。审阅结束后才使用
私有映射解盲；hash 不一致时脚本拒绝合并，旧 schema v1 导出仍可由解码器读取：

```bash
python experiments/world_model_quant/decode_b1_review.py \
  --review /path/to/b1-review-reviewer01.json \
  --private-mapping results/world_model_quant/review/b1_moviegen32_20260729/private_mapping.json \
  --output results/world_model_quant/review/b1_moviegen32_20260729/reviewer01_decoded.csv
```

同 bit 下，只有当 TRQ 的配对指标优于 naive/RTN，并且没有新增 TRQ-only catastrophe
时，才认为方法通过。B1 若区间较宽但效应方向未反转，允许进入 B2 以增加统计功效；
出现 BF16 失败、量化静默回退、不可解码输出或重复 TRQ-only catastrophe 时，只停止
对应 baseline，保留其他队列和全部故障证据。

MovieGen32 与 HY-WAN holdout 的双 GPU 串行队列入口如下。两张卡并行，但每张卡内部
严格串行；默认继续 `expansion_a_20260724`，仅补缺失的 10-31 和 cases 6-10：

```bash
CAUSAL_HY_GPU=6 LONGCAT_GPU=7 \
  RUN_ID=expansion_a_20260724 \
  STAGE_ID=expansion_b1_20260727 \
  bash experiments/world_model_quant/run_expansion_b1_serial.sh
```

该 B1 生成队列已在 2026-07-28 完成，命令保留用于复现和故障恢复，不应在现有结果上
重复启动。LongCat 最后 57 条曾按互斥索引拆到两张 GPU：一张完成剩余 TRQ INT2 并生成
naive INT2 indices 10-24，另一张生成 naive INT4 全 22 条及 naive INT2 indices 25-31。
输出文件与 manifest 区间互斥，最终由统一可解码门验收。

正式启动前用 `DRY_RUN=1` 检查 GPU、索引、prompt 文件与 HY holdout manifest。阶段状态
写入 `results/world_model_quant/orchestration/<STAGE_ID>/`，生成日志写入
`results/world_model_quant/logs/<STAGE_ID>/orchestrator/`。

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

单场景 `turn_left` BF16 gate 随后已生成可解码视频。第一次全矩阵接续因监督命令没有
导出 `RUN_ID`，误写入独立的 `action_control_full/`；该任务已停止，目录保留作故障证据，
不与 `action_control_expansion_a_20260724/` 混合。第二次接续暴露新终端处于 `(base)`，
导致 `torchrun` 使用系统入口并缺少 `remote_pdb`。恢复协议因此再增加运行时门：

1. `common.sh` 的 `configure_videoquant_runtime` 默认探测
   `$HOME/miniconda3/envs/videoquant`，也接受显式 `VIDEOQUANT_ENV_PREFIX`；
2. 生成、Causal pilot、两卡编排和评测入口统一将该环境的 `bin/` 放到 `PATH` 首位，
   并固定 `CONDA_PREFIX` 与 `PYTHON_BIN`；显式路径不存在或缺少可执行的
   `python/torchrun` 时在 GPU 启动前失败；
3. 正确 recovery 使用 `RUN_ID=expansion_a_20260724`，保留已通过的 BF16 gate，只补
   其余视频。监督进程独立记录 PID、PGID、状态和日志；完成条件仍是正确目录 100 个
   MP4 全部通过 `ffprobe`。

2026-07-25 14:56，正确 recovery 已在 GPU 2 启动，状态为 `running`，启动检查时显存
约 7.6 GiB，日志已完成 CUDA 分布式初始化与权重加载。此状态只证明工程续跑已恢复，
不代表 100-video 矩阵或动作可控性评测完成。

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
