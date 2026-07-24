# 当前状态

## 当前目标

- 以 `temporalresidualkvquant` 为主方法库，用统一的 TRQ codec 推进 Self-Forcing
  长视频 KV cache 量化，并继续研究 head-wise / role-aware precision policy。

## 正在做什么

- **三基线 TRQ smoke 已完成，扩展队列正在故障恢复（2026-07-24）**：
  - Causal Forcing 与 LongCat 使用 MovieGen10；HY-WorldPlay 使用多场景动作条件协议；
  - 五档精度统一为 BF16、TRQ INT4/INT2、packed-naive INT4/INT2；
  - `experiments/world_model_quant/` 提供 smoke、Causal 长度 pilot、两张 A100
    队列、配对指标、VBench 和 HY action proxy 入口；
  - LongCat 配对指标跳过 13 个共享 conditioning frames；
  - 远端已完成 15 个 smoke 视频，三条 GPU backend 的五档调用链均能产出视频；
  - 人工中点帧检查中，Causal packed-naive INT2 出现结构崩坏，TRQ INT2 保留主体与
    街景；HY 和 LongCat 中点帧未见灾难，但动作可控性、长时质量和统计指标仍未验证；
  - GPU 0 已关闭；使用 GPU 2/4 自动运行 MovieGen10、Causal length pilot 和 HY
    官方 10 场景 dev/holdout 协议，先启动 dev，holdout 仍需人工批准；
  - `expansion_a_20260724` 已完成 Causal 21 帧五档各 3 条；42 帧 BF16 因 cache
    容量仍固定为 21 帧而失败，GPU 4 已退出；GPU 2 的 LongCat 子进程继续保留结果；
  - 修复将 cache 容量显式绑定 `num_output_frames`，并把编排改为可解码文件级恢复、
    独立阶段状态与进程组清理；先通过 42/84 帧单 prompt gate 再恢复正式队列；
  - 修复后 42/84 帧五档单 prompt gate 均通过；84 帧四个量化模式峰值 CUDA 为
    23.44-26.46 GB，恢复编排已接管既有 LongCat 并补齐剩余 Causal pilot；
  - 远端代码同步改为 GitHub commit 后的一次性 fast-forward pull，不做远端轮询。

- **仓库所有权边界已整理（2026-07-24）**：
  - `temporalresidualkvquant/` 继续作为 TRQ 方法主库；
  - Causal Forcing、HY-WorldPlay、LongCat-Video 和 Rolling Forcing 的启动器、
    配置与补丁统一进入 `integrations/`；
  - QVG 与 focused-forcing 上游快照降级为 `references/` 只读参考；
  - 当前记录进入 `docs/project/`，历史执行记录进入 `docs/history/`；
  - `forcing/causalforcing` 已明确为历史 Rolling/Long-video 变体，官方
    Causal Forcing 由 `integrations/causal_forcing/setup.sh` 在远端安装；
  - LongCat 的 SDPA fallback 已保存为可重放补丁，不再依赖未记录的嵌套仓修改。

- **TRQ 在线分叉 E0-E5 实验框架已落地（2026-07-20）**：
  - E0 新增 absolute metrics、真实 boundary jump、事件标线曲线和 prompt heatmap；
  - E1 新增逐 prompt/seed paired VBench、latent-quality Spearman 和人工 failure tags；2026-07-22 起取消 VBench 数值硬阈值，只保留人工 catastrophe gate；
  - E2/E3 支持 Delay-48/72、BF16 sink、Gradual-3、K-only/V-only、layer groups 和 sampled attention trace；
  - E4 gate 同时检查质量、相对 K2V2 的 boundary-jump 降幅和 actual KV saving；
  - E5 的 183/501/699 分阶段执行，501/699 不再由 latent gate 自动放行；
  - 当前 EPIC launcher 只接受物理 GPU 2/3，默认 GPU 2；E6 留到质量 winner 冻结后的独立分支。

- **TRQ 统一完成（2026-07-11）**：
  - 本机主目录改为 `/Users/moweile/Code/LAB/videoquant-trq`。
  - `src/trq/real/trq.py` 是 temporal residual quantization 唯一稳定实现；Python 包入口统一为 `trq`。
  - v1 稳定 predictor 为 `identity`、`affine_channel`；RoPE 留待实验。
  - `hrq-*`、`s2pp-*` 保留兼容入口，学弟的 `qvg` 仓库不再作为运行时依赖。
  - CPU codec、真实 affine 参数、head-wise、K/V 位宽、cache 生命周期和诊断测试共 36 项通过。
  - 协作者入口已更新：`temporalresidualkvquant/README.md` 和 `docs/getting_started.md`。

### 历史状态

- **工作区已改为 Git worktree 隔离模式**（2026-05-23）：
  - 原目录 `/mnt/workspace/caipeiliang/code/moweile/videoquant` 只作为 detached HEAD 管理入口。
  - `videoquant-main` 对应 `main`。
  - `videoquant-prompt` 对应 `HWQ_prompt_router`，commit `345ecf7`。
  - `videoquant-online` 对应 `hwq_online_calibration`，commit `d1bf3eb`。
  - `videoquant-hrq` 对应 `feature/hwq-residual-quant`，commit `3ee4612`。
  - prompt / online 两分支已做 `git merge-tree` dry-run，未来互相合并无文本冲突 marker。

- **32-prompt 大规模 VBench 全矩阵对比完成**（2026-05-22）：12 条实验线（含 k=8 int4+int2 修复），180 frames，MovieGenVideoBench 前 32 prompts
- 核心发现：
  - **int8+int4**: 全部变体在 ↓0.44% 以内，TK8 最优 ↓0.23%，近乎无损；Top-K vs Random 增益 +0.19pp
  - **int4+int2**: TK8 ↓2.89% 最优，TK2 ↓5.20% 最差，top-k 优势明确（k 越大质量越好）；Top-K vs Random 增益 +0.34pp (TK4 vs Rand4)
  - **k 的收益递减**: int8+int4 下 k=2/4/6/8 几乎无差（0.7609-0.7616），说明 int8+int4 精度足够高，2 个 high-precision head 就够；int4+int2 下 k 越大越好（TK2 ↓5.20% → TK8 ↓2.89%）
- 当前最重要的研究工作聚焦三块：
  - `importance metric`：定义 head 重要性，例如量化敏感性、attention output 变化、denoising prediction 影响、跨 chunk 稳定性，或 identity/scene/motion 相关敏感性。
  - `importance collection`：确定离线 calibration、在线估计，或前几个 chunk calibration 后固定 policy。
  - `policy granularity`：确定全模型统一、per-layer、per-chunk、sink/history/tail、K/V 分开，或按 prompt 类型自适应的 top-k 策略。

## 最近完成

- **32-prompt 全矩阵 VBench 对比 + k 消融 + int4+int2 sweep**（2026-05-22）：
  - 完成 12 条实验线的 180-frames MovieGenVideoBench 评估（前 32 prompts）：
  - **int8+int4 组**（6 线）：
    | 实验线 | Final Score | vs BF16 | Peak VRAM | KV Cache |
    |--------|------------|---------|-----------|----------|
    | BF16 Baseline (32p) | 0.7633 | — | 61.9 GB | 48.2 GB |
    | TK8 int8+int4 | 0.7616 | ↓0.23% | 37.4 GB | 23.4 GB |
    | TK4 int8+int4 | 0.7615 | ↓0.25% | 33.6 GB | 19.7 GB |
    | TK6 int8+int4 | 0.7609 | ↓0.31% | 35.4 GB | 22.1 GB |
    | TK2 int8+int4 | 0.7609 | ↓0.32% | 31.7 GB | 18.2 GB |
    | Rand4 int8+int4 | 0.7600 | ↓0.44% | 33.6 GB | 19.7 GB |
  - **int4+int2 组**（5 线）：
    | 实验线 | Final Score | vs BF16 | Peak VRAM | KV Cache |
    |--------|------------|---------|-----------|----------|
    | TK8 int4+int2 | 0.7413 | ↓2.89% | 28.6 GB | 14.4 GB |
    | TK6 int4+int2 | 0.7348 | ↓3.75% | 27.6 GB | 13.4 GB |
    | TK4 int4+int2 | 0.7302 | ↓4.34% | 26.7 GB | 12.4 GB |
    | Rand4 int4+int2 | 0.7268 | ↓4.79% | 26.7 GB | 12.4 GB |
    | TK2 int4+int2 | 0.7236 | ↓5.20% | 25.8 GB | 11.5 GB |
  - 关键结论：
    - int8+int4 精度下 top-k 优势微弱（k=2/4/6/8 几乎持平），说明 int8+int4 精度已经足够高
    - int4+int2 精度下 top-k 优势显著且 k 越大越好（↓5.20% → ↓2.89%），head importance 的价值在低精度场景更突出
    - Top-K vs Random 增益：int8+int4 +0.19pp，int4+int2 +0.34pp (TK4 vs Rand4)
    - KV Cache 压缩与 k 近似线性：int8+int4 每增加 2 heads ≈ +2 GB，int4+int2 每增加 2 heads ≈ +1 GB
  - k=8 int4+int2 评估遇 "Too many open files" 错误，修复后所有 8 维度评估完成
  - 新增 top-k policies: `assets/head_importance/top2_dmd_loss.json`, `top6_dmd_loss.json`, `top8_dmd_loss.json`
  - 更新 `aggregate_results.py` 支持分组相对退化比较

- **Top-K HWQ int8+int4 32-prompt VBench 大规模对比**（2026-05-20）：
  - 使用 MovieGenVideoBench 前 32 prompts，180 frames，评估 2 条实验线各 32 条视频：
    | 实验线 | Final Score | vs BF16 | Peak VRAM | KV Cache |
    |--------|------------|---------|-----------|----------|
    | BF16 Baseline (32p) | 0.7633 | — | 61.9 GB | 48.2 GB |
    | **Top-K HWQ int8+int4 (32p)** | **0.7615** | **↓0.23%** | **33.6 GB** | **19.7 GB (2.45×)** |
  - 8 个维度几乎全部持平：subject_consistency ↓0.01%, background_consistency ↓0.12%, motion_smoothness ↓0.06%
  - 结论：Top-K HWQ int8+int4 几乎无损（↓0.23%），且 KV Cache 压缩 2.45×，Peak VRAM 降低 45.7%
  - 该结果有力支撑了论文核心主张：head-wise 混合精度 + importance top-k 策略可以在不损失视频质量的前提下实现显著显存压缩
  - 新增：`assets/moviegenbench_32.txt` (32-prompt subset)、`results/selfforcing/vbench_eval_mb32/comparison_summary.json`
  - 更新 `aggregate_results.py` 支持新实验标签
- **拷贝 Focused-Forcing 参考代码与 DMD loss 数据**（2026-05-17）：
  - 将 `/data2/moweile-20251213/workspace/focused-forcing-code` 拷贝到 `external/focused-forcing-code/`
  - 新增 `external/README.md`，明确 `external/` 用于存放外部参考代码和分析结果，`temporalresidualkvquant/` 继续作为主方法代码库
  - 确认 `focusedforcing_sf/cf/rf/longlive/dm_loss.json` 均包含 360 个 global head 的 DMD loss 分数，可用于生成 top-k policy
- **新增每周实验汇报目录**（2026-05-17）：
  - 新增 `temporalresidualkvquant/weekly_reports/README.md`，规范每周汇报的命名、结构和维护方式
  - 新增 `temporalresidualkvquant/weekly_reports/2026-W20.md`，整理本周六条实验线、两阶段 head importance 进展、问题和下周计划，方便和老师同步并请老师指导方向
- **补充两条 packed-naive R-HWQ-4h 实验及 VBench 评估**（2026-05-17）：
  - 跑通两条新实验线：
    | 实验线 | 配置 | 最终得分 | vs BF16 |
    |--------|------|---------|---------|
    | R-HWQ-4h Packed (int8+int4) | HIGH=packed-naive-int8, LOW=packed-naive-int4, 4 random heads | 0.6479 | ↓0.10% |
    | R-HWQ-4h Packed (int4+int2) | HIGH=packed-naive-int4, LOW=packed-naive-int2, 4 random heads | 0.6279 | ↓3.19% |
  - int8+int4 packed-naive 几乎无损（↓0.10%），是无需 k-means 的轻量化选项
  - int4+int2 介于 PRQ（↓1.07%）和更大退化之间，提供 5.3× 真压缩
  - 输出目录：`results/selfforcing/rhwq_seed_0_hi_4_packed-naive-int4_lo_packed-naive-int2_64/` 和 `results/selfforcing/rhwq_seed_0_hi_4_packed-naive-int8_lo_packed-naive-int4_64/`
  - 更新 `aggregate_results.py` 至 5 条实验线（不含 naive）
- **四条实验线 VBench 视频质量对比**（2026-05-17）：
  - 使用 VBench-Long 8 维度（subject_consistency, background_consistency, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality, overall_consistency, clip_score）评估 4 条实验线各 2 条视频
  - 修复 2 个 VBench 兼容性 bug（视频名排序解析、split_clip 缓存正则）
  - 评估结果：
    | 实验线 | Final Score | vs BF16 |
    |--------|------------|---------|
    | BF16 Baseline | 0.6486 | - |
    | QVG INT2 (PRQ) | 0.6469 | ↓0.26% |
    | R-HWQ-4h (PRQ) | 0.6416 | ↓1.07% |
  - PRQ-based 量化质量退化极小（全 INT2 ~0.3%，R-HWQ-4h ~1%），naive blockwise 退化显著（~8%）
  - 新增评估脚本：`scripts/eval/evaluate_experiments.sh`、`scripts/eval/aggregate_results.py`
  - 结果存档：`results/selfforcing/vbench_eval/comparison_summary.json`
  - 现有 `run_head_importance_analysis.sh` 已按 `PHASE=inference/scoring/all` 调用两阶段脚本
  - 修复 Phase 2 resume 逻辑：按 chunk 判断已完成 head，避免某个 head 只在部分 chunk 出现时被错误整体跳过
  - launcher 新增实测参数：`HEAD_START`、`HEAD_END`、`ALLOW_INCOMPLETE`、`DELETE_LATENTS_AFTER_SCORING`、`SKIP_EXISTING`
  - 更新 `docs/head_importance_topk.md`，补充两阶段运行、smoke test 和恢复方式
  - 已通过 `py_compile`、`bash -n`、11 个单测
- **head importance analysis 脚本调试**（2026-05-14）：
  - 发现并修复 3 个代码 bug：
    - `self_forcing_dmd.yaml:5`：`real_name: Wan2.1-T2V-14B` → `1.3B`（本地只有 1.3B 权重）
    - `wan_wrapper.py:150-151`：`enable_gradient_checkpointing(enable=True)` 签名不兼容，改为直接设 `self.model.gradient_checkpointing = True`
    - `analyze_head_importance.py:118-121`：DMD text_encoder 未移到 GPU，加入短暂 GPU 编码后立即回 CPU
  - 尝试 3 种内存优化（DMD score CPU offloading、DMD T5 短暂 GPU 使用、heads_per_batch=1），均未能解决 OOM
  - 根因：Self-Forcing 推理阶段 KV cache 膨胀到 78+ GB（126 frames），剩余空间不足以加载 DMD score 模型（~5 GB）
  - 结论：需拆成两阶段脚本 — Phase 1 只推理存 latent，Phase 2 只算 DMD loss
- **新增 per-layer top-k fixed policy 初版**（2026-05-12）：
  - 新增 `TopKHeadPolicy` 和 `load_topk_head_policy()`，支持从 JSON/CSV/TXT 读取 head 重要性或显式 top heads
  - 新增 `hwq.head_importance` 选头模块，库内支持读取 focused-forcing head-ablation DMD loss JSON、聚合均值、per-layer 选择 top-k heads、写出 policy JSON
  - 新增 vendored Self-Forcing head-ablation 分析链路：
    - `wan/modules/causal_model.py` 支持 `ablation_global_head_ids`，按 sample mask 指定 global head
    - `pipeline/causal_inference.py` 支持 latent-only analysis，避免 DMD 分析时额外 VAE decode
    - `backends/self_forcing/analyze_head_importance.py` 可直接跑 mask-head generation + DMD loss + policy 聚合
    - `scripts/self_forcing/run_head_importance_analysis.sh` 一键生成 `assets/head_importance/top4_dmd_loss.json`
  - `CausalInferencePipeline` 新增 `headwise_mode=topk`，量化每层 KV cache 时按 `layer_idx` 使用对应 top-k high-precision heads
  - packed-naive launcher 支持 `HEADWISE_MODE=topk`、`HEAD_IMPORTANCE_PATH`、`HEAD_IMPORTANCE_SCORE_DIRECTION`
  - 新增脚本：`temporalresidualkvquant/scripts/self_forcing/run_packed_naive_topk_hwq.sh`
  - 新增聚合脚本：`temporalresidualkvquant/scripts/aggregate_head_importance.py`，作为 `hwq.head_importance` 的 CLI wrapper，可把 focused-forcing ablation JSON 聚合成 top-k policy JSON
  - 新增文档：`temporalresidualkvquant/docs/head_importance_topk.md`，包含跨机器路径处理和运行命令
  - 已通过 `py_compile`、`bash -n`、聚合脚本 smoke test、11 个单测
- **新增 packed-naive real-compression 支路**（2026-05-11）：
  - 新增 quant types：`packed-naive-int2`、`packed-naive-int4`、`packed-naive-int8`
  - 区别于旧 `naive-int2/int4` fake quant，packed-naive 会存储 uint8 packed codes + per-block min/scale metadata
  - 解压路径已接入 `uncompress_single_cache()`，head-wise mixed groups 可直接复用
  - 新增脚本：`temporalresidualkvquant/scripts/self_forcing/run_packed_naive_hwq.sh`
  - 已通过 `py_compile`、`bash -n`、5 个单测和 `packed-naive-int8` smoke test
- **R-HWQ-4h naive int2/int4 跑通**（2026-05-10）：
  - 配置：4 high-precision heads (naive-int4) + 8 low-precision heads (naive-int2)，block_size=64
  - 不用 QVG 的 triton-nstages-kmeans PRQ，直接用 blockwise 量化
  - 需 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 才能在 A100 80G 上跑通（naive 量化返回 bf16 张量，不压缩显存，峰值 >76G）
  - 产出 2 条视频：`results/selfforcing/rhwq_seed_0_hi_4_naive-int4_lo_naive-int2_64/kc_256_vc_256_nstages_1/`
- **R-HWQ-4h（triton PRQ）首次真实 Self-Forcing 推理跑通**（2026-05-09）：
  - 配置：4 high-precision heads (int4) + remaining heads (int2)，block_size=64，256 K/V centroids，1 PRQ stage，seed=0
  - 产出 2 条视频：`results/selfforcing/rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/`
- **QVG INT2 baseline 在 A100 上跑通**：
  - 对应 Slurm 作业：`24309`，状态 `COMPLETED`，耗时 `00:24:58`
  - 产出 2 条视频：`results/selfforcing/triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/`
- **修复 A100 兼容性**：
  - `fp8e4nv` 自动回退：`quant_pack.py` 新增 `_gpu_supports_fp8e4nv()`，非 Hopper GPU 自动降级 bf16
  - 视频保存：`inference.py` 从已废弃的 `torchvision.io.write_video` 切到 `imageio.mimsave`
- 从 `QVG` 的 `quant_videogen` 中抽出可复用量化核心，现已统一到独立库 `temporalresidualkvquant/src/trq/`。
- 新增 `hwq.headwise`：`RandomHeadPolicy`、`compress_headwise_kv_cache`。
- 新增 `hwq.self_forcing`：`compress_self_forcing_cache_span`。
- 已通过：`py_compile`、`python -m unittest discover -s tests -v`。
- 实验产物目录统一为 `temporalresidualkvquant/results/`（替代 `outputs/`）。
- **引入 external focused-forcing-code + 产出完整 top-k policy**（2026-05-17）：
  - Pull 入 `external/focused-forcing-code/`，包含上游 4 份 DMD loss JSON（cf/sf/rf/longlive，360 heads 各一份，内容相同）
  - 用 `scripts/aggregate_head_importance.py` 直接将 `focusedforcing_sf/dm_loss.json` 聚合成 top-4 policy
  - 产出 `assets/head_importance/top4_dmd_loss.json`：30 layers，每层 top-4 heads
  - 新增文档 `docs/quantization_approaches.md`：Naive / Packed-naive / PRQ 三类量化方案对比
  - 两阶段 smoke test 也已验证通过（42 frames, 6 heads），但因 external 已有现成 360-heads 结果，无需自己跑全量

## 当前阻塞 / 未完成

- E1 仍需在集群复用现有 48 条视频完成 VBench，并人工填写定性 failure tags；
- E2/E3 只完成代码与 CPU/dry-run 验证，尚未生成新 GPU 结果；
- E4 候选必须依据 E2/E3 结果选择，当前没有预先指定 winner；
- E6 Triton/decoded-cache 优化尚未启动，符合质量 winner 冻结后另开分支的计划。

- head importance 目前采用 focused-forcing head ablation 的 DMD loss 聚合；后续仍需评估它和 identity / scene / motion 质量维度的相关性。
- Top-K vs random 的增益在 int4+int2 下仅 +0.38pp（0.6303 vs 0.6279），在 int8+int4 下尚未有 random 对照
- 当前 per-layer top-4 DMD-loss 策略可能过于粗略：per-chunk top-k、K/V 分开选头、不同 prompt 类型自适应等更细粒度策略尚未实验

## 下一步

- **优先**: 统一实验矩阵已基本完成，接下来聚焦：
  - Top-K × PRQ 叠加：DMD top-4 + PRQ int4+int2，可能的 SOTA 路线
  - QVG PRQ INT2 的 32-prompt 结果，完成 BF16 / PRQ / Top-K 三足对照
  - 探索更优 importance metric（当前 DMD loss 在 int8+int4 下 top-k vs random 仅 +0.19pp）
  - Prompt-adaptive policy：在 `videoquant-prompt` 中基于 `HWQ_prompt_router` 继续构造 bucket-specific policies。
  - First-chunks online calibration：在 `videoquant-online` 中继续验证 runtime policy 的质量和 selected-head overlap。
- **论文叙事方向**：
  - int8+int4 全部方案近乎无损（<0.5%），可作为 "安全压缩" 定位
  - int4+int2 需要 top-k head importance（↓2.89% vs ↓5.20%），展示 head-wise 价值
  - k 的收益递减分析：int8+int4 仅需 k=2，int4+int2 需 k=8
- 统一实验矩阵当前状态：
  - BF16 baseline：✅ ↓0.00%，~80 GB (2p) / ✅ 61.9 GB (32p)
  - QVG INT2 (PRQ)：✅ ↓0.26%，~20 GB (2p) / ❌ 待跑 32p
  - R-HWQ-4h PRQ (int4+int2)：✅ ↓1.07%，~20 GB (2p)
  - R-HWQ-4h Packed (int8+int4)：✅ ↓0.10%，~40 GB (2p) / ✅ Rand4 ↓0.44% (32p)
  - R-HWQ-4h Packed (int4+int2)：✅ ↓3.19%，~26 GB (2p) / ✅ Rand4 ↓4.79% (32p)
  - Top-K HWQ Packed (int4+int2)：✅ ↓2.82%，26 GB (2p) / ✅ TK4 ↓4.34% (32p)
  - Top-K HWQ Packed (int8+int4)：✅ ↓0.23%, 33.6 GB (32p) / ✅ k-sweep 全完成
  - R-HWQ-2h：❌ 待跑
