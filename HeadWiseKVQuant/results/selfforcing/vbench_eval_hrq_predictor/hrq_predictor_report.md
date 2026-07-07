# HRQ 预测器实验报告

**分支：** `feature/hwq-residual-quant`
**开始日期：** 2026-05-28
**更新日期：** 2026-06-04
**配置：** topk-8，hrq-int4（高精度）/ hrq-int2（低精度），block_size=64，stride=1560

---

## Identity 基线（参考值，请勿覆盖）

来源：`hrq_topk_top8_dmd_loss_mb32_hi_8_hrq-int4_lo_hrq-int2_64`（MB32）

| 指标 | 数值 |
|------|------|
| VBench 综合评分 | **0.7616** |
| KV Cache 显存 | **14067.8 MB** |
| 峰值显存 | **28364.8 MB** |
| KV 压缩率（vs BF16） | **71.5%** |

---

## 第一阶段：预测器诊断

### 实验配置

- **训练数据：** `kv_dumps/train/*.pt`（moviegenbench_15.txt 前 10 条）
- **测试数据：** `kv_dumps/heldout/*.pt`（剩余 5 条）
- **预测器步长：** 1560 token（1 帧）
- **完整逐层报告：** `predictor_diagnostics.txt`（30 层 × K/V × identity/affine）
- **分析日志：** `logs/analyze_hrq_predictor_20260529_220350.log`

### 预量化残差 Rel-L2 汇总

30 层宏平均 Rel-L2：

| 预测器 | K 训练集 | K 测试集 | V 训练集 | V 测试集 | 训练→测试 gap（K / V） |
|--------|---------|---------|---------|---------|----------------------|
| identity | 0.3600 | 0.3479 | 0.6940 | 0.6783 | −0.0121 / −0.0157 |
| affine_channel | 0.3389 | 0.3294 | 0.6356 | 0.6269 | −0.0095 / −0.0087 |
| tiny_mlp | — | — | — | — | （已跳过，见结论） |

**affine 相对 identity 在测试集上的提升：K 5.3%，V 7.6%，均值 6.4%**

### 关键观测

- [x] **affine_channel 能否降低测试集残差？** 能。K：0.3479→0.3294（−5.3%），V：0.6783→0.6269（−7.6%）
- [x] **训练→测试 gap 是否过大（过拟合）？** 不。gap 为小幅负值，测试集 Rel-L2 略低于训练集，说明 affine 参数泛化良好，无过拟合。
- [ ] **tiny_mlp 是否优于 affine？** 未评估——affine 提升仅 6.4%，低于 10% 门槛，不值得继续训练 tiny_mlp（见 Q4）。

---

## 第二、三阶段：已拟合参数文件

- **Affine 参数：** `assets/hrq_predictors/affine_channel_self_forcing_dmd.pt` ✓ 已保存（756 KB）
- **Tiny MLP 参数：** `assets/hrq_predictors/tiny_mlp_self_forcing_dmd.pt` — **未生成**（提升 < 10%，按决策框架跳过）

> 由 `analyze_hrq_predictor.py --skip_mlp` 生成。

---

## 第五阶段：Quick15 VBench 消融实验

> 更新：早期报告曾按 10% 残差提升门槛将 Quick15 标记为“已跳过”；后续实际补跑了 `affine_channel` Quick15。本节以 2026-06-02 前后落盘的结果文件为准。

### affine_channel Quick15 生成结果

- **生成目录：** `results/selfforcing/vbench_eval_hrq_predictor/quick15_topk8_hrq-int4_hrq-int2_pred_affine_channel/`
- **基础视频：** 12 个 `*-0_ema.mp4`，生成时间 2026-05-30 13:05–17:01
- **VBench split clip：** 312 个 mp4，用于分项评估
- **推理日志：** `affine_channel_inference.log`、`affine_resume.log`
- **评估日志：** `vbench_affine_channel.log`

### affine_channel Quick15 VBench 分项

| 指标 | 分数 | 结果文件 |
|------|------:|----------|
| subject_consistency | 0.934849 | `pred_affine_channel_q12_subject_consistency_eval_results.json` |
| background_consistency | 0.935717 | `pred_affine_channel_q12_background_consistency_eval_results.json` |
| motion_smoothness | 0.914331 | `pred_affine_channel_q12_motion_smoothness_eval_results.json` |
| dynamic_degree | 0.931159 | `pred_affine_channel_q12_dynamic_degree_eval_results.json` |
| aesthetic_quality | 0.423799 | `pred_affine_channel_q12_aesthetic_quality_eval_results.json` |
| imaging_quality | 0.525848 | `pred_affine_channel_q12_imaging_quality_eval_results.json` |
| overall_consistency | 0.053911 | `pred_affine_channel_q12_overall_consistency_eval_results.json` |
| clip_score | 0.218827 | `pred_affine_channel_q12_clip_score_eval_results.json` |

说明：当前目录中只有 VBench 分项 JSON，没有发现可复现的综合分聚合脚本或权重配置；因此本报告不手工合成“VBench 综合分”。与 MB32 identity 基线的 `0.7616` 不应直接横向比较，因为一个是 MB32 综合结果，一个是 Quick15 分项评估。

### Quick15 同规模对比：baseline vs affine_channel

对照来源：`results/selfforcing/vbench_eval_hrq_quick15/`，对应 `hrq_topk_top8_dmd_loss_quick15_hi_8_hrq-int4_lo_hrq-int2_64`。该 baseline 与 affine_channel 同为 Quick15 分项评估，可直接比较 8 个 VBench 指标。

| 指标 | baseline quick15 | affine_channel quick15 | 差值 affine-baseline | 相对变化 |
|------|------:|------:|------:|------:|
| subject_consistency | 0.985644 | 0.934849 | -0.050795 | -5.15% |
| background_consistency | 0.971236 | 0.935717 | -0.035518 | -3.66% |
| motion_smoothness | 0.990793 | 0.914331 | -0.076462 | -7.72% |
| dynamic_degree | 0.289855 | 0.931159 | +0.641304 | +221.25% |
| aesthetic_quality | 0.625835 | 0.423799 | -0.202036 | -32.28% |
| imaging_quality | 0.708939 | 0.525848 | -0.183092 | -25.83% |
| overall_consistency | 0.229023 | 0.053911 | -0.175112 | -76.46% |
| clip_score | 0.315596 | 0.218827 | -0.096769 | -30.66% |

### Quick15 聚合总分对比

聚合方式沿用 `scripts/eval/aggregate_results.py`：先按 VBench 维度 min/max 归一化，再计算 temporal / frame-wise / text 三类分数，最后按 2:2:1 聚合为 final score。

| 聚合项 | baseline quick15 | affine_channel quick15 | 差值 affine-baseline | 相对变化 |
|--------|------:|------:|------:|------:|
| temporal_quality | 0.876044 | 0.861975 | -0.014069 | -1.61% |
| frame_wise_quality | 0.667387 | 0.474823 | -0.192564 | -28.85% |
| text_alignment | 0.758219 | 0.381654 | -0.376565 | -49.66% |
| final_score | 0.769016 | 0.611050 | -0.157966 | -20.54% |
| six_dims_mean_raw | 0.762050 | 0.777617 | +0.015567 | +2.04% |

注意：`six_dims_mean_raw` 只是 6 个非文本 raw 指标的简单平均，不是正式 VBench final。它在 affine_channel 下略升，主要由 `dynamic_degree` 大幅上升拉动；正式 final score 仍然明显下降，因为 frame-wise 和 text_alignment 损失更大。

### Quick15 结论

affine_channel 的预量化残差确实下降，但同规模 Quick15 对比显示它只显著提高 `dynamic_degree`，其余 7 个 VBench 分项全部下降；按项目聚合公式计算，final score 从 0.769016 降到 0.611050，下降 0.157966（-20.54%）。因此，Quick15 结果明确不支持将默认预测器从 identity 切换到 affine_channel。若后续仍想利用 affine_channel，方向更像是动态场景专用策略，而不是全局默认策略。

---

## 第六阶段：MB32 扩展实验

**尚未运行 affine_channel MB32 扩展。** 当前只有 identity MB32 参考值和 affine_channel Quick15 分项结果。

| 预测器模式 | 评估规模 | VBench 综合分 | KV 显存（MB） | 峰值显存（MB） | vs BF16 压缩率 | 备注 |
|-----------|----------|-------------|-------------|-------------|--------------|------|
| identity | MB32 | 0.7616 | 14067.8 | 28364.8 | 71.5% | 参考基线 |
| affine_channel | Quick15 | 0.611050 | — | — | — | 同规模 baseline Quick15 为 0.769016 |
| affine_channel | MB32 | — | — | — | — | 未运行 |
| tiny_mlp | — | — | — | — | — | 未生成/未评估 |

---

## 结论

### Q1：alpha/beta 是否跨帧/块共享？

本实现中，alpha/beta 离线拟合后在推理时**保持固定**——相同的 alpha/beta 用于每个新视频的所有帧块（t=1..N-1），**不随样本更新**。

**结论：** 是，共享且固定。关键在于最小二乘解是否泛化。结果表明确实泛化：训练→测试 gap 为小幅负值（测试集 Rel-L2 反而更低），不存在过拟合。

### Q2：最小二乘解能否泛化到新样本？

**结论：泛化良好。** affine_channel 的训练→测试 gap 为 −0.0095（K）和 −0.0087（V），均为小幅负值，说明 alpha/beta 在训练数据上略有欠拟合，在测试集表现反而稍好。无可测过拟合，拟合参数在未见提示词上稳定可靠。

### Q3：tiny_mlp 是否比 affine 更可靠？

**结论：未评估。** affine 相对 identity 在测试集的提升仅 6.4%（K：5.3%，V：7.6%），远低于 10% 门槛。在 affine 基线本身提升有限的情况下，为 30 层 × K/V = 60 个 MLP 投入 GPU 训练时间不合算，故跳过 tiny_mlp。

### Q4：是否值得将预测器从 identity 升级？

**决策框架结果：**
- 测试集残差提升（affine vs identity）：**均值 6.4%**（K：5.3%，V：7.6%）→ **低于 10% 门槛**
- VBench quick15：**已有同规模 baseline 对照**；affine_channel final score 0.611050 vs baseline 0.769016，下降 0.157966（-20.54%）
- MB32 扩展：**未运行 affine_channel**

**结论：暂不值得升级默认预测器。**

基于当前证据，affine_channel 虽然能稳定降低预量化残差，但幅度只有约 6.4%；同规模 Quick15 对比中，它以明显牺牲一致性、美学、画质和 CLIP 对齐为代价换来更高的动态程度，最终聚合分从 0.769016 降到 0.611050。**Identity（等价于 alpha=1、beta=0 的固定预测）仍保持默认预测器。** 已拟合的 `affine_channel_self_forcing_dmd.pt` 参数文件和 Quick15 结果建议作为后续对照实验素材保留；若要重新打开这条线，下一步应考虑“动态场景专用触发”或直接运行 affine_channel MB32 验证是否存在规模效应。

---

## 附录：逐层 K/V 残差剖析

完整逐层 Rel-L2 表见 `predictor_diagnostics.txt`，主要规律如下：

- **K cache 残差整体低于 V cache**（各层均如此）。
- **中间层（L8–L19）残差最大**（K：0.30–0.62，V：0.68–0.97）；早期（L0–L2）和末期（L27–L29）层残差明显更低。
- **Affine 提升在中间层最为显著**，例如 L18 V：identity 0.9215 → affine 0.8078（测试集 −12.4%）。
- **训练→测试 gap 各层均小**（|gap| < 0.04），证实泛化稳定。

逐头 Affine 残差（测试集，Layer 0 和 Layer 29）：

| 层 | KV | 各头范围 | 备注 |
|----|----|---------|------|
| L00 | K | h0=0.152 … h4=0.028 | 各头差异大（0.028–0.286） |
| L00 | V | h0=0.393 … h9=0.408 | 相对均匀（0.238–0.408） |
| L29 | K | h0=0.101 … h7=0.360 | 各头差异大（0.036–0.360） |
| L29 | V | h0=0.567 … h3=0.659 | 整体偏高（0.478–0.659） |
