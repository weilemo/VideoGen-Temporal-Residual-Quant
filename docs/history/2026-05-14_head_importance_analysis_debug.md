# 2026-05-14 head importance analysis 脚本调试

## 任务

测试 `bash scripts/self_forcing/run_head_importance_analysis.sh`

## 发现与修复

3 个代码 bug 已修复（未提交，3 files modified in working tree）：

| 文件 | 问题 | 修复 |
|------|------|------|
| `self_forcing_dmd.yaml:5` | `real_name: Wan2.1-T2V-14B`，本地只有 1.3B | 改为 `Wan2.1-T2V-1.3B` |
| `wan_wrapper.py:150-151` | `enable_gradient_checkpointing(enable=True)` 与 `CausalWanModel._set_gradient_checkpointing(self, module, value=False)` 签名不兼容 | 直接设 `self.model.gradient_checkpointing = True` |
| `analyze_head_importance.py:118-121` | DMD text_encoder 未移到 GPU | 短暂 move to GPU 编码后立即回 CPU |

## 内存优化尝试

3 轮优化，均未能解决 OOM：

1. DMD score 模型 CPU offloading（推理时在 CPU，loss 计算时才上 GPU）
2. DMD T5 text_encoder 短暂 GPU 使用（编码完立即回 CPU，节省 ~6 GB）
3. Smoke test：`NUM_OUTPUT_FRAMES=42 HEAD_END=6 HEADS_PER_BATCH=1`（仅 6 heads，42 frames）仍 OOM

## 根因

Self-Forcing 推理阶段 KV cache 随 frames 增长膨胀到 ~78 GB（126 frames），A100 80GB 仅剩 ~2 GB。DMD score 模型（2× Wan 1.3B ≈ 5 GB bf16）无论何时加载都会超限。

## 结论

`analyze_head_importance.py` 单进程方案在 A100 80GB 上不可行。需拆成两阶段：

- **Phase 1**：只跑 Self-Forcing 推理（无 DMD），每个 head ablation 后存 latent 到磁盘
- **Phase 2**：加载 DMD 模型（无需 pipeline），读 latent 文件独立算 DMD loss，聚合 policy

## 当前状态

- 代码修复保留在工作区（未提交）
- STATUS.md / HANDOFF.md / MEMORY.md 已更新
- 等待用户决定是否进行两阶段拆分
