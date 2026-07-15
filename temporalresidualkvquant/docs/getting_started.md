# TRQ 上手与实验指南

这份文档面向第一次使用统一代码库的同学。目标是依次完成：CPU codec 验证、
Self-Forcing 冒烟、TRQ 视频生成、诊断实验和结果记录。

## 1. 先确认你在主方法库

```bash
cd /path/to/videoquant-trq/temporalresidualkvquant
git status --short --branch
```

只在这个目录开发 TRQ。仓库内 `Quant-VideoGen/` 和仓库外 `qvg/` 用于查看原始
QVG、PRQ、S2++ 机制，不参与当前运行。

主要修改位置：

| 任务 | 位置 |
|---|---|
| TRQ 编码、解码、state contract | `src/trq/real/trq.py` |
| HRQ/S2++ 历史兼容 | `src/trq/real/hrq.py`, `src/trq/real/s2pp.py` |
| cache 压缩路由 | `src/trq/compress.py`, `src/trq/uncompress.py` |
| head-wise policy | `src/trq/headwise.py` |
| Self-Forcing CLI | `backends/self_forcing/inference.py` |
| rollout 中的 KV 调度 | `backends/self_forcing/pipeline/causal_inference.py` |
| 离线诊断 | `src/trq/analysis/`, `scripts/analysis/` |

## 2. 安装方法包

最低要求：Python 3.10、PyTorch 2.1+。GPU 路径还需要与机器匹配的 CUDA、
Triton 和 Self-Forcing 运行环境。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Linux GPU 环境再安装 Triton 可选依赖；Self-Forcing 的其余依赖沿用已配置的
CUDA/conda 环境：

```bash
python -m pip install -e '.[gpu]'
```

需要生成诊断图时：

```bash
python -m pip install -e '.[analysis]'
```

如果 `python -c 'import trq'` 失败，通常是没有在本目录执行 editable install，
或当前 shell 使用了另一个 Python：

```bash
which python
python -m pip show temporal-residual-kv-quant
python -c 'import trq; print(trq.__file__)'
```

## 3. 理解 TRQ 的数据契约

TRQ codec 接收 `[B,H,S,D]`。Self-Forcing cache 原始布局是 `[B,S,H,D]`，
转换由集成层负责，不要在 codec 内重复转置。

```python
import torch
from trq import trq_dequantize_tensor, trq_quantize_tensor

x = torch.randn(1, 2, 12, 16)
state, encoder_recon = trq_quantize_tensor(
    x,
    num_bits=2,
    anchor_bits=4,
    block_size=8,
    predictor_stride=4,
    predictor_mode="identity",
    return_reconstruction=True,
)
decoder_recon = trq_dequantize_tensor(state, output_dtype=x.dtype)
assert torch.equal(encoder_recon, decoder_recon)
```

这个相等性是链式 residual 的硬约束。编码端若用原始上一单元预测、解码端却只能
用重建上一单元，rollout 会产生无法复现的漂移。

`affine_channel` 的参数可以是 `[D]`、`[H,D]` 或 `[L,H,D]`；最后一种必须传
`layer_idx`。参数会写入 state，解码不能静默回退到 identity。

## 4. 准备 Self-Forcing 权重

推荐目录：

```text
ckpts/Self-Forcing/
├── self_forcing_dmd.pt
└── Wan2.1-T2V-1.3B/
    ├── Wan2.1_VAE.pth
    ├── diffusion_pytorch_model.safetensors
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── google/umt5-xxl/
```

不复制权重也可以：

```bash
export SELF_FORCING_CKPT_ROOT=/shared/path/ckpts/Self-Forcing
# 或只覆盖 DMD checkpoint
export CKPT_PATH=/shared/path/self_forcing_dmd.pt
```

下载来源和 rsync 示例见 [checkpoint_sync.md](checkpoint_sync.md)。大权重、
KV dump 和生成视频都不应提交到 git。

## 5. 按顺序跑生成实验

先 BF16 smoke，确认模型、prompt 和输出路径没有问题：

```bash
NUM_OUTPUT_FRAMES=42 \
PROMPTS_PATH=assets/t2v.txt \
OUTPUT_FOLDER=results/smoke/bf16 \
  bash scripts/self_forcing/run_bf16.sh
```

再跑 TRQ identity。当前 launcher 的文件名仍含 `hrq`，只是兼容历史任务名：

```bash
PROMPTS_PATH=assets/moviegenbench_32.txt \
NUM_OUTPUT_FRAMES=42 \
  bash scripts/self_forcing/run_hrq_predictor_ablation.sh identity
```

最后跑 affine 对照：

```bash
TRQ_PREDICTOR_PARAMS_DIR=assets/trq_predictors \
PROMPTS_PATH=assets/moviegenbench_32.txt \
NUM_OUTPUT_FRAMES=42 \
  bash scripts/self_forcing/run_hrq_predictor_ablation.sh affine_channel
```

常用参数：

| 环境变量 / CLI | 含义 | 当前默认 |
|---|---|---|
| `NUM_OUTPUT_FRAMES` | 输出帧数 | 180 |
| `LOCAL_ATTN_SIZE` | local attention window | 180 |
| `PROMPTS_PATH` | prompt 文件 | launcher 各自定义 |
| `OUTPUT_FOLDER` | 输出目录 | launcher 各自定义 |
| `--quant_type` | `trq-int2/int4/int8` | 视脚本而定 |
| `--trq_group_size` | 最后一维量化 block | 64 |
| `--trq_anchor_bits` | anchor 位宽 | 4 |
| `--trq_predictor_stride` | 时间预测单元长度 | 1560 tokens |
| `--trq_predictor_mode` | `identity/affine_channel` | identity |
| `--trq_k_bits`, `--trq_v_bits` | K/V residual 位宽覆盖 | 0，跟随 quant type |

## 6. 跑两项诊断实验

诊断必须基于 BF16 raw KV dump，不能从已经量化的视频反推。

小规模 smoke：

```bash
CKPT_PATH=/path/to/self_forcing_dmd.pt \
COLLECT_DUMPS=1 \
MAX_PROMPTS=2 \
TRAIN_N=0 \
DUMP_LAYERS=0 \
LAYERS=0 \
SAMPLE_CAPACITY=50000 \
BOOTSTRAP_RESAMPLES=100 \
  bash scripts/analysis/run_trq_diagnostics.sh
```

正式实验：

```bash
CKPT_PATH=/path/to/self_forcing_dmd.pt \
COLLECT_DUMPS=1 \
DUMP_LAYERS=0,5,11,17,23,29 \
LAYERS=0,5,11,17,23,29 \
RESET_INTERVALS=1,2,4,8,24,none \
  bash scripts/analysis/run_trq_diagnostics.sh
```

输出默认位于 `results/trq_diagnostics/<run_id>/`，包含 CSV、JSON、PNG 和
`run.log`。首先检查：

- residual 的 std、P99、entropy 和 2-bit NRMSE 是否一致优于 raw KV；
- chain NRMSE/cosine 是否随 rollout step 单调恶化；
- `none` 与 runtime reset interval 的差距；
- K/V、浅层/中层/深层是否表现一致；
- decoder parity 是否通过。

4 个完整 180-frame dump 约 200 GiB。MovieGenBench-32 全层 dump 可能达到 TiB
级，先选择层，再决定是否扩大 prompt 数。统计定义见
[trq_diagnostic_experiments.md](trq_diagnostic_experiments.md)。

如果 TRQ identity 与 QVG S2++ identity 的生成效果不一致，先按
[Identity 差异定位实验](identity_parity_experiments.md) 跑同输入 codec parity 和
`实现 × 配置` 在线矩阵，不要直接用 VBench 总分猜测原因。

## 7. 评估与记录

正式比较至少包含：

| 类别 | 必记内容 |
|---|---|
| 数据 | prompt 集、prompt 数、帧数、seed |
| 方法 | quant type、K/V bits、anchor bits、block/stride、predictor、reset |
| 质量 | VBench-Long 分项及 aggregate；长视频重点看 consistency 和 motion |
| 资源 | Peak VRAM、KV cache physical bytes、推理时间 |
| 正确性 | encoder/decoder parity、是否出现 NaN/Inf、失败样本 |

主表使用 MovieGenBench-32。Quick15、2-prompt 和短帧结果标为 smoke/ablation，
不要与主表直接合并。当前 `scripts/eval/evaluate_experiments.sh` 仍有旧服务器
绝对路径，运行前需要改成当前 VBench checkout；它不是开箱即用的跨机器脚本。

## 8. 常见问题

`Checkpoint does not exist`：检查 `SELF_FORCING_CKPT_ROOT` 的目录层级，或直接
传 `CKPT_PATH`。

`Predictor params not found`：identity 不需要参数；affine 需要
`assets/trq_predictors/affine_channel_self_forcing_dmd.pt`。

`RoPE prediction is experimental`：这是预期行为。RoPE 尚未通过对照实验，不能
作为 TRQ v1 predictor。

显存节省与预期不符：确认运行的不是 `naive-int2/int4`。这两个是 fake quant；
真实压缩使用 `trq-*`、`packed-naive-*` 或 PRQ。

诊断找不到 dump：确认 `DUMPS_GLOB` 指向 `heldout/*.pt`，并且 dump 格式为
`layer_shards`。

## 9. 提交前检查

```bash
python -m unittest discover -s tests -v
git diff --check
git status --short
```

同时确认：新代码只写 TRQ 名称；没有提交 checkpoint、dump、视频和服务器日志；
文档中的命令在仓库根目录可执行；新 predictor 有独立 decoder parity 测试。
