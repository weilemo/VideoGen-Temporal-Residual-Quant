# TRQ: Temporal Residual Quantization for Video KV Cache

本目录是统一后的方法代码库，研究 Self-Forcing 长视频生成中的 KV cache
低比特压缩。当前主线是 **TRQ（Temporal Residual Quantization）**：用前一个
已重建时间单元预测当前单元，并只量化预测残差。

如果你是第一次接手，先读本文并跑通 CPU 测试，再看
[上手与实验指南](docs/getting_started.md)。

新机器从零配置请直接看 [新机器初始化](docs/new_machine_setup.md)。

## 当前稳定范围

TRQ v1 的输入布局是 `[B, H, S, D]`，链式重建过程为：

$$
\hat{x}_0 = Q_{anchor}(x_0), \qquad
p_t = f(\hat{x}_{t-1}), \qquad
r_t = x_t - p_t, \qquad
\hat{x}_t = p_t + Q_{residual}(r_t).
$$

- 稳定 predictor：`identity`、`affine_channel`
- 支持位宽：2/4/8 bit；anchor 默认 4 bit
- residual：按 block 的非对称 zero-point 量化并真实位打包
- 解码依赖：前一个**重建值**，不是前一个原始值
- 状态格式：自包含的 `format="trq"`, `version=1`
- `hrq-*`、`s2pp-*` 只作为历史量化类型兼容别名
- RoPE predictor 尚未进入稳定实现，需要后续对照实验决定是否采用

PRQ、packed-naive 和 head-wise Top-K 仍作为对照方法保留，但新残差量化开发
统一放在 `src/trq/real/trq.py`。

## 仓库结构

```text
temporalresidualkvquant/
├── src/trq/                  # 量化包；TRQ、PRQ、head-wise policy、cache
│   ├── real/trq.py           # TRQ v1 唯一稳定 codec
│   ├── real/hrq.py           # 历史 HRQ 兼容适配
│   ├── real/s2pp.py          # 稳定 S2++ 调用兼容适配
│   └── analysis/             # residual 分布与 chain drift 分析
├── backends/self_forcing/    # vendored Self-Forcing 推理后端
├── scripts/self_forcing/     # 视频生成与 KV dump 脚本
├── scripts/analysis/         # TRQ 诊断实验入口
├── scripts/eval/             # VBench/参考指标辅助脚本
├── assets/                   # prompts、head policy、predictor 参数
├── docs/                     # 机制、集成与实验文档
└── tests/                    # CPU codec 与集成单元测试
```

旁边的 `Quant-VideoGen/` 和 `/Users/moweile/Code/LAB/qvg` 都是参考实现，
不是当前运行依赖。不要在其中新增第二套 TRQ codec。

## 安装与 CPU 冒烟

要求 Python 3.10+。在本目录执行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

诊断绘图额外安装：

```bash
python -m pip install -e '.[analysis]'
```

最小 codec 示例：

```python
import torch
from trq import trq_dequantize_tensor, trq_quantize_tensor, trq_state_nbytes

x = torch.randn(1, 12, 3120, 128)  # [B, H, S, D]
state = trq_quantize_tensor(
    x,
    num_bits=2,
    anchor_bits=4,
    block_size=64,
    predictor_stride=1560,
    predictor_mode="identity",
)
x_hat = trq_dequantize_tensor(state, output_dtype=torch.float32)

print(x_hat.shape)
print("physical state bytes:", trq_state_nbytes(state))
```

## Self-Forcing 快速开始

完整视频生成需要 CUDA 环境、Self-Forcing 依赖和模型权重。`pip install -e .`
只安装方法包，不负责建立完整的 GPU 推理环境。

Linux GPU 环境可安装本库声明的 Triton 依赖：

```bash
python -m pip install -e '.[gpu]'
```

Self-Forcing 其余依赖仍以项目已有的 CUDA/conda 环境为准。

权重默认布局：

```text
ckpts/Self-Forcing/
├── self_forcing_dmd.pt
└── Wan2.1-T2V-1.3B/
```

权重在其他位置时，设置 `SELF_FORCING_CKPT_ROOT` 或直接设置 `CKPT_PATH`。
建议先用少量 prompt 和短视频冒烟：

```bash
export SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing

NUM_OUTPUT_FRAMES=42 \
PROMPTS_PATH=assets/t2v.txt \
OUTPUT_FOLDER=results/smoke/bf16 \
  bash scripts/self_forcing/run_bf16.sh
```

TRQ identity 的推荐入口：

```bash
export SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing

PROMPTS_PATH=assets/moviegenbench_32.txt \
  bash scripts/self_forcing/run_hrq_predictor_ablation.sh identity
```

脚本名 `run_hrq_predictor_ablation.sh` 暂时为历史名称，内部实际使用
`trq-int2/trq-int4` 和 `--trq_*` 参数。`affine_channel` 需要
`assets/trq_predictors/affine_channel_self_forcing_dmd.pt`。

## 当前两项诊断实验

统一入口同时覆盖：

1. Raw KV vs residual 分布：比较方差、标准差、分位数、熵和 2-bit 误差。
2. Residual chain drift：比较不同 rollout step 和 reset interval 下的误差累积。

GPU 到位后运行：

```bash
python -m pip install -e '.[analysis]'

CKPT_PATH=/path/to/self_forcing_dmd.pt \
COLLECT_DUMPS=1 \
DUMP_LAYERS=0,5,11,17,23,29 \
LAYERS=0,5,11,17,23,29 \
  bash scripts/analysis/run_trq_diagnostics.sh
```

默认收集 4 个 180-frame prompt 约需 200 GiB 磁盘；先限制层数做 smoke test。
完整协议与输出字段见
[TRQ 诊断实验](docs/trq_diagnostic_experiments.md)。

在线 trajectory drift 使用同 prompt/seed 的 BF16-A、BF16-B 与 TRQ 三路生成：

```bash
DRY_RUN=1 bash scripts/analysis/run_online_paired_rollout.sh
bash scripts/analysis/run_online_paired_rollout.sh
```

该入口会保存 clean latent、结构化 runtime/cache metrics，并输出 prompt-level
bootstrap drift 判据。协议见
[Online Paired Rollout](docs/online_paired_rollout.md)。

## 结果与评估

- 新实验产物统一写入 `results/`。
- 报告至少记录：prompt 集、帧数、seed、TRQ 完整配置、Peak VRAM、KV cache
  实际字节、视频质量指标。
- 主比较集使用 MovieGenBench-32；Quick15 和历史 2-prompt 结果只能做 smoke，
  不与主表混合下结论。
- `scripts/eval/evaluate_experiments.sh` 含历史服务器绝对路径，当前只能作为
  VBench-Long 流程参考，换机器时必须显式配置 VBench 路径与实验目录。

## 文档索引

- [新机器初始化](docs/new_machine_setup.md)：Miniconda、clone、依赖和 GPU 验证
- [上手与实验指南](docs/getting_started.md)：交给新同学的完整操作手册
- [TRQ 统一设计](docs/trq_unification.md)：稳定边界、兼容层和准入条件
- [TRQ 诊断实验](docs/trq_diagnostic_experiments.md)：分布图和 chain drift 协议
- [Online Paired Rollout](docs/online_paired_rollout.md)：BF16 repeat、在线 latent drift 与效率指标
- [Identity 差异定位](docs/identity_parity_experiments.md)：同输入 codec parity 与跨库在线矩阵
- [量化方案对比](docs/quantization_approaches.md)：naive、packed-naive、PRQ、TRQ
- [Self-Forcing 集成](docs/self_forcing_integration.md)：缓存布局与调用路径
- [Checkpoint 配置](docs/checkpoint_sync.md)：权重目录与跨机器设置
- [Workspace 结构](docs/workspace_structure.md)：主库和参考库的边界

## 开发约定

1. 新状态和新实验统一使用 `TRQ` 命名。
2. codec 逻辑只改 `src/trq/real/trq.py`；模型调度改
   `backends/self_forcing/pipeline/causal_inference.py`。
3. predictor 必须保证 encoder reconstruction 与独立 decoder 完全一致。
4. 任何显存结论都报告物理 state bytes，不能只按位宽估算。
5. RoPE、Cross-KV、AR(2)、error feedback 等先放实验分支，通过质量、漂移、
   延迟和状态字节对照后再进入稳定 codec。

本库继承了 Quant-VideoGen 的低比特量化与 PRQ 实现，并将学弟 S2++ 与原 HRQ
中可复用的时间残差机制统一为 TRQ v1。
