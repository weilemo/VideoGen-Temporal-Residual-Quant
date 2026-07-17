# Identity 差异定位实验

目标：判断 TRQ identity 与学弟 QVG S2++ identity 的效果差异来自 codec、参数，
还是 Self-Forcing 在线集成。先跑同输入离线 parity，再跑小规模在线矩阵，最后才跑
MovieGenBench-32/VBench。

## 1. 集群准备

两个仓库放在同一级目录：

```bash
cd ~/workspace
git clone https://github.com/weilemo/VideoGen-Temporal-Residual-Quant.git
git clone https://github.com/jiahui1021/qvg.git

cd VideoGen-Temporal-Residual-Quant/temporalresidualkvquant
python -m pip install -e '.[gpu,analysis]'
```

先记录版本，结果报告也会自动保存 commit 和学弟 codec SHA256：

```bash
git -C ~/workspace/VideoGen-Temporal-Residual-Quant rev-parse HEAD
git -C ~/workspace/qvg rev-parse HEAD
```

### 指定可用 GPU

如果集群只允许使用物理 GPU 2，运行跨仓库实验时同时设置
`CUDA_VISIBLE_DEVICES=2` 和 `GPU_IDS=2`：

```bash
CUDA_VISIBLE_DEVICES=2 GPU_IDS=2 \
QVG_ROOT=~/workspace/qvg \
bash scripts/analysis/run_bf16_cross_repo.sh

CUDA_VISIBLE_DEVICES=2 GPU_IDS=2 \
QVG_ROOT=~/workspace/qvg \
bash scripts/analysis/run_identity_online_matrix.sh
```

单独运行 TRQ 时只需：

```bash
CUDA_VISIBLE_DEVICES=2 \
bash scripts/self_forcing/run_hrq_predictor_ablation.sh identity
```

设置后，物理 GPU 2 会在当前进程内映射成逻辑 `cuda:0`，日志显示 `cuda:0`
是正常现象。可以先验证：

```bash
CUDA_VISIBLE_DEVICES=2 python -c \
'import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'
```

预期设备数量为 `1`。不要设置 `CUDA_VISIBLE_DEVICES=2,3`，否则程序会同时看到
两张卡。

## 2. BF16 基线

先确认两个 Self-Forcing 后端本身一致：

```bash
QVG_ROOT=~/workspace/qvg \
PROMPTS_PATH="$PWD/assets/t2v.txt" \
NUM_OUTPUT_FRAMES=42 \
SEED=0 \
bash scripts/analysis/run_bf16_cross_repo.sh
```

默认输出 `video_parity.json`，包含 paired PSNR、SSIM 和可用时的 LPIPS。如果
BF16 已明显不同，应先检查 checkpoint、后端 commit 和随机数，不继续归因 codec。

## 3. 首事件同输入 Codec parity

先用 TRQ 在线运行采集第一次量化事件。42 帧足够跨过首次 cache 量化边界：

```bash
cd ~/workspace/VideoGen-Temporal-Residual-Quant/temporalresidualkvquant

TRQ_PARITY_CAPTURE_DIR="$PWD/results/identity_first_event/raw" \
TRQ_PARITY_CAPTURE_LAYERS=0,5,11,17,23,29 \
PROMPTS_PATH="$PWD/assets/t2v.txt" \
NUM_OUTPUT_FRAMES=42 \
HEADWISE_MODE=none \
TRQ_BITS=2 \
TRQ_ANCHOR_BITS=4 \
TRQ_GROUP_SIZE=64 \
TRQ_PREDICTOR_STRIDE=1560 \
OUTPUT_FOLDER="$PWD/results/identity_first_event/trq_video" \
bash scripts/self_forcing/run_hrq_predictor_ablation.sh identity
```

快照包含同一时刻的 raw K/V、TRQ decoded K/V、实际 token span、state metadata
和 quant config。随后把同一 raw KV 同时送入两个 codec：

```bash
QVG_ROOT=~/workspace/qvg \
DUMPS_GLOB="$PWD/results/identity_first_event/raw/*.pt" \
LAYERS=all \
MAX_DUMPS=0 \
DEVICE=cuda \
STUDENT_TRITON=1 \
OUTPUT_DIR="$PWD/results/identity_first_event/parity" \
bash scripts/analysis/run_identity_codec_parity.sh
```

默认比较三套 profile：

| Profile | bits | anchor | group | stride |
|---|---:|---:|---:|---:|
| common | 2 | 4 | 64 | 1560 |
| stride448 | 2 | 4 | 64 | 448 |
| legacy_student | 4 | 8 | 32 | 448 |

自定义配置格式为 `名称:bits:anchor:group:stride`：

```bash
CONFIGS='mine:2:4:64:1560,student:2:4:64:448' \
bash scripts/analysis/run_identity_codec_parity.sh
```

首先看 `report.md`：

- `TRQ/S2 raw Rel-L2`：各自相对同一 raw KV 的误差；
- `TRQ vs S2 Rel-L2`：相同配置下两个实现是否已经离线分叉；
- decoder parity：TRQ encoder/decoder、S2 Torch/TRQ compatibility decoder；
- `STUDENT_TRITON=1` 时额外检查 S2 Torch/Triton parity；
- `unit_rows.csv`：确定第一次出现差异的 predictor unit。

## 4. 在线 `实现 × 配置` 矩阵

先用单 prompt、42 帧。仓库内 `assets/t2v.txt` 只含一条 prompt：

```bash
QVG_ROOT=~/workspace/qvg \
PROMPTS_PATH="$PWD/assets/t2v.txt" \
NUM_OUTPUT_FRAMES=42 \
SEED=0 \
PROFILES='mine:2:4:64:1560,student:2:4:64:448' \
IMPLEMENTATIONS='trq,student' \
bash scripts/analysis/run_identity_online_matrix.sh
```

这会得到四组：TRQ×mine、TRQ×student、S2++×mine、S2++×student。默认关闭
head-wise，K/V 都使用纯 identity + asymmetric zero-point。每个输出目录保存
`resolved_env.txt`；TRQ 目录还保存首事件 parity snapshot。

先检查命令而不运行：

```bash
DRY_RUN=1 bash scripts/analysis/run_identity_online_matrix.sh
```

矩阵脚本不调用学弟的 `run_s2.sh`，因为该脚本当前硬编码 prompt 文件且没有转发
非零 `SEED`。矩阵会直接调用学弟的 `inference.py`，显式传入 prompt、seed、
checkpoint 和全部 S2++ 参数。权重不在统一目录时设置 `STUDENT_CKPT_PATH` 和
`FORCING_WAN_MODEL_DIR`。

## 5. 结论顺序

1. 两边 BF16 视频先一致；否则先修 Self-Forcing/checkpoint/随机数。
2. 同 raw KV、同配置的 codec 已不同：定位量化舍入、zero-point、packing、dtype。
3. S2 Torch 与 Triton 不同：先修 decoder parity。
4. 离线 codec 一致、在线视频不同：检查量化 span、cache read、RoPE、BF16 tail。
5. 单 prompt 定位完成后，再扩大到 4 prompts，最后跑 MovieGenBench-32 和
   VBench-Long。不要用 VBench 平均分代替首次分叉定位。

## 输出结构

```text
results/identity_first_event/
├── raw/                         # 第一次在线量化事件的层快照
├── parity/
│   ├── manifest.json
│   ├── resolved_config.json
│   ├── parity_rows.csv
│   ├── unit_rows.csv
│   ├── summary.json
│   └── report.md
└── trq_video/

results/identity_online_matrix/
├── trq_mine_identity/
├── trq_student_identity/
├── student_mine_identity/
└── student_student_identity/
```
