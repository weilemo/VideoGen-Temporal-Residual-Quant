# Self-Forcing QVG 准备

日期：`2026-05-06`

## 本次目标

- 为 `QVG` 的 `Self-Forcing` 复现线打通最小可执行前置路径。

## 已完成

- 在 `Quant-VideoGen` 下建立 `Self-Forcing` 所需 `ckpts` 入口：
  - `Quant-VideoGen/ckpts/Self-Forcing/self_forcing_dmd.pt`
  - `Quant-VideoGen/ckpts/Self-Forcing/Wan2.1-T2V-1.3B`
- `self_forcing_dmd.pt` 已软链到用户模型目录：
  - `/mnt/users/moweile-20251213/models/huggingface/hub/models--gdhe17--Self-Forcing/snapshots/2f8b779212da279d212c22a509b66ad6552f350e/checkpoints/self_forcing_dmd.pt`
- `Wan2.1-T2V-1.3B` 已软链到公共模型目录：
  - `/mnt/public/pretrained_models/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a`

## 已验证

- 下列关键文件存在且可解析：
  - `self_forcing_dmd.pt`
  - `Wan2.1_VAE.pth`
  - `models_t5_umt5-xxl-enc-bf16.pth`
  - `diffusion_pytorch_model.safetensors`
  - `google/umt5-xxl`

## 当前可直接执行

- 在 `Quant-VideoGen` 根目录下可以直接尝试：
  - `bash scripts/Self-Forcing/run_bf16.sh`
  - `bash scripts/Self-Forcing/run_qvg.sh`

## 注意

- 这一步只验证了路径与权重入口，不代表环境依赖已经完全满足。
- 真正开始跑之前，还需要确认 `QVG` 运行环境、`flash-attn`、`triton` 与 `torch` 版本匹配。
