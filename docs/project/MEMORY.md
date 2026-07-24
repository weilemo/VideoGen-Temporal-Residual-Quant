# 长期记忆

## 项目定位

- 项目管理入口：`/mnt/workspace/caipeiliang/code/moweile/videoquant`（detached HEAD，仅管理 worktree）
- 这是一个面向长视频自回归 diffusion 生成的研究项目工作区。
- 项目目标：研究 KV cache quantization，尽量在显著压缩历史 KV 显存占用的同时，保持长时视频质量，重点关注 `identity consistency`、`scene consistency` 和 `motion continuity`。
- 当前可见子目录：
  - `temporalresidualkvquant` — 论文方法主代码库（head-wise KV cache quantization）
  - `forcing` — 上游模型和 VBench 评估代码
  - `integrations` — 各基线的 TRQ 适配、补丁与启动入口
  - `references/quant-videogen` — QVG 原始实验仓（保留作参考）

## Worktree 隔离布局

- 当前本机主仓库：`/Users/moweile/Code/LAB/videoquant-trq`。
- 当前残差量化唯一稳定实现：`temporalresidualkvquant/src/trq/real/trq.py`；Python 包名为 `trq`。
- `hrq-*` / `s2pp-*` 是兼容别名；QVG 与学弟的 `qvg` 仓库保留作参考。
- RoPE predictor 尚未进入稳定 codec，需后续独立实验决定。

- 以下是 2026-05-23 服务器环境的历史 worktree 布局：
  - `videoquant-main` → `main`
  - `videoquant-prompt` → `HWQ_prompt_router`
  - `videoquant-online` → `hwq_online_calibration`
  - `videoquant-hrq` → `feature/hwq-residual-quant`
- 原目录 `videoquant/` 用于 `git worktree list`、`git branch -vv`、`git worktree add/remove/prune` 等管理操作。
- 若 agent 要改某个方向，必须进入对应 worktree，避免共享脏工作树。

## 当前问题定义

- 研究对象不是普通 `LLM` 推理，而是 `forcing-based long video generation`。
- 核心矛盾是：视频越长，历史 `KV cache` 越大，显存很快成为瓶颈，进而限制可生成时长和长程一致性。
- 当前需要系统回答的问题包括：
  - `LLM` 的通用 `KV quantization` 方法能否直接迁移到长视频 diffusion。
  - 如果不能直接迁移，主要质量损伤更集中出现在 `motion`、`identity` 还是 `scene layout`。
  - 视频中不同时间段历史、不同 attention heads 是否对量化敏感性不同。

## 当前技术路线

- 技术路线分成两条主线并行推进：
  - 视频原生量化线
  - `LLM KV quant` 方法迁移线

## 视频原生量化线

- 当前最贴题的起点是 `QVG (Quant VideoGen)`。
- `QVG` 直接研究 autoregressive video generation 中的 `2-bit KV-cache quantization`。
- `QVG` 不是简单照搬 `LLM` 量化，而是采用视频特化设计，当前需要重点关注：
  - `Semantic-Aware Smoothing`
  - `Progressive Residual Quantization`

## LLM 方法迁移线

- 这条线的目标是把成熟的 `LLM KV quant` 方法迁移到视频生成框架中，作为可复现 baseline。
- 当前优先级建议：
  - `KIVI`
  - `KVQuant`
  - `GEAR`
- 当前默认优先从 `KIVI` 开始，因为它实现相对简单、`training-free`，更适合作为现有视频代码的第一版迁移基线。

## 当前研究判断

- 单纯把 `LLM KV quant` 原样迁到视频生成中，可能可以节省显存，但不一定能保住长时视频质量，尤其可能先伤害 `motion dynamics`。
- 视频量化更可能需要结构感知设计，而不是全局统一 bit 配置。
- 当前最有希望的后续创新方向：
  - `importance-based top-k head-wise mixed precision`
  - `role-aware quantization`，即按 `sink / history / tail` 等时间角色分配不同 bit
- `RandomHeadPolicy` 只是 head-wise quant 的 sanity-check baseline；后续论文方法重点应转向判断 head 重要性，再按 top-k 选择高精度 heads。
- `naive-int2/int4` fake-quant 分支仅用作开发阶段的精度调试工具（量化后立即反量化回 bf16，不节省显存），不纳入任何实验线对比表格。论文中只比较 real-compression 方法（PRQ、packed-naive）。
- 当前 head importance 主线重点关注三件事：
  - `importance metric`：怎样定义每个 head 的重要性分数。
  - `importance collection`：怎样离线或在线收集这些分数。
  - `policy granularity`：top-k 是全模型统一、per-layer、per-chunk，还是按 sink/history/tail、K/V 或 prompt 类型细分。

## 协作记录约定

- 本目录下的记录文件只服务 `videoquant` 项目。
- `docs/project/MEMORY.md` 只记录长期稳定事实。
- `docs/project/STATUS.md` 记录当前任务状态。
- `docs/project/HANDOFF.md` 用于 agent 快速交接。
- `docs/history/` 保存单次任务日志。
- 实验产物 Git 管理约定：
  - 少量样例视频可以直接提交到 GitHub 仓库。
  - 成批实验结果视频默认使用 `Git LFS`，避免主仓库持续膨胀。
- 维护代码的同步边界是 GitHub：本地完成 source、脚本、测试和文档后提交并推送；
  远端 code-server 只人工执行一次 `git pull --ff-only`，不运行 Git 轮询，也不绕过
  Git 直接覆盖维护代码。远端权重、环境、上游仓库、结果和日志不进入该同步链路。

## 服务器执行约定

- 训练、推理、评测、批处理等长任务，默认通过 Slurm 运行。
- 登录节点只做编辑、查看日志、提交作业、短时间调试。
- 正式任务尽量记录：脚本路径、环境名、资源申请、日志路径、输出路径、`jobid`。
- `videoquant` 相关 Slurm 日志与实验输出不一定落在当前 worktree，可能落在共享盘运行副本或 `temporalresidualkvquant/results/` 下；查实验结果时需要同时检查运行脚本里的 `OUTPUT_FOLDER`。
- 后续如果实验结果或 Slurm 日志先输出到共享盘运行副本，默认同步一份回当前方向对应的 `videoquant-*` worktree 或 `temporalresidualkvquant/results/` 记录目录，避免结果只留在临时运行副本。

## Slurm 与服务器规则

- 总规则参考：[服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)
- 如果数据在 `/mnt`，优先考虑 `a100_global`
- 如果数据在本地盘 `/data` 或 `/data2`，优先考虑对应本地分区
- 正式任务建议显式记录：分区、账号、GPU、CPU、内存、时长、日志路径

## Self-Forcing 推理环境

- 环境：Python 3.12，CUDA 12.8，8× A100-80GB SXM
  - Self-Forcing 推理用 `conda activate forcing`
  - VBench 评估用 `conda activate vbench`
- 安装历史（2026-05-09 凌晨）：
  - 01:08-01:09：基础依赖 `diffusers==0.38.0`, `einops==0.8.2`, `easydict==1.13`, `safetensors==0.8.0rc0`, `regex`
  - 01:12：视频/图像链 `decord==0.6.0`, `imageio==2.37.3`, `opencv-python==4.13.0.92`, `scipy==1.17.1`, `lmdb==2.2.0`, `ftfy==6.3.1`
  - 01:14：HF 全家桶 `transformers==5.8.0`, `accelerate==1.13.0`, `tokenizers==0.22.2`
  - 01:21：编译工具 `ninja==1.13.0`, `wheel==0.47.0`
  - 01:23：`flash_attn==2.8.3`
  - 02:06：`imageio-ffmpeg==0.6.0`（配合 imageio 替代已废弃的 torchvision write_video）
- 核心版本速查：
  - `torch==2.9.0+cu128`, `triton==3.5.0`, `flash_attn==2.8.3`, `diffusers==0.38.0`, `transformers==5.8.0`
- 显存限制（2026-05-14 实测）：
  - Self-Forcing 推理 126 frames 时 KV cache 峰值 ~78 GB（A100 80GB），仅剩 ~2 GB 余量
  - `analyze_head_importance.py` 需同时加载推理 pipeline + DMD（3× Wan 1.3B + 2× T5），即使激进 CPU offloading 仍 OOM
  - 结论：head importance analysis 必须拆成两阶段（推理存 latent + 离线算 DMD loss），单进程无法完成
  - 两阶段方案已于 2026-05-17 smoke test 验证通过：Phase 1 推理 ~25.5 GB，Phase 2 DMD scoring ~1.2 GB
  - 全量 360 heads 两阶段命令已就绪：`PHASE=inference bash scripts/self_forcing/run_head_importance_analysis.sh` + `PHASE=scoring`
- 已知问题与修复：
  - A100 (SM 8.0) 不支持 `float8_e4m3fn`，`hwq/real/quant_pack.py` 已加入 GPU 能力检测自动回退 `bfloat16`
  - `torchvision.io.write_video` 已废弃，`inference.py` 已切到 `imageio.mimsave`
- 一键复现：
  ```bash
  pip install torch==2.9.0+cu128 triton==3.5.0 flash-attn==2.8.3 \
      diffusers==0.38.0 transformers==5.8.0 accelerate==1.13.0 \
      einops==0.8.2 imageio==2.37.3 imageio-ffmpeg==0.6.0 \
      decord==0.6.0 opencv-python scipy ninja wheel ftfy safetensors lmdb
  ```

## 待后续补充的长期信息

- 主代码仓路径与分工
- 常用训练脚本路径
- 常用评测脚本路径
- 常用环境名与激活方式
- 常用数据路径与输出路径
- 常用日志目录规范
