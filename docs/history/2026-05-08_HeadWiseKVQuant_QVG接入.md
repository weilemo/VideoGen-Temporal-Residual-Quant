# 2026-05-08 HeadWiseKVQuant 接入 QVG Self-Forcing

## 背景

- 目标是把基于 `Self-Forcing` KV cache 的低精度量化代码从 `QVG` 中独立出来，后续以 `HeadWiseKVQuant` 作为论文方法代码库。
- 上一步已建立独立库 `/data2/moweile-20251213/workspace/videoquant/HeadWiseKVQuant`，但 QVG 的 Self-Forcing pipeline 仍保留 inline head-wise 量化实现。

## 本次修改

- 修改 `Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py`：
  - 从 `hwq` 导入 `ChunkedKVCache`、`compress_kv_cache`、`uncompress_kv_cache`
  - 从 `hwq` 导入 `RandomHeadPolicy`、`compress_headwise_kv_cache`
  - 删除 pipeline 内部临时的 mixed headwise pack/compress 逻辑
  - `headwise_mode=random` 时直接调用独立库完成按 head-group 压缩
- 修改脚本：
  - `scripts/Self-Forcing/run_bf16.sh`
  - `scripts/Self-Forcing/run_qvg.sh`
  - `scripts/Self-Forcing/run_random_hwq.sh`
  - 三者均加入 `PYTHONPATH=../HeadWiseKVQuant/src:experiments/Self-Forcing:.`

## 验证

- `causal_inference.py` 通过 `py_compile`。
- 三个 Self-Forcing shell 脚本通过 `bash -n`。
- `HeadWiseKVQuant` 单测通过：`python -m unittest discover -s tests -v`，共 3 个测试。
- 从 `Quant-VideoGen` 目录用脚本同款 `PYTHONPATH` 导入 `CausalInferencePipeline` 成功。

## 下一步

- 在真实 GPU 环境运行 `scripts/Self-Forcing/run_random_hwq.sh`。
- 重点检查：
  - 是否生成 `R-HWQ-4h` 视频
  - mixed precision 解压是否稳定
  - 日志中的 Rel L2 是否合理
  - 输出视频是否出现明显 identity / scene / motion 退化
