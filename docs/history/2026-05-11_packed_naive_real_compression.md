# 2026-05-11 packed-naive real-compression 支路

## 背景

- 旧 `naive-int2/int4` 是 fake quant：量化后立即反量化回 BF16 tensor，适合作质量对照，但不节省 KV cache 显存。
- 新需求是补一条真正压缩 KV cache 的 blockwise naive 支路，并支持 `int2`、`int4`、`int8`。

## 本次实现

- 新增 `HeadWiseKVQuant/src/hwq/packed_naive.py`：
  - `packed_naive_quantize_tensor`
  - `packed_naive_dequantize_tensor`
  - int2：4 个 code packed 到 1 byte
  - int4：2 个 code packed 到 1 byte
  - int8：1 个 code 存 1 byte
  - 每个 block 保存 `min` 和 `scale`
- 新增 quant types：
  - `packed-naive-int2`
  - `packed-naive-int4`
  - `packed-naive-int8`
- 接入：
  - `compress.py`：新增 `QuantizeFunctions.PACKED_NAIVE`
  - `uncompress.py`：支持 packed-naive dict 解压
  - `kv_cache.py`：修复 `_move_item()` 递归移动 nested dict/list/tuple
- 新增脚本：
  - `scripts/self_forcing/run_packed_naive_hwq.sh`

## 验证

- `py_compile` 通过：
  - `packed_naive.py`
  - `compress.py`
  - `uncompress.py`
  - `kv_cache.py`
- `bash -n` 通过全部 Self-Forcing launcher。
- 单测通过：5 tests OK。
- `packed-naive-int8` 压缩/解压 smoke test 通过，输出 shape 正确。

## 后续实验

默认 R-HWQ packed naive：

```bash
cd HeadWiseKVQuant
bash scripts/self_forcing/run_packed_naive_hwq.sh
```

全 int8 packed naive：

```bash
cd HeadWiseKVQuant
QUANT_TYPE=packed-naive-int8 \
HIGH_PRECISION_QUANT_TYPE=packed-naive-int8 \
LOW_PRECISION_QUANT_TYPE=packed-naive-int8 \
  bash scripts/self_forcing/run_packed_naive_hwq.sh
```
