# 量化方案对比：Naive / Packed-Naive / PRQ / TRQ

## 四类量化

### 1. Naive blockwise（`naive-int2/int4`）

**假量化。** 量化后立刻反量化，输出 bf16 张量，shape 和输入完全一样。没有存储收益。

```
x → round(x / scale) → clamp → × scale → x' (bf16)
```

- 用途：快速测量量化数值精度损失
- 显存：0 节省，需要 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 才能在 A100 80GB 上跑 126 frames
- 量化类型：blockwise 对称 INT 量化，scale 为 E4M3 fp8
- 代码：`trq/sim/quant/lowbit_quantize.py` (`_blockwise_intx_quantize_triton`)

### 2. Packed-naive（`packed-naive-int2/int4/int8`）

**真压缩。** 按 block 做 min-max 非对称量化，结果位打包成 uint8，返回 packed dict。

```
对每个 block (block_size=64):
  scale = (max - min) / (2^n - 1)
  code  = round((x - min) / scale)      → 0 ~ 2^n-1
  # int4: 两个 4-bit 挤一个 uint8
  # int2: 四个 2-bit 挤一个 uint8

存储: {packed_codes (uint8, bit-packed), scales (fp16), mins (fp16)}
解压: unpack → code × scale + min
```

- 用途：无 codebook 的真压缩 baseline
- 压缩率：~4x（int4）/ ~8x（int2）
- 无 codebook / 无 centroids，纯标量量化 + 位打包
- 代码：`trq/packed_naive.py` (`packed_naive_quantize_tensor`)

### 3. PRQ / triton-nstages-kmeans（`triton-nstages-kmeans-int2/int4`）

**真压缩 + 向量量化。** 多阶段 K-Means 先抓粗粒度结构（向量级 centroids），残差再 blockwise INT 量化。

```
Stage 1:  K-Means(n_clusters) → centroid + cluster_id
          residual = x - centroid[cluster_id]
Stage 2:  K-Means 对 residual 再做
...
Stage N:  最终残差 → blockwise INT 量化 + 位打包

存储: {centroids (bf16), cluster_ids (uint8), packed_residual (uint8), scales}
```

- 用途：高质量真压缩主线
- 压缩率：最高（向量量化 + 标量量化叠加）
- centroids 是向量级的（如 d=128 维度空间内的聚类中心），比纯 blockwise 更能捕获结构
- VBench 实测 vs BF16：全 INT2 仅降 ~0.3%，R-HWQ-4h 降 ~1%
- 代码：`trq/real/prq.py` (`prq_quant`), `trq/functions.py` (`triton_prq_quantize_tensor`)
- 解压：`trq/real/accumulate.py` (`nstage_accum`)

### 4. TRQ（`trq-int2/int4/int8`）

**真压缩 + 时间预测残差量化。** 首个时间单元作为 anchor，后续单元由前一个重建单元预测，只量化预测残差。

```text
anchor_hat = Q_anchor(x_0)
prediction_t = predictor(x_hat_{t-1})
residual_t = x_t - prediction_t
x_hat_t = prediction_t + Q_residual(residual_t)
```

- v1 稳定 predictor：`identity`、`affine_channel`
- residual：按 block 的非对称 zero-point 量化，2/4 bit 位打包
- 状态格式：`format="trq"`, `version=1`，包含解码所需 predictor 参数
- `hrq-*` 和 `s2pp-*` 仅作为量化类型兼容别名，新状态一律写为 TRQ
- RoPE predictor 尚未定型，明确保留为实验项，稳定入口会报错
- 代码：`trq/real/trq.py`；兼容适配：`trq/real/hrq.py`、`trq/real/s2pp.py`

---

## 对比

| | Naive | Packed-naive | PRQ | TRQ |
|---|---|---|---|---|
| 真压缩? | 否 | 是 | 是 | 是 |
| 输出格式 | bf16 tensor | packed dict | packed dict | versioned packed dict |
| 压缩方式 | - | 标量 min-max | 向量 K-Means + 标量残差 | 时间预测 + 标量残差 |
| 压缩率 | 1x | ~4-8x | 高 | 高，取决于 anchor/metadata |
| VBench vs BF16 (全INT2) | ↓8.2% | 未单独测 | ↓0.3% | 待按 v1 复验 |
| VBench vs BF16 (R-HWQ-4h) | - | ↓0.1% (int8+int4) / ↓3.2% (int4+int2) | ↓1.1% | 待按 v1 复验 |
| 显存需求 | 同 BF16 (~78GB) | 显著降低 | 显著降低 | 显著降低，需统计完整状态 |
| 用途 | 精度调试 | 快速压缩 baseline | 向量量化主线 | 时间冗余研究主线 |

---

## 关键结论

- **Naive 不能省显存**：是纯数值模拟，VBench 退化也最大（~8%），说明仅做伪量化 + 不做压缩不是一个有竞争力的路径
- **PRQ 质量最好**：向量量化捕获了 attention head 内的结构，不是简单的逐元素量化，适合作为论文主力方法
- **Packed-naive 是折中**：压缩率接近 PRQ，但实现简单，没有 codebook 开销，适合作为 baseline 和消融对比
- **Packed-naive int8+int4 意外优秀**：R-HWQ-4h 配置下仅比 BF16 低 0.10%，几乎无损；4 个 int8 高精度 heads + 8 个 int4 低精度 heads 的组合在 block_size=64 下提供了接近 PRQ 的视频质量
- **Packed-naive int4+int2 中等退化**：R-HWQ-4h 配置下降 ↓3.19%，介于 naive（↓8.19%）和 PRQ（↓1.07%）之间，说明 2-bit 低精度组在 packed-naive 下仍会引入不可忽略的量化噪声
