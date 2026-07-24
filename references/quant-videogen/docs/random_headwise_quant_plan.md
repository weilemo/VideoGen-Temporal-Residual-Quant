# Random Head-Wise Quantization 实验方案

这份文档定义 `head-wise quant` 的第一版实验方案。

目标不是一开始就找出“最重要的 head”，而是先回答一个更基础的问题：

- 在不分析 head 重要性的前提下，随机混合精度是否已经比全 `INT2` 更稳。
- 当前 `Self-Forcing + QVG` 框架能否稳定支持 `head-wise` 混合量化。

本文把这个第一版 baseline 记作：

- `R-HWQ`
- 全称：`Random Head-Wise Quantization`

## 1. 实验目标

当前我们已经有两条清晰的参考线：

1. `BF16` baseline
2. `QVG INT2-all` baseline

第一版 `R-HWQ` 的目标是插入第三条线：

3. `Random mixed precision head-wise quant`

核心问题是：

- 如果固定总量化预算，只把随机选中的少量 head 提升到更高精度，其余 head 保持低精度，结果是否会优于全 `INT2`。

如果答案是肯定的，说明：

- `head-wise quant` 方向在工程上值得继续推进。
- 后续再做 head importance 分析是有意义的。

## 2. 第一版策略

### 2.1 不做 importance 分析

第一版不做：

- attention score 统计
- gradient / saliency 分析
- loss 增量排序
- quality 驱动的 head ranking

原因：

1. 先验证框架是否支持 mixed head policy。
2. 先验证“少量高精度 head”这个思路本身是否有效。
3. 降低第一阶段实现复杂度。

### 2.2 采用两组混合精度

第一版建议只做两组：

1. `high-precision group`
2. `low-precision group`

推荐默认设置：

- high-precision group: `INT4`
- low-precision group: `INT2`

暂时不建议第一版直接让部分 head 保持 `BF16`，因为：

1. 显存预算会上升更明显。
2. 不利于和现有 `INT2-all` 量化线做公平比较。
3. `INT4 + INT2` 更接近“纯量化策略改进”的问题设定。

## 3. 随机分组策略

### 3.1 第一版推荐：全层共享同一组随机 head

在 Wan 1.3B 当前配置下：

- `num_heads = 12`

第一版推荐做法：

- 用一个随机 seed 采样出若干 `head_ids`
- 这组 `head_ids` 在全部 transformer layers 中保持一致

例如：

- `seed = 0`
- `high_precision_heads = [1, 4, 9, 11]`
- 其余 heads 用 `INT2`

这样做的优点：

1. 实现简单。
2. 实验变量更少。
3. 结果更容易解释。
4. 后面和“importance-based head selection”对比时更干净。

### 3.2 不推荐第一版每层单独随机

每层单独随机虽然更自由，但会带来三个问题：

1. 配置和元数据复杂度明显提高。
2. 不同 seed 之间波动更难解释。
3. 一旦效果变化，很难判断是层间差异还是 head 差异。

因此第一版不建议做 per-layer random allocation。

## 4. 第一轮推荐实验矩阵

第一轮实验不要铺太大，先跑最小矩阵：

1. `BF16`
2. `INT2-all`
3. `R-HWQ-2h`
4. `R-HWQ-4h`

其中：

- `R-HWQ-2h`：随机选 `2 / 12` 个 head 用 `INT4`，其余 `10 / 12` 个 head 用 `INT2`
- `R-HWQ-4h`：随机选 `4 / 12` 个 head 用 `INT4`，其余 `8 / 12` 个 head 用 `INT2`

### 4.1 随机 seed

每种配置至少跑 `3` 个随机 seed：

- `seed = 0`
- `seed = 1`
- `seed = 2`

目的不是追求统计显著性，而是先看随机分配是否有稳定趋势。

## 5. 评测重点

第一轮优先做定性对比，重点看三类质量现象：

1. `identity consistency`
2. `scene consistency`
3. `motion continuity`

统一对比同一批 prompt 下的：

- `BF16`
- `INT2-all`
- `R-HWQ`

第一轮最值得观察的问题：

1. `R-HWQ` 是否比 `INT2-all` 更少闪烁。
2. 主体身份是否更稳定。
3. 场景结构是否更不容易漂。
4. 长时运动是否更连续。

如果有余力，再补：

- 显存
- 量化耗时
- 总生成时间

## 6. 代码实现方案

### 6.1 第一版实现原则

第一版实现只做一件事：

- 支持两组 head 的随机 mixed precision

不在第一版做：

- importance-based head ranking
- 多于两组的复杂策略
- per-layer 独立随机
- BF16 / INT4 / INT2 三种以上精度同时混合

### 6.2 推荐新增的配置项

在 `Self-Forcing` 推理入口上增加以下配置：

- `headwise_mode`
- `headwise_seed`
- `num_high_precision_heads`
- `high_precision_quant_type`
- `low_precision_quant_type`

推荐第一版默认值：

```text
headwise_mode=random
headwise_seed=0
num_high_precision_heads=4
high_precision_quant_type=triton-nstages-kmeans-int4
low_precision_quant_type=triton-nstages-kmeans-int2
```

### 6.3 推荐实现步骤

第一步，生成随机分组：

- 在 `CausalInferencePipeline` 初始化时，根据 `num_heads` 和 `headwise_seed` 生成一次 `head_group_map`
- 例如：
  - `high_precision_heads = [1, 4, 9, 11]`
  - `low_precision_heads = [0, 2, 3, 5, 6, 7, 8, 10]`

第二步，改造 `quantize_kv_cache()`：

- 读取原始 `k, v`，shape 为 `[B, H, S, D]`
- 按 `head_group_map` 切分成两块
- 对每块分别调用量化路径
- 把量化后的 payload 和 head id 映射一起写回 cache

第三步，改造反量化路径：

- 在 `uncompress_kv_cache()` 中按 group 分别解压
- 再把各组 head 放回完整 `[B, H, S, D]`

第四步，保证 `ChunkedKVCache.read()` 仍然对上层透明：

- 上层 attention 仍然只看到完整重建后的 full-precision tensor
- mixed policy 的复杂度尽量封装在压缩 / 解压层

## 7. 推荐的量化 span 存储格式

当前量化实现默认一个 span 只有一个 `quant_type`。

为了支持 `R-HWQ`，建议把量化 span 改成按 group 存储：

```python
{
  "groups": [
    {
      "head_ids": [1, 4, 9, 11],
      "quant_type": "triton-nstages-kmeans-int4",
      "payload": ...
    },
    {
      "head_ids": [0, 2, 3, 5, 6, 7, 8, 10],
      "quant_type": "triton-nstages-kmeans-int2",
      "payload": ...
    }
  ],
  "info": {
    "output_dtype": ...,
    "group_mode": "random",
    "seed": 0
  }
}
```

这种设计的优点是：

1. 兼容后续 importance-based grouping。
2. 不需要把 mixed policy 写死在 `quant_type` 字符串里。
3. 后面从 random baseline 过渡到真正的 `HWQ` 策略时，不用再改存储协议。

## 8. 预期结果与判断标准

第一轮对 `R-HWQ` 的合理预期不是“接近 BF16”，而是：

1. 比 `INT2-all` 更稳一些。
2. 不同随机 seed 之间会有波动。
3. 如果波动明显，反而说明不同 heads 的量化敏感性不均匀。

### 8.1 如果 `R-HWQ` 优于 `INT2-all`

说明：

- mixed precision head-wise 方向可行
- 后续值得做 head importance 分析

### 8.2 如果 `R-HWQ` 没明显优于 `INT2-all`

优先考虑两种解释：

1. 高精度 head 数量太少
2. 随机选中的 head 没覆盖真正敏感的 head

这时不应该直接否定方向，而应进入下一阶段：

- importance-based allocation

## 9. 实施顺序建议

建议按下面顺序推进：

1. 先实现 `R-HWQ-4h`
2. 先跑 `seed=0`
3. 和 `BF16`、`INT2-all` 做定性对比
4. 如果结果有希望，再补 `seed=1,2`
5. 最后再决定是否进入 head importance 分析

## 10. 一句话结论

`Random mixed precision` 是当前 `head-wise quant` 方向最合适的第一版 sanity-check baseline：它不要求先知道哪些 head 重要，但能快速验证 `Self-Forcing + QVG` 这套框架是否值得继续向真正的 `HWQ` 推进。
