# 基于 QVG BF16 / Self-Forcing 代码推进 Head-Wise Quant 的可行性说明

这份文档用于回答一个具体工程问题：

`Self-Forcing + QVG` 当前这套 BF16 / KV cache 量化代码，是否适合作为后续 `head-wise quant` 的主代码框架？

结论先写在前面：

- 适合。
- 不需要重写整套量化接入。
- 最合理的路线是基于现有 `Self-Forcing + QVG` 代码，先做 `head-group mixed precision`，再视实验结果推进到更细粒度的 `per-head` 策略。

## 1. 结论

当前 `QVG` 在 `Self-Forcing` 上的接入方式，已经具备推进 `head-wise quant` 所需的核心工程条件：

1. `BF16` baseline 已经跑通，说明生成链路本身稳定可用。
2. 量化入口集中在 KV cache 管理层，而不是深埋在 attention kernel 内部。
3. KV 张量天然保留 `head` 维度，现有聚类 / PRQ 路径本质上已经按 head 独立处理。

因此，后续主线应当直接基于这套代码做增量改造，而不是另起一套量化框架。

## 2. 已确认的工程基础

### 2.1 BF16 baseline 已完成有效输出

`Self-Forcing` 的 BF16 baseline 已成功运行并生成视频，结果位于共享盘：

- `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/results/selfforcing/bf16/0-0_ema.mp4`
- `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/results/selfforcing/bf16/1-0_ema.mp4`

对应日志：

- `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/slurm_logs/qvg_sf_bf16_g-24046.out`

这说明当前 `Self-Forcing -> causal inference -> Wan -> KV cache` 的整条推理链路已经可作为后续 `head-wise quant` 的稳定起点。

### 2.2 BF16 启动脚本就是后续主入口的无量化基线

文件：[scripts/Self-Forcing/run_bf16.sh](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/scripts/Self-Forcing/run_bf16.sh)

这条脚本通过：

```bash
--quant_type none
```

运行同一套 `Self-Forcing` 推理代码，因此它不仅是 baseline，也是后续 `head-wise quant` 改造前后的直接对照线。

## 3. 为什么这套框架适合做 Head-Wise Quant

### 3.1 量化入口集中在 cache 管理层

`Self-Forcing` 中最关键的量化入口在：

- [experiments/Self-Forcing/pipeline/causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py)

尤其是 `quantize_kv_cache()` 这一段：

- 读取历史 KV cache
- 整理成 `[B, H, S, D]`
- 调用 `compress_kv_cache()`
- 再把压缩结果写回 cache

对应关键位置：

- [causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py:62)
- [causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py:98)
- [causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py:118)

这意味着后续如果要做 `head-wise quant`，主改动点会集中在这层，而不是去重写底层 attention 计算。

### 3.2 KV cache 的张量组织天然支持按 head 改造

当前 KV cache 初始化时，已经把 head 维度作为一等结构保存：

- [causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py:478)
- [causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py:480)

底层容器 `ChunkedKVCache` 也显式保存：

- `num_heads`
- `head_dim`
- layout 为 `BHSD` 或 `BSHD`

对应文件：

- [quant_videogen/kv_cache.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/kv_cache.py:37)

因此，从数据结构上看，这套实现本来就是围绕 `[B, H, S, D]` / `[B, S, H, D]` 组织的，适合按 head 或 head-group 拆分策略。

### 3.3 现有 KMeans / PRQ 实现本质上已经是逐 head 聚类

现有 KMeans 路径会把输入从 `[B, H, S, D]` reshape 成 `[B*H, S, D]` 后做聚类：

- [quant_videogen/functions.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/functions.py:86)
- [quant_videogen/functions.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/functions.py:114)

真实 PRQ 也是同样的处理方式：

- [quant_videogen/real/prq.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/real/prq.py:10)
- [quant_videogen/real/prq.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/real/prq.py:52)

这说明现有实现虽然对外暴露的是全局统一配置，但在张量处理语义上，其实已经具备“每个 head 独立建模”的基础。

换句话说：

- 现在缺的不是 `head-wise` 的数据路径。
- 现在缺的是 `head-wise` 的配置路径和元数据路径。

## 4. 当前不支持真正 Head-Wise Mixed Policy 的约束

### 4.1 `quant_config` 目前只有一份全局配置

`inference.py` 当前会把量化参数统一塞进 `config.quant_config`：

- [experiments/Self-Forcing/inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/inference.py:75)

其中配置项包括：

- `quant_type`
- `cache_num_k_centroids`
- `cache_num_v_centroids`
- `kmeans_max_iters`
- `quant_block_size`
- `num_prq_stages`

这套结构默认所有 head 共用一组参数。

### 4.2 `compress_kv_cache()` 默认对所有 head 使用同一种量化策略

文件：

- [quant_videogen/compress.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/compress.py:171)

当前接口签名本质上是：

```python
compress_kv_cache(k, v, quant_type, quant_config, quantize_fn)
```

这里没有任何 `per-head policy` 或 `group mapping` 的表达能力，因此默认所有 head 共享：

- bitwidth
- centroids 数量
- PRQ stage 数量
- block size

### 4.3 反量化路径默认所有 head 共用一个 `quant_type`

文件：

- [quant_videogen/uncompress.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/uncompress.py:18)

当前 `num_bits` 是通过解析单个 `quant_config.quant_type` 得到的。也就是说：

- 如果压缩结果里混有 `int2` 和 `int4`
- 现有 `uncompress_kv_cache()` 没有能力知道每个 head 应该按哪种 bitwidth 去还原

这是当前实现不支持 mixed head policy 的关键硬约束之一。

### 4.4 `Self-Forcing` 侧 head 数和 head dim 仍有硬编码

当前 KV cache 初始化里直接写死：

- `num_heads = 12`
- `head_dim = 128`

位置在：

- [causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py:480)

对于当前 Wan 1.3B 配置这没问题，但如果后续要迁移到其他模型，最好把这两个值从模型本身读取，而不是继续硬编码。

## 5. 最合理的第一版实现路线

### 5.1 不建议一上来就做 12 个 head 完全独立

第一版最适合做的是：

- `head-group mixed precision`

而不是：

- 12 个 head 各自独立配置

原因很简单：

1. 配置更简单。
2. 压缩结果元数据更容易定义。
3. 调试和回归验证成本更低。
4. 足够验证“不同 head 的量化敏感性不同”这个核心研究假设。

### 5.2 推荐的第一版策略

先做两组：

- sensitive heads: `BF16` 或 `INT4`
- other heads: `INT2`

这样有两个好处：

1. 研究上可解释。
2. 工程上可以先验证“少量 head 抬精度是否能显著保质量”。

### 5.3 推荐的改造顺序

第一阶段：

1. 把 `quant_config` 扩成能表达 `per_head_policy` 或 `head_group_policy`。
2. 在 `quantize_kv_cache()` 中按 head-group 拆分 KV。
3. 对不同 group 分别调用压缩逻辑。
4. 在压缩产物里保存 group 配置与 group 到 head 的映射。

第二阶段：

1. 改 `uncompress_kv_cache()`，支持按 group 独立反量化。
2. 把各组结果重新拼回完整 `[B, H, S, D]`。
3. 确保 `ChunkedKVCache.read()` 读取量化 span 时仍能透明返回全精度张量。

第三阶段：

1. 统一 BF16、现有 QVG INT2、新的 head-wise 结果对比。
2. 观察质量损伤主要落在：
   - identity consistency
   - scene consistency
   - motion continuity
3. 再决定是否继续把策略细化到真正的 `per-head`。

## 6. 建议优先改的文件

后续实现优先从下面几个文件入手：

1. [experiments/Self-Forcing/pipeline/causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py)
2. [quant_videogen/compress.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/compress.py)
3. [quant_videogen/uncompress.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/uncompress.py)
4. [quant_videogen/kv_cache.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/quant_videogen/kv_cache.py)

必要时再补：

5. [experiments/Self-Forcing/inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/inference.py)

## 7. 当前项目层面的建议

基于现有事实，项目主线应当明确切换为：

1. 以 `Self-Forcing + QVG` 为主要实验底座。
2. 以 `BF16 baseline` 为质量参考线。
3. 以现有 `QVG INT2` 为统一量化基线。
4. 以新的 `head-wise quant` 为下一阶段主要研发方向。

这条主线比继续横向铺更多 benchmark 更有效，因为它直接服务当前最核心的问题：

- 视频长时生成中，不同 attention heads 的历史 KV 是否具有不同的量化敏感性。

## 8. 一句话版本

`QVG BF16 / Self-Forcing` 当前这套代码不是只能“继续复现”的临时脚手架，而是已经足够作为后续 `head-wise quant` 的主代码框架；真正需要补的不是底层生成链路，而是面向 `per-head / head-group` 的配置、压缩元数据和反量化路径。
