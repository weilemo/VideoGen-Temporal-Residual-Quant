# Self-Forcing 接入 QVG 详解

这份文档聚焦 `experiments/Self-Forcing/` 目录，目标是把两件事讲清楚：

1. 这个实验目录里每一层代码分别负责什么。
2. 从 `scripts/Self-Forcing/run_qvg.sh` 开始，到一次 chunk 完成生成，中间经过哪些函数、哪些文件、张量 shape 如何变化。

本文默认你已经知道这个仓库的总体目标：给长视频生成中的 KV cache 做量化压缩，从而降低长时序推理的显存占用。

## 1. 目录分层

`experiments/Self-Forcing/` 可以按职责分成 7 层：

1. 入口层：`inference.py`
2. 配置层：`configs/`
3. 推理编排层：`pipeline/causal_inference.py`
4. 适配层：`utils/wan_wrapper.py`
5. 底层生成模型层：`wan/`
6. Self-Forcing 方法层：`model/`
7. 数据与辅助工具层：`utils/`、`demo_utils/`

它们之间的关系可以概括成：

`run_qvg.sh` -> `inference.py` -> `CausalInferencePipeline` -> `WanDiffusionWrapper` -> `CausalWanModel` -> 每层注意力读写 KV cache  
同时 `CausalInferencePipeline` 在合适时机调用 `quant_videogen` 对旧的 KV cache 做压缩。

## 2. 各部分作用

### 2.1 `inference.py`

文件：[experiments/Self-Forcing/inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/inference.py)

这是 Self-Forcing 实验的命令行推理入口，主要职责有 6 个：

1. 解析命令行参数。
2. 初始化单卡或分布式推理环境。
3. 读取并合并配置文件。
4. 将 QVG 的量化参数写入 `config.quant_config`。
5. 构建推理 pipeline。
6. 按 prompt 循环执行推理并保存视频。

这里最关键的 QVG 接入点是：

```python
config.quant_config = {
    "quant_type": args.quant_type,
    "cache_num_k_centroids": args.cache_num_k_centroids,
    "cache_num_v_centroids": args.cache_num_v_centroids,
    "kmeans_max_iters": args.kmeans_max_iters,
    "quant_block_size": args.quant_block_size,
    "num_prq_stages": args.num_prq_stages
}
```

这意味着后面所有量化行为都不再直接依赖 CLI，而统一依赖 `config.quant_config`。

### 2.2 `configs/`

目录：`experiments/Self-Forcing/configs/`

主要有 3 个文件：

- [default_config.yaml](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/configs/default_config.yaml)
- [self_forcing_dmd.yaml](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/configs/self_forcing_dmd.yaml)
- [self_forcing_sid.yaml](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/configs/self_forcing_sid.yaml)

作用区分如下：

- `default_config.yaml`
  提供默认值，比如是否 `causal`、默认视频分辨率、默认总帧数、`context_noise` 等。
- `self_forcing_dmd.yaml`
  DMD 版本实验配置。当前 `run_qvg.sh` 默认就使用它。
- `self_forcing_sid.yaml`
  SID 版本实验配置。

这层最关键的推理参数包括：

- `denoising_step_list`
- `num_frame_per_block`
- `model_kwargs.local_attn_size`
- `independent_first_frame`
- `context_noise`

其中 `num_frame_per_block: 3` 决定了 Self-Forcing 是按 3 帧一个块来滚动生成。

### 2.3 `pipeline/causal_inference.py`

文件：[experiments/Self-Forcing/pipeline/causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py)

这是 Self-Forcing 接入 QVG 的核心文件，主要负责：

1. 初始化文本编码器、Wan 生成器、VAE。
2. 构建 `kv_cache1` 和 `crossattn_cache`。
3. 以 block/chunk 为单位组织长视频生成。
4. 在生成过程中维护上下文缓存。
5. 定期把更旧的 KV cache 压缩成量化表示。

这个文件里最重要的成员变量有：

- `self.num_transformer_blocks = 30`
- `self.frame_seq_length = 1560`
- `self.num_frame_per_block`
- `self.local_attn_size`
- `self.quant_config`
- `self.kv_cache1`
- `self.crossattn_cache`

这里要特别理解两个概念：

- `frame_seq_length = 1560`
  表示 1 帧 latent 在底层 transformer 中对应 1560 个 token。
- `kv_cache1`
  是一个长度为 30 的列表，对应 30 个 transformer block；每个元素都是一层的 K/V cache 容器。

### 2.4 `utils/wan_wrapper.py`

文件：[experiments/Self-Forcing/utils/wan_wrapper.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/utils/wan_wrapper.py)

这是 Self-Forcing 和底层 Wan 模型之间的适配层，主要定义 3 个 wrapper：

- `WanTextEncoder`
- `WanVAEWrapper`
- `WanDiffusionWrapper`

职责分别是：

- `WanTextEncoder`
  把 prompt 文本变成 `prompt_embeds`。
- `WanVAEWrapper`
  负责 latent 和 pixel 之间的编码/解码。
- `WanDiffusionWrapper`
  负责把上层 pipeline 的输入整理成底层 `CausalWanModel` 能直接消费的形式。

这层的重要意义是屏蔽底层细节，让 `CausalInferencePipeline` 只需要关心：

- 输入 latent 是什么 shape
- 当前 timestep 是什么
- 当前 KV cache/cross-attn cache 是什么

### 2.5 `wan/`

目录：`experiments/Self-Forcing/wan/`

这是 Self-Forcing 依赖的底层 Wan 模型实现，真正执行扩散预测、注意力、RoPE、KV cache 读写。

和 QVG 接入最相关的文件是：

- [wan/modules/causal_model.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/wan/modules/causal_model.py)
- [wan/modules/model.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/wan/modules/model.py)
- `wan/modules/attention.py`

其中：

- `model.py`
  更接近原始 Wan 模型公共组件定义。
- `causal_model.py`
  是面向长视频递推生成的改造版，支持：
  - self-attention KV cache
  - cross-attention cache
  - local attention sliding window
  - 与 QVG 的 `ChunkedKVCache` 协同工作

### 2.6 `model/`

目录：`experiments/Self-Forcing/model/`

这一层更多是 Self-Forcing 方法本身的训练/算法层，不是 QVG 推理接入的主战场。

常见文件大致作用如下：

- `base.py`：公共基类
- `diffusion.py`：扩散训练/推理相关组织
- `dmd.py`：DMD 路线
- `sid.py`：SID 路线
- `gan.py`：GAN 相关逻辑
- `ode_regression.py`：ODE 回归相关逻辑
- `causvid.py`：与因果视频生成相关的上层逻辑

如果你的目标是理解 QVG 推理接入，可以先把这一层放在次优先级。

### 2.7 `utils/` 与 `demo_utils/`

- `utils/dataset.py`
  定义文本数据集、图文对数据集、LMDB latent 数据集。
- `utils/scheduler.py`
  调度器封装。
- `utils/misc.py`
  通用工具。
- `utils/distributed.py`
  分布式辅助。
- `demo_utils/memory.py`
  显存相关工具，比如动态 swap、剩余显存估算等。

这些模块更多是基础设施和运行辅助。

## 3. Self-Forcing 中 QVG 是怎么接进去的

QVG 在 Self-Forcing 中的接入重点只有两处：

1. `inference.py` 中注入 `config.quant_config`
2. `pipeline/causal_inference.py` 中维护 `ChunkedKVCache` 并定期调用 `compress_kv_cache()`

因此接入思路不是“直接改底层 attention 让它每次都量化”，而是：

1. 正常生成并缓存最近上下文。
2. 当上下文逐渐变长后，挑选更旧的一段历史 cache。
3. 把这段历史 cache 从 BF16 重新编码成量化表示。
4. 后续读 cache 时，如果读到量化段，就通过 `ChunkedKVCache.read()` 和 `uncompress_kv_cache()` 自动恢复成可用于注意力计算的张量。

这是一种“缓存管理层接入”，而不是“算子内部硬编码接入”。

## 4. 从 `run_qvg.sh` 到生成视频的总调用链

这一节先给出总览，再在下一节展开到“一次 chunk”的粒度。

### 4.1 Shell 层

入口脚本：[scripts/Self-Forcing/run_qvg.sh](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/scripts/Self-Forcing/run_qvg.sh)

它做三件事：

1. 定义实验输入
   - `prompts_path=assets/t2v.txt`
   - `ckpt_path=ckpts/Self-Forcing/self_forcing_dmd.pt`
   - `num_output_frames=180`
   - `local_attn_size=180`

2. 定义量化配置
   - `quant_type="triton-nstages-kmeans-int2"`
   - `cache_num_k_centroids=256`
   - `cache_num_v_centroids=256`
   - `kmeans_max_iters=2`
   - `quant_block_size=64`
   - `num_prq_stages=1`

3. 调 `torchrun ... experiments/Self-Forcing/inference.py`

### 4.2 Python 入口层

`inference.py` 的主线如下：

1. 解析 CLI 参数
2. 初始化 DDP / device
3. 读取并合并 YAML 配置
4. 覆盖 `local_attn_size`
5. 构建 `config.quant_config`
6. 创建 `CausalInferencePipeline`
7. 加载 Self-Forcing checkpoint 到 `pipeline.generator`
8. 构建 `TextDataset`
9. 为每个 prompt 构建噪声：
   - shape: `[B, T, C, H, W]`
   - 默认示例：`[1, 180, 16, 60, 104]`
10. 调用 `pipeline.inference(...)`
11. 保存视频

### 4.3 Pipeline 层

`CausalInferencePipeline.inference()` 的主线如下：

1. 文本编码
2. 初始化输出 latent 缓冲区
3. 初始化 `kv_cache1`
4. 初始化 `crossattn_cache`
5. 若有初始视频上下文，先缓存上下文
6. 按 chunk 循环生成
7. 在 chunk 之间定期量化旧 KV cache
8. 所有 latent 生成完后用 VAE 解码
9. 保存视频

### 4.4 底层模型层

当 pipeline 调 `self.generator(...)` 时，实际调用链为：

1. `WanDiffusionWrapper.forward()`
2. `CausalWanModel.forward()`
3. `CausalWanModel._forward_inference()`
4. 逐层执行 `CausalWanAttentionBlock.forward()`
5. 其中的 `self_attn.forward()`
6. 进入 `attn_kv_cache_prerope(...)`
7. 写入/读取 `ChunkedKVCache`
8. 执行注意力
9. 最后 head + unpatchify，返回当前 chunk 的 flow 预测

## 5. 一次 chunk 完成生成的逐函数调用链

这一节以 `run_qvg.sh` 的默认 T2V 推理为例，假设：

- batch size = 1
- `num_output_frames = 180`
- `num_frame_per_block = 3`
- `initial_latent = None`
- 单次 chunk 生成 3 帧

### 5.1 `run_qvg.sh`

文件：[scripts/Self-Forcing/run_qvg.sh](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/scripts/Self-Forcing/run_qvg.sh)

启动命令本质上是：

```bash
torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path assets/t2v.txt \
  --num_output_frames 180 \
  --local_attn_size 180 \
  --quant_type triton-nstages-kmeans-int2 \
  ...
```

### 5.2 `inference.py` 准备输入

核心路径：

1. 读取 prompt 文本
2. 构建 `TextDataset`
3. DataLoader 每次取一条 prompt
4. 生成随机噪声：

```python
sampled_noise = torch.randn(
    [args.num_samples, args.num_output_frames, 16, 60, 104],
    device=device,
    dtype=torch.bfloat16
)
```

因此输入给 pipeline 的 `noise` shape 是：

`[B, T, C, H, W] = [1, 180, 16, 60, 104]`

这里：

- `B = 1`
- `T = 180` latent frames
- `C = 16` latent channels
- `H = 60`
- `W = 104`

然后调用：

```python
video, latents = pipeline.inference(
    noise=sampled_noise,
    text_prompts=prompts,
    return_latents=True,
    initial_latent=None,
    low_memory=low_memory,
)
```

### 5.3 `CausalInferencePipeline.inference()` 开始

文件：[experiments/Self-Forcing/pipeline/causal_inference.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/pipeline/causal_inference.py)

第一步会解析输入 shape：

```python
batch_size, num_frames, num_channels, height, width = noise.shape
```

得到：

- `batch_size = 1`
- `num_frames = 180`
- `num_channels = 16`
- `height = 60`
- `width = 104`

由于当前配置：

- `independent_first_frame = false`
- `num_frame_per_block = 3`

所以：

```python
num_blocks = num_frames // num_frame_per_block = 180 // 3 = 60
```

也就是说，整个视频会分成 60 个 chunk 生成，每个 chunk 3 帧。

### 5.4 文本编码

调用：

```python
conditional_dict = self.text_encoder(text_prompts=text_prompts)
```

对应文件：[utils/wan_wrapper.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/utils/wan_wrapper.py)

`WanTextEncoder.forward()` 会：

1. tokenizer 编码文本
2. 喂给 UMT5 encoder
3. 返回：

```python
{
    "prompt_embeds": context
}
```

这里 `prompt_embeds` 的 shape 可以理解为：

`[B, L_text, C_text]`

其中：

- `B = 1`
- `L_text` 最长被 pad 到 512
- `C_text` 是文本编码维度，后续会在底层模型里再映射到 Wan 的 hidden dim

### 5.5 初始化输出 latent、KV cache、cross-attn cache

#### 输出 latent

```python
output = torch.zeros(
    [batch_size, num_output_frames, num_channels, height, width],
    device=noise.device,
    dtype=noise.dtype
)
```

shape：

`[1, 180, 16, 60, 104]`

#### KV cache

调用：

```python
self._initialize_kv_cache(
    batch_size=batch_size,
    dtype=noise.dtype,
    device=noise.device,
    target_frames=target_frames
)
```

每层 cache 的结构是：

```python
{
    "k": ChunkedKVCache(..., layout="BSHD"),
    "v": ChunkedKVCache(..., layout="BSHD"),
    "global_end_index": tensor([0]),
    "local_end_index": tensor([0])
}
```

共有 30 层。

`ChunkedKVCache` 的单 chunk 单位就是 1 帧，因此：

- 每帧 token 数 = `frame_seq_length = 1560`
- 每层 head 数 = 12
- 每个 head dim = 128

因此它存的单帧 K/V 数据 shape 是：

`[B, S_frame, H, D] = [1, 1560, 12, 128]`

注意这里使用的是 `BSHD` 布局，而 QVG 压缩函数内部常用的是 `BHSD`，所以在量化前后会看到 permute。

#### Cross-attn cache

每层 cross-attn cache 初始化为：

- `k`: `[B, 512, 12, 128]`
- `v`: `[B, 512, 12, 128]`

文本长度固定 pad 到 512。

### 5.6 进入 chunk 循环

由于没有 `initial_latent`，因此：

```python
all_num_frames = [3] * 60
```

对于第一个 chunk：

- `chunk_index = 0`
- `current_num_frames = 3`
- `current_start_frame = 0`

构造当前 chunk 的噪声输入：

```python
noisy_input = noise[:, 0:3]
```

shape：

`[1, 3, 16, 60, 104]`

这是第一次 chunk 进入 denoising 的输入。

### 5.7 chunk 内部的 denoising step 循环

默认 `self_forcing_dmd.yaml` 中：

```yaml
denoising_step_list:
- 1000
- 750
- 500
- 250
warp_denoising_step: true
```

因此一个 chunk 会走多次时刻的迭代。每次都会调用一次：

```python
self.generator(
    noisy_image_or_video=noisy_input,
    conditional_dict=conditional_dict,
    timestep=timestep,
    kv_cache=self.kv_cache1,
    crossattn_cache=self.crossattn_cache,
    current_start=current_start_frame * self.frame_seq_length
)
```

对第一个 chunk 来说：

`current_start = 0 * 1560 = 0`

对于第 `n` 个 chunk，如果前面已经生成了 `3n` 帧，则：

`current_start = 3n * 1560`

### 5.8 `WanDiffusionWrapper.forward()`

文件：[utils/wan_wrapper.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/utils/wan_wrapper.py)

这个函数接收到的输入 shape 是：

- `noisy_image_or_video`: `[1, 3, 16, 60, 104]`
- `timestep`: `[1, 3]`
- `prompt_embeds`: `[1, L_text, C_text]`

它会先做一个重要维度变换：

```python
noisy_image_or_video.permute(0, 2, 1, 3, 4)
```

于是 shape 从：

`[B, T, C, H, W] = [1, 3, 16, 60, 104]`

变成：

`[B, C, T, H, W] = [1, 16, 3, 60, 104]`

这是底层 `CausalWanModel` 期望的输入布局。

然后调用：

```python
self.model(
    noisy_image_or_video.permute(0, 2, 1, 3, 4),
    t=input_timestep,
    context=prompt_embeds,
    seq_len=self.seq_len,
    kv_cache=kv_cache,
    crossattn_cache=crossattn_cache,
    current_start=current_start,
    cache_start=cache_start
)
```

底层返回的是 flow prediction，shape 仍是 `[B, C, T, H, W]`，然后再 permute 回：

`[B, T, C, H, W]`

最终 `WanDiffusionWrapper.forward()` 返回：

- `flow_pred`
- `pred_x0`

shape 都是：

`[1, 3, 16, 60, 104]`

### 5.9 `CausalWanModel._forward_inference()`

文件：[wan/modules/causal_model.py](/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/experiments/Self-Forcing/wan/modules/causal_model.py)

这是底层真正执行当前 chunk 的地方。

输入时 `x` 是一个 batch 内视频列表，每个样本 shape：

`[C, F, H, W] = [16, 3, 60, 104]`

#### 第一步：patch embedding

```python
x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
```

接着：

```python
grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
x = [u.flatten(2).transpose(1, 2) for u in x]
seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
x = torch.cat(x)
```

这里得到：

- `grid_sizes`: `[B, 3]`
- 对当前 3 帧 chunk，通常是 `[3, 30, 52]`
- 因为 `30 * 52 = 1560`，所以：
  - 每帧 token 数 = 1560
  - 3 帧总 token 数 = 4680

于是：

- `seq_lens = [4680]`
- `x` 的 shape 变成 `[1, 4680, dim]`

其中 `dim` 是 Wan transformer hidden size，当前 1.3B 配置下通常是 1536 或 2048 量级，具体由模型配置决定；对理解缓存流程来说，最重要的是第二维 `4680`。

#### 第二步：time embedding

模型根据当前 timestep 做时间嵌入，得到 `e`，后面逐层调制各个 attention / FFN block。

#### 第三步：text embedding

`prompt_embeds` 会被映射到 Wan 内部文本上下文表示，形成：

`context: [B, text_len, dim]`

其中 `text_len` 会 pad 到 512。

#### 第四步：逐层 transformer block

模型遍历所有 block：

```python
for block_index, block in enumerate(self.blocks):
    x = block(x, **kwargs)
```

此时：

- `x`: `[1, 4680, dim]`
- `kv_cache[block_index]`: 当前层缓存
- `crossattn_cache[block_index]`: 当前层文本交叉注意力缓存
- `current_start = 0` 对于第一个 chunk

### 5.10 `CausalWanAttentionBlock.forward()`

这一层接收到：

- `x`: `[B, L, C] = [1, 4680, dim]`
- `e`: `[B, F, 6, C] = [1, 3, 6, dim]`

它先把 `x` 重新看成按帧组织：

```python
num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
```

于是：

- `num_frames = 3`
- `frame_seqlen = 4680 // 3 = 1560`

然后进入 self-attention：

```python
self.self_attn(...)
```

### 5.11 `CausalWanSelfAttention.forward()`

这里先做 Q/K/V 投影：

```python
q = self.norm_q(self.q(x)).view(b, s, n, d)
k = self.norm_k(self.k(x)).view(b, s, n, d)
v = self.v(x).view(b, s, n, d)
```

当前 shape：

- `b = 1`
- `s = 4680`
- `n = 12`
- `d = 128`

所以：

- `q`: `[1, 4680, 12, 128]`
- `k`: `[1, 4680, 12, 128]`
- `v`: `[1, 4680, 12, 128]`

如果带有缓存，则进入：

```python
x, kv_cache = self.attn_kv_cache_prerope(...)
```

### 5.12 `attn_kv_cache_prerope()`：把当前 chunk 接入缓存

这是 Self-Forcing QVG 接入最关键的底层节点之一。

逻辑如下：

1. 根据 `grid_sizes` 计算单帧 token 数：

```python
frame_seqlen = math.prod(grid_sizes[0][1:]).item()
```

得到：

`30 * 52 = 1560`

2. 对 query 施加按帧偏移的因果 RoPE：

```python
roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
```

shape 仍然是：

`[1, 4680, 12, 128]`

3. 计算当前 chunk 对应的 token 区间：

- `current_start`
- `current_end = current_start + roped_query.shape[1]`

对第一个 chunk：

- `current_start = 0`
- `current_end = 4680`

4. 将当前 chunk 的 K/V 写入 `ChunkedKVCache`

```python
kv_cache["k"].write(local_start_index, local_end_index, k)
kv_cache["v"].write(local_start_index, local_end_index, v)
```

注意这里写入的是：

- `k`: `[1, 4680, 12, 128]`
- `v`: `[1, 4680, 12, 128]`

而 `ChunkedKVCache` 的布局是 `BSHD`，正好匹配。

5. 读出从 0 到 `local_end_index` 的全部历史缓存

```python
k_all = kv_cache["k"].read(0, local_end_index)
v_all = kv_cache["v"].read(0, local_end_index)
```

对第一个 chunk：

- `k_all`: `[1, 4680, 12, 128]`
- `v_all`: `[1, 4680, 12, 128]`

对后续 chunk，它们会越来越长，直到达到 local window 上限。

6. 对读出的整段历史 K 应用长输入 RoPE：

```python
k_input = causal_rope_apply_long_input(k_all, grid_sizes, freqs)
```

shape 不变：

`[1, total_tokens_so_far, 12, 128]`

7. 做注意力：

```python
x = attention(roped_query, k_input, v_all)
```

输出 shape：

`[1, 4680, 12, 128]`

8. 回到上层：

```python
x = x.flatten(2)
x = self.o(x)
```

shape 重新回到：

`[1, 4680, dim]`

### 5.13 block 结束，整层 transformer 执行完

每一层都会这样更新自己的 `kv_cache[layer_idx]`。

因此在一次 chunk 内，30 层 transformer 会分别把这一 chunk 的 K/V 写进各自那层缓存。

此时缓存中的索引语义是：

- `global_end_index`
  当前视频在全局时间上的 token 终点
- `local_end_index`
  当前实际存储在本层缓存中的 token 终点

对于第一个 chunk 结束后，大致会是：

- `global_end_index = 4680`
- `local_end_index = 4680`

### 5.14 底层模型输出回到 `WanDiffusionWrapper`

`CausalWanModel._forward_inference()` 最后：

1. 经过 `head`
2. `unpatchify`
3. 返回 `[B, C, F, H, W]`

对当前 chunk：

`[1, 16, 3, 60, 104]`

回到 `WanDiffusionWrapper.forward()` 后又 permute 回：

`[1, 3, 16, 60, 104]`

然后根据 flow matching 关系计算：

- `flow_pred`
- `pred_x0`

shape 都是：

`[1, 3, 16, 60, 104]`

### 5.15 当前 denoising step 结束，开始下一个 timestep

在 `CausalInferencePipeline.inference()` 中：

- 如果当前还不是最后一个 denoising step
  - 用 `pred_x0` 作为较干净的估计
  - 调 scheduler 重新加噪
  - 进入下一 timestep
- 如果已经是最后一个 denoising step
  - 当前 chunk 的最终 latent 就是 `denoised_pred`

因此，一个 chunk 会经历多次：

`noisy_input -> generator -> pred_x0 -> scheduler add_noise -> noisy_input`

直到最后一次输出真正的 chunk latent。

### 5.16 当前 chunk 的最终结果落盘到输出 latent

最后一个 denoising step 完成后：

```python
output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
```

对于第一个 chunk：

```python
output[:, 0:3] = denoised_pred
```

所以 `output` 中前 3 帧被填满。

### 5.17 用“干净上下文”再次跑一遍以更新缓存

这一点非常关键，也是 Self-Forcing 设计的重要特征：

```python
self.generator(
    noisy_image_or_video=denoised_pred,
    conditional_dict=conditional_dict,
    timestep=context_timestep,
    kv_cache=self.kv_cache1,
    crossattn_cache=self.crossattn_cache,
    current_start=current_start_frame * self.frame_seq_length,
)
```

这里不是为了再生成一份输出，而是为了让缓存中存的是“当前 chunk 的 clean context 表征”，而不是某个中间噪声时刻的状态。

也就是说，一个 chunk 完整完成的定义其实是：

1. 用多个 denoising step 把 chunk latent 生成出来
2. 将最终 `denoised_pred` 写入总输出
3. 再以 context 模式走一遍模型，更新各层缓存

只有这三步都做完，这个 chunk 才真正成为后续 chunk 的历史上下文。

## 6. QVG 量化在 chunk 之间何时触发

量化不发生在 chunk 内部的每个 timestep，而发生在 chunk 之间。

在 `CausalInferencePipeline.inference()` 里：

```python
QUANT_FACTOR = 8
if chunk_index < QUANT_FACTOR or chunk_index % QUANT_FACTOR != 0:
    pass
else:
    ...
    self.quantize_kv_cache(tokens_to_quantize_start, tokens_to_quantize_end, max_tokens_to_quantize)
```

意思是：

- 前 8 个 chunk 不量化
- 之后每隔 8 个 chunk 触发一次
- 量化的是更旧的一段历史 token 区间，而不是最新 chunk

这样做的动机是：

1. 最新上下文对质量更敏感，先保留 BF16
2. 更旧的上下文对误差容忍度更高，更适合压缩
3. 不在最内层 attention 路径中加入量化逻辑，工程侵入性更低

## 7. `ChunkedKVCache` 在 Self-Forcing 里的实际角色

虽然 `ChunkedKVCache` 定义在 `quant_videogen/kv_cache.py`，但它在 Self-Forcing 中扮演的是核心状态容器，而不是普通工具类。

在当前接入中，它承担 4 个角色：

1. 以 frame 为单位分块存储 K/V
2. 允许某段历史仍是 BF16，而另一段已经量化
3. 在读取缓存时自动恢复 full-precision tensor
4. 支持 CPU/GPU offload

因此它是 Self-Forcing 长视频推理和 QVG 量化之间的关键桥梁。

## 8. 一句话总结

Self-Forcing 的这套接入并不是把量化硬塞进某个 attention kernel，而是把量化放在“长视频递推生成的缓存管理层”上：

- `inference.py` 负责把量化参数送进配置
- `CausalInferencePipeline` 负责决定何时、量化哪段历史 cache
- `ChunkedKVCache` 负责以可分块、可混合精度的形式保存历史 K/V
- `CausalWanModel` 负责在后续 chunk 中真正消费这些缓存

所以从工程结构上看，Self-Forcing + QVG 的核心思想是：

“把长视频生成看成一个持续增长的上下文缓存系统，然后在缓存系统层面对旧上下文做压缩，而不是在生成算子里直接做强侵入改造。”
