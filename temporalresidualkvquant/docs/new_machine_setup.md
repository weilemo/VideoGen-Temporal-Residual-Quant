# 新机器初始化

适用于 Linux + NVIDIA GPU。命令默认在用户目录安装 Miniconda，不需要 `sudo`。

## 1. 安装 Miniconda

```bash
cd ~
curl -fsSL -o miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

接受 Anaconda channel 条款并创建 Python 3.10 环境：

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n videoquant python=3.10 pip git -y
conda activate videoquant
```

新终端若找不到 `conda`，先执行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate videoquant
```

## 2. Clone 并安装

```bash
cd ~
git clone https://github.com/weilemo/VideoGen-Temporal-Residual-Quant.git
cd VideoGen-Temporal-Residual-Quant

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e './temporalresidualkvquant[gpu,analysis]'
python -m pip install \
  omegaconf tqdm einops imageio imageio-ffmpeg pillow \
  opencv-python-headless transformers diffusers accelerate safetensors \
  sentencepiece easydict ftfy regex scikit-image lpips lmdb
```

仓库为私有且 HTTPS 无法认证时，配置 GitHub SSH key 后改用：

```bash
git clone git@github.com:weilemo/VideoGen-Temporal-Residual-Quant.git
```

## 3. 验证

```bash
cd ~/VideoGen-Temporal-Residual-Quant/temporalresidualkvquant

python -c "import torch, triton, trq; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

必须确认 `torch.cuda.is_available()` 输出 `True`。`nvcc` 不是常规推理的前置条件；
只有编译自定义 CUDA extension 报错时再安装 CUDA Toolkit。

## 4. 配置权重

Git 不包含模型权重。推荐通过共享盘设置：

```bash
export SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing
```

目录应至少包含：

```text
Self-Forcing/
├── self_forcing_dmd.pt
└── Wan2.1-T2V-1.3B/
```

完成后参考 [上手与实验指南](getting_started.md) 运行 BF16 smoke test。
