## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Install CUDA 12.4 Toolkit.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

Set the environment variables in ```.bashrc``` file.
```bash
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export XDG_CACHE_HOME="/path/to/.cache"
export VBENCH_CACHE_DIR="/path/to/.cache/vbench"
export TMPDIR="/path/to/.cache/tmp"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="YOUR_HF_TOKEN"
```

Create a conda environment.
```bash
conda config --prepend envs_dirs /path/to/miniconda3/envs
conda config --prepend pkgs_dirs /path/to/miniconda3/pkgs

conda create -n forcing python=3.10 -y
conda activate forcing

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://pypi.org/simple https://pypi.ngc.nvidia.com"
pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.org pypi.ngc.nvidia.com"

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install pip==23.0 && pip install nvidia-pyindex && pip install --upgrade pip && pip install tensorrt-cu12 pycuda
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python setup.py develop
```

Download checkpoints
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing
```

## Quick Start
### Change model path
```python
# Wan-AI/Wan2.1-T2V-1.3B
# demo.py line 103
vae_state_dict = torch.load('/path/to/Wan2.1_VAE.pth', map_location="cpu")
# demo_utils/vae_torch2trt.py line 44
vae_state_dict = torch.load('/path/to/Wan2.1_VAE.pth', map_location="cpu")
# utils/wan_wrapper.py line 25, 30, 69, 128 and 130
torch.load("/path/to/models_t5_umt5-xxl-enc-bf16.pth",
name="/path/to/google/umt5-xxl/", seq_len=512, clean='whitespace')
pretrained_path="/path/to/Wan2.1_VAE.pth",
"/path/to", local_attn_size=local_attn_size, sink_size=sink_size)
self.model = WanModel.from_pretrained("/path/to")

# gdhe17/Self-Forcing
# demo.py line 35
parser.add_argument("--checkpoint_path", type=str, default='/path/to/checkpoints/self_forcing_dmd.pt')
```

### GUI demo
```bash
python demo.py
```
Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.
* You may want to adjust FPS so it plays smoothly on your device.
* The speed can be improved by enabling `torch.compile`, [TAEHV-VAE](https://github.com/madebyollin/taehv/), or using FP8 Linear layers, although the latter two options may sacrifice quality. It is recommended to use `torch.compile` if possible and enable TAEHV-VAE if further speedup is needed.

### CLI Inference
Example inference script using the chunk-wise autoregressive checkpoint trained with DMD:
```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path /path/to/checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench/MovieGenVideoBench_extended.txt \
    --use_ema
```
Other config files and corresponding checkpoints can be found in [configs](configs) folder and our [huggingface repo](https://huggingface.co/gdhe17/Self-Forcing/tree/main/checkpoints).
