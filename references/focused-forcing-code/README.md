
```bash
conda create -n forcing python=3.10 -y
conda activate forcing

python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 

python -m pip install -r requirements.txt
python -m pip install pip==23.0 && python -m pip install nvidia-pyindex && python -m pip install --upgrade pip && python -m pip install tensorrt-cu12 pycuda

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -m pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2/flashinfer_python-0.2.2+cu124torch2.6-cp38-abi3-linux_x86_64.whl#sha256=5e1cdb2fb7c0e9e9a2a2241becc52b771dc0093dd5f54e10f8bf612e46ef93a9
python -m pip install flashinfer_python-0.2.2+cu124torch2.6-cp38-abi3-linux_x86_64.whl
rm flashinfer_python-0.2.2+cu124torch2.6-cp38-abi3-linux_x86_64.whl

python -m pip install xfuser
conda install -c conda-forge ffmpeg
```