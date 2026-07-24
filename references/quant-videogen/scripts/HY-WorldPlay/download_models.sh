#!/bin/bash
BASE=https://huggingface.co/tencent/HY-WorldPlay/resolve/main

mkdir -p ckpts/HY-WorldPlay/wan_transformer ckpts/HY-WorldPlay/wan_distilled_model

wget -O ckpts/HY-WorldPlay/wan_transformer/config.json "$BASE/wan_transformer/config.json"
wget -O ckpts/HY-WorldPlay/wan_transformer/diffusion_pytorch_model.safetensors "$BASE/wan_transformer/diffusion_pytorch_model.safetensors"
wget -O ckpts/HY-WorldPlay/wan_distilled_model/model.pt "$BASE/wan_distilled_model/model.pt"
