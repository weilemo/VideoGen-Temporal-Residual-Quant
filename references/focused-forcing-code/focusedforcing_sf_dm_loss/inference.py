import argparse
import torch
import os
import glob
import json
import math
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from wan.modules.dmd_loss.dmd import DMD
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml", help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, default="/path/to/.cache/huggingface/hub/models--gdhe17--Self-Forcing/snapshots/2f8b779212da279d212c22a509b66ad6552f350e/checkpoints/self_forcing_dmd.pt", help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=126, help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", default=True, help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true", default=True, help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--method", type=str)
parser.add_argument("--index", type=int)
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

print(f'Rank {local_rank}: Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

if args.method is not None:
    config.method = getattr(config, args.method, {})
    if 'method' not in config.method:
        config.method["method"] = args.method
    if 'num_samples' not in config.method:
        config.method["num_samples"] = args.num_samples

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)
state_dict = torch.load(args.checkpoint_path, map_location="cpu")
pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

dmd = DMD(config, device=device)
dmd.eval()

# Create dataset
dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)

num_prompts = len(dataset)
chunk = math.ceil(num_prompts / world_size)
start = local_rank * chunk
end = min(start + chunk, num_prompts)
local_indices = list(range(start, end))
print(f"Rank {local_rank}: Number of prompts: {len(local_indices)}")

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for idx in tqdm(local_indices, disable=(local_rank != 0)):
    batch = dataset[idx]
    idx = idx + 128 + args.index * 8
    seed = args.seed + idx    
    set_seed(seed)

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts']
    extended_prompt = batch['extended_prompts'] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * args.num_samples
    else:
        prompts = [prompt] * args.num_samples

    conditional_dict = dmd.text_encoder(text_prompts=[prompt])
    unconditional_dict = dmd.text_encoder(text_prompts=[args.negative_prompt])

    for part in range(38, math.ceil(360 / args.num_samples)):
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

        # Generate 126 frames
        _, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
            video_index=idx,
            part=part,
        )

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

        latent = torch.tensor_split(latents, 6, dim=1)
        for chunk_id, chunk in enumerate(latent):
            local_results = {}
            for sample_id in range(args.num_samples):
                x = chunk[sample_id]
                if x.dim() == 4:
                    x = x.unsqueeze(0)  # [1,F,C,H,W]
                x = x.to(device=device, dtype=torch.bfloat16)
                dm_loss, _ = dmd.compute_distribution_matching_loss(
                    image_or_video=x,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict
                )
                key = f"{part * args.num_samples + sample_id}"
                local_results[key] = float(dm_loss.item())
            
            out_path = f"{args.output_folder}/{idx}_{seed}/{chunk_id}.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    merged = json.load(f)
            else:
                merged = {}

            merged.update(local_results)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

if dist.is_initialized():
    dist.barrier()

if local_rank == 0:
    print(f"All prompts generated, results saved to: {args.output_folder}")

if dist.is_initialized():
    dist.destroy_process_group()

# CUDA_VISIBLE_DEVICES=0 python inference.py --data_path prompts\compute_dmd_loss\0.txt --output_folder videos/test --num_output_frames 126 --method focusedforcing
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 inference.py --data_path prompts\compute_dmd_loss\0.txt --output_folder videos/test --num_output_frames 126 --method focusedforcing
