import argparse
import math
import os

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from collections import OrderedDict
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed


def main(args):
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

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    if args.method is not None:
        config.method = getattr(config, args.method, {})
        if 'method' not in config.method:
            config.method["method"] = args.method

    def remove_fsdp_prefix(state_dict):
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if "_fsdp_wrapped_module." in key:
                new_key = key.replace("_fsdp_wrapped_module.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    # Initialize pipeline
    pipeline = CausalInferencePipeline(config, device=device)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(remove_fsdp_prefix(state_dict['generator' if not args.use_ema else 'generator_ema']))
    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Create dataset
    dataset = TextDataset(prompt_path=args.prompt_path)

    num_prompts = len(dataset)
    chunk = math.ceil(num_prompts / world_size)
    start = local_rank * chunk
    end = min(start + chunk, num_prompts)
    local_indices = list(range(start, end))
    print(f"Rank {local_rank}: Number of prompts: {len(local_indices)}")

    # Create output directory (only on main process to avoid race conditions)
    profile = False
    if local_rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        profile = False
    if dist.is_initialized():
        dist.barrier()

    from wan.modules.focusedforcing import prepare_meta
    meta = prepare_meta(loss_path="dm_loss.json", max_budget=args.max_budget, min_budget=args.min_budget, attn_weight=args.attn_weight)

    for video_index in tqdm(local_indices, disable=(local_rank != 0)):
        prompt = dataset[video_index]['prompts']

        video_index = video_index + args.video_index_offset
        seed = args.seed + video_index
        set_seed(seed)

        # if args.save_with_index:
        #     all_exist = all(
        #         os.path.exists(os.path.join(args.output_path, f'{video_index}_{seed}_{sample_index}.mp4'))
        #         for sample_index in range(args.num_samples)
        #     )
        # else:
        #     all_exist = all(
        #         os.path.exists(os.path.join(args.output_path, f'{prompt[:100]}_{seed}_{sample_index}.mp4'))
        #         for sample_index in range(args.num_samples)
        #     )

        # if all_exist:
        #     continue

        latent_height = math.ceil(args.height / 8)
        latent_width = math.ceil(args.width / 8)
        num_channels = 16
        sampled_noise = torch.randn([args.num_samples, args.num_latent_frames, num_channels, latent_height, latent_width], device=device, dtype=torch.bfloat16)

        # Generate 126 frames
        video, _ = pipeline.inference_rolling_forcing(
            noise=sampled_noise,
            text_prompts=prompt * args.num_samples,
            return_latents=True,
            profile=profile,
            video_index=video_index,
            meta=meta,
        )
        video = video[:, :486]
        
        video = 255.0 * rearrange(video, 'b t c h w -> b t h w c').cpu()

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

        #! Make sure each sample is mapped to the correct seed
        for sample_index in range(args.num_samples):
            if args.save_with_index:
                video_path = os.path.join(args.output_path, f'{video_index}_{seed}_{sample_index}.mp4')
            else:
                video_path = os.path.join(args.output_path, f'{prompt[:100]}_{seed}_{sample_index}.mp4')

            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            write_video(video_path, video[sample_index], fps=args.fps)

    if dist.is_initialized():
        dist.barrier()

    if local_rank == 0:
        print(f"All prompts generated, results saved to: {args.output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/rolling_forcing_dmd.yaml", help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="/path/to/.cache/huggingface/hub/models--zhuhz22--Causal-Forcing/snapshots/634dc0df042be8133173c3eb4d9efb5da1fba9ca/chunkwise/longvideo.pt", help="Path to the checkpoint folder")
    parser.add_argument("--prompt_path", type=str, help="Path to the prompt file")
    parser.add_argument("--output_path", type=str, help="Output folder")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video")
    parser.add_argument("--width", type=int, default=832, help="Width of the output video")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second of the output video")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--num_latent_frames", type=int, default=126, help="Number of frames to generate, 21 for 5s")
    parser.add_argument("--video_index_offset", type=int, default=0, help="Offset of the video index")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Whether to use EMA parameters")
    parser.add_argument("--profile", action="store_true", default=False, help="Whether to profile the inference")
    parser.add_argument("--save_with_index", action="store_true", default=True, help="Whether to save the video using the index or prompt as the filename")
    parser.add_argument("--method", type=str, default="focusedforcing")
    parser.add_argument("--max_budget", type=int, default=12, help="Maximum budget")
    parser.add_argument("--min_budget", type=int, default=4, help="Minimum budget")
    parser.add_argument("--attn_weight", type=float, default=0.46, help="Weight for attention")
    args = parser.parse_args()

    main(args)

# CUDA_VISIBLE_DEVICES=0 python inference.py --prompt_path prompts/test/test_1.txt --output_path videos/test
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 inference.py --prompt_path prompts/test/test_0.txt --output_path videos/test
