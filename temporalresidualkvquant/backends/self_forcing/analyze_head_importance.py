import warnings
warnings.warn(
    "analyze_head_importance.py is superseded by run_inference_ablation.py + run_dmd_scoring.py. "
    "Use run_head_importance_analysis.sh (with PHASE env var) instead.",
    DeprecationWarning,
)

import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from model import DMD
from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from trq.head_importance import build_topk_policy_from_focused_forcing, write_topk_policy


parser = argparse.ArgumentParser(
    description="Run Self-Forcing single-head ablation, compute DMD loss, and optionally write a top-k policy."
)
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the Self-Forcing checkpoint")
parser.add_argument("--data_path", type=str, required=True, help="Prompt file")
parser.add_argument("--extended_prompt_path", type=str, default="", help="Optional extended prompt file")
parser.add_argument("--output_folder", type=str, required=True, help="Directory for per-chunk head-loss JSON files")
parser.add_argument("--policy_output_path", type=str, default="", help="Optional output path for the aggregated top-k policy")
parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形，毁容，杂乱背景")
parser.add_argument("--num_output_frames", type=int, default=126)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--num_layers", type=int, default=30)
parser.add_argument("--top_k", type=int, default=4)
parser.add_argument("--head_start", type=int, default=0)
parser.add_argument("--head_end", type=int, default=-1)
parser.add_argument("--heads_per_batch", type=int, default=3)
parser.add_argument("--num_loss_chunks", type=int, default=6)
parser.add_argument("--local_attn_size", type=int, default=180)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_ema", action="store_true")
parser.add_argument("--score_direction", choices=["higher", "lower"], default="higher")
parser.add_argument("--allow_incomplete", action="store_true")
args = parser.parse_args()


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank, dist.get_world_size()
    return torch.device("cuda"), 0, 1


def main():
    device, local_rank, world_size = setup_distributed()
    set_seed(args.seed + local_rank)
    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = OmegaConf.load(os.path.join(script_dir, "configs/default_config.yaml"))
    config = OmegaConf.merge(default_config, config)
    config.quant_config = {
        "quant_type": "none",
        "headwise_mode": "none",
    }
    if args.local_attn_size >= 0:
        if "model_kwargs" not in config or config.model_kwargs is None:
            config.model_kwargs = OmegaConf.create()
        config.model_kwargs.local_attn_size = args.local_attn_size

    print(f"Rank {local_rank}: free VRAM {get_cuda_free_memory_gb(gpu)} GB")
    low_memory = get_cuda_free_memory_gb(gpu) < 40

    pipeline = CausalInferencePipeline(config, device=device)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        pipeline.generator.load_state_dict(state_dict["generator_ema" if args.use_ema else "generator"])
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    dmd = DMD(config, device=device)
    dmd.eval()
    # DMD text_encoder only needed briefly for encoding; DMD score models stay CPU
    # to avoid competing with pipeline models for VRAM (each T5 text_encoder is ~6 GB in bf16)

    dataset = TextDataset(
        prompt_path=args.data_path,
        extended_prompt_path=args.extended_prompt_path or None,
    )
    num_prompts = len(dataset)
    prompt_chunk = math.ceil(num_prompts / world_size)
    start = local_rank * prompt_chunk
    end = min(start + prompt_chunk, num_prompts)
    local_indices = list(range(start, end))

    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    total_heads = args.num_layers * args.num_heads
    head_end = total_heads if args.head_end < 0 else min(args.head_end, total_heads)
    head_ids = list(range(args.head_start, head_end))

    for prompt_idx in tqdm(local_indices, disable=(local_rank != 0)):
        batch = dataset[prompt_idx]
        prompt = batch["prompts"]
        extended_prompt = batch.get("extended_prompts")
        # Encode with DMD text_encoder briefly, then free VRAM
        dmd.text_encoder.to(device=gpu)
        conditional_dict = dmd.text_encoder(text_prompts=[prompt])
        unconditional_dict = dmd.text_encoder(text_prompts=[args.negative_prompt])
        dmd.text_encoder.to(device="cpu")
        torch.cuda.empty_cache()

        for offset in range(0, len(head_ids), args.heads_per_batch):
            current_head_ids = head_ids[offset:offset + args.heads_per_batch]
            num_samples = len(current_head_ids)
            prompts = [extended_prompt or prompt] * num_samples
            set_seed(args.seed + prompt_idx * total_heads + current_head_ids[0])
            noise = torch.randn(
                [num_samples, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )
            ablation_ids = torch.tensor(current_head_ids, device=device, dtype=torch.long)

            _, latents = pipeline.inference(
                noise=noise,
                text_prompts=prompts,
                return_latents=True,
                low_memory=low_memory,
                ablation_global_head_ids=ablation_ids,
                decode_video=False,
            )

            if hasattr(pipeline.vae.model, "clear_cache"):
                pipeline.vae.model.clear_cache()

            # Offload pipeline models to CPU, bring DMD scores to GPU
            pipeline.generator.to(device="cpu")
            pipeline.vae.to(device="cpu")
            torch.cuda.empty_cache()
            dmd.real_score.to(device=gpu)
            dmd.fake_score.to(device=gpu)

            latent_chunks = torch.tensor_split(latents, args.num_loss_chunks, dim=1)
            for chunk_id, chunk in enumerate(latent_chunks):
                local_results = {}
                for sample_id, global_head_id in enumerate(current_head_ids):
                    x = chunk[sample_id]
                    if x.dim() == 4:
                        x = x.unsqueeze(0)
                    x = x.to(device=device, dtype=torch.bfloat16)
                    dm_loss, _ = dmd.compute_distribution_matching_loss(
                        image_or_video=x,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                    )
                    local_results[str(global_head_id)] = float(dm_loss.item())

                out_path = os.path.join(args.output_folder, f"prompt_{prompt_idx:05d}", f"chunk_{chunk_id:02d}.json")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8") as f:
                        merged = json.load(f)
                else:
                    merged = {}
                merged.update(local_results)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)

            # Offload DMD scores back to CPU, bring pipeline models back to GPU for next inference
            dmd.real_score.to(device="cpu")
            dmd.fake_score.to(device="cpu")
            torch.cuda.empty_cache()
            pipeline.generator.to(device=gpu)
            pipeline.vae.to(device=gpu)

    if dist.is_initialized():
        dist.barrier()

    if local_rank == 0 and args.policy_output_path:
        policy = build_topk_policy_from_focused_forcing(
            args.output_folder,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            top_k=args.top_k,
            score_direction=args.score_direction,
            allow_incomplete=args.allow_incomplete,
        )
        out = write_topk_policy(policy, args.policy_output_path)
        print(f"Wrote top-k head policy to {out}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
