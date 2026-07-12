import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


parser = argparse.ArgumentParser(
    description="Phase 1: Run Self-Forcing inference with head ablation, save latents to disk."
)
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the Self-Forcing checkpoint")
parser.add_argument("--data_path", type=str, required=True, help="Prompt file")
parser.add_argument("--extended_prompt_path", type=str, default="", help="Optional extended prompt file")
parser.add_argument("--output_folder", type=str, required=True, help="Directory for saved latents and metadata")
parser.add_argument("--num_output_frames", type=int, default=126)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--num_layers", type=int, default=30)
parser.add_argument("--num_loss_chunks", type=int, default=6)
parser.add_argument("--head_start", type=int, default=0)
parser.add_argument("--head_end", type=int, default=-1)
parser.add_argument("--heads_per_batch", type=int, default=3)
parser.add_argument("--local_attn_size", type=int, default=180)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_ema", action="store_true")
args = None


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

        out_dir = os.path.join(args.output_folder, f"prompt_{prompt_idx:05d}")
        os.makedirs(out_dir, exist_ok=True)

        batches_meta = []

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

            # Save latents to disk
            batch_idx = offset // args.heads_per_batch
            batch_file = f"batch_{batch_idx:05d}.pt"
            torch.save(latents.cpu(), os.path.join(out_dir, batch_file))

            batches_meta.append({
                "batch_idx": batch_idx,
                "global_head_ids": current_head_ids,
                "file": batch_file,
            })

            del latents, noise, ablation_ids
            torch.cuda.empty_cache()

        metadata = {
            "prompt": prompt,
            "extended_prompt": extended_prompt or "",
            "num_output_frames": args.num_output_frames,
            "seed": args.seed,
            "total_heads": total_heads,
            "num_loss_chunks": args.num_loss_chunks,
            "batches": batches_meta,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    main()
