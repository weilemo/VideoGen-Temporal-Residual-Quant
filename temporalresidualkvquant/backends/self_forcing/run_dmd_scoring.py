import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from model import DMD
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb

from trq.head_importance import build_topk_policy_from_focused_forcing, write_topk_policy


parser = argparse.ArgumentParser(
    description="Phase 2: Load saved ablation latents, compute DMD loss, and optionally build a top-k policy."
)
parser.add_argument("--config_path", type=str, help="Path to the config file (for DMD model init)")
parser.add_argument("--output_folder", type=str, required=True, help="Directory containing saved latents from Phase 1")
parser.add_argument("--policy_output_path", type=str, default="", help="Optional output path for the aggregated top-k policy")
parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形，毁容，杂乱背景")
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--num_layers", type=int, default=30)
parser.add_argument("--top_k", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--score_direction", choices=["higher", "lower"], default="higher")
parser.add_argument("--allow_incomplete", action="store_true")
parser.add_argument("--delete_latents_after_scoring", action="store_true",
                    help="Delete .pt files after scoring to free disk space")
parser.add_argument("--skip_existing", "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                    help="Skip heads already present in chunk JSONs (resumption support)")
args = None


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank, dist.get_world_size()
    return torch.device("cuda"), 0, 1


def _heads_by_chunk_json(prompt_dir: str, num_loss_chunks: int) -> dict[int, set[str]]:
    """Return global_head_id strings already present in each chunk JSON."""
    existing = {}
    for chunk_id in range(num_loss_chunks):
        chunk_path = os.path.join(prompt_dir, f"chunk_{chunk_id:02d}.json")
        if os.path.exists(chunk_path):
            with open(chunk_path, "r", encoding="utf-8") as f:
                existing[chunk_id] = set(json.load(f).keys())
        else:
            existing[chunk_id] = set()
    return existing


def _heads_completed_in_all_chunks(existing_by_chunk: dict[int, set[str]]) -> set[str]:
    chunks = list(existing_by_chunk.values())
    if not chunks:
        return set()
    completed = set(chunks[0])
    for chunk_heads in chunks[1:]:
        completed &= chunk_heads
    return completed


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

    print(f"Rank {local_rank}: free VRAM before DMD init: {get_cuda_free_memory_gb(gpu):.1f} GB")

    dmd = DMD(config, device=device)
    dmd = dmd.to(dtype=torch.bfloat16)
    dmd.eval()
    # Offload generator and VAE — not needed for scoring
    dmd.generator.to(device="cpu")
    dmd.vae.to(device="cpu")
    torch.cuda.empty_cache()

    print(f"Rank {local_rank}: free VRAM after DMD init + offload: {get_cuda_free_memory_gb(gpu):.1f} GB")

    # Discover prompt directories
    if not os.path.isdir(args.output_folder):
        raise FileNotFoundError(
            f"Output folder {args.output_folder} not found. Run Phase 1 (inference ablation) first."
        )
    prompt_dirs = sorted(
        d for d in os.listdir(args.output_folder)
        if d.startswith("prompt_") and os.path.isdir(os.path.join(args.output_folder, d))
    )
    if not prompt_dirs:
        raise FileNotFoundError(
            f"No prompt_XXXXX directories found in {args.output_folder}. "
            "Run Phase 1 (inference ablation) first."
        )

    num_prompts = len(prompt_dirs)
    prompt_chunk = math.ceil(num_prompts / world_size)
    start = local_rank * prompt_chunk
    end = min(start + prompt_chunk, num_prompts)
    local_dirs = prompt_dirs[start:end]

    if local_rank == 0:
        print(f"Found {num_prompts} prompt directories, processing {len(local_dirs)} on this rank")

    for prompt_dir in tqdm(local_dirs, disable=(local_rank != 0)):
        prompt_path = os.path.join(args.output_folder, prompt_dir)

        with open(os.path.join(prompt_path, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        prompt_text = metadata.get("extended_prompt") or metadata["prompt"]
        total_heads = metadata["total_heads"]
        num_loss_chunks = metadata.get("num_loss_chunks", 6)
        prompt_idx = int(prompt_dir.split("_")[1])

        # Encode text once per prompt
        dmd.text_encoder.to(device=device)
        conditional_dict = dmd.text_encoder(text_prompts=[prompt_text])
        unconditional_dict = dmd.text_encoder(text_prompts=[args.negative_prompt])
        dmd.text_encoder.to(device="cpu")
        torch.cuda.empty_cache()

        existing_by_chunk = _heads_by_chunk_json(prompt_path, num_loss_chunks) if args.skip_existing else {
            chunk_id: set() for chunk_id in range(num_loss_chunks)
        }
        completed_heads = _heads_completed_in_all_chunks(existing_by_chunk)

        for batch_info in metadata["batches"]:
            global_head_ids = batch_info["global_head_ids"]
            batch_file = batch_info["file"]

            # Skip batch if all heads already scored
            if args.skip_existing and all(str(h) in completed_heads for h in global_head_ids):
                continue

            latents = torch.load(
                os.path.join(prompt_path, batch_file), map_location="cpu", weights_only=True
            )

            dmd.real_score.to(device=device)
            dmd.fake_score.to(device=device)

            latent_chunks = torch.tensor_split(latents, num_loss_chunks, dim=1)
            for chunk_id, chunk in enumerate(latent_chunks):
                local_results = {}
                for sample_id, global_head_id in enumerate(global_head_ids):
                    if args.skip_existing and str(global_head_id) in existing_by_chunk.get(chunk_id, set()):
                        continue
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

                if local_results:
                    out_path = os.path.join(prompt_path, f"chunk_{chunk_id:02d}.json")
                    if os.path.exists(out_path):
                        with open(out_path, "r", encoding="utf-8") as f:
                            merged = json.load(f)
                    else:
                        merged = {}
                    merged.update(local_results)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(merged, f, ensure_ascii=False, indent=2)
                    existing_by_chunk.setdefault(chunk_id, set()).update(local_results.keys())

            completed_heads = _heads_completed_in_all_chunks(existing_by_chunk)

            dmd.real_score.to(device="cpu")
            dmd.fake_score.to(device="cpu")
            torch.cuda.empty_cache()

            if args.delete_latents_after_scoring:
                os.remove(os.path.join(prompt_path, batch_file))

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
    args = parser.parse_args()
    main()
