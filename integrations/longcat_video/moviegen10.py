#!/usr/bin/env python3
"""Generate paired LongCat-Video continuation samples for KV-cache experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist


NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, "
    "still picture, worst quality, low quality, JPEG artifacts, deformed, disfigured, "
    "misshapen limbs, fused fingers, messy background"
)

OUTPUT_NAMES = {
    "bf16": "bf16",
    "trq_int4": "trq_int4",
    "trq_int2": "trq_int2",
    "naive_int4": "naive_int4",
    "naive_int2": "naive_int2",
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--longcat-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--prompts",
        default=str(repo_root / "temporalresidualkvquant/assets/moviegenbench_32.txt"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prefix-dir", required=True)
    parser.add_argument("--task", choices=("prefix", "continuation"), default="continuation")
    parser.add_argument("--mode", choices=tuple(OUTPUT_NAMES), default="bf16")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-cond-frames", type=int, default=13)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--predictor-stride", type=int, default=1560)
    parser.add_argument("--anchor-bits", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(), repo_root


def initialize_distributed(context_parallel_size, longcat_repo):
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    sys.path.insert(0, str(longcat_repo))
    from longcat_video.context_parallel.context_parallel_util import init_context_parallel

    init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=rank,
        world_size=dist.get_world_size(),
    )
    return rank, local_rank


def load_pipeline(checkpoint_dir, local_rank):
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    cp_split_hw = context_parallel_util.get_optimal_split(context_parallel_util.get_cp_size())
    common = {"torch_dtype": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", **common)
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", **common
    )
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", **common)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", **common
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, **common
    )
    pipeline = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipeline.to(local_rank)
    return pipeline


def save_video(video, path):
    tensor = torch.from_numpy(np.asarray(video))
    if tensor.dtype != torch.uint8:
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    iio.imwrite(str(path), tensor.cpu().numpy(), fps=15, codec="libx264")


def is_decodable_video(path: Path) -> bool:
    if not path.is_file():
        return False
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "video" in result.stdout.splitlines()


def main():
    args, repo_root = parse_args()
    longcat_repo = Path(args.longcat_repo).resolve()
    if not (longcat_repo / "longcat_video/pipeline_longcat_video.py").exists():
        raise FileNotFoundError(f"invalid LongCat repository: {longcat_repo}")
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not (checkpoint_dir / "dit").is_dir():
        raise FileNotFoundError(f"LongCat checkpoint is incomplete: missing {checkpoint_dir / 'dit'}")

    rank, local_rank = initialize_distributed(args.context_parallel_size, longcat_repo)
    sys.path.insert(0, str(repo_root / "temporalresidualkvquant/src"))
    from trq.backends.longcat_video import QuantizedLongCatCache, install_longcat_kv_quant

    pipeline = load_pipeline(checkpoint_dir, local_rank)
    install_longcat_kv_quant(
        pipeline,
        args.mode,
        group_size=args.group_size,
        predictor_stride=args.predictor_stride,
        anchor_bits=args.anchor_bits,
        device=torch.device("cuda", local_rank),
    )

    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    selected = list(enumerate(prompts))[args.start_index: args.start_index + args.limit]
    prefix_dir = Path(args.prefix_dir).resolve()
    prefix_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_root).resolve() / OUTPUT_NAMES[args.mode]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    if args.task == "prefix" and args.mode != "bf16":
        raise ValueError("prefix generation must use --mode bf16")

    from diffusers.utils import load_video

    for prompt_idx, prompt in selected:
        target = (prefix_dir if args.task == "prefix" else output_dir) / f"{prompt_idx}-0.mp4"
        if is_decodable_video(target) and not args.overwrite:
            print(f"[skip] {target}")
            continue
        generator = torch.Generator(device=local_rank).manual_seed(args.seed + prompt_idx)
        torch.cuda.reset_peak_memory_stats(local_rank)

        if args.task == "prefix":
            output = pipeline.generate_t2v(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=480,
                width=832,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=4.0,
                generator=generator,
            )[0]
            cache_stats = None
        else:
            prefix_path = prefix_dir / f"{prompt_idx}-0.mp4"
            if not prefix_path.exists():
                raise FileNotFoundError(
                    f"missing BF16 prefix {prefix_path}; run --task prefix --mode bf16 first"
                )
            output = pipeline.generate_vc(
                video=load_video(str(prefix_path)),
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                resolution="480p",
                num_frames=args.num_frames,
                num_cond_frames=args.num_cond_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=4.0,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=True,
            )[0]
            cache = pipeline._get_kv_cache_dict()
            if isinstance(cache, QuantizedLongCatCache):
                cache_stats = {
                    "native_bytes": cache.native_nbytes,
                    "packed_bytes": cache.packed_nbytes(),
                    "compression_ratio": cache.native_nbytes / cache.packed_nbytes(),
                }
            else:
                native_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for pair in cache.values()
                    for tensor in pair
                )
                cache_stats = {
                    "native_bytes": native_bytes,
                    "packed_bytes": native_bytes,
                    "compression_ratio": 1.0,
                }

        if rank == 0:
            save_video(output, target)
            row = {
                "prompt_idx": prompt_idx,
                "seed": args.seed + prompt_idx,
                "mode": args.mode,
                "video": str(target),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(local_rank),
                "cache": cache_stats,
            }
            manifest.append(row)
            print(json.dumps(row, ensure_ascii=False))
        pipeline._clear_cache()
        del output
        torch.cuda.empty_cache()

    if rank == 0:
        manifest_path = output_dir / f"manifest_{args.task}_{args.start_index}_{args.limit}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
