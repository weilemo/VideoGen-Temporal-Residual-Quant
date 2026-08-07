import argparse
import json
import torch
import os
import time
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
import imageio
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
# temporal sliding window size; set -1 for global attention
parser.add_argument("--local_attn_size", type=int, default=-1,
                    help="Sliding-window length used by causal inference (−1 = global attention)")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--profile", action="store_true", help="Collect structured runtime and memory metrics")
parser.add_argument("--save_rollout_latents", action="store_true", help="Save clean rollout latents for paired analysis")
parser.add_argument("--rollout_metrics_dir", type=str, default="", help="Output root for latent and runtime metric files")
parser.add_argument("--prompt_indices", type=str, default="", help="Optional comma-separated prompt indices to run without renumbering")

#########################################################
# Quantization Configuration
#########################################################
parser.add_argument("--quant_type", type=str, default="none", help="Quantization type")
parser.add_argument("--cache_num_k_centroids", type=int, default=256, help="Number of K-Means centroids for K tensor")
parser.add_argument("--cache_num_v_centroids", type=int, default=256, help="Number of K-Means centroids for V tensor")
parser.add_argument("--kmeans_max_iters", type=int, default=100, help="Maximum iterations for K-Means clustering")
parser.add_argument("--quant_block_size", type=int, default=16, help="Block size for quantization")
parser.add_argument("--num_prq_stages", type=int, default=1, help="Number of PRQ stages for nstages-kmeans quantization")
parser.add_argument("--trq_group_size", "--hrq_group_size", dest="trq_group_size", type=int, default=64, help="Group size for TRQ")
parser.add_argument("--trq_anchor_bits", "--hrq_anchor_bits", dest="trq_anchor_bits", type=int, default=4, help="Anchor bit width for TRQ")
parser.add_argument("--trq_predictor_stride", "--hrq_predictor_stride", dest="trq_predictor_stride", type=int, default=1560, help="Predictor stride for TRQ")
parser.add_argument("--trq_predictor_mode", "--hrq_predictor_mode", dest="trq_predictor_mode", type=str, default="identity", choices=["identity", "affine_channel", "affine"], help="Default TRQ predictor")
parser.add_argument("--trq_k_predictor_mode", type=str, default="", help="Optional K-specific predictor override")
parser.add_argument("--trq_v_predictor_mode", type=str, default="", choices=["", "identity", "affine_channel", "affine", "cross_kv", "hybrid_kv_innovation"], help="Optional V-specific predictor override; cross/hybrid requires s2pp quant_type")
parser.add_argument("--trq_predictor_params_path", "--hrq_predictor_params_path", dest="trq_predictor_params_path", type=str, default="", help="Path to fitted affine predictor params (.pt or .npz)")
parser.add_argument("--trq_v_predictor_params_path", type=str, default="", help="Optional V-specific predictor params (.npz)")
parser.add_argument("--trq_scale_precision", "--hrq_scale_precision", dest="trq_scale_precision", type=str, default="bf16", help="Scale precision for TRQ")
parser.add_argument("--trq_residual_quant_mode", "--hrq_residual_quant_mode", dest="trq_residual_quant_mode", type=str, default="asym_zero_point", help="Residual quantization mode for TRQ")
parser.add_argument("--trq_k_bits", type=int, default=0, help="Optional K residual bit override")
parser.add_argument("--trq_v_bits", type=int, default=0, help="Optional V residual bit override")
parser.add_argument("--trq_first_quant_frame", type=int, default=24, help="First scheduled online quantization boundary in latent-frame steps")
parser.add_argument("--trq_quant_interval_frames", type=int, default=24, help="Frame span converted by each bulk quantization event")
parser.add_argument("--trq_quant_schedule", type=str, default="bulk", choices=["bulk", "gradual"], help="Online conversion schedule")
parser.add_argument("--trq_gradual_frames", type=int, default=3, help="Frames converted per block by the gradual schedule")
parser.add_argument("--trq_protected_sink_frames", type=int, default=0, help="Earliest frames that remain BF16")
parser.add_argument("--trq_cache_roles", type=str, default="both", choices=["both", "k", "v"], help="Cache roles quantized by TRQ")
parser.add_argument("--trq_quantized_layers", type=str, default="all", help="Layer selection such as all, 0-7, or 8,10-12")
parser.add_argument("--attention_sink_frames", type=int, default=0, help="Frames retained as the attention sink during cache eviction")
parser.add_argument("--headwise_mode", type=str, default="none", help="Head-wise policy mode: none, random, or topk")
parser.add_argument("--headwise_seed", type=int, default=0, help="Random seed used for head-wise grouping")
parser.add_argument("--num_high_precision_heads", type=int, default=0, help="How many heads use the high-precision quant type")
parser.add_argument("--high_precision_quant_type", type=str, default="triton-nstages-kmeans-int4",
                    help="Quant type used for the sampled high-precision head group")
parser.add_argument("--low_precision_quant_type", type=str, default="",
                    help="Quant type used for the remaining heads; defaults to --quant_type when empty")
parser.add_argument("--head_importance_path", type=str, default="",
                    help="JSON/CSV/TXT file used by headwise_mode=topk")
parser.add_argument("--head_importance_score_direction", type=str, default="higher", choices=["higher", "lower"],
                    help="Whether larger or smaller importance scores should be kept at high precision")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
_script_dir = os.path.dirname(os.path.abspath(__file__))
default_config = OmegaConf.load(os.path.join(_script_dir, "configs/default_config.yaml"))
config = OmegaConf.merge(default_config, config)

# Put the quantization configuration into the config
config.quant_config = {
    "quant_type": args.quant_type,
    "cache_num_k_centroids": args.cache_num_k_centroids,
    "cache_num_v_centroids": args.cache_num_v_centroids,
    "kmeans_max_iters": args.kmeans_max_iters,
    "quant_block_size": args.quant_block_size,
    "num_prq_stages": args.num_prq_stages,
    "trq_group_size": args.trq_group_size,
    "trq_anchor_bits": args.trq_anchor_bits,
    "trq_predictor_stride": args.trq_predictor_stride,
    "trq_predictor_mode": args.trq_predictor_mode,
    "trq_k_predictor_mode": args.trq_k_predictor_mode or None,
    "trq_v_predictor_mode": args.trq_v_predictor_mode or None,
    "trq_predictor_params_path": args.trq_predictor_params_path,
    "trq_v_predictor_params_path": args.trq_v_predictor_params_path or None,
    "trq_scale_precision": args.trq_scale_precision,
    "trq_residual_quant_mode": args.trq_residual_quant_mode,
    "trq_k_bits": args.trq_k_bits,
    "trq_v_bits": args.trq_v_bits,
    "trq_first_quant_frame": args.trq_first_quant_frame,
    "trq_quant_interval_frames": args.trq_quant_interval_frames,
    "trq_quant_schedule": args.trq_quant_schedule,
    "trq_gradual_frames": args.trq_gradual_frames,
    "trq_protected_sink_frames": args.trq_protected_sink_frames,
    "trq_cache_roles": args.trq_cache_roles,
    "trq_quantized_layers": args.trq_quantized_layers,
    "attention_sink_frames": args.attention_sink_frames,
    "headwise_mode": args.headwise_mode,
    "headwise_seed": args.headwise_seed,
    "num_high_precision_heads": args.num_high_precision_heads,
    "high_precision_quant_type": args.high_precision_quant_type,
    "low_precision_quant_type": args.low_precision_quant_type or args.quant_type,
    "head_importance_path": args.head_importance_path,
    "head_importance_score_direction": args.head_importance_score_direction,
}

# ------------------------------------------------------------------
# Override sliding-window size from CLI when provided (>=0)
# ------------------------------------------------------------------
if args.local_attn_size >= 0:
    if "model_kwargs" not in config or config.model_kwargs is None:
        config.model_kwargs = OmegaConf.create()
    config.model_kwargs.local_attn_size = args.local_attn_size
if args.attention_sink_frames < 0:
    raise ValueError("attention_sink_frames must be non-negative")
if "model_kwargs" not in config or config.model_kwargs is None:
    config.model_kwargs = OmegaConf.create()
config.model_kwargs.sink_size = args.attention_sink_frames

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
selected_prompt_indices = None
if args.prompt_indices.strip():
    selected_prompt_indices = {int(value.strip()) for value in args.prompt_indices.split(",") if value.strip()}
    invalid_prompt_indices = sorted(index for index in selected_prompt_indices if not 0 <= index < num_prompts)
    if invalid_prompt_indices:
        raise ValueError(f"prompt_indices outside [0, {num_prompts - 1}]: {invalid_prompt_indices}")
    if not selected_prompt_indices:
        raise ValueError("prompt_indices selected no prompts")
print(f"Number of prompts: {num_prompts}")
if selected_prompt_indices is not None:
    print(f"Selected prompt indices: {sorted(selected_prompt_indices)}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
metrics_root = Path(args.rollout_metrics_dir or os.path.join(args.output_folder, "rollout_metrics"))
if args.profile or args.save_rollout_latents:
    (metrics_root / "runtime").mkdir(parents=True, exist_ok=True)
    if args.save_rollout_latents:
        (metrics_root / "latents").mkdir(parents=True, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()
    if selected_prompt_indices is not None and idx not in selected_prompt_indices:
        continue

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    if args.profile:
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
        profile=args.profile,
    )
    if args.profile:
        torch.cuda.synchronize()
    wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            artifact_stem = f'{idx}-{seed_idx}_{model}'
            metadata = {
                "format": "trq_online_rollout",
                "version": 1,
                "prompt_index": int(idx),
                "prompt": prompt,
                "seed": int(args.seed),
                "effective_seed": int(args.seed + local_rank),
                "sample_index": int(seed_idx),
                "num_output_frames": int(args.num_output_frames),
                "local_attn_size": int(args.local_attn_size),
                "data_path": str(Path(args.data_path).expanduser().resolve()),
                "config_path": str(Path(args.config_path).expanduser().resolve()),
                "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()) if args.checkpoint_path else "",
                "quant_type": str(args.quant_type),
                "trq_k_bits": int(args.trq_k_bits),
                "trq_v_bits": int(args.trq_v_bits),
                "trq_anchor_bits": int(args.trq_anchor_bits),
                "trq_group_size": int(args.trq_group_size),
                "trq_predictor_stride": int(args.trq_predictor_stride),
                "trq_predictor_mode": str(args.trq_predictor_mode),
                "trq_scale_precision": str(args.trq_scale_precision),
                "trq_residual_quant_mode": str(args.trq_residual_quant_mode),
                "trq_first_quant_frame": int(args.trq_first_quant_frame),
                "trq_quant_interval_frames": int(args.trq_quant_interval_frames),
                "trq_quant_schedule": str(args.trq_quant_schedule),
                "trq_gradual_frames": int(args.trq_gradual_frames),
                "trq_protected_sink_frames": int(args.trq_protected_sink_frames),
                "trq_cache_roles": str(args.trq_cache_roles),
                "trq_quantized_layers": str(args.trq_quantized_layers),
                "attention_sink_frames": int(args.attention_sink_frames),
                "quantization_events": list(getattr(pipeline, "last_quantization_events", [])),
                "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda),
                "gpu_name": str(torch.cuda.get_device_name(device)),
                "latent_shape": list(latents[seed_idx].shape),
            }
            if args.save_rollout_latents:
                torch.save(
                    {
                        "metadata": metadata,
                        "latents": latents[seed_idx].detach().to(
                            device="cpu", dtype=torch.bfloat16
                        ).contiguous(),
                    },
                    metrics_root / "latents" / f"{artifact_stem}.pt",
                )
            if args.profile:
                runtime_payload = {
                    "metadata": metadata,
                    "wall_time_e2e_ms": float(wall_time_ms),
                    "pipeline": pipeline.last_runtime_metrics,
                }
                with open(metrics_root / "runtime" / f"{artifact_stem}.json", "w", encoding="utf-8") as handle:
                    json.dump(runtime_payload, handle, indent=2, ensure_ascii=False)

            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{artifact_stem}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')

            import termcolor
            print(termcolor.colored(f"Saving video to {output_path}", "green"))
            video_np = video[seed_idx].numpy().astype('uint8')
            imageio.mimsave(output_path, video_np, fps=16, codec='libx264')
