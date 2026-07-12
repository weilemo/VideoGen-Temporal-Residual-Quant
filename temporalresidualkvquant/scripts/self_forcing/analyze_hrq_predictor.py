"""Offline predictor diagnostics for HRQ (Phase 1–3).

Loads BF16 KV-cache dumps produced with DUMP_KV_LEVEL=1 and quant_type=none,
then for each layer:
  - Computes pre-quant residual Rel-L2 for identity / affine_channel / tiny_mlp
  - Fits per-layer, per-head, per-channel affine params (least-squares)
  - Trains a tiny MLP (D→hidden→D, per-layer, shared across heads)
  - Reports train vs held-out gap and residual statistics

Outputs:
  assets/trq_predictors/affine_channel_self_forcing_dmd.pt
  assets/trq_predictors/tiny_mlp_self_forcing_dmd.pt
  results/selfforcing/vbench_eval_hrq_predictor/predictor_diagnostics.txt

Usage:
    python analyze_hrq_predictor.py \\
        --train_dumps kv_dumps/train_*.pt \\
        --heldout_dumps kv_dumps/heldout_*.pt \\
        [--predictor_stride 1560] \\
        [--output_dir assets/trq_predictors] \\
        [--report_path results/selfforcing/vbench_eval_hrq_predictor/predictor_diagnostics.txt] \\
        [--device cuda] \\
        [--mlp_hidden_mult 1] \\
        [--mlp_epochs 30] \\
        [--mlp_lr 1e-3] \\
        [--mlp_batch_size 4096] \\
        [--mlp_patience 5] \\
        [--skip_mlp]
"""

from __future__ import annotations

import argparse
import copy
import glob
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── PYTHONPATH setup ────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_hwq_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
_sf_root = os.path.join(_hwq_root, "backends", "self_forcing")
for _p in [os.path.join(_hwq_root, "src"), _sf_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trq.kv_cache import ChunkState  # noqa: E402 – must come after path setup


# ══════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AffineStats:
    """Incremental sufficient statistics for least-squares affine fitting."""
    sum_xx: Optional[torch.Tensor] = None   # [H, D]
    sum_xy: Optional[torch.Tensor] = None
    sum_x: Optional[torch.Tensor] = None
    sum_y: Optional[torch.Tensor] = None
    N: int = 0

    def update(self, x_prev: torch.Tensor, x_curr: torch.Tensor):
        """Accumulate one (prev, curr) chunk pair.

        Args:
            x_prev: [B, H, S, D] float tensor
            x_curr: [B, H, S, D] float tensor
        """
        # Sum over B and S, keeping [H, D]
        xx = (x_prev * x_prev).sum(dim=(0, 2))
        xy = (x_prev * x_curr).sum(dim=(0, 2))
        sx = x_prev.sum(dim=(0, 2))
        sy = x_curr.sum(dim=(0, 2))
        n = x_prev.shape[0] * x_prev.shape[2]   # B * S

        if self.sum_xx is None:
            self.sum_xx = xx
            self.sum_xy = xy
            self.sum_x = sx
            self.sum_y = sy
        else:
            self.sum_xx = self.sum_xx + xx
            self.sum_xy = self.sum_xy + xy
            self.sum_x = self.sum_x + sx
            self.sum_y = self.sum_y + sy
        self.N += n

    def solve(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha, beta) each [H, D]; falls back to identity where ill-conditioned."""
        N = float(self.N)
        denom = N * self.sum_xx - self.sum_x ** 2
        safe = denom.abs() > 1e-6 * N
        alpha = torch.where(safe, (N * self.sum_xy - self.sum_x * self.sum_y) / denom.clamp_min(1e-30),
                            torch.ones_like(denom))
        beta = torch.where(safe, (self.sum_y - alpha * self.sum_x) / N,
                           torch.zeros_like(denom))
        return alpha, beta


# ══════════════════════════════════════════════════════════════════════
# Tiny MLP
# ══════════════════════════════════════════════════════════════════════

class TinyMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def to_param_dict(self) -> dict:
        """Export as a plain dict of tensors for serialization and fast inference."""
        return {
            "fc1_weight": self.fc1.weight.detach().cpu(),
            "fc1_bias": self.fc1.bias.detach().cpu(),
            "fc2_weight": self.fc2.weight.detach().cpu(),
            "fc2_bias": self.fc2.bias.detach().cpu(),
        }


# ══════════════════════════════════════════════════════════════════════
# KV dump loading helpers
# ══════════════════════════════════════════════════════════════════════

def count_filled_tokens(kv_cache_obj) -> int:
    """Return how many tokens are written in a ChunkedKVCache."""
    n = sum(1 for s in kv_cache_obj.chunk_state if s != ChunkState.EMPTY)
    return n * kv_cache_obj.frame_seq_length


def load_layer_kv(dump_path: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load K and V tensors for *layer_idx* from a KV dump file.

    Returns:
        k, v: float32 tensors of shape [B, H, S, D] (BHSD layout).
    """
    data = torch.load(dump_path, map_location="cpu", weights_only=False)
    kv_cache = data["kv_cache"]
    layer = kv_cache[layer_idx]
    total_tokens = count_filled_tokens(layer["k"])
    if total_tokens == 0:
        raise ValueError(f"Layer {layer_idx} has 0 filled tokens in {dump_path}")
    # read returns BSHD [B, S, H, D] (layout="BSHD")
    k = layer["k"].read(0, total_tokens).permute(0, 2, 1, 3).float().contiguous()
    v = layer["v"].read(0, total_tokens).permute(0, 2, 1, 3).float().contiguous()
    return k, v


def inspect_dump(dump_path: str) -> dict:
    """Return metadata from a dump without loading tensor data."""
    data = torch.load(dump_path, map_location="cpu", weights_only=False)
    kv_cache = data["kv_cache"]
    layer = kv_cache[0]
    k_cache = layer["k"]
    total_tokens = count_filled_tokens(k_cache)
    return {
        "num_layers": len(kv_cache),
        "num_heads": k_cache.num_heads,
        "head_dim": k_cache.head_dim,
        "frame_seq_length": k_cache.frame_seq_length,
        "total_tokens": total_tokens,
    }


# ══════════════════════════════════════════════════════════════════════
# Predictor evaluation helpers
# ══════════════════════════════════════════════════════════════════════

def rel_l2(residual: torch.Tensor, target: torch.Tensor) -> float:
    return (residual.norm() / target.norm().clamp_min(1e-8)).item()


def iter_chunk_pairs(x: torch.Tensor, stride: int):
    """Yield (x_prev, x_curr) pairs by chunking x along S-dim.

    x: [B, H, S, D]; stride is in tokens.
    Yields float32 [B, H, stride, D] pairs (may be shorter for the last chunk).
    """
    S = x.shape[2]
    num_chunks = math.ceil(S / stride)
    for t in range(1, num_chunks):
        s0, e0 = (t - 1) * stride, t * stride
        s1, e1 = t * stride, min((t + 1) * stride, S)
        if e1 <= s1:
            break
        # Keep same size for prev and curr
        length = e1 - s1
        x_prev = x[:, :, s0:s0 + length, :]
        x_curr = x[:, :, s1:e1, :]
        yield x_prev, x_curr


def eval_identity(dump_paths: List[str], layer_idx: int, kv_key: str,
                  stride: int, device: torch.device) -> dict:
    """Compute identity-predictor residuals over all dumps."""
    rel_l2_vals = []
    for path in dump_paths:
        k, v = load_layer_kv(path, layer_idx)
        x = (k if kv_key == "K" else v).to(device)
        for x_prev, x_curr in iter_chunk_pairs(x, stride):
            r = x_curr - x_prev
            rel_l2_vals.append(rel_l2(r, x_curr))
        del k, v, x
    return _aggregate_stats(rel_l2_vals)


def eval_affine(dump_paths: List[str], layer_idx: int, kv_key: str,
                alpha: torch.Tensor, beta: torch.Tensor,
                stride: int, device: torch.device) -> dict:
    """Compute affine-predictor residuals."""
    alpha = alpha.to(device)
    beta = beta.to(device)
    rel_l2_vals = []
    for path in dump_paths:
        k, v = load_layer_kv(path, layer_idx)
        x = (k if kv_key == "K" else v).to(device)
        for x_prev, x_curr in iter_chunk_pairs(x, stride):
            pred = alpha.unsqueeze(0).unsqueeze(2) * x_prev + beta.unsqueeze(0).unsqueeze(2)
            r = x_curr - pred
            rel_l2_vals.append(rel_l2(r, x_curr))
        del k, v, x
    return _aggregate_stats(rel_l2_vals)


def eval_mlp(dump_paths: List[str], layer_idx: int, kv_key: str,
             mlp: TinyMLP, stride: int, device: torch.device) -> dict:
    """Compute MLP-predictor residuals."""
    mlp.eval()
    rel_l2_vals = []
    with torch.no_grad():
        for path in dump_paths:
            k, v = load_layer_kv(path, layer_idx)
            x = (k if kv_key == "K" else v).to(device)
            D = x.shape[-1]
            for x_prev, x_curr in iter_chunk_pairs(x, stride):
                B, H, S, _ = x_prev.shape
                flat = x_prev.reshape(-1, D)
                pred = mlp(flat).reshape(B, H, S, D)
                r = x_curr - pred
                rel_l2_vals.append(rel_l2(r, x_curr))
            del k, v, x
    return _aggregate_stats(rel_l2_vals)


def eval_affine_per_head(dump_paths: List[str], layer_idx: int, kv_key: str,
                         alpha: torch.Tensor, beta: torch.Tensor,
                         stride: int, device: torch.device) -> torch.Tensor:
    """Return per-head Rel-L2 [H] for the affine predictor."""
    alpha = alpha.to(device)
    beta = beta.to(device)
    H = alpha.shape[0]
    per_head_num = torch.zeros(H, device=device)
    per_head_den = torch.zeros(H, device=device)
    for path in dump_paths:
        k, v = load_layer_kv(path, layer_idx)
        x = (k if kv_key == "K" else v).to(device)
        for x_prev, x_curr in iter_chunk_pairs(x, stride):
            pred = alpha.unsqueeze(0).unsqueeze(2) * x_prev + beta.unsqueeze(0).unsqueeze(2)
            r = x_curr - pred
            per_head_num += r.reshape(r.shape[0], H, -1).norm(dim=-1).sum(dim=0)
            per_head_den += x_curr.reshape(x_curr.shape[0], H, -1).norm(dim=-1).sum(dim=0)
        del k, v, x
    return (per_head_num / per_head_den.clamp_min(1e-8))


def _aggregate_stats(vals: List[float]) -> dict:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "p95": float("nan"), "max": float("nan"), "n": 0}
    t = torch.tensor(vals)
    return {
        "mean": t.mean().item(),
        "std": t.std().item() if len(vals) > 1 else 0.0,
        "p95": t.quantile(0.95).item(),
        "max": t.max().item(),
        "n": len(vals),
    }


# ══════════════════════════════════════════════════════════════════════
# Affine fitting
# ══════════════════════════════════════════════════════════════════════

def fit_affine_layer(train_dumps: List[str], layer_idx: int, kv_key: str,
                     stride: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-head, per-channel alpha/beta via least squares."""
    stats = AffineStats()
    for path in train_dumps:
        k, v = load_layer_kv(path, layer_idx)
        x = (k if kv_key == "K" else v).to(device)
        for x_prev, x_curr in iter_chunk_pairs(x, stride):
            stats.update(x_prev, x_curr)
        del k, v, x
    return stats.solve()


# ══════════════════════════════════════════════════════════════════════
# MLP training
# ══════════════════════════════════════════════════════════════════════

def _heldout_rel_l2_mlp(heldout_dumps: List[str], layer_idx: int, kv_key: str,
                          mlp: TinyMLP, stride: int, device: torch.device) -> float:
    mlp.eval()
    numer, denom = 0.0, 0.0
    with torch.no_grad():
        for path in heldout_dumps:
            k, v = load_layer_kv(path, layer_idx)
            x = (k if kv_key == "K" else v).to(device)
            D = x.shape[-1]
            for x_prev, x_curr in iter_chunk_pairs(x, stride):
                B, H, S, _ = x_prev.shape
                flat = x_prev.reshape(-1, D)
                pred = mlp(flat).reshape(B, H, S, D)
                r = x_curr - pred
                numer += r.norm().item() ** 2
                denom += x_curr.norm().item() ** 2
            del k, v, x
    return math.sqrt(numer / max(denom, 1e-8))


def train_mlp_layer(train_dumps: List[str], heldout_dumps: List[str],
                    layer_idx: int, kv_key: str,
                    head_dim: int, hidden_dim: int,
                    stride: int, device: torch.device,
                    max_epochs: int, lr: float,
                    batch_size: int, patience: int,
                    verbose: bool = False) -> TinyMLP:
    """Train a TinyMLP for one layer (K or V)."""
    mlp = TinyMLP(head_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    best_heldout = float("inf")
    best_state = copy.deepcopy(mlp.state_dict())
    patience_counter = 0

    for epoch in range(max_epochs):
        mlp.train()
        epoch_loss = 0.0
        n_steps = 0

        for path in train_dumps:
            k, v = load_layer_kv(path, layer_idx)
            x = (k if kv_key == "K" else v).to(device)
            D = x.shape[-1]

            for x_prev, x_curr in iter_chunk_pairs(x, stride):
                B, H, S, _ = x_prev.shape
                flat_prev = x_prev.reshape(-1, D)
                flat_curr = x_curr.reshape(-1, D)

                # Mini-batch gradient steps
                perm = torch.randperm(flat_prev.shape[0], device=device)
                for i in range(0, flat_prev.shape[0], batch_size):
                    idx = perm[i:i + batch_size]
                    loss = F.mse_loss(mlp(flat_prev[idx]), flat_curr[idx])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_steps += 1

            del k, v, x

        if heldout_dumps:
            h_rl2 = _heldout_rel_l2_mlp(heldout_dumps, layer_idx, kv_key, mlp, stride, device)
        else:
            h_rl2 = epoch_loss / max(n_steps, 1)

        if verbose:
            print(f"    [L{layer_idx} {kv_key} ep{epoch+1:02d}] "
                  f"train_loss={epoch_loss/max(n_steps,1):.4e}  held_rl2={h_rl2:.4f}")

        if h_rl2 < best_heldout - 1e-5:
            best_heldout = h_rl2
            best_state = copy.deepcopy(mlp.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    [L{layer_idx} {kv_key}] Early stop at epoch {epoch+1}")
                break

    mlp.load_state_dict(best_state)
    return mlp


# ══════════════════════════════════════════════════════════════════════
# Main analysis loop
# ══════════════════════════════════════════════════════════════════════

def _fmt_stats(s: dict) -> str:
    return f"mean={s['mean']:.4f}  std={s['std']:.4f}  p95={s['p95']:.4f}  max={s['max']:.4f}"


def _load_dump_all_layers(dump_path: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load all layers from a dump file.  Returns list of (k, v) in BHSD float32."""
    data = torch.load(dump_path, map_location="cpu", weights_only=False)
    kv_cache = data["kv_cache"]
    out = []
    for layer in kv_cache:
        total_tokens = count_filled_tokens(layer["k"])
        k = layer["k"].read(0, total_tokens).permute(0, 2, 1, 3).float().contiguous()
        v = layer["v"].read(0, total_tokens).permute(0, 2, 1, 3).float().contiguous()
        out.append((k, v))
    # free the kv_cache objects so GC can reclaim memory
    for layer in kv_cache:
        for obj in layer.values():
            if hasattr(obj, "chunks"):
                obj.chunks = [None] * len(obj.chunks)
    del data, kv_cache
    return out


def run_analysis(args):
    import gc

    train_dumps = sorted(glob.glob(args.train_dumps) if "*" in args.train_dumps else args.train_dumps)
    heldout_dumps = sorted(glob.glob(args.heldout_dumps) if "*" in args.heldout_dumps else args.heldout_dumps)

    if isinstance(train_dumps, str):
        train_dumps = [train_dumps]
    if isinstance(heldout_dumps, str):
        heldout_dumps = [heldout_dumps]

    if not train_dumps:
        raise FileNotFoundError(f"No train dumps matched: {args.train_dumps!r}\n"
                                "Run collect_kv_dumps.sh first to generate KV dumps.")
    if not heldout_dumps:
        print("WARNING: no held-out dumps found; train==heldout (no gap analysis)")
        heldout_dumps = train_dumps

    print(f"Train dumps:   {len(train_dumps)}")
    print(f"Heldout dumps: {len(heldout_dumps)}")

    meta = inspect_dump(train_dumps[0])
    num_layers = meta["num_layers"]
    num_heads = meta["num_heads"]
    head_dim = meta["head_dim"]
    stride = args.predictor_stride
    hidden_dim = head_dim * args.mlp_hidden_mult
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    print(f"Model: {num_layers} layers, {num_heads} heads, head_dim={head_dim}")
    print(f"Predictor stride: {stride}")
    print(f"MLP hidden: {hidden_dim}  |  Device: {device}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)

    # ── Per-layer accumulators (file-major loop avoids re-reading 49 GB files) ───
    # affine sufficient statistics
    affine_stats: dict[tuple, AffineStats] = {
        (l, kv): AffineStats() for l in range(num_layers) for kv in ("K", "V")
    }
    # per-layer rel-l2 lists: identity, affine
    id_train_rl2:     dict[tuple, list] = {(l, kv): [] for l in range(num_layers) for kv in ("K", "V")}
    id_heldout_rl2:   dict[tuple, list] = {(l, kv): [] for l in range(num_layers) for kv in ("K", "V")}
    af_train_rl2:     dict[tuple, list] = {(l, kv): [] for l in range(num_layers) for kv in ("K", "V")}
    af_heldout_rl2:   dict[tuple, list] = {(l, kv): [] for l in range(num_layers) for kv in ("K", "V")}
    af_ph_num:        dict[tuple, Optional[torch.Tensor]] = {(l, kv): None for l in range(num_layers) for kv in ("K", "V")}
    af_ph_den:        dict[tuple, Optional[torch.Tensor]] = {(l, kv): None for l in range(num_layers) for kv in ("K", "V")}

    # ── Pass 1: fit affine on train dumps ──────────────────────────────
    print("=== Pass 1: Accumulating affine statistics (train) ===")
    for fi, path in enumerate(train_dumps):
        t0 = time.time()
        print(f"  Loading train[{fi}]: {os.path.basename(path)} ...", flush=True)
        layers_kv = _load_dump_all_layers(path)
        for l_idx, (k, v) in enumerate(layers_kv):
            for kv_key, x in (("K", k), ("V", v)):
                x_dev = x.to(device)
                for x_prev, x_curr in iter_chunk_pairs(x_dev, stride):
                    affine_stats[(l_idx, kv_key)].update(x_prev, x_curr)
                del x_dev
        del layers_kv
        gc.collect()
        print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Solve affine params for all layers
    print("\nSolving affine params ...", flush=True)
    alpha_all: dict[tuple, torch.Tensor] = {}
    beta_all: dict[tuple, torch.Tensor] = {}
    for l in range(num_layers):
        for kv in ("K", "V"):
            a, b = affine_stats[(l, kv)].solve()
            alpha_all[(l, kv)] = a.to(device)
            beta_all[(l, kv)] = b.to(device)

    # ── Pass 2: evaluate identity + affine on train dumps ─────────────
    print("\n=== Pass 2: Evaluating identity + affine on train dumps ===")
    for fi, path in enumerate(train_dumps):
        t0 = time.time()
        print(f"  Loading train[{fi}]: {os.path.basename(path)} ...", flush=True)
        layers_kv = _load_dump_all_layers(path)
        for l_idx, (k, v) in enumerate(layers_kv):
            for kv_key, x in (("K", k), ("V", v)):
                x_dev = x.to(device)
                alpha = alpha_all[(l_idx, kv_key)]
                beta = beta_all[(l_idx, kv_key)]
                for x_prev, x_curr in iter_chunk_pairs(x_dev, stride):
                    # identity
                    id_train_rl2[(l_idx, kv_key)].append(rel_l2(x_curr - x_prev, x_curr))
                    # affine
                    pred = alpha.unsqueeze(0).unsqueeze(2) * x_prev + beta.unsqueeze(0).unsqueeze(2)
                    af_train_rl2[(l_idx, kv_key)].append(rel_l2(x_curr - pred, x_curr))
                del x_dev
        del layers_kv
        gc.collect()
        print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # ── Pass 3: evaluate identity + affine on heldout dumps ───────────
    print("\n=== Pass 3: Evaluating identity + affine on heldout dumps ===")
    for fi, path in enumerate(heldout_dumps):
        t0 = time.time()
        print(f"  Loading heldout[{fi}]: {os.path.basename(path)} ...", flush=True)
        layers_kv = _load_dump_all_layers(path)
        for l_idx, (k, v) in enumerate(layers_kv):
            for kv_key, x in (("K", k), ("V", v)):
                x_dev = x.to(device)
                alpha = alpha_all[(l_idx, kv_key)]
                beta = beta_all[(l_idx, kv_key)]
                H = x_dev.shape[1]
                for x_prev, x_curr in iter_chunk_pairs(x_dev, stride):
                    # identity
                    id_heldout_rl2[(l_idx, kv_key)].append(rel_l2(x_curr - x_prev, x_curr))
                    # affine
                    pred = alpha.unsqueeze(0).unsqueeze(2) * x_prev + beta.unsqueeze(0).unsqueeze(2)
                    af_heldout_rl2[(l_idx, kv_key)].append(rel_l2(x_curr - pred, x_curr))
                    # per-head affine rel-l2 accumulation
                    r = x_curr - pred
                    ph_num = r.reshape(r.shape[0], H, -1).norm(dim=-1).sum(dim=0)
                    ph_den = x_curr.reshape(x_curr.shape[0], H, -1).norm(dim=-1).sum(dim=0)
                    if af_ph_num[(l_idx, kv_key)] is None:
                        af_ph_num[(l_idx, kv_key)] = ph_num
                        af_ph_den[(l_idx, kv_key)] = ph_den
                    else:
                        af_ph_num[(l_idx, kv_key)] = af_ph_num[(l_idx, kv_key)] + ph_num
                        af_ph_den[(l_idx, kv_key)] = af_ph_den[(l_idx, kv_key)] + ph_den
                del x_dev
        del layers_kv
        gc.collect()
        print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # ── MLP training (file-major, all layers simultaneously) ──────────
    mlp_train_rl2:   dict[tuple, list] = {}
    mlp_heldout_rl2: dict[tuple, list] = {}
    mlp_params_out: dict[tuple, dict] = {}

    if not args.skip_mlp:
        print("\n=== Pass 4: Training tiny_mlp (all layers, file-major) ===")
        mlps = {(l, kv): TinyMLP(head_dim, hidden_dim).to(device)
                for l in range(num_layers) for kv in ("K", "V")}
        optims = {k: torch.optim.Adam(v.parameters(), lr=args.mlp_lr)
                  for k, v in mlps.items()}
        best_heldout = {k: float("inf") for k in mlps}
        best_state = {k: copy.deepcopy(v.state_dict()) for k, v in mlps.items()}
        patience_ctr = {k: 0 for k in mlps}
        active = set(mlps.keys())  # keys still training

        for epoch in range(args.mlp_epochs):
            if not active:
                break
            # ─ train pass
            for fi, path in enumerate(train_dumps):
                if not active:
                    break
                layers_kv = _load_dump_all_layers(path)
                for l_idx, (k, v) in enumerate(layers_kv):
                    for kv_key, x in (("K", k), ("V", v)):
                        key = (l_idx, kv_key)
                        if key not in active:
                            continue
                        x_dev = x.to(device)
                        D = x_dev.shape[-1]
                        mlps[key].train()
                        for x_prev, x_curr in iter_chunk_pairs(x_dev, stride):
                            B, H, S, _ = x_prev.shape
                            fp = x_prev.reshape(-1, D)
                            fc = x_curr.reshape(-1, D)
                            perm = torch.randperm(fp.shape[0], device=device)
                            for i in range(0, fp.shape[0], args.mlp_batch_size):
                                idx = perm[i:i+args.mlp_batch_size]
                                loss = F.mse_loss(mlps[key](fp[idx]), fc[idx])
                                optims[key].zero_grad(set_to_none=True)
                                loss.backward()
                                optims[key].step()
                        del x_dev
                del layers_kv
                gc.collect()

            # ─ heldout evaluation pass
            h_numer = {k: 0.0 for k in active}
            h_denom = {k: 0.0 for k in active}
            for fi, path in enumerate(heldout_dumps):
                layers_kv = _load_dump_all_layers(path)
                for l_idx, (k, v) in enumerate(layers_kv):
                    for kv_key, x in (("K", k), ("V", v)):
                        key = (l_idx, kv_key)
                        if key not in active:
                            continue
                        x_dev = x.to(device)
                        D = x_dev.shape[-1]
                        mlps[key].eval()
                        with torch.no_grad():
                            for x_prev, x_curr in iter_chunk_pairs(x_dev, stride):
                                B, H, S, _ = x_prev.shape
                                pred = mlps[key](x_prev.reshape(-1, D)).reshape(B, H, S, D)
                                r = x_curr - pred
                                h_numer[key] += r.norm().item() ** 2
                                h_denom[key] += x_curr.norm().item() ** 2
                        del x_dev
                del layers_kv
                gc.collect()

            # ─ early stopping check
            newly_stopped = set()
            for key in list(active):
                h_rl2 = math.sqrt(h_numer[key] / max(h_denom[key], 1e-8))
                if args.verbose_mlp:
                    print(f"  [L{key[0]:02d} {key[1]} ep{epoch+1:02d}] held_rl2={h_rl2:.4f}", flush=True)
                if h_rl2 < best_heldout[key] - 1e-5:
                    best_heldout[key] = h_rl2
                    best_state[key] = copy.deepcopy(mlps[key].state_dict())
                    patience_ctr[key] = 0
                else:
                    patience_ctr[key] += 1
                    if patience_ctr[key] >= args.mlp_patience:
                        newly_stopped.add(key)
            active -= newly_stopped
            if newly_stopped and args.verbose_mlp:
                print(f"  Epoch {epoch+1}: {len(newly_stopped)} MLPs converged, "
                      f"{len(active)} still active", flush=True)

        # Restore best states
        for key, mlp in mlps.items():
            mlp.load_state_dict(best_state[key])

        # ─ MLP eval pass (train)
        print("\n=== Pass 5: Evaluating tiny_mlp on train dumps ===")
        for fi, path in enumerate(train_dumps):
            layers_kv = _load_dump_all_layers(path)
            for l_idx, (k, v) in enumerate(layers_kv):
                for kv_key, x in (("K", k), ("V", v)):
                    key = (l_idx, kv_key)
                    x_dev = x.to(device)
                    D = x_dev.shape[-1]
                    mlps[key].eval()
                    with torch.no_grad():
                        for x_prev, x_curr in iter_chunk_pairs(x_dev, stride):
                            B, H, S, _ = x_prev.shape
                            pred = mlps[key](x_prev.reshape(-1, D)).reshape(B, H, S, D)
                            r = x_curr - pred
                            if key not in mlp_train_rl2:
                                mlp_train_rl2[key] = []
                            mlp_train_rl2[key].append(rel_l2(r, x_curr))
                    del x_dev
            del layers_kv
            gc.collect()

        # ─ MLP eval pass (heldout)
        print("\n=== Pass 6: Evaluating tiny_mlp on heldout dumps ===")
        for fi, path in enumerate(heldout_dumps):
            layers_kv = _load_dump_all_layers(path)
            for l_idx, (k, v) in enumerate(layers_kv):
                for kv_key, x in (("K", k), ("V", v)):
                    key = (l_idx, kv_key)
                    x_dev = x.to(device)
                    D = x_dev.shape[-1]
                    mlps[key].eval()
                    with torch.no_grad():
                        for x_prev, x_curr in iter_chunk_pairs(x_dev, stride):
                            B, H, S, _ = x_prev.shape
                            pred = mlps[key](x_prev.reshape(-1, D)).reshape(B, H, S, D)
                            r = x_curr - pred
                            if key not in mlp_heldout_rl2:
                                mlp_heldout_rl2[key] = []
                            mlp_heldout_rl2[key].append(rel_l2(r, x_curr))
                    del x_dev
            del layers_kv
            gc.collect()

        # ─ Export MLP params
        for key, mlp in mlps.items():
            mlp_params_out[key] = mlp.to_param_dict()

    # ── Assemble result rows ──────────────────────────────────────────
    rows = []
    per_head_rows = []

    header = ("Layer | KV | Pred         | "
              "Train (mean/std/p95/max)                  | "
              "Held (mean/std/p95/max)")
    print()
    print(header)
    print("-" * len(header))

    affine_params_out = {"predictor_mode": "affine_channel",
                         "num_layers": num_layers, "num_heads": num_heads, "head_dim": head_dim,
                         "K": {}, "V": {}}
    mlp_params_dict   = {"predictor_mode": "tiny_mlp",
                         "num_layers": num_layers, "head_dim": head_dim, "hidden_dim": hidden_dim,
                         "K": {}, "V": {}}

    for l_idx in range(num_layers):
        for kv_key in ("K", "V"):
            key = (l_idx, kv_key)

            tr_id = _aggregate_stats(id_train_rl2[key])
            he_id = _aggregate_stats(id_heldout_rl2[key])
            rows.append((l_idx, kv_key, "identity", tr_id, he_id))
            print(f"  {l_idx:2d}  | {kv_key} | identity      | "
                  f"{_fmt_stats(tr_id)} | {_fmt_stats(he_id)}")

            alpha = alpha_all[key].cpu()
            beta  = beta_all[key].cpu()
            affine_params_out[kv_key][l_idx] = {"alpha": alpha, "beta": beta}

            tr_af = _aggregate_stats(af_train_rl2[key])
            he_af = _aggregate_stats(af_heldout_rl2[key])
            rows.append((l_idx, kv_key, "affine_channel", tr_af, he_af))
            print(f"  {l_idx:2d}  | {kv_key} | affine_channel | "
                  f"{_fmt_stats(tr_af)} | {_fmt_stats(he_af)}")

            if af_ph_num[key] is not None:
                ph = (af_ph_num[key] / af_ph_den[key].clamp_min(1e-8)).cpu()
            else:
                ph = torch.zeros(num_heads)
            per_head_rows.append((l_idx, kv_key, "affine_channel", ph))

            if not args.skip_mlp:
                tr_ml = _aggregate_stats(mlp_train_rl2.get(key, []))
                he_ml = _aggregate_stats(mlp_heldout_rl2.get(key, []))
                rows.append((l_idx, kv_key, "tiny_mlp", tr_ml, he_ml))
                print(f"  {l_idx:2d}  | {kv_key} | tiny_mlp      | "
                      f"{_fmt_stats(tr_ml)} | {_fmt_stats(he_ml)}")
                mlp_params_dict[kv_key][l_idx] = mlp_params_out[key]

    # ── Save fitted params ───────────────────────────────────────────
    affine_path = os.path.join(args.output_dir, "affine_channel_self_forcing_dmd.pt")
    torch.save(affine_params_out, affine_path)
    print(f"\nSaved affine params → {affine_path}")

    if not args.skip_mlp:
        mlp_path = os.path.join(args.output_dir, "tiny_mlp_self_forcing_dmd.pt")
        torch.save(mlp_params_dict, mlp_path)
        print(f"Saved tiny_mlp params → {mlp_path}")

    # ── Aggregate summary ────────────────────────────────────────────
    _write_report(args.report_path, rows, per_head_rows,
                  num_layers, num_heads, stride, train_dumps, heldout_dumps,
                  not args.skip_mlp)


def _write_report(report_path, rows, per_head_rows,
                  num_layers, num_heads, stride, train_dumps, heldout_dumps, has_mlp):
    lines = []
    lines += ["# HRQ Predictor Diagnostics Report",
              "",
              f"num_layers={num_layers}  num_heads={num_heads}  "
              f"stride={stride}",
              f"train_dumps={len(train_dumps)}  heldout_dumps={len(heldout_dumps)}",
              ""]

    lines += ["## Per-Layer Rel-L2 Summary",
              "",
              "Layer | KV | Predictor      | Train_mean | Heldout_mean | Gap",
              "------|----|-----------     |-----------|--------------|----"]
    for l_idx, kv_key, pred, tr, he in rows:
        gap = he["mean"] - tr["mean"]
        lines.append(f"{l_idx:5d} | {kv_key}  | {pred:<14s} | "
                     f"{tr['mean']:.4f}     | {he['mean']:.4f}       | {gap:+.4f}")

    lines += ["", "## Aggregated per-predictor (across all layers, held-out)", ""]
    for pred_name in (["identity", "affine_channel"] + (["tiny_mlp"] if has_mlp else [])):
        kv_vals: dict[str, list] = {"K": [], "V": []}
        for l_idx, kv_key, pred, tr, he in rows:
            if pred == pred_name:
                kv_vals[kv_key].append(he["mean"])
        for kv_key in ("K", "V"):
            v = torch.tensor(kv_vals[kv_key]) if kv_vals[kv_key] else torch.tensor([float("nan")])
            lines.append(f"  {pred_name:<16s} {kv_key}  mean={v.mean():.4f}  "
                         f"std={v.std():.4f}  p95={v.quantile(0.95):.4f}  max={v.max():.4f}")

    lines += ["", "## Per-Head Affine Rel-L2 (held-out, layer 0 and last layer)", ""]
    for l_idx, kv_key, pred, ph in per_head_rows:
        if l_idx in (0, num_layers - 1):
            ph_str = "  ".join(f"h{h}={ph[h]:.3f}" for h in range(len(ph)))
            lines.append(f"  L{l_idx:02d} {kv_key} {pred}: {ph_str}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written → {report_path}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def _parse_dump_list(s: str) -> List[str]:
    """Accept comma-separated paths or glob patterns."""
    paths = []
    for token in s.split(","):
        token = token.strip()
        expanded = glob.glob(token)
        paths.extend(expanded if expanded else [token])
    return sorted(set(paths))


def parse_args():
    p = argparse.ArgumentParser(description="HRQ predictor diagnostics")
    p.add_argument("--train_dumps", required=True,
                   help="Comma-separated or glob of train KV dump .pt files")
    p.add_argument("--heldout_dumps", default="",
                   help="Comma-separated or glob of held-out dump .pt files")
    p.add_argument("--predictor_stride", type=int, default=1560)
    p.add_argument("--output_dir", default="assets/trq_predictors")
    p.add_argument("--report_path",
                   default="results/selfforcing/vbench_eval_hrq_predictor/predictor_diagnostics.txt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mlp_hidden_mult", type=int, default=1,
                   help="MLP hidden_dim = head_dim * hidden_mult")
    p.add_argument("--mlp_epochs", type=int, default=30)
    p.add_argument("--mlp_lr", type=float, default=1e-3)
    p.add_argument("--mlp_batch_size", type=int, default=4096)
    p.add_argument("--mlp_patience", type=int, default=5)
    p.add_argument("--skip_mlp", action="store_true",
                   help="Skip MLP training (affine only)")
    p.add_argument("--verbose_mlp", action="store_true",
                   help="Print per-epoch MLP training progress")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve dump paths
    args.train_dumps = _parse_dump_list(args.train_dumps)
    args.heldout_dumps = _parse_dump_list(args.heldout_dumps) if args.heldout_dumps else []

    if not args.train_dumps:
        raise FileNotFoundError(
            f"No train dump files found. "
            "Run collect_kv_dumps.sh to generate them, then rerun this script."
        )

    run_analysis(args)


if __name__ == "__main__":
    main()
