"""Prompt-disjoint analysis for K-conditioned temporal V innovation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_QUANTILE_MAX_SAMPLES = 262_144


@dataclass(frozen=True)
class CrossKVModel:
    """Per-head affine map ``V = K W + b``."""

    weight: torch.Tensor  # [H, D, D]
    bias: torch.Tensor  # [H, D]

    def predict(self, key: torch.Tensor) -> torch.Tensor:
        if key.ndim != 4:
            raise ValueError(f"expected K in BHSD layout, got {tuple(key.shape)}")
        if tuple(key.shape[1::2]) != (self.weight.shape[0], self.weight.shape[1]):
            raise ValueError(
                f"K geometry {tuple(key.shape)} does not match predictor "
                f"{tuple(self.weight.shape)}"
            )
        return torch.einsum("bhsd,hde->bhse", key.float(), self.weight) + self.bias[None, :, None, :]


class CrossKVAccumulator:
    """Streaming float64 sufficient statistics for a per-head affine fit."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.xtx = torch.zeros(num_heads, head_dim + 1, head_dim + 1, dtype=torch.float64)
        self.xty = torch.zeros(num_heads, head_dim + 1, head_dim, dtype=torch.float64)
        self.samples = 0

    def update(self, key: torch.Tensor, value: torch.Tensor) -> None:
        _validate_pair(key, value)
        if (key.shape[1], key.shape[-1]) != (self.num_heads, self.head_dim):
            raise ValueError("K/V geometry changed within one layer fit")
        x = key.permute(1, 0, 2, 3).reshape(self.num_heads, -1, self.head_dim).double()
        y = value.permute(1, 0, 2, 3).reshape(self.num_heads, -1, self.head_dim).double()
        ones = torch.ones(self.num_heads, x.shape[1], 1, dtype=torch.float64)
        design = torch.cat((x, ones), dim=-1)
        self.xtx += torch.matmul(design.transpose(1, 2), design)
        self.xty += torch.matmul(design.transpose(1, 2), y)
        self.samples += int(x.shape[1])

    def fit(self, ridge: float) -> CrossKVModel:
        if self.samples == 0:
            raise ValueError("cannot fit Cross-KV without samples")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        regularizer = torch.zeros_like(self.xtx)
        scale = self.xtx[:, : self.head_dim, : self.head_dim].diagonal(dim1=1, dim2=2).mean(dim=1)
        regularizer[:, : self.head_dim, : self.head_dim] = torch.eye(
            self.head_dim, dtype=torch.float64
        )[None] * (scale * float(ridge)).view(-1, 1, 1)
        solution = torch.linalg.solve(self.xtx + regularizer, self.xty)
        return CrossKVModel(
            weight=solution[:, : self.head_dim].float().contiguous(),
            bias=solution[:, self.head_dim].float().contiguous(),
        )


class GammaAccumulator:
    """Streaming ridge estimate of per-head/per-channel AR(1) coefficients."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self.numerator = torch.zeros(num_heads, head_dim, dtype=torch.float64)
        self.denominator = torch.zeros(num_heads, head_dim, dtype=torch.float64)
        self.pairs = 0

    def update(self, innovations: list[torch.Tensor]) -> None:
        for previous, current in zip(innovations, innovations[1:]):
            previous = previous.double()
            current = current.double()
            self.numerator += (previous * current).sum(dim=(0, 2))
            self.denominator += previous.square().sum(dim=(0, 2))
            self.pairs += int(previous.shape[0] * previous.shape[2])

    def fit(self, ridge: float, rho: float) -> torch.Tensor:
        if self.pairs == 0:
            raise ValueError("cannot fit innovation gamma without adjacent full units")
        if ridge < 0:
            raise ValueError("gamma ridge must be non-negative")
        if not 0 < rho < 1:
            raise ValueError("rho must satisfy 0 < rho < 1")
        scale = float(self.denominator.mean().item())
        gamma = self.numerator / (self.denominator + float(ridge) * max(scale, 1e-12))
        return gamma.clamp(-float(rho), float(rho)).float().contiguous()


def full_units(tensor: torch.Tensor, unit_size: int) -> list[torch.Tensor]:
    if tensor.ndim != 4:
        raise ValueError(f"expected BHSD tensor, got {tuple(tensor.shape)}")
    if unit_size <= 0:
        raise ValueError("unit_size must be positive")
    count = int(tensor.shape[2] // unit_size)
    return [tensor[:, :, index * unit_size : (index + 1) * unit_size] for index in range(count)]


def innovation_units(
    key: torch.Tensor,
    value: torch.Tensor,
    model: CrossKVModel,
    unit_size: int,
) -> list[torch.Tensor]:
    _validate_pair(key, value)
    return [v_unit - model.predict(k_unit) for k_unit, v_unit in zip(full_units(key, unit_size), full_units(value, unit_size))]


def evaluate_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    model: CrossKVModel,
    *,
    unit_size: int,
    gamma: torch.Tensor | None = None,
    shuffled_innovations: list[torch.Tensor] | None = None,
    shuffle_seed: int = 0,
    quantile_max_samples: int = DEFAULT_QUANTILE_MAX_SAMPLES,
) -> list[dict[str, float | int]]:
    """Return one row per head using only adjacent full units.

    The preferred shuffled control supplies innovations from another prompt.
    The within-prompt fallback exists for unit tests and interactive analysis,
    but the experiment CLI always uses a prompt-disjoint donor.
    """
    _validate_pair(key, value)
    if quantile_max_samples <= 0:
        raise ValueError("quantile_max_samples must be positive")
    k_units = full_units(key, unit_size)
    v_units = full_units(value, unit_size)
    if len(k_units) < 2:
        raise ValueError(
            f"sequence length {key.shape[2]} provides fewer than two full units of {unit_size}"
        )
    cross_units = [model.predict(unit) for unit in k_units]
    z_units = [target - prediction for target, prediction in zip(v_units, cross_units)]
    pair_count = len(k_units) - 1

    if shuffled_innovations is not None:
        if len(shuffled_innovations) < pair_count:
            raise ValueError("shuffled donor has fewer full units than the evaluated prompt")
        shuffled_donors = list(shuffled_innovations[:pair_count])
        for donor, expected in zip(shuffled_donors, z_units[:-1]):
            if donor.shape != expected.shape:
                raise ValueError(
                    f"shuffled donor geometry {tuple(donor.shape)} does not match "
                    f"{tuple(expected.shape)}"
                )
    else:
        shuffled_donors = list(z_units[:-1])
        if pair_count > 1:
            generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
            shift = int(torch.randint(1, pair_count, (1,), generator=generator).item())
            shuffled_donors = shuffled_donors[-shift:] + shuffled_donors[:-shift]
        else:
            shuffled_donors = [torch.flip(shuffled_donors[0], dims=(2,))]

    num_heads = key.shape[1]
    accumulators = {
        name: torch.zeros(num_heads, dtype=torch.float64)
        for name in ("target", "temporal", "cross", "oracle", "shuffled")
    }
    distributions: dict[str, list[torch.Tensor]] = {"raw": [], "cross_residual": [], "innovation": []}
    gamma_view = None
    if gamma is not None:
        if tuple(gamma.shape) != (key.shape[1], key.shape[-1]):
            raise ValueError(f"gamma must be [H,D], got {tuple(gamma.shape)}")
        gamma_view = gamma[None, :, None, :].float()

    for index in range(1, len(k_units)):
        target = v_units[index].float()
        temporal = v_units[index - 1].float()
        cross = cross_units[index]
        cross_residual = z_units[index]
        if gamma_view is None:
            oracle = cross
            shuffled = cross
            innovation = cross_residual
        else:
            oracle = cross + gamma_view * z_units[index - 1]
            shuffled = cross + gamma_view * shuffled_donors[index - 1]
            innovation = target - oracle
        for name, prediction in (
            ("temporal", temporal),
            ("cross", cross),
            ("oracle", oracle),
            ("shuffled", shuffled),
        ):
            accumulators[name] += (target - prediction).double().square().sum(dim=(0, 2, 3))
        accumulators["target"] += target.double().square().sum(dim=(0, 2, 3))
        distributions["raw"].append(target.detach().cpu())
        distributions["cross_residual"].append(cross_residual.detach().cpu())
        distributions["innovation"].append(innovation.detach().cpu())

    values_per_head = int(pair_count * key.shape[0] * unit_size * key.shape[-1])
    rows: list[dict[str, float | int]] = []
    for head in range(num_heads):
        row: dict[str, float | int] = {"head": head, "pairs": pair_count, "values": values_per_head}
        for name in ("temporal", "cross", "oracle", "shuffled"):
            row[f"{name}_mse"] = float(accumulators[name][head].item() / values_per_head)
            row[f"{name}_nmse"] = _ratio(accumulators[name][head], accumulators["target"][head])
        row["cross_over_temporal"] = _ratio(accumulators["cross"][head], accumulators["temporal"][head])
        row["oracle_over_cross"] = _ratio(accumulators["oracle"][head], accumulators["cross"][head])
        row["shuffled_over_cross"] = _ratio(accumulators["shuffled"][head], accumulators["cross"][head])
        for distribution_index, (name, chunks) in enumerate(distributions.items()):
            stats = distribution_stats(
                chunks,
                head,
                quantile=0.99,
                max_quantile_samples=quantile_max_samples,
                seed=shuffle_seed + 104_729 * distribution_index + head,
            )
            row[f"{name}_std"] = stats["std"]
            row[f"{name}_p99_abs"] = stats["quantile_abs"]
            row[f"{name}_p99_samples"] = stats["quantile_samples"]
        rows.append(row)
    return rows


def distribution_stats(
    chunks: Iterable[torch.Tensor],
    head: int,
    *,
    quantile: float,
    max_quantile_samples: int = DEFAULT_QUANTILE_MAX_SAMPLES,
    seed: int = 0,
) -> dict[str, float | int]:
    """Compute exact standard deviation and a bounded sampled abs quantile.

    The gate metrics remain full-data reductions. Only the diagnostic quantile
    is sampled, which avoids materializing and sorting a multi-billion-element
    tensor for each head.
    """
    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("distribution chunks must be non-empty")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    if max_quantile_samples <= 0:
        raise ValueError("max_quantile_samples must be positive")

    flattened: list[torch.Tensor] = []
    sizes: list[int] = []
    count = 0
    mean = 0.0
    m2 = 0.0
    for chunk in chunk_list:
        if chunk.ndim != 4 or not 0 <= head < chunk.shape[1]:
            raise ValueError(
                f"invalid distribution chunk/head: shape={tuple(chunk.shape)}, head={head}"
            )
        values = chunk[:, head].reshape(-1).detach().to(device="cpu", dtype=torch.float32)
        size = int(values.numel())
        if size == 0:
            continue
        chunk_variance, chunk_mean = torch.var_mean(values, correction=0)
        chunk_mean_value = float(chunk_mean.item())
        chunk_m2 = float(chunk_variance.item()) * size
        if count == 0:
            mean = chunk_mean_value
            m2 = chunk_m2
        else:
            combined = count + size
            delta = chunk_mean_value - mean
            m2 += chunk_m2 + delta * delta * count * size / combined
            mean += delta * size / combined
        count += size
        flattened.append(values)
        sizes.append(size)

    if count == 0:
        raise ValueError("distribution chunks contain no values")

    sample_total = min(count, int(max_quantile_samples))
    quotas = _proportional_quotas(sizes, sample_total)
    samples: list[torch.Tensor] = []
    for index, (values, quota) in enumerate(zip(flattened, quotas)):
        if quota <= 0:
            continue
        if quota >= values.numel():
            sample = values
        else:
            generator = torch.Generator(device="cpu").manual_seed(
                int(seed) + 1_000_003 * (index + 1)
            )
            indices = torch.randint(values.numel(), (quota,), generator=generator)
            sample = values[indices]
        samples.append(sample.abs())

    quantile_values = torch.cat(samples)
    return {
        "count": count,
        "std": float(max(m2 / count, 0.0) ** 0.5),
        "quantile_abs": float(torch.quantile(quantile_values, quantile).item()),
        "quantile_samples": int(quantile_values.numel()),
    }


def _proportional_quotas(sizes: list[int], total: int) -> list[int]:
    if total <= 0 or not sizes or sum(sizes) <= 0:
        raise ValueError("quota inputs must be positive")
    population = sum(sizes)
    numerators = [total * size for size in sizes]
    quotas = [numerator // population for numerator in numerators]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(sizes)),
        key=lambda index: numerators[index] % population,
        reverse=True,
    )
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas


def prompt_aggregates(rows: Iterable[dict[str, Any]]) -> list[dict[str, float | str]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        prompt_id = str(row["prompt_id"])
        target = grouped.setdefault(
            prompt_id,
            {"cross": 0.0, "temporal": 0.0, "oracle": 0.0, "shuffled": 0.0, "values": 0.0},
        )
        count = float(row["values"])
        target["values"] += count
        for name in ("cross", "temporal", "oracle", "shuffled"):
            target[name] += float(row[f"{name}_mse"]) * count
    result = []
    for prompt_id, sums in sorted(grouped.items()):
        result.append(
            {
                "prompt_id": prompt_id,
                "cross_over_temporal": _ratio(sums["cross"], sums["temporal"]),
                "oracle_over_cross": _ratio(sums["oracle"], sums["cross"]),
                "shuffled_over_cross": _ratio(sums["shuffled"], sums["cross"]),
            }
        )
    return result


def bootstrap_median_ci(
    values: Iterable[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(resamples, array.size))
    medians = np.median(array[indices], axis=1)
    return {
        "prompts": int(array.size),
        "median": float(np.median(array)),
        "ci95_lower": float(np.quantile(medians, 0.025)),
        "ci95_upper": float(np.quantile(medians, 0.975)),
    }


def _validate_pair(key: torch.Tensor, value: torch.Tensor) -> None:
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError(f"expected matching BHSD K/V, got {tuple(key.shape)} and {tuple(value.shape)}")
    if not key.is_floating_point() or not value.is_floating_point():
        raise ValueError("K/V must be floating point")
    if not torch.isfinite(key).all() or not torch.isfinite(value).all():
        raise ValueError("K/V contains NaN or Inf")


def _ratio(numerator: torch.Tensor | float, denominator: torch.Tensor | float) -> float:
    numerator_value = float(numerator.item()) if isinstance(numerator, torch.Tensor) else float(numerator)
    denominator_value = float(denominator.item()) if isinstance(denominator, torch.Tensor) else float(denominator)
    if denominator_value <= 0:
        return float("nan")
    return numerator_value / denominator_value
