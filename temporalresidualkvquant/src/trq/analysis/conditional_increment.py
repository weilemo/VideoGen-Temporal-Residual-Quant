"""Held-out conditional predictive gain of current K beyond previous V."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


MODEL_NAMES = (
    "temporal",
    "cross",
    "joint",
    "wrong_space",
    "wrong_time",
    "wrong_prompt",
)
JOINT_MODEL_NAMES = MODEL_NAMES[2:]


@dataclass(frozen=True)
class AffineModel:
    """Per-head affine model with input and output dimensions allowed to differ."""

    weight: torch.Tensor  # [H, input_dim, output_dim]
    bias: torch.Tensor  # [H, output_dim]

    def squared_error(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_features(features, target)
        if features.shape[1] != self.weight.shape[0]:
            raise ValueError("feature head count does not match affine model")
        if features.shape[-1] != self.weight.shape[1]:
            raise ValueError("feature dimension does not match affine model")
        if target.shape[-1] != self.weight.shape[2]:
            raise ValueError("target dimension does not match affine model")
        prediction = (
            torch.einsum("bhsf,hfd->bhsd", features.float(), self.weight)
            + self.bias[None, :, None, :]
        )
        return (target.float() - prediction).double().square().sum(dim=(0, 2, 3))


class AffineAccumulator:
    """Streaming float64 sufficient statistics for per-head ridge regression."""

    def __init__(self, num_heads: int, input_dim: int, output_dim: int) -> None:
        if min(num_heads, input_dim, output_dim) <= 0:
            raise ValueError("affine dimensions must be positive")
        self.num_heads = int(num_heads)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.xtx = torch.zeros(
            num_heads, input_dim + 1, input_dim + 1, dtype=torch.float64
        )
        self.xty = torch.zeros(
            num_heads, input_dim + 1, output_dim, dtype=torch.float64
        )
        self.samples = 0

    def update(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_chunk: int = 4096,
    ) -> None:
        _validate_features(features, target)
        if sample_chunk <= 0:
            raise ValueError("sample_chunk must be positive")
        if features.shape[1] != self.num_heads:
            raise ValueError("feature head count changed within affine fit")
        if features.shape[-1] != self.input_dim or target.shape[-1] != self.output_dim:
            raise ValueError("feature geometry changed within affine fit")
        x = features.permute(1, 0, 2, 3).reshape(
            self.num_heads, -1, self.input_dim
        )
        y = target.permute(1, 0, 2, 3).reshape(
            self.num_heads, -1, self.output_dim
        )
        for start in range(0, x.shape[1], sample_chunk):
            x_chunk = x[:, start : start + sample_chunk].double()
            y_chunk = y[:, start : start + sample_chunk].double()
            ones = torch.ones(
                self.num_heads, x_chunk.shape[1], 1, dtype=torch.float64
            )
            design = torch.cat((x_chunk, ones), dim=-1)
            self.xtx += torch.matmul(design.transpose(1, 2), design)
            self.xty += torch.matmul(design.transpose(1, 2), y_chunk)
        self.samples += int(x.shape[1])

    def fit(self, ridge: float) -> AffineModel:
        if self.samples == 0:
            raise ValueError("cannot fit affine model without samples")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        regularizer = torch.zeros_like(self.xtx)
        scale = self.xtx[:, : self.input_dim, : self.input_dim].diagonal(
            dim1=1, dim2=2
        ).mean(dim=1)
        regularizer[:, : self.input_dim, : self.input_dim] = (
            torch.eye(self.input_dim, dtype=torch.float64)[None]
            * (scale * float(ridge)).view(-1, 1, 1)
        )
        solution = torch.linalg.solve(self.xtx + regularizer, self.xty)
        return AffineModel(
            weight=solution[:, : self.input_dim].float().contiguous(),
            bias=solution[:, self.input_dim].float().contiguous(),
        )


def conditional_feature_batches(
    key: torch.Tensor,
    value: torch.Tensor,
    donor_key: torch.Tensor,
    *,
    unit_size: int,
    wrong_space_shift: int = 1,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Build aligned temporal, joint, and matched-control feature tensors."""
    _validate_kv(key, value)
    if donor_key.ndim != 4 or not donor_key.is_floating_point():
        raise ValueError("donor K must be a floating-point BHSD tensor")
    if (donor_key.shape[0], donor_key.shape[1], donor_key.shape[3]) != (
        key.shape[0],
        key.shape[1],
        key.shape[3],
    ):
        raise ValueError("donor K geometry does not match target K")
    if unit_size <= 1:
        raise ValueError("unit_size must be greater than one for wrong-space control")
    if wrong_space_shift == 0 or abs(wrong_space_shift) >= unit_size:
        raise ValueError("wrong_space_shift must be non-zero and smaller than unit_size")

    units = min(key.shape[2] // unit_size, donor_key.shape[2] // unit_size)
    if units < 2:
        raise ValueError("target and donor must contain at least two full units")
    token_count = (units - 1) * unit_size

    key_units = key[:, :, : units * unit_size].reshape(
        key.shape[0], key.shape[1], units, unit_size, key.shape[3]
    )
    value_units = value[:, :, : units * unit_size].reshape(
        value.shape[0], value.shape[1], units, unit_size, value.shape[3]
    )
    donor_units = donor_key[:, :, : units * unit_size].reshape(
        donor_key.shape[0], donor_key.shape[1], units, unit_size, donor_key.shape[3]
    )

    previous_value = value_units[:, :, :-1].reshape(
        value.shape[0], value.shape[1], token_count, value.shape[3]
    )
    current_value = value_units[:, :, 1:].reshape(
        value.shape[0], value.shape[1], token_count, value.shape[3]
    )
    previous_key = key_units[:, :, :-1].reshape(
        key.shape[0], key.shape[1], token_count, key.shape[3]
    )
    current_key_units = key_units[:, :, 1:]
    current_key = current_key_units.reshape(
        key.shape[0], key.shape[1], token_count, key.shape[3]
    )
    wrong_space_key = torch.roll(
        current_key_units, shifts=wrong_space_shift, dims=3
    ).reshape(key.shape[0], key.shape[1], token_count, key.shape[3])
    wrong_prompt_key = donor_units[:, :, 1:].reshape(
        donor_key.shape[0], donor_key.shape[1], token_count, donor_key.shape[3]
    )

    features = {
        "temporal": previous_value,
        "cross": current_key,
        "joint": torch.cat((previous_value, current_key), dim=-1),
        "wrong_space": torch.cat((previous_value, wrong_space_key), dim=-1),
        "wrong_time": torch.cat((previous_value, previous_key), dim=-1),
        "wrong_prompt": torch.cat((previous_value, wrong_prompt_key), dim=-1),
    }
    return features, current_value, units - 1


def fit_conditional_models(
    records: Iterable[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]],
    *,
    ridge: float,
    wrong_space_shift: int = 1,
    sample_chunk: int = 4096,
) -> dict[int, dict[str, AffineModel]]:
    """Fit every pre-registered model per layer from calibration records."""
    accumulators: dict[int, dict[str, AffineAccumulator]] = {}
    for layer, key, value, donor_key, unit_size in records:
        features, target, _ = conditional_feature_batches(
            key,
            value,
            donor_key,
            unit_size=unit_size,
            wrong_space_shift=wrong_space_shift,
        )
        layer_accumulators = accumulators.setdefault(layer, {})
        for name, feature in features.items():
            accumulator = layer_accumulators.setdefault(
                name,
                AffineAccumulator(
                    num_heads=feature.shape[1],
                    input_dim=feature.shape[-1],
                    output_dim=target.shape[-1],
                ),
            )
            accumulator.update(feature, target, sample_chunk=sample_chunk)
    if not accumulators:
        raise ValueError("no calibration records were supplied")
    return {
        layer: {
            name: accumulator.fit(ridge)
            for name, accumulator in sorted(layer_accumulators.items())
        }
        for layer, layer_accumulators in sorted(accumulators.items())
    }


def evaluate_conditional_record(
    key: torch.Tensor,
    value: torch.Tensor,
    donor_key: torch.Tensor,
    models: dict[str, AffineModel],
    *,
    unit_size: int,
    wrong_space_shift: int = 1,
) -> list[dict[str, float | int | str]]:
    """Return one SSE/MSE row per method and head for one layer record."""
    features, target, pairs = conditional_feature_batches(
        key,
        value,
        donor_key,
        unit_size=unit_size,
        wrong_space_shift=wrong_space_shift,
    )
    if set(models) != set(MODEL_NAMES):
        raise ValueError("fitted model set does not match the pre-registered matrix")
    values_per_head = int(target.shape[0] * target.shape[2] * target.shape[3])
    rows: list[dict[str, float | int | str]] = []
    for name in MODEL_NAMES:
        sse = models[name].squared_error(features[name], target)
        for head in range(target.shape[1]):
            value_sse = float(sse[head].item())
            rows.append(
                {
                    "method": name,
                    "head": head,
                    "pairs": pairs,
                    "values": values_per_head,
                    "sse": value_sse,
                    "mse": value_sse / values_per_head,
                }
            )
    return rows


def aggregate_prompt_rows(
    rows: Iterable[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    """Aggregate layer/head SSE and derive prompt-level partial R-squared."""
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        prompt_id = str(row["prompt_id"])
        method = str(row["method"])
        target = grouped.setdefault(prompt_id, {name: 0.0 for name in MODEL_NAMES})
        target[method] += float(row["sse"])
    result: list[dict[str, float | str]] = []
    for prompt_id, values in sorted(grouped.items()):
        temporal = values["temporal"]
        if temporal <= 0:
            raise ValueError(f"temporal SSE is not positive for prompt {prompt_id}")
        row: dict[str, float | str] = {
            "prompt_id": prompt_id,
            **{f"{name}_sse": values[name] for name in MODEL_NAMES},
            "cross_over_temporal": values["cross"] / temporal,
        }
        for name in JOINT_MODEL_NAMES:
            row[f"partial_r2_{name}"] = (temporal - values[name]) / temporal
        for control in JOINT_MODEL_NAMES[1:]:
            row[f"joint_minus_{control}"] = (
                float(row["partial_r2_joint"]) - float(row[f"partial_r2_{control}"])
            )
        result.append(row)
    return result


def _validate_features(features: torch.Tensor, target: torch.Tensor) -> None:
    if features.ndim != 4 or target.ndim != 4:
        raise ValueError("features and target must use BHSD-like rank-4 layout")
    if features.shape[:3] != target.shape[:3]:
        raise ValueError("features and target sample geometry must match")
    if not features.is_floating_point() or not target.is_floating_point():
        raise ValueError("features and target must be floating point")
    if not torch.isfinite(features).all() or not torch.isfinite(target).all():
        raise ValueError("features or target contains NaN/Inf")


def _validate_kv(key: torch.Tensor, value: torch.Tensor) -> None:
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("K/V must be matching BHSD tensors")
    if not key.is_floating_point() or not value.is_floating_point():
        raise ValueError("K/V must be floating point")
    if not torch.isfinite(key).all() or not torch.isfinite(value).all():
        raise ValueError("K/V contains NaN/Inf")


__all__ = [
    "AffineAccumulator",
    "AffineModel",
    "JOINT_MODEL_NAMES",
    "MODEL_NAMES",
    "aggregate_prompt_rows",
    "conditional_feature_batches",
    "evaluate_conditional_record",
    "fit_conditional_models",
]
