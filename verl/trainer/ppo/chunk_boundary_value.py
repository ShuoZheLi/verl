# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as torch_F

__all__ = [
    "build_chunk_boundary_mask",
    "build_chunk_boundary_training_tensors",
    "build_response_state_mask",
    "compute_chunk_boundary_metric_sums",
    "compute_chunk_boundary_prediction_metrics",
    "finalize_chunk_boundary_metric_sums",
]


def _validate_2d_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"{name} must have shape [batch, seq_len], got {tuple(tensor.shape)}.")


def _coerce_rollout_scalars(
    rollout_scalars: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if rollout_scalars.dim() == 2:
        if rollout_scalars.shape[1] != 1:
            raise ValueError(f"{name} must have shape [batch] or [batch, 1], got {tuple(rollout_scalars.shape)}.")
        rollout_scalars = rollout_scalars.squeeze(-1)
    if rollout_scalars.dim() != 1 or rollout_scalars.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have shape [batch] for batch size {batch_size}, got {tuple(rollout_scalars.shape)}."
        )
    return rollout_scalars.to(device=device, dtype=dtype)


def build_response_state_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """Return a mask over prefix states 0..T for each response trajectory."""
    _validate_2d_tensor(response_mask, "response_mask")
    valid_lengths = response_mask.to(dtype=torch.bool).sum(dim=-1, dtype=torch.long)
    state_positions = torch.arange(response_mask.shape[1] + 1, device=response_mask.device, dtype=torch.long)
    return state_positions.unsqueeze(0) <= valid_lengths.unsqueeze(-1)


def build_chunk_boundary_mask(
    response_mask: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Mark prompt-only, every chunk boundary, and the final response boundary."""
    _validate_2d_tensor(response_mask, "response_mask")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    state_mask = build_response_state_mask(response_mask)
    state_positions = torch.arange(state_mask.shape[1], device=response_mask.device, dtype=torch.long)
    boundary_mask = state_mask & (state_positions.unsqueeze(0) % chunk_size == 0)

    response_lengths = response_mask.to(dtype=torch.bool).sum(dim=-1, dtype=torch.long)
    batch_indices = torch.arange(response_mask.shape[0], device=response_mask.device)
    boundary_mask[batch_indices, response_lengths] = True
    return boundary_mask


def build_chunk_boundary_training_tensors(
    response_mask: torch.Tensor,
    rollout_targets: torch.Tensor,
    *,
    chunk_size: int,
    sample_weight: torch.Tensor | None = None,
    uniform_per_state_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build chunk-boundary supervision tensors from rollout-level targets."""
    _validate_2d_tensor(response_mask, "response_mask")
    batch_size = response_mask.shape[0]
    boundary_mask = build_chunk_boundary_mask(response_mask=response_mask, chunk_size=chunk_size)

    targets_1d = _coerce_rollout_scalars(
        rollout_targets,
        batch_size=batch_size,
        device=response_mask.device,
        dtype=torch.float32,
        name="rollout_targets",
    )
    boundary_targets = targets_1d.unsqueeze(-1).expand_as(boundary_mask).to(dtype=torch.float32)

    if sample_weight is None:
        sample_weight_1d = torch.ones(batch_size, device=response_mask.device, dtype=torch.float32)
    else:
        sample_weight_1d = _coerce_rollout_scalars(
            sample_weight,
            batch_size=batch_size,
            device=response_mask.device,
            dtype=torch.float32,
            name="sample_weight",
        )
    if torch.any(sample_weight_1d < 0):
        raise ValueError(
            "sample_weight must be non-negative for chunk-boundary supervision, "
            f"got minimum {sample_weight_1d.min().item():.6f}."
        )

    boundary_counts = boundary_mask.sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0)
    if uniform_per_state_weight:
        boundary_weights = boundary_mask.to(dtype=torch.float32) * sample_weight_1d.unsqueeze(-1)
    else:
        per_rollout_weight = sample_weight_1d / boundary_counts
        boundary_weights = boundary_mask.to(dtype=torch.float32) * per_rollout_weight.unsqueeze(-1)

    effective_rollout_mask = sample_weight_1d > 0
    response_lengths = response_mask.to(dtype=torch.bool).sum(dim=-1).to(dtype=torch.float32)
    effective_response_lengths = response_lengths[effective_rollout_mask]
    effective_boundary_counts = boundary_mask.sum(dim=-1)[effective_rollout_mask].to(dtype=torch.float32)

    valid_partial_mask = effective_response_lengths > 0
    if valid_partial_mask.any():
        effective_response_lengths_long = effective_response_lengths.to(dtype=torch.long)
        partial_fraction = (
            (effective_response_lengths_long % chunk_size != 0).to(dtype=torch.float32)[valid_partial_mask].mean().item()
        )
    else:
        partial_fraction = 0.0

    effective_rollout_weight_sum = sample_weight_1d[effective_rollout_mask].sum()
    if effective_rollout_weight_sum.item() > 0:
        mean_rollout_target = (
            (targets_1d[effective_rollout_mask] * sample_weight_1d[effective_rollout_mask]).sum()
            / effective_rollout_weight_sum
        ).item()
    else:
        mean_rollout_target = 0.0

    metrics = {
        "chunk_boundary/chunk_size": float(chunk_size),
        "chunk_boundary/rollouts_in_batch": float(batch_size),
        "chunk_boundary/effective_rollouts": float(effective_rollout_mask.sum().item()),
        "chunk_boundary/boundary_states_total": float((boundary_weights > 0).sum().item()),
        "chunk_boundary/boundaries_per_rollout_mean": (
            effective_boundary_counts.mean().item() if effective_boundary_counts.numel() > 0 else 0.0
        ),
        "chunk_boundary/mean_rollout_reward": mean_rollout_target,
        "chunk_boundary/mean_response_length": (
            effective_response_lengths.mean().item() if effective_response_lengths.numel() > 0 else 0.0
        ),
        "chunk_boundary/final_partial_chunk_fraction": partial_fraction,
        "chunk_boundary/uniform_per_state_weight": float(uniform_per_state_weight),
    }
    return boundary_mask, boundary_targets, boundary_weights, metrics


def compute_chunk_boundary_metric_sums(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_weights: torch.Tensor,
    *,
    loss_type: str,
) -> dict[str, torch.Tensor]:
    """Accumulate weighted chunk-boundary diagnostic sums."""
    _validate_2d_tensor(predictions, "predictions")
    _validate_2d_tensor(targets, "targets")
    _validate_2d_tensor(boundary_mask.to(dtype=torch.float32), "boundary_mask")
    _validate_2d_tensor(boundary_weights, "boundary_weights")
    if predictions.shape != targets.shape or predictions.shape != boundary_mask.shape or predictions.shape != boundary_weights.shape:
        raise ValueError(
            "predictions, targets, boundary_mask, and boundary_weights must share the same shape, "
            f"got {tuple(predictions.shape)}, {tuple(targets.shape)}, {tuple(boundary_mask.shape)}, "
            f"and {tuple(boundary_weights.shape)}."
        )
    if loss_type not in {"bce", "mse"}:
        raise ValueError(f"Unsupported chunk-boundary loss_type: {loss_type}.")

    predictions = predictions.float()
    targets = targets.float()
    effective_weights = boundary_weights.float() * boundary_mask.to(dtype=torch.float32)
    state_count = (effective_weights > 0).sum().to(dtype=torch.float32)
    weight_sum = effective_weights.sum()

    positive_weights = effective_weights * (targets >= 0.5).to(dtype=torch.float32)
    negative_weights = effective_weights * (targets < 0.5).to(dtype=torch.float32)

    metric_sums = {
        "weight_sum": weight_sum,
        "state_count": state_count,
        "prediction_sum": (predictions * effective_weights).sum(),
        "positive_prediction_sum": (predictions * positive_weights).sum(),
        "positive_weight_sum": positive_weights.sum(),
        "negative_prediction_sum": (predictions * negative_weights).sum(),
        "negative_weight_sum": negative_weights.sum(),
        "mse_sum": (((predictions - targets) ** 2) * effective_weights).sum(),
    }
    if loss_type == "bce":
        clipped_predictions = predictions.clamp(1e-6, 1.0 - 1e-6)
        metric_sums["bce_sum"] = (
            torch_F.binary_cross_entropy(clipped_predictions, targets, reduction="none") * effective_weights
        ).sum()
    return metric_sums


def finalize_chunk_boundary_metric_sums(
    metric_sums: dict[str, torch.Tensor],
    *,
    prefix: str = "chunk_boundary",
) -> dict[str, float]:
    """Convert chunk-boundary diagnostic sums into scalar metrics."""

    def _safe_ratio(numerator_key: str, denominator_key: str) -> float:
        denominator = metric_sums[denominator_key]
        if denominator.item() <= 0:
            return 0.0
        return (metric_sums[numerator_key] / denominator).item()

    metrics = {
        f"{prefix}/prediction_mean": _safe_ratio("prediction_sum", "weight_sum"),
        f"{prefix}/prediction_mean_positive": _safe_ratio("positive_prediction_sum", "positive_weight_sum"),
        f"{prefix}/prediction_mean_negative": _safe_ratio("negative_prediction_sum", "negative_weight_sum"),
        f"{prefix}/mse": _safe_ratio("mse_sum", "weight_sum"),
        f"{prefix}/weight_sum": metric_sums["weight_sum"].item(),
        f"{prefix}/state_count": metric_sums["state_count"].item(),
    }
    if "bce_sum" in metric_sums:
        metrics[f"{prefix}/bce"] = _safe_ratio("bce_sum", "weight_sum")
    return metrics


def compute_chunk_boundary_prediction_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_weights: torch.Tensor,
    *,
    loss_type: str,
    prefix: str = "chunk_boundary",
) -> dict[str, float]:
    metric_sums = compute_chunk_boundary_metric_sums(
        predictions=predictions,
        targets=targets,
        boundary_mask=boundary_mask,
        boundary_weights=boundary_weights,
        loss_type=loss_type,
    )
    return finalize_chunk_boundary_metric_sums(metric_sums, prefix=prefix)
