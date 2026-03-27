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

import verl.utils.torch_functional as verl_F

__all__ = [
    "build_response_chunk_ids",
    "broadcast_chunk_scalar_to_tokens",
    "compute_chunked_advantages",
    "reduce_values_by_chunk",
]


# Keep the logged reduce mode numeric so it remains compatible with scalar-only trackers.
_CHUNK_REDUCE_MODE_TO_ID = {
    "first": 0.0,
    "last": 1.0,
    "mean": 2.0,
}


def _validate_2d_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"{name} must have shape [batch, response_length], got {tuple(tensor.shape)}.")


def _chunk_boundaries(valid_chunk_ids: torch.Tensor) -> torch.Tensor:
    if valid_chunk_ids.numel() == 0:
        return valid_chunk_ids.new_zeros(0)
    change_points = torch.nonzero(valid_chunk_ids[1:] != valid_chunk_ids[:-1], as_tuple=False).flatten() + 1
    return torch.cat(
        [
            valid_chunk_ids.new_zeros(1),
            change_points,
            valid_chunk_ids.new_tensor([valid_chunk_ids.numel()]),
        ]
    )


def _mean_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return values.float().mean().item()


def _std_or_zero(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return values.float().std(unbiased=False).item()


def _var_or_zero(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return values.float().var(unbiased=False).item()


def build_response_chunk_ids(
    response_mask: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Assign chunk ids over valid response tokens only."""
    _validate_2d_tensor(response_mask, "response_mask")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    valid_mask = response_mask.to(dtype=torch.bool)
    token_ranks = torch.cumsum(valid_mask.to(dtype=torch.long), dim=-1) - 1
    token_ranks = torch.where(valid_mask, token_ranks, torch.zeros_like(token_ranks))
    return torch.div(token_ranks, chunk_size, rounding_mode="floor")


def reduce_values_by_chunk(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_ids: torch.Tensor,
    reduce: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce token-aligned values to one baseline scalar per chunk.

    Returns:
        chunk_baselines: [B, Kmax]
        chunk_mask: [B, Kmax]
    """
    _validate_2d_tensor(values, "values")
    _validate_2d_tensor(response_mask, "response_mask")
    _validate_2d_tensor(chunk_ids, "chunk_ids")
    if values.shape != response_mask.shape or values.shape != chunk_ids.shape:
        raise ValueError(
            "values, response_mask, and chunk_ids must have identical shapes for chunk reduction, "
            f"got {tuple(values.shape)}, {tuple(response_mask.shape)}, and {tuple(chunk_ids.shape)}."
        )
    if reduce not in _CHUNK_REDUCE_MODE_TO_ID:
        raise ValueError(f"Unsupported chunk reduction mode: {reduce}.")

    valid_mask = response_mask.to(dtype=torch.bool)
    batch_size = values.size(0)
    num_chunks = torch.zeros(batch_size, dtype=torch.long, device=values.device)

    for batch_idx in range(batch_size):
        valid_chunk_ids = chunk_ids[batch_idx][valid_mask[batch_idx]]
        if valid_chunk_ids.numel() == 0:
            continue
        num_chunks[batch_idx] = valid_chunk_ids[-1] + 1

    max_num_chunks = int(num_chunks.max().item()) if batch_size > 0 else 0
    chunk_baselines = values.new_zeros((batch_size, max_num_chunks))
    chunk_mask = torch.zeros((batch_size, max_num_chunks), dtype=torch.bool, device=values.device)

    for batch_idx in range(batch_size):
        row_valid_mask = valid_mask[batch_idx]
        if not row_valid_mask.any():
            continue

        row_values = values[batch_idx][row_valid_mask]
        row_chunk_ids = chunk_ids[batch_idx][row_valid_mask]
        boundaries = _chunk_boundaries(row_chunk_ids)

        for chunk_idx in range(max(boundaries.numel() - 1, 0)):
            start = int(boundaries[chunk_idx].item())
            end = int(boundaries[chunk_idx + 1].item())
            chunk_values = row_values[start:end]

            if reduce == "first":
                baseline = chunk_values[0]
            elif reduce == "last":
                baseline = chunk_values[-1]
            else:
                baseline = chunk_values.mean()

            chunk_baselines[batch_idx, chunk_idx] = baseline
            chunk_mask[batch_idx, chunk_idx] = True

    return chunk_baselines, chunk_mask


def broadcast_chunk_scalar_to_tokens(
    chunk_values: torch.Tensor,
    chunk_mask: torch.Tensor,
    chunk_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast per-chunk scalars back onto token positions."""
    _validate_2d_tensor(chunk_values, "chunk_values")
    _validate_2d_tensor(chunk_mask, "chunk_mask")
    _validate_2d_tensor(chunk_ids, "chunk_ids")
    _validate_2d_tensor(response_mask, "response_mask")
    if chunk_values.shape != chunk_mask.shape:
        raise ValueError(
            "chunk_values and chunk_mask must have identical shapes for broadcasting, "
            f"got {tuple(chunk_values.shape)} and {tuple(chunk_mask.shape)}."
        )
    if chunk_values.size(0) != chunk_ids.size(0) or chunk_ids.shape != response_mask.shape:
        raise ValueError(
            "chunk_values batch size must match chunk_ids batch size and chunk_ids must match response_mask. "
            f"Got {tuple(chunk_values.shape)}, {tuple(chunk_ids.shape)}, and {tuple(response_mask.shape)}."
        )

    if chunk_values.size(1) == 0:
        return torch.zeros_like(response_mask, dtype=chunk_values.dtype, device=chunk_values.device)

    safe_chunk_ids = torch.clamp(chunk_ids, min=0, max=chunk_values.size(1) - 1)
    token_values = chunk_values.gather(dim=1, index=safe_chunk_ids)
    return token_values * response_mask.to(dtype=chunk_values.dtype)


def _compute_chunk_advantage_metrics(
    *,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_ids: torch.Tensor,
    chunk_baselines: torch.Tensor,
    chunk_mask: torch.Tensor,
    chunk_advantages: torch.Tensor,
    raw_token_advantages: torch.Tensor,
    chunk_size: int,
    chunk_reduce: str,
) -> dict[str, Any]:
    valid_mask = response_mask.to(dtype=torch.bool)
    valid_rows = valid_mask.any(dim=-1)
    response_lengths = valid_mask.sum(dim=-1)
    num_chunks = chunk_mask.sum(dim=-1)

    remainder = torch.remainder(response_lengths, chunk_size)
    last_chunk_lengths = torch.where(
        response_lengths > 0,
        torch.where(remainder == 0, torch.full_like(response_lengths, chunk_size), remainder),
        torch.zeros_like(response_lengths),
    )

    valid_chunk_counts = num_chunks[valid_rows].float()
    valid_last_chunk_lengths = last_chunk_lengths[valid_rows].float()
    flat_chunk_baselines = chunk_baselines[chunk_mask]
    flat_chunk_advantages = chunk_advantages[chunk_mask]
    flat_token_advantages = raw_token_advantages[valid_mask]

    within_chunk_stds: list[torch.Tensor] = []
    within_chunk_ranges: list[torch.Tensor] = []
    for batch_idx in range(values.size(0)):
        row_valid_mask = valid_mask[batch_idx]
        if not row_valid_mask.any():
            continue

        row_values = values[batch_idx][row_valid_mask].float()
        row_chunk_ids = chunk_ids[batch_idx][row_valid_mask]
        boundaries = _chunk_boundaries(row_chunk_ids)

        for chunk_idx in range(max(boundaries.numel() - 1, 0)):
            start = int(boundaries[chunk_idx].item())
            end = int(boundaries[chunk_idx + 1].item())
            chunk_values = row_values[start:end]
            within_chunk_stds.append(chunk_values.std(unbiased=False))
            within_chunk_ranges.append(chunk_values.max() - chunk_values.min())

    within_chunk_stds_tensor = (
        torch.stack(within_chunk_stds)
        if within_chunk_stds
        else values.new_zeros(0, dtype=torch.float32)
    )
    within_chunk_ranges_tensor = (
        torch.stack(within_chunk_ranges)
        if within_chunk_ranges
        else values.new_zeros(0, dtype=torch.float32)
    )

    return {
        "chunk/chunk_size": float(chunk_size),
        "chunk/num_chunks_mean": _mean_or_zero(valid_chunk_counts),
        "chunk/num_chunks_max": float(valid_chunk_counts.max().item()) if valid_chunk_counts.numel() > 0 else 0.0,
        "chunk/last_chunk_len_mean": _mean_or_zero(valid_last_chunk_lengths),
        "chunk/baseline_mean": _mean_or_zero(flat_chunk_baselines),
        "chunk/baseline_std": _std_or_zero(flat_chunk_baselines),
        "chunk/adv_mean": _mean_or_zero(flat_chunk_advantages),
        "chunk/adv_std": _std_or_zero(flat_chunk_advantages),
        "chunk/within_chunk_value_std_mean": _mean_or_zero(within_chunk_stds_tensor),
        "chunk/within_chunk_value_range_mean": _mean_or_zero(within_chunk_ranges_tensor),
        "chunk/token_adv_var_after_broadcast": _var_or_zero(flat_token_advantages),
        "chunk/reduce_mode": _CHUNK_REDUCE_MODE_TO_ID[chunk_reduce],
    }


def compute_chunked_advantages(
    values: torch.Tensor,
    rollout_returns: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_size: int,
    chunk_reduce: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute chunk-level rollout-return-minus-baseline actor advantages."""
    _validate_2d_tensor(values, "values")
    _validate_2d_tensor(response_mask, "response_mask")
    if values.shape != response_mask.shape:
        raise ValueError(
            "values and response_mask must have identical shapes for chunk advantages, "
            f"got {tuple(values.shape)} and {tuple(response_mask.shape)}."
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    if chunk_reduce not in _CHUNK_REDUCE_MODE_TO_ID:
        raise ValueError(f"Unsupported chunk reduction mode: {chunk_reduce}.")

    if rollout_returns.dim() == 2:
        if rollout_returns.shape[1] != 1:
            raise ValueError(
                "rollout_returns must have shape [batch] or [batch, 1] for chunk advantages, "
                f"got {tuple(rollout_returns.shape)}."
            )
        rollout_returns = rollout_returns.squeeze(-1)
    if rollout_returns.dim() != 1 or rollout_returns.shape[0] != values.shape[0]:
        raise ValueError(
            "rollout_returns must have shape [batch] for chunk advantages, "
            f"got {tuple(rollout_returns.shape)} for batch size {values.shape[0]}."
        )

    chunk_ids = build_response_chunk_ids(response_mask=response_mask, chunk_size=chunk_size)
    chunk_baselines, chunk_mask = reduce_values_by_chunk(
        values=values,
        response_mask=response_mask,
        chunk_ids=chunk_ids,
        reduce=chunk_reduce,
    )
    chunk_advantages = rollout_returns.unsqueeze(-1).to(dtype=values.dtype) - chunk_baselines
    raw_token_advantages = broadcast_chunk_scalar_to_tokens(
        chunk_values=chunk_advantages,
        chunk_mask=chunk_mask,
        chunk_ids=chunk_ids,
        response_mask=response_mask,
    )
    advantages = verl_F.masked_whiten(raw_token_advantages, response_mask)
    advantages = advantages * response_mask.to(dtype=advantages.dtype)

    metrics = _compute_chunk_advantage_metrics(
        values=values,
        response_mask=response_mask,
        chunk_ids=chunk_ids,
        chunk_baselines=chunk_baselines,
        chunk_mask=chunk_mask,
        chunk_advantages=chunk_advantages,
        raw_token_advantages=raw_token_advantages,
        chunk_size=chunk_size,
        chunk_reduce=chunk_reduce,
    )
    return advantages, metrics
