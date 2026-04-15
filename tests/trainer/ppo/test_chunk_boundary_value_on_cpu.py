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

import torch

from verl.trainer.ppo.chunk_boundary_value import (
    build_chunk_boundary_mask,
    build_chunk_boundary_training_tensors,
)
from verl.trainer.ppo.core_algos import (
    compute_chunk_boundary_bce_value_loss,
    compute_chunk_boundary_mse_value_loss,
)


def test_build_chunk_boundary_mask_includes_prompt_every_chunk_and_final_boundary():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    boundary_mask = build_chunk_boundary_mask(response_mask=response_mask, chunk_size=2)

    expected = torch.tensor(
        [
            [1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(boundary_mask, expected)


def test_build_chunk_boundary_training_tensors_default_weighting_equalizes_rollout_mass():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    rollout_targets = torch.tensor([1.0, 0.0], dtype=torch.float32)

    boundary_mask, boundary_targets, boundary_weights, metrics = build_chunk_boundary_training_tensors(
        response_mask=response_mask,
        rollout_targets=rollout_targets,
        chunk_size=2,
    )

    expected_mask = torch.tensor(
        [
            [1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    expected_weights = torch.tensor(
        [
            [0.25, 0.0, 0.25, 0.0, 0.25, 0.25],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(boundary_mask, expected_mask)
    torch.testing.assert_close(boundary_targets, torch.tensor([[1.0] * 6, [0.0] * 6], dtype=torch.float32))
    torch.testing.assert_close(boundary_weights, expected_weights)
    torch.testing.assert_close(boundary_weights.sum(dim=-1), torch.ones(2, dtype=torch.float32))
    assert metrics["chunk_boundary/boundaries_per_rollout_mean"] == 3.0
    assert metrics["chunk_boundary/boundary_states_total"] == 6.0


def test_build_chunk_boundary_training_tensors_respects_zero_sample_weight_rollouts():
    response_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32)
    rollout_targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    sample_weight = torch.tensor([1.0, 0.0], dtype=torch.float32)

    _, _, boundary_weights, metrics = build_chunk_boundary_training_tensors(
        response_mask=response_mask,
        rollout_targets=rollout_targets,
        chunk_size=2,
        sample_weight=sample_weight,
    )

    torch.testing.assert_close(boundary_weights[1], torch.zeros_like(boundary_weights[1]))
    assert metrics["chunk_boundary/effective_rollouts"] == 1.0


def test_chunk_boundary_bce_value_loss_ignores_non_boundary_positions():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    boundary_mask, boundary_targets, boundary_weights, _ = build_chunk_boundary_training_tensors(
        response_mask=response_mask,
        rollout_targets=torch.tensor([1.0, 0.0], dtype=torch.float32),
        chunk_size=2,
    )

    logits = torch.tensor(
        [
            [2.1972246, -100.0, 2.1972246, -100.0, 2.1972246],
            [-1.3862944, 100.0, -1.3862944, 100.0, 100.0],
        ],
        dtype=torch.float32,
    )

    vf_loss, vf_clipfrac, vpreds, metrics = compute_chunk_boundary_bce_value_loss(
        vpred_logits=logits,
        boundary_targets=boundary_targets,
        boundary_mask=boundary_mask,
        boundary_weights=boundary_weights,
    )

    expected_loss = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.1972246], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([-1.3862944], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        )
    )

    torch.testing.assert_close(vf_loss, expected_loss)
    torch.testing.assert_close(vf_clipfrac, torch.tensor(0.0))
    torch.testing.assert_close(vpreds[0, boundary_mask[0]], torch.full((3,), 0.9))
    torch.testing.assert_close(vpreds[1, boundary_mask[1]], torch.full((2,), 0.2))
    assert abs(metrics["critic/chunk_boundary/prediction_mean_positive"] - 0.9) < 1e-6
    assert abs(metrics["critic/chunk_boundary/prediction_mean_negative"] - 0.2) < 1e-6
    assert abs(metrics["critic/chunk_boundary/bce"] - expected_loss.item()) < 1e-6


def test_chunk_boundary_mse_value_loss_uses_equal_rollout_weighting():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    boundary_mask, boundary_targets, boundary_weights, _ = build_chunk_boundary_training_tensors(
        response_mask=response_mask,
        rollout_targets=torch.tensor([1.0, 0.0], dtype=torch.float32),
        chunk_size=2,
    )

    predictions = torch.tensor(
        [
            [0.75, -999.0, 0.75, -999.0, 0.75],
            [0.25, 999.0, 0.25, 999.0, 999.0],
        ],
        dtype=torch.float32,
    )

    vf_loss, vf_clipfrac, masked_predictions, metrics = compute_chunk_boundary_mse_value_loss(
        vpreds=predictions,
        boundary_targets=boundary_targets,
        boundary_mask=boundary_mask,
        boundary_weights=boundary_weights,
    )

    expected_loss = torch.tensor(0.0625, dtype=torch.float32)
    torch.testing.assert_close(vf_loss, expected_loss)
    torch.testing.assert_close(vf_clipfrac, torch.tensor(0.0))
    torch.testing.assert_close(masked_predictions[0, boundary_mask[0]], torch.full((3,), 0.75))
    torch.testing.assert_close(masked_predictions[1, boundary_mask[1]], torch.full((2,), 0.25))
    assert abs(metrics["critic/chunk_boundary/prediction_mean_positive"] - 0.75) < 1e-6
    assert abs(metrics["critic/chunk_boundary/prediction_mean_negative"] - 0.25) < 1e-6
    assert abs(metrics["critic/chunk_boundary/mse"] - expected_loss.item()) < 1e-6
