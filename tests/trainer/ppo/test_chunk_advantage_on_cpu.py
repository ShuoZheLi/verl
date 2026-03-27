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

from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.chunk_advantage import (
    build_response_chunk_ids,
    compute_chunked_advantages,
    reduce_values_by_chunk,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_gae_advantage_return
from verl.trainer.ppo.ray_trainer import compute_advantage
import verl.utils.torch_functional as verl_F


def test_build_response_chunk_ids_respects_valid_token_order():
    response_mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )

    chunk_ids = build_response_chunk_ids(response_mask=response_mask, chunk_size=2)

    expected = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(chunk_ids, expected)


def test_reduce_values_by_chunk_supports_first_last_and_mean():
    values = torch.tensor([[10.0, 11.0, 99.0, 20.0, 21.0, 22.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.float32)
    chunk_ids = build_response_chunk_ids(response_mask=response_mask, chunk_size=2)

    first_baselines, first_mask = reduce_values_by_chunk(
        values=values,
        response_mask=response_mask,
        chunk_ids=chunk_ids,
        reduce="first",
    )
    last_baselines, last_mask = reduce_values_by_chunk(
        values=values,
        response_mask=response_mask,
        chunk_ids=chunk_ids,
        reduce="last",
    )
    mean_baselines, mean_mask = reduce_values_by_chunk(
        values=values,
        response_mask=response_mask,
        chunk_ids=chunk_ids,
        reduce="mean",
    )

    expected_mask = torch.tensor([[True, True, True]])
    torch.testing.assert_close(first_mask, expected_mask)
    torch.testing.assert_close(last_mask, expected_mask)
    torch.testing.assert_close(mean_mask, expected_mask)
    torch.testing.assert_close(first_baselines, torch.tensor([[10.0, 20.0, 22.0]]))
    torch.testing.assert_close(last_baselines, torch.tensor([[11.0, 21.0, 22.0]]))
    torch.testing.assert_close(mean_baselines, torch.tensor([[10.5, 20.5, 22.0]]))


def test_compute_chunked_advantages_broadcasts_chunk_baselines_then_whitens():
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    rollout_returns = torch.tensor([1.0], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)

    advantages, metrics = compute_chunked_advantages(
        values=values,
        rollout_returns=rollout_returns,
        response_mask=response_mask,
        chunk_size=2,
        chunk_reduce="first",
    )

    raw_token_advantages = torch.tensor([[0.9, 0.9, 0.7, 0.7]], dtype=torch.float32)
    expected_advantages = verl_F.masked_whiten(raw_token_advantages, response_mask) * response_mask

    torch.testing.assert_close(advantages, expected_advantages)
    assert metrics["chunk/chunk_size"] == 2.0
    assert metrics["chunk/reduce_mode"] == 0.0
    assert metrics["chunk/num_chunks_mean"] == 2.0


def test_compute_advantage_chunk_mode_keeps_gae_returns_and_overrides_actor_advantages():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.float32),
            "values": torch.tensor([[0.2, 0.4, 0.6, 0.0]], dtype=torch.float32),
        }
    )

    config = AlgoConfig(adv_estimator="gae", adv_mode="chunk", chunk_size=2, chunk_reduce="first")
    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        config=config,
    )

    _, expected_returns = compute_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=1.0,
        lam=1.0,
    )
    raw_chunk_advantages = torch.tensor([[0.8, 0.8, 0.4, 0.0]], dtype=torch.float32)
    expected_advantages = verl_F.masked_whiten(raw_chunk_advantages, data.batch["response_mask"])
    expected_advantages = expected_advantages * data.batch["response_mask"]

    torch.testing.assert_close(output.batch["returns"], expected_returns)
    torch.testing.assert_close(output.batch["advantages"], expected_advantages)
    assert output.meta_info["advantage_metrics"]["chunk/chunk_size"] == 2.0


def test_compute_advantage_chunk_mode_handles_empty_responses_without_crashing():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "response_mask": torch.tensor(
                [
                    [0, 0, 0],
                    [1, 1, 0],
                ],
                dtype=torch.float32,
            ),
            "values": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.3, 0.7, 0.0],
                ],
                dtype=torch.float32,
            ),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        config=AlgoConfig(adv_estimator="gae", adv_mode="chunk", chunk_size=2, chunk_reduce="mean"),
    )

    torch.testing.assert_close(output.batch["advantages"][0], torch.zeros_like(output.batch["advantages"][0]))
    assert torch.isfinite(output.batch["advantages"]).all()


def test_chunk_size_one_matches_token_level_rollout_minus_value_baseline_for_final_reward_rlvr():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.float32),
            "values": torch.tensor([[0.2, 0.5, 0.7]], dtype=torch.float32),
        }
    )

    standard_output = compute_advantage(
        DataProto.from_single_dict(
            {
                "token_level_rewards": data.batch["token_level_rewards"].clone(),
                "response_mask": data.batch["response_mask"].clone(),
                "values": data.batch["values"].clone(),
            }
        ),
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        config=AlgoConfig(adv_estimator="gae", adv_mode="token"),
    )
    chunk_output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        config=AlgoConfig(adv_estimator="gae", adv_mode="chunk", chunk_size=1, chunk_reduce="first"),
    )

    torch.testing.assert_close(chunk_output.batch["returns"], standard_output.batch["returns"])
    torch.testing.assert_close(chunk_output.batch["advantages"], standard_output.batch["advantages"])
