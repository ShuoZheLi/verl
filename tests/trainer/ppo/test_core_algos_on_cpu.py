# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import random
import unittest

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.config import AlgoConfig
import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_prompt_baseline_bce_value_loss,
    compute_prompt_baseline_advantage_return,
    compute_prompt_baseline_regression_value_loss,
    compute_prompt_residual_advantage_return,
    compute_prompt_residual_regression_value_loss,
    compute_reverse_decay_gae_advantage_return,
    compute_grpo_vectorized_outcome_advantage,
    compute_zero_critic_advantage_return,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)
from verl.trainer.ppo.ray_trainer import compute_advantage
import verl.utils.torch_functional as verl_F


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        self._original_registry = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.copy()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = self._original_registry
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def test_zero_critic_matches_zero_value_gae():
    rewards = torch.tensor([[0.2, 0.3, 0.0, 0.0], [1.0, -0.5, 0.4, 0.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
    gamma = 0.9

    advantages, returns = compute_zero_critic_advantage_return(
        token_level_rewards=rewards,
        response_mask=response_mask,
        gamma=gamma,
    )

    expected_advantages, expected_returns = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=torch.zeros_like(rewards),
        response_mask=response_mask,
        gamma=gamma,
        lam=1.0,
    )

    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


def test_reverse_decay_gae_masks_invalid_tokens_and_skips_masked_values():
    rewards = torch.tensor(
        [
            [0.4, 0.0, 1.0, 0.0],
            [0.2, 0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    values_a = torch.tensor(
        [
            [0.3, 999.0, 0.7, -888.0],
            [0.1, 0.4, 123.0, 456.0],
        ],
        dtype=torch.float32,
    )
    values_b = torch.tensor(
        [
            [0.3, -999.0, 0.7, 888.0],
            [0.1, 0.4, -123.0, -456.0],
        ],
        dtype=torch.float32,
    )

    advantages_a, returns_a = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=rewards,
        values=values_a,
        response_mask=response_mask,
        gamma=0.7,
        lam=0.6,
    )
    advantages_b, returns_b = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=rewards,
        values=values_b,
        response_mask=response_mask,
        gamma=0.7,
        lam=0.6,
    )

    torch.testing.assert_close(advantages_a, advantages_b)
    torch.testing.assert_close(returns_a, returns_b)
    torch.testing.assert_close(advantages_a * (1.0 - response_mask), torch.zeros_like(advantages_a))
    torch.testing.assert_close(returns_a * (1.0 - response_mask), torch.zeros_like(returns_a))


def test_reverse_decay_gae_matches_standard_gae_when_lambda_one_and_gamma_one():
    torch.manual_seed(0)
    rewards = torch.randn(3, 6, dtype=torch.float32)
    values = torch.randn(3, 6, dtype=torch.float32)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    reverse_advantages, reverse_returns = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lam=1.0,
    )
    gae_advantages, gae_returns = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lam=1.0,
    )

    torch.testing.assert_close(reverse_returns, gae_returns * response_mask)
    torch.testing.assert_close(reverse_advantages, gae_advantages * response_mask)


def test_reverse_decay_gae_matches_standard_gae_when_lambda_one_with_general_gamma():
    torch.manual_seed(1)
    rewards = torch.randn(4, 5, dtype=torch.float32)
    values = torch.randn(4, 5, dtype=torch.float32)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1],
        ],
        dtype=torch.float32,
    )

    reverse_advantages, reverse_returns = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.83,
        lam=1.0,
    )
    gae_advantages, gae_returns = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.83,
        lam=1.0,
    )

    torch.testing.assert_close(reverse_returns, gae_returns * response_mask)
    torch.testing.assert_close(reverse_advantages, gae_advantages * response_mask)


def test_reverse_decay_gae_matches_hand_computed_toy_example():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 0.0]], dtype=torch.float32)
    values = torch.tensor([[0.5, 1.0, 1.5, 10.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)

    advantages, returns = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.9,
        lam=0.5,
    )

    expected_raw_advantages = torch.tensor([[2.6225, 2.5250, 1.5000, 0.0]], dtype=torch.float32)
    expected_returns = torch.tensor([[3.1225, 3.5250, 3.0000, 0.0]], dtype=torch.float32)
    expected_advantages = verl_F.masked_whiten(expected_raw_advantages, response_mask) * response_mask

    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


def test_reverse_decay_gae_remains_finite_for_tiny_positive_lambda():
    torch.manual_seed(2)
    rewards = torch.randn(2, 6, dtype=torch.float32)
    values = torch.randn(2, 6, dtype=torch.float32)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )

    advantages, returns = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.95,
        lam=1e-3,
    )

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()


def test_compute_advantage_reverse_decay_gae_matches_core_algo():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor(
                [
                    [0.4, 0.0, 1.0, 0.0],
                    [0.2, 0.5, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "response_mask": torch.tensor(
                [
                    [1, 0, 1, 0],
                    [1, 1, 0, 0],
                ],
                dtype=torch.float32,
            ),
            "values": torch.tensor(
                [
                    [0.3, 0.0, 0.7, 0.0],
                    [0.1, 0.4, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.REVERSE_DECAY_GAE,
        gamma=0.7,
        lam=0.6,
        config=AlgoConfig(adv_estimator="reverse_decay_gae", lam=0.6),
    )

    expected_advantages, expected_returns = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=0.7,
        lam=0.6,
    )

    torch.testing.assert_close(output.batch["advantages"], expected_advantages)
    torch.testing.assert_close(output.batch["returns"], expected_returns)


def test_compute_advantage_gae_supports_split_actor_and_critic_lam():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.0],
                    [0.4, -0.2, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 1, 0],
                    [1, 1, 0, 0],
                ],
                dtype=torch.float32,
            ),
            "values": torch.tensor(
                [
                    [0.5, 0.4, 0.2, 0.0],
                    [0.1, -0.1, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        }
    )

    config = AlgoConfig(adv_estimator="gae", lam=0.95, actor_lam=0.5, critic_lam=0.9)
    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GAE,
        gamma=0.8,
        lam=config.lam,
        config=config,
    )

    expected_advantages, _ = compute_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=0.8,
        lam=0.5,
    )
    _, expected_returns = compute_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=0.8,
        lam=0.9,
    )

    torch.testing.assert_close(output.batch["advantages"], expected_advantages)
    torch.testing.assert_close(output.batch["returns"], expected_returns)


def test_compute_advantage_reverse_decay_gae_uses_algorithm_lam_as_critic_fallback():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.4, 0.0, 1.0, 0.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 0, 1, 0]], dtype=torch.float32),
            "values": torch.tensor([[0.3, 0.0, 0.7, 0.0]], dtype=torch.float32),
        }
    )

    config = AlgoConfig(adv_estimator="reverse_decay_gae", lam=0.9, actor_lam=0.6, critic_lam=None)
    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.REVERSE_DECAY_GAE,
        gamma=0.7,
        lam=config.lam,
        config=config,
    )

    expected_advantages, _ = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=0.7,
        lam=0.6,
    )
    _, expected_returns = compute_reverse_decay_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=0.7,
        lam=0.9,
    )

    torch.testing.assert_close(output.batch["advantages"], expected_advantages)
    torch.testing.assert_close(output.batch["returns"], expected_returns)


def test_compute_advantage_rejects_split_lams_for_prompt_baseline():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.float32),
            "values": torch.tensor([[0.5, 1.0, 1.0]], dtype=torch.float32),
        }
    )

    with pytest.raises(ValueError, match="Different actor_lam and critic_lam"):
        compute_advantage(
            data,
            adv_estimator=AdvantageEstimator.PROMPT_BASELINE,
            gamma=1.0,
            lam=1.0,
            config=AlgoConfig(adv_estimator="prompt_baseline", lam=1.0, actor_lam=0.8, critic_lam=1.0),
        )


def test_reverse_decay_gae_rejects_nonpositive_lambda():
    rewards = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    values = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="lam > 0"):
        compute_reverse_decay_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=response_mask,
            gamma=1.0,
            lam=0.0,
        )


def test_compute_advantage_zero_critic_injects_zero_values():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.2, 0.3, 0.0], [1.0, -0.5, 0.4]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.ZERO_CRITIC,
        gamma=0.9,
        lam=1.0,
        config=AlgoConfig(adv_estimator="zero_critic"),
    )

    assert "values" in output.batch
    torch.testing.assert_close(output.batch["values"], torch.zeros_like(output.batch["returns"]))


def test_compute_advantage_zero_critic_rejects_nonunit_lambda():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.2, 0.3, 0.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.float32),
        }
    )

    with pytest.raises(ValueError, match="effective actor and critic lambdas"):
        compute_advantage(
            data,
            adv_estimator=AdvantageEstimator.ZERO_CRITIC,
            gamma=0.9,
            lam=0.95,
            config=AlgoConfig(adv_estimator="zero_critic"),
        )


def test_prompt_baseline_advantage_uses_prompt_end_value_only():
    rewards = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    values = torch.tensor([[0.25, 99.0, 99.0], [0.75, -99.0, -99.0]], dtype=torch.float32)

    advantages, returns = compute_prompt_baseline_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lam=1.0,
    )

    expected_returns = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    raw_advantages = torch.tensor([[0.75, 0.75, 0.0], [0.25, 0.25, 0.25]], dtype=torch.float32)
    expected_advantages = verl_F.masked_whiten(raw_advantages, response_mask) * response_mask

    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


def test_compute_advantage_prompt_baseline_matches_core_algo():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
            "values": torch.tensor([[0.2, 5.0, 5.0], [0.6, 7.0, 7.0]], dtype=torch.float32),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.PROMPT_BASELINE,
        gamma=1.0,
        lam=1.0,
        config=AlgoConfig(adv_estimator="prompt_baseline"),
    )

    expected_advantages, expected_returns = compute_prompt_baseline_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=1.0,
        lam=1.0,
    )

    torch.testing.assert_close(output.batch["returns"], expected_returns)
    torch.testing.assert_close(output.batch["advantages"], expected_advantages)


def test_compute_advantage_prompt_baseline_rejects_nonunit_lambda():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.float32),
            "values": torch.tensor([[0.5, 1.0, 1.0]], dtype=torch.float32),
        }
    )

    with pytest.raises(ValueError, match="lam=1.0"):
        compute_advantage(
            data,
            adv_estimator=AdvantageEstimator.PROMPT_BASELINE,
            gamma=1.0,
            lam=0.95,
            config=AlgoConfig(adv_estimator="prompt_baseline", lam=0.95),
        )


def test_compute_advantage_prompt_baseline_bce_matches_prompt_baseline_core_algo():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
            "values": torch.tensor([[0.2, 0.9, 0.9], [0.6, 0.1, 0.1]], dtype=torch.float32),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.PROMPT_BASELINE_BCE,
        gamma=1.0,
        lam=1.0,
        config=AlgoConfig(adv_estimator="prompt_baseline_bce"),
    )

    expected_advantages, expected_returns = compute_prompt_baseline_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=1.0,
        lam=1.0,
    )

    torch.testing.assert_close(output.batch["returns"], expected_returns)
    torch.testing.assert_close(output.batch["advantages"], expected_advantages)


def test_compute_advantage_prompt_baseline_regression_matches_prompt_baseline_core_algo():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
            "values": torch.tensor([[0.2, 5.0, 5.0], [0.6, 7.0, 7.0]], dtype=torch.float32),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.PROMPT_BASELINE_REGRESSION,
        gamma=1.0,
        lam=1.0,
        config=AlgoConfig(adv_estimator="prompt_baseline_regression"),
    )

    expected_advantages, expected_returns = compute_prompt_baseline_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=1.0,
        lam=1.0,
    )

    torch.testing.assert_close(output.batch["returns"], expected_returns)
    torch.testing.assert_close(output.batch["advantages"], expected_advantages)


def test_compute_prompt_residual_advantage_return_uses_combined_baseline():
    rewards = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    prompt_prior_values = torch.tensor([0.2, 0.6], dtype=torch.float32)
    residual_values = torch.tensor([[0.0, 0.3, 0.0], [0.1, 0.2, 0.4]], dtype=torch.float32)

    advantages, returns = compute_prompt_residual_advantage_return(
        token_level_rewards=rewards,
        prompt_prior_values=prompt_prior_values,
        residual_values=residual_values,
        response_mask=response_mask,
        gamma=1.0,
        alpha=0.5,
        lam=1.0,
    )

    expected_returns = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    expected_values = torch.tensor([[0.2, 0.35, 0.0], [0.65, 0.7, 0.8]], dtype=torch.float32)
    raw_advantages = (expected_returns - expected_values) * response_mask
    expected_advantages = verl_F.masked_whiten(raw_advantages, response_mask) * response_mask

    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_advantages)


def test_compute_advantage_prompt_residual_baseline_sets_actor_baseline_and_rollout_return():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
            "prompt_prior_values": torch.tensor([0.2, 0.6], dtype=torch.float32),
            "residual_values": torch.tensor([[0.0, 0.3, 0.0], [0.1, 0.2, 0.4]], dtype=torch.float32),
            "values": torch.zeros((2, 3), dtype=torch.float32),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.PROMPT_RESIDUAL_BASELINE,
        gamma=1.0,
        lam=1.0,
        config=AlgoConfig(adv_estimator="prompt_residual_baseline", prompt_residual_alpha=0.5),
    )

    expected_values = torch.tensor([[0.2, 0.35, 0.0], [0.65, 0.7, 0.8]], dtype=torch.float32)
    expected_rollout_returns = torch.tensor([1.0, 1.0], dtype=torch.float32)

    torch.testing.assert_close(output.batch["values"], expected_values)
    torch.testing.assert_close(output.batch["rollout_returns"], expected_rollout_returns)
    assert output.meta_info["advantage_metrics"]["actor/residual_weight_alpha"] == 0.5


def test_compute_advantage_prompt_residual_baseline_ramp_scales_alpha_by_step():
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.float32),
            "prompt_prior_values": torch.tensor([0.4], dtype=torch.float32),
            "residual_values": torch.tensor([[0.2, 0.6, 0.0]], dtype=torch.float32),
            "values": torch.zeros((1, 3), dtype=torch.float32),
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.PROMPT_RESIDUAL_BASELINE_RAMP,
        gamma=1.0,
        lam=1.0,
        global_step=50,
        config=AlgoConfig(
            adv_estimator="prompt_residual_baseline_ramp",
            prompt_residual_alpha=1.0,
            prompt_residual_alpha_ramp_steps=100,
        ),
    )

    expected_values = torch.tensor([[0.5, 0.7, 0.0]], dtype=torch.float32)

    torch.testing.assert_close(output.batch["values"], expected_values)
    assert output.meta_info["advantage_metrics"]["actor/residual_weight_alpha"] == 0.5


def test_prompt_baseline_bce_value_loss_uses_prompt_end_logit_only():
    vpred_logits_a = torch.logit(torch.tensor([[0.8, 0.01], [0.3, 0.99]], dtype=torch.float32))
    vpred_logits_b = torch.logit(torch.tensor([[0.8, 0.99], [0.3, 0.01]], dtype=torch.float32))
    old_values = torch.tensor([[0.2, 0.2], [0.6, 0.6]], dtype=torch.float32)
    returns = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)

    vf_loss_a, vf_clipfrac_a, vpreds_a, metrics_a = compute_prompt_baseline_bce_value_loss(
        vpred_logits=vpred_logits_a,
        returns=returns,
        values=old_values,
        response_mask=response_mask,
        cliprange_value=10.0,
    )
    vf_loss_b, vf_clipfrac_b, vpreds_b, metrics_b = compute_prompt_baseline_bce_value_loss(
        vpred_logits=vpred_logits_b,
        returns=returns,
        values=old_values,
        response_mask=response_mask,
        cliprange_value=10.0,
    )

    expected_loss = torch.nn.functional.binary_cross_entropy(
        torch.tensor([[0.8], [0.3]], dtype=torch.float32),
        torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        reduction="mean",
    )

    torch.testing.assert_close(vf_loss_a, expected_loss)
    torch.testing.assert_close(vf_loss_b, expected_loss)
    torch.testing.assert_close(vpreds_a, vpreds_b)
    torch.testing.assert_close(metrics_a["critic/prompt_success_prob_mean"], torch.tensor(0.55))
    torch.testing.assert_close(metrics_b["critic/prompt_success_prob_mean"], torch.tensor(0.55))
    torch.testing.assert_close(vf_clipfrac_a, torch.tensor(0.0))
    torch.testing.assert_close(vf_clipfrac_b, torch.tensor(0.0))


def test_prompt_baseline_regression_value_loss_uses_prompt_end_value_only():
    vpreds_a = torch.tensor([[0.8, 10.0], [0.3, -10.0]], dtype=torch.float32)
    vpreds_b = torch.tensor([[0.8, -10.0], [0.3, 10.0]], dtype=torch.float32)
    old_values = torch.tensor([[0.2, 0.2], [0.6, 0.6]], dtype=torch.float32)
    returns = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)

    vf_loss_a, vf_clipfrac_a, mirrored_vpreds_a, metrics_a = compute_prompt_baseline_regression_value_loss(
        vpreds=vpreds_a,
        returns=returns,
        values=old_values,
        response_mask=response_mask,
        cliprange_value=10.0,
    )
    vf_loss_b, vf_clipfrac_b, mirrored_vpreds_b, metrics_b = compute_prompt_baseline_regression_value_loss(
        vpreds=vpreds_b,
        returns=returns,
        values=old_values,
        response_mask=response_mask,
        cliprange_value=10.0,
    )

    expected_loss = 0.5 * torch.mean((torch.tensor([[0.8], [0.3]], dtype=torch.float32) - torch.tensor([[1.0], [0.0]], dtype=torch.float32)) ** 2)
    expected_mirrored_vpreds = torch.tensor([[0.8, 0.8], [0.3, 0.3]], dtype=torch.float32)

    torch.testing.assert_close(vf_loss_a, expected_loss)
    torch.testing.assert_close(vf_loss_b, expected_loss)
    torch.testing.assert_close(mirrored_vpreds_a, expected_mirrored_vpreds)
    torch.testing.assert_close(mirrored_vpreds_b, expected_mirrored_vpreds)
    torch.testing.assert_close(metrics_a["critic/prompt_value_pred_mean"], torch.tensor(0.55))
    torch.testing.assert_close(metrics_b["critic/prompt_value_pred_mean"], torch.tensor(0.55))
    torch.testing.assert_close(metrics_a["critic/prompt_value_target_mean"], torch.tensor(0.5))
    torch.testing.assert_close(metrics_b["critic/prompt_value_target_mean"], torch.tensor(0.5))
    torch.testing.assert_close(vf_clipfrac_a, torch.tensor(0.0))
    torch.testing.assert_close(vf_clipfrac_b, torch.tensor(0.0))


def test_prompt_residual_regression_value_loss_uses_stop_grad_prompt_target_decomposition():
    prompt_prior_vpreds = torch.tensor([0.2, 0.6], dtype=torch.float32)
    residual_vpreds = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    rollout_returns = torch.tensor([1.0, 0.0], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)

    vf_loss, vf_clipfrac, combined_vpreds, metrics = compute_prompt_residual_regression_value_loss(
        prompt_prior_vpreds=prompt_prior_vpreds,
        residual_vpreds=residual_vpreds,
        rollout_returns=rollout_returns,
        response_mask=response_mask,
        prompt_loss_weight=1.0,
        residual_loss_weight=1.0,
    )

    expected_prompt_loss = torch.tensor((0.64 + 0.36) / 2, dtype=torch.float32)
    expected_residual_loss = torch.tensor((0.49 + 0.36 + 0.81) / 3, dtype=torch.float32)
    expected_total_loss = expected_prompt_loss + expected_residual_loss
    expected_combined_vpreds = torch.tensor([[0.3, 0.4], [0.9, 1.0]], dtype=torch.float32)

    torch.testing.assert_close(vf_loss, expected_total_loss)
    torch.testing.assert_close(vf_clipfrac, torch.tensor(0.0))
    torch.testing.assert_close(combined_vpreds, expected_combined_vpreds)
    torch.testing.assert_close(metrics["critic/prompt_prior_loss"], expected_prompt_loss)
    torch.testing.assert_close(metrics["critic/residual_loss"], expected_residual_loss)


def test_prompt_baseline_bce_value_loss_rejects_targets_outside_unit_interval():
    with pytest.raises(ValueError, match=r"requires prompt-level targets in \[0, 1\]"):
        compute_prompt_baseline_bce_value_loss(
            vpred_logits=torch.zeros((1, 2), dtype=torch.float32),
            returns=torch.tensor([[1.2, 1.2]], dtype=torch.float32),
            values=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            response_mask=torch.tensor([[1, 1]], dtype=torch.float32),
            cliprange_value=0.2,
        )


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
