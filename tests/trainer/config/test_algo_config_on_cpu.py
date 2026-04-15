# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.config import AlgoConfig, ChunkBoundaryCriticConfig, KLControlConfig
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    get_adv_estimator_fn,
    resolve_advantage_lambdas,
)
from verl.trainer.ppo.utils import need_critic
from verl.utils.config import omega_conf_to_dataclass, validate_config


class TestAlgoConfig(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample algorithm config as DictConfig (similar to what comes from YAML)
        self.config_dict = {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": 0.99,
            "lam": 0.95,
            "adv_estimator": "gae",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": True,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "adaptive",
                "kl_coef": 0.002,
                "horizon": 5000,
                "target_kl": 0.05,
            },
            "use_pf_ppo": True,
            "pf_ppo": {"reweight_method": "max_min", "weight_pow": 3.0},
        }
        self.omega_config = OmegaConf.create(self.config_dict)

    def test_dataclass_creation_from_dict(self):
        """Test creating AlgoConfig from dictionary."""
        config = omega_conf_to_dataclass(self.config_dict)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)
        self.assertEqual(config.adv_estimator, "gae")
        self.assertTrue(config.norm_adv_by_std_in_grpo)
        self.assertTrue(config.use_kl_in_reward)
        self.assertEqual(config.kl_penalty, "kl")
        self.assertTrue(config.use_pf_ppo)

    def test_dataclass_creation_from_omega_config(self):
        """Test creating AlgoConfig from OmegaConf DictConfig."""
        config = omega_conf_to_dataclass(self.omega_config)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)

    def test_nested_configs(self):
        """Test that nested configurations are properly converted."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test KL control config
        self.assertIsInstance(config.kl_ctrl, KLControlConfig)
        self.assertEqual(config.kl_ctrl.type, "adaptive")
        self.assertEqual(config.kl_ctrl.kl_coef, 0.002)
        self.assertEqual(config.kl_ctrl.horizon, 5000)
        self.assertEqual(config.kl_ctrl.target_kl, 0.05)

        # Test PF PPO config
        self.assertEqual(config.pf_ppo.get("reweight_method"), "max_min")
        self.assertEqual(config.pf_ppo.get("weight_pow"), 3.0)

    def test_default_values(self):
        """Test that default values are properly set."""
        minimal_config = {"gamma": 0.8}
        config = omega_conf_to_dataclass(minimal_config, AlgoConfig)

        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(config.lam, 1.0)  # default value
        self.assertIsNone(config.actor_lam)  # default value
        self.assertIsNone(config.critic_lam)  # default value
        self.assertEqual(config.adv_estimator, "gae")  # default value
        self.assertEqual(config.adv_mode, "token")  # default value
        self.assertEqual(config.chunk_size, 8)  # default value
        self.assertEqual(config.chunk_reduce, "first")  # default value
        self.assertTrue(config.norm_adv_by_std_in_grpo)  # default value
        self.assertFalse(config.use_kl_in_reward)  # default value
        self.assertEqual(config.kl_penalty, "kl")  # default value
        self.assertFalse(config.use_pf_ppo)  # default value
        self.assertIsInstance(config.chunk_boundary_critic, ChunkBoundaryCriticConfig)
        self.assertFalse(config.chunk_boundary_critic.enable)
        self.assertEqual(config.chunk_boundary_critic.chunk_size, 32)
        self.assertEqual(config.chunk_boundary_critic.loss_type, "bce")

    def test_adv_mode_token_ignores_chunk_reduce_validation(self):
        config = AlgoConfig(adv_mode="token", chunk_reduce="not_used")
        self.assertEqual(config.adv_mode, "token")
        self.assertEqual(config.chunk_reduce, "not_used")

    def test_chunk_mode_rejects_nonpositive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "chunk_size must be >= 1"):
            AlgoConfig(adv_mode="chunk", chunk_size=0)

    def test_validate_config_rejects_chunk_mode_without_gae(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                "data": {"train_batch_size": 1},
                "algorithm": {
                    "adv_estimator": "prompt_baseline",
                    "adv_mode": "chunk",
                    "chunk_size": 8,
                    "chunk_reduce": "first",
                    "lam": 1.0,
                    "gamma": 1.0,
                    "use_kl_in_reward": False,
                },
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": "fsdp",
                        "rollout_n": 1,
                        "ppo_mini_batch_size": 1,
                        "ppo_micro_batch_size_per_gpu": 1,
                        "use_dynamic_bsz": False,
                    },
                    "rollout": {
                        "log_prob_micro_batch_size": None,
                        "log_prob_micro_batch_size_per_gpu": 1,
                    },
                    "model": {},
                },
                "critic": {
                    "enable": True,
                    "strategy": "fsdp",
                    "ppo_micro_batch_size_per_gpu": 1,
                    "value_head_type": "scalar",
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "adv_mode=chunk currently requires algorithm.adv_estimator=gae"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_get_method_backward_compatibility(self):
        """Test the get method for backward compatibility."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test existing attribute
        self.assertEqual(config.get("gamma"), 0.99)
        self.assertEqual(config.get("gamma", 1.0), 0.99)

        # Test non-existing attribute
        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_post_init_nested_configs(self):
        """Test that __post_init__ properly initializes nested configs when None."""
        # Create config without nested configs
        minimal_config = AlgoConfig(gamma=0.9)

        # Check that nested configs are initialized
        self.assertIsNotNone(minimal_config.kl_ctrl)
        self.assertIsInstance(minimal_config.kl_ctrl, KLControlConfig)
        assert not minimal_config.pf_ppo

    def test_zero_critic_disables_critic_even_if_enabled(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "zero_critic"},
                "critic": {"enable": True},
            }
        )

        with self.assertWarns(UserWarning):
            self.assertFalse(need_critic(config))

    def test_prompt_baseline_enables_critic_by_default(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "prompt_baseline"},
                "critic": {"enable": None},
            }
        )

        self.assertTrue(need_critic(config))

    def test_prompt_baseline_bce_enables_critic_by_default(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "prompt_baseline_bce"},
                "critic": {"enable": None},
            }
        )

        self.assertTrue(need_critic(config))

    def test_chunk_boundary_critic_enables_critic_by_default(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "grpo", "chunk_boundary_critic": {"enable": True}},
                "critic": {"enable": None},
            }
        )

        self.assertTrue(need_critic(config))

    def test_reverse_decay_gae_enables_critic_by_default(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "reverse_decay_gae"},
                "critic": {"enable": None},
            }
        )

        self.assertTrue(need_critic(config))

    def test_prompt_baseline_regression_enables_critic_by_default(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "prompt_baseline_regression"},
                "critic": {"enable": None},
            }
        )

        self.assertTrue(need_critic(config))

    def test_prompt_baseline_regression_validate_config_requires_lam_one(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                "algorithm": {
                    "adv_estimator": "prompt_baseline_regression",
                    "lam": 0.95,
                    "gamma": 1.0,
                    "use_kl_in_reward": False,
                },
                "critic": {"value_head_type": "scalar"},
            }
        )

        with self.assertRaisesRegex(ValueError, "effective actor and critic lambdas"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_reverse_decay_gae_validate_config_requires_positive_lambda(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                "algorithm": {
                    "adv_estimator": "reverse_decay_gae",
                    "lam": 0.0,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "> 0"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_reverse_decay_gae_lambda_resolution_uses_actor_and_critic_overrides(self):
        actor_lam, critic_lam = resolve_advantage_lambdas(
            lam=0.0,
            config=OmegaConf.create({"actor_lam": 0.8, "critic_lam": 0.9}),
        )

        self.assertEqual(actor_lam, 0.8)
        self.assertEqual(critic_lam, 0.9)

    def test_prompt_baseline_regression_validate_config_requires_scalar_value_head(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                "algorithm": {
                    "adv_estimator": "prompt_baseline_regression",
                    "lam": 1.0,
                    "gamma": 1.0,
                    "use_kl_in_reward": False,
                },
                "critic": {"value_head_type": "categorical"},
            }
        )

        with self.assertRaisesRegex(ValueError, "critic.value_head_type=scalar"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_chunk_boundary_critic_validate_config_rejects_kl_in_reward(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "n_gpus_per_node": 1,
                    "nnodes": 1,
                    "actor_update_interval": 1,
                    "critic_warmup": 0,
                    "use_legacy_worker_impl": "enable",
                },
                "data": {"train_batch_size": 1},
                "algorithm": {
                    "adv_estimator": "gae",
                    "lam": 1.0,
                    "gamma": 1.0,
                    "use_kl_in_reward": True,
                    "chunk_boundary_critic": {"enable": True, "chunk_size": 32, "loss_type": "bce"},
                },
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": "fsdp",
                        "rollout_n": 1,
                        "ppo_mini_batch_size": 1,
                        "ppo_micro_batch_size_per_gpu": 1,
                        "use_dynamic_bsz": False,
                        "use_kl_loss": False,
                    },
                    "rollout": {
                        "n": 1,
                        "log_prob_micro_batch_size": None,
                        "log_prob_micro_batch_size_per_gpu": 1,
                    },
                    "model": {},
                },
                "critic": {
                    "enable": True,
                    "strategy": "fsdp",
                    "ppo_micro_batch_size_per_gpu": 1,
                    "value_head_type": "scalar",
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "chunk_boundary_critic.enable=True does not support algorithm.use_kl_in_reward=True"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_prompt_residual_baseline_enables_critic_by_default(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "prompt_residual_baseline"},
                "critic": {"enable": None},
            }
        )

        self.assertTrue(need_critic(config))

    def test_prompt_residual_baseline_validate_config_requires_lam_one(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1, "use_legacy_worker_impl": "enable"},
                "algorithm": {
                    "adv_estimator": "prompt_residual_baseline",
                    "lam": 0.95,
                    "gamma": 1.0,
                    "use_kl_in_reward": False,
                    "prompt_residual_alpha": 1.0,
                },
                "critic": {"value_head_type": "scalar", "strategy": "fsdp"},
            }
        )

        with self.assertRaisesRegex(ValueError, "effective actor and critic lambdas"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_prompt_residual_baseline_validate_config_requires_scalar_value_head(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1, "use_legacy_worker_impl": "enable"},
                "algorithm": {
                    "adv_estimator": "prompt_residual_baseline",
                    "lam": 1.0,
                    "gamma": 1.0,
                    "use_kl_in_reward": False,
                    "prompt_residual_alpha": 1.0,
                },
                "critic": {"value_head_type": "categorical", "strategy": "fsdp"},
            }
        )

        with self.assertRaisesRegex(ValueError, "critic.value_head_type=scalar"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_prompt_residual_baseline_ramp_validate_config_requires_positive_ramp_steps(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1, "use_legacy_worker_impl": "enable"},
                "algorithm": {
                    "adv_estimator": "prompt_residual_baseline_ramp",
                    "lam": 1.0,
                    "gamma": 1.0,
                    "use_kl_in_reward": False,
                    "prompt_residual_alpha": 1.0,
                    "prompt_residual_alpha_ramp_steps": 0,
                },
                "critic": {"value_head_type": "scalar", "strategy": "fsdp"},
            }
        )

        with self.assertRaisesRegex(ValueError, "prompt_residual_alpha_ramp_steps > 0"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_prompt_baseline_bce_validate_config_requires_gamma_one(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                "algorithm": {
                    "adv_estimator": "prompt_baseline_bce",
                    "lam": 1.0,
                    "gamma": 0.9,
                    "use_kl_in_reward": False,
                },
                "critic": {"value_head_type": "scalar"},
            }
        )

        with self.assertRaisesRegex(ValueError, "algorithm.gamma=1.0"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_prompt_baseline_bce_validate_config_rejects_kl_in_reward(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                "algorithm": {
                    "adv_estimator": "prompt_baseline_bce",
                    "lam": 1.0,
                    "gamma": 1.0,
                    "use_kl_in_reward": True,
                },
                "critic": {"value_head_type": "scalar"},
            }
        )

        with self.assertRaisesRegex(ValueError, "does not support algorithm.use_kl_in_reward=True"):
            validate_config(config, use_reference_policy=False, use_critic=True)

    def test_config_init_from_yaml(self):
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(config_name="ppo_trainer")
        algo_config = omega_conf_to_dataclass(cfg.algorithm)
        from verl.trainer.config import AlgoConfig

        assert isinstance(algo_config, AlgoConfig)


class TestAlgoCompute(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.algo_config = AlgoConfig(
            gamma=0.99,
            lam=0.95,
            adv_estimator="gae",
            norm_adv_by_std_in_grpo=True,
            use_kl_in_reward=True,
            kl_penalty="kl",
            kl_ctrl=KLControlConfig(type="adaptive", kl_coef=0.002, horizon=5000, target_kl=0.05),
            use_pf_ppo=True,
            pf_ppo={"reweight_method": "max_min", "weight_pow": 3.0},
        )

    def test_advantage_estimator_with_cfg(self):
        """Test integration with advantage estimators from core_algos."""
        config = self.algo_config

        # Test GAE advantage estimator
        adv_fn = get_adv_estimator_fn(config.adv_estimator)
        self.assertIsNotNone(adv_fn)

        # Test with actual GAE computation
        batch_size, seq_len = 2, 5
        token_level_rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)

        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=config.gamma,
            lam=config.lam,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))

    def test_grpo_advantage_estimator_with_cfg(self):
        """Test integration with GRPO advantage estimator."""
        grpo_config = AlgoConfig(adv_estimator="grpo", norm_adv_by_std_in_grpo=True)

        # Test GRPO advantage computation
        batch_size, seq_len = 4, 3
        token_level_rewards = torch.tensor([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [0.5, 0.2, 0.0], [1.5, 0.8, 0.0]])
        response_mask = torch.ones(batch_size, seq_len)
        index = np.array([0, 0, 1, 1])  # Two groups

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=grpo_config.norm_adv_by_std_in_grpo,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))


if __name__ == "__main__":
    unittest.main()
