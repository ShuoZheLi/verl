# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.chunk_advantage import compute_chunked_advantages
from verl.trainer.ppo.chunk_boundary_value import (
    build_chunk_boundary_training_tensors,
    build_response_state_mask,
    compute_chunk_boundary_metric_sums,
    finalize_chunk_boundary_metric_sums,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_grpo_baseline_metrics,
    compute_overlong_filtering_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    compute_zero_critic_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    valid_seq_mask = response_mask.sum(dim=-1) > 0
    valid_batch_size = int(valid_seq_mask.sum().item())

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    if valid_batch_size > 0:
        current_kl = torch.mean(current_kl[valid_seq_mask], dim=0).item()
    else:
        current_kl = 0.0

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    if valid_batch_size > 0:
        kl_ctrl.update(current_kl=current_kl, n_steps=valid_batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


_OVERLONG_TERMINATION_REASONS = {"max_length", "length", "max_tokens", "max_new_tokens", "truncated"}
_NON_OVERLONG_TERMINATION_REASONS = {"eos", "stop", "aborted", "turn_limit", "tool_call", "interaction_end"}


def _normalize_eos_token_ids(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


def infer_overlong_mask(
    data: DataProto,
    eos_token_id: Any = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Infer whether each sample ended because it exhausted the response-length budget."""
    raw_response_mask = data.batch.get("response_mask")
    if raw_response_mask is None:
        raw_response_mask = compute_response_mask(data)

    responses = data.batch["responses"]
    batch_size, max_response_length = responses.shape
    response_lengths = raw_response_mask.sum(dim=-1).to(dtype=torch.long)
    eos_token_ids = _normalize_eos_token_ids(eos_token_id)

    termination_reasons = data.non_tensor_batch.get("termination_reason")
    if termination_reasons is None:
        inferred_reasons = np.empty(batch_size, dtype=object)
        inferred_reasons[:] = None
    else:
        inferred_reasons = np.asarray(termination_reasons, dtype=object).copy()

    is_overlong = torch.zeros(batch_size, dtype=torch.bool, device=responses.device)

    for idx in range(batch_size):
        reason = inferred_reasons[idx]
        if isinstance(reason, bytes):
            reason = reason.decode("utf-8")
        if isinstance(reason, str):
            reason = reason.strip().lower()
        else:
            reason = None

        if reason in _OVERLONG_TERMINATION_REASONS:
            is_overlong[idx] = True
            inferred_reasons[idx] = "max_length"
            continue
        if reason in _NON_OVERLONG_TERMINATION_REASONS:
            inferred_reasons[idx] = reason
            continue

        response_length = int(response_lengths[idx].item())
        if response_length <= 0:
            inferred_reasons[idx] = "aborted"
            continue

        hit_response_limit = response_length >= max_response_length
        last_token_id = int(responses[idx, response_length - 1].item())
        ended_with_eos = last_token_id in eos_token_ids if eos_token_ids else False

        overlong = hit_response_limit and not ended_with_eos
        is_overlong[idx] = overlong
        if overlong:
            inferred_reasons[idx] = "max_length"
        elif ended_with_eos:
            inferred_reasons[idx] = "eos"
        else:
            inferred_reasons[idx] = "unknown"

    return is_overlong, inferred_reasons


def apply_overlong_filtering(data: DataProto, eos_token_id: Any = None) -> DataProto:
    raw_response_mask = data.batch.get("response_mask")
    if raw_response_mask is None:
        raw_response_mask = compute_response_mask(data)
    raw_response_mask = raw_response_mask.clone()

    is_overlong, termination_reasons = infer_overlong_mask(data=data, eos_token_id=eos_token_id)
    sample_weight = (~is_overlong).to(dtype=raw_response_mask.dtype)
    filtered_response_mask = raw_response_mask * sample_weight.unsqueeze(-1)

    data.batch["raw_response_mask"] = raw_response_mask
    data.batch["is_overlong"] = is_overlong
    data.batch["sample_weight"] = sample_weight
    data.batch["overlong_filtered_response_mask"] = filtered_response_mask.clone()
    data.batch["response_mask"] = filtered_response_mask
    data.non_tensor_batch["termination_reason"] = termination_reasons
    data.non_tensor_batch["is_overlong"] = is_overlong.cpu().numpy()
    return data


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    actor_lam: Optional[float] = None,
    critic_lam: Optional[float] = None,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    global_step: Optional[int] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Backward-compatible lambda fallback. Defaults to 1.0.
        actor_lam (float, optional): Optional actor-side lambda override for policy advantages.
        critic_lam (float, optional): Optional critic-side lambda override for value-learning returns.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.
        global_step (int, optional): Current PPO step, used by scheduled prompt-residual baselines.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    adv_estimator_name = adv_estimator.value if isinstance(adv_estimator, AdvantageEstimator) else str(adv_estimator)
    actor_lam, critic_lam = core_algos.resolve_advantage_lambdas(
        lam=lam,
        config=config,
        actor_lam=actor_lam,
        critic_lam=critic_lam,
    )
    if actor_lam != critic_lam and adv_estimator_name not in (
        AdvantageEstimator.GAE.value,
        AdvantageEstimator.REVERSE_DECAY_GAE.value,
    ):
        raise ValueError(
            "Different actor_lam and critic_lam are currently supported only for "
            "algorithm.adv_estimator in {'gae', 'reverse_decay_gae'}."
        )

    def _apply_chunk_advantage_override() -> None:
        adv_mode = config.get("adv_mode", "token") if config is not None else "token"
        if adv_mode != "chunk":
            return

        if adv_estimator_name != AdvantageEstimator.GAE.value:
            raise ValueError(
                "algorithm.adv_mode=chunk currently requires algorithm.adv_estimator=gae so only the actor "
                "advantage tensor changes while the critic path stays on standard PPO targets/losses."
            )
        if "values" not in data.batch:
            raise ValueError(
                "algorithm.adv_mode=chunk requires critic values. Ensure the critic worker is enabled and "
                "token-level value predictions were computed before advantage calculation."
            )

        rollout_returns = core_algos.compute_outcome_rollout_returns(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        advantages, chunk_metrics = compute_chunked_advantages(
            values=data.batch["values"],
            rollout_returns=rollout_returns,
            response_mask=data.batch["response_mask"],
            chunk_size=int(config.get("chunk_size", 8)) if config is not None else 8,
            chunk_reduce=str(config.get("chunk_reduce", "first")) if config is not None else "first",
        )
        data.batch["advantages"] = advantages
        advantage_metrics = data.meta_info.get("advantage_metrics")
        if advantage_metrics is None:
            advantage_metrics = {}
            data.meta_info["advantage_metrics"] = advantage_metrics
        advantage_metrics.update(chunk_metrics)

    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    valid_sample_mask = data.batch["response_mask"].sum(dim=-1) > 0

    def scatter_valid_results(
        valid_advantages: torch.Tensor,
        valid_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_advantages = torch.zeros_like(data.batch["token_level_rewards"])
        full_returns = torch.zeros_like(data.batch["token_level_rewards"])
        full_advantages[valid_sample_mask] = valid_advantages
        full_returns[valid_sample_mask] = valid_returns
        return full_advantages, full_returns

    grouped_adv_estimators = {
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_VECTORIZED,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.OPO,
        AdvantageEstimator.GPG,
        AdvantageEstimator.RLOO_VECTORIZED,
        AdvantageEstimator.OPTIMAL_TOKEN_BASELINE,
        AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE,
    }

    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_advantage_return_with_separate_lams(
            core_algos.compute_gae_advantage_return,
            actor_lam=actor_lam,
            critic_lam=critic_lam,
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        _apply_chunk_advantage_override()
        if config.get("use_pf_ppo", False) and valid_sample_mask.any():
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.REVERSE_DECAY_GAE:
        if "values" not in data.batch:
            raise ValueError(
                f"algorithm.adv_estimator={adv_estimator.value} requires critic values. "
                "Ensure the critic worker is enabled and values were computed before advantage calculation."
            )
        advantages, returns = core_algos.compute_advantage_return_with_separate_lams(
            core_algos.compute_reverse_decay_gae_advantage_return,
            actor_lam=actor_lam,
            critic_lam=critic_lam,
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        _apply_chunk_advantage_override()
        if config.get("use_pf_ppo", False) and valid_sample_mask.any():
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator in (
        AdvantageEstimator.PROMPT_BASELINE,
        AdvantageEstimator.PROMPT_BASELINE_REGRESSION,
        AdvantageEstimator.PROMPT_BASELINE_BCE,
    ):
        if "values" not in data.batch:
            raise ValueError(
                f"algorithm.adv_estimator={adv_estimator.value} requires critic values. "
                "Ensure the critic worker is enabled and values were computed before advantage calculation."
            )
        advantages, returns = core_algos.compute_prompt_baseline_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=actor_lam,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.batch["rollout_returns"] = core_algos.compute_outcome_rollout_returns(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["prompt_prior_values"] = data.batch["values"][:, 0]
        _apply_chunk_advantage_override()
        if config.get("use_pf_ppo", False) and valid_sample_mask.any():
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator in (
        AdvantageEstimator.PROMPT_RESIDUAL_BASELINE,
        AdvantageEstimator.PROMPT_RESIDUAL_BASELINE_RAMP,
    ):
        if "prompt_prior_values" not in data.batch or "residual_values" not in data.batch:
            raise ValueError(
                f"algorithm.adv_estimator={adv_estimator.value} requires both prompt_prior_values and "
                "residual_values from the critic. Ensure the prompt-residual critic is enabled and values "
                "were computed before advantage calculation."
            )
        alpha = core_algos.get_prompt_residual_alpha(
            adv_estimator,
            config=config,
            global_step=global_step,
        )
        advantages, returns = core_algos.compute_prompt_residual_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            prompt_prior_values=data.batch["prompt_prior_values"],
            residual_values=data.batch["residual_values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=actor_lam,
            alpha=alpha,
            config=config,
        )
        rollout_returns = core_algos.compute_outcome_rollout_returns(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        combined_values = core_algos.combine_prompt_residual_values(
            prompt_prior_values=data.batch["prompt_prior_values"],
            residual_values=data.batch["residual_values"],
            response_mask=data.batch["response_mask"],
            alpha=alpha,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.batch["rollout_returns"] = rollout_returns
        data.batch["values"] = combined_values
        data.meta_info["advantage_metrics"] = {"actor/residual_weight_alpha": alpha}
        _apply_chunk_advantage_override()
        if config.get("use_pf_ppo", False) and valid_sample_mask.any():
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.ZERO_CRITIC:
        if actor_lam != 1.0 or critic_lam != 1.0:
            raise ValueError(
                "algorithm.adv_estimator=zero_critic requires effective actor and critic lambdas to be 1.0."
            )
        advantages, returns = core_algos.compute_zero_critic_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.batch["values"] = torch.zeros_like(returns)
        _apply_chunk_advantage_override()
        if config.get("use_pf_ppo", False) and valid_sample_mask.any():
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        if valid_sample_mask.any():
            valid_advantages, valid_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"][valid_sample_mask],
                response_mask=grpo_calculation_mask[valid_sample_mask],
                index=data.non_tensor_batch["uid"][valid_sample_mask.cpu().numpy()],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            advantages, returns = scatter_valid_results(valid_advantages, valid_returns)
        else:
            advantages = torch.zeros_like(data.batch["token_level_rewards"])
            returns = torch.zeros_like(data.batch["token_level_rewards"])
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        _apply_chunk_advantage_override()
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        if adv_estimator in grouped_adv_estimators and not torch.all(valid_sample_mask):
            if valid_sample_mask.any():
                valid_adv_kwargs = {
                    "token_level_rewards": adv_kwargs["token_level_rewards"][valid_sample_mask],
                    "response_mask": adv_kwargs["response_mask"][valid_sample_mask],
                    "config": adv_kwargs["config"],
                }
                if "index" in adv_kwargs:
                    valid_adv_kwargs["index"] = adv_kwargs["index"][valid_sample_mask.cpu().numpy()]
                if "reward_baselines" in adv_kwargs:
                    valid_adv_kwargs["reward_baselines"] = adv_kwargs["reward_baselines"][valid_sample_mask]
                if "sum_pi_squared" in adv_kwargs:
                    valid_adv_kwargs["sum_pi_squared"] = adv_kwargs["sum_pi_squared"][valid_sample_mask]
                if "rollout_is_weights" in adv_kwargs and adv_kwargs["rollout_is_weights"] is not None:
                    valid_adv_kwargs["rollout_is_weights"] = adv_kwargs["rollout_is_weights"][valid_sample_mask]

                valid_advantages, valid_returns = adv_estimator_fn(**valid_adv_kwargs)
                advantages, returns = scatter_valid_results(valid_advantages, valid_returns)
            else:
                advantages = torch.zeros_like(data.batch["token_level_rewards"])
                returns = torch.zeros_like(data.batch["token_level_rewards"])
        else:
            # calculate advantage estimator
            advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        _apply_chunk_advantage_override()
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        chunk_boundary_cfg = self.config.algorithm.get("chunk_boundary_critic", None)
        self.use_chunk_boundary_critic = bool(chunk_boundary_cfg is not None and chunk_boundary_cfg.get("enable", False))
        self.use_zero_critic = self.config.algorithm.adv_estimator == AdvantageEstimator.ZERO_CRITIC
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _get_chunk_boundary_critic_cfg(self):
        return self.config.algorithm.chunk_boundary_critic

    def _chunk_boundary_loss_type(self) -> str:
        return str(self._get_chunk_boundary_critic_cfg().loss_type)

    def _freeze_actor_updates(self) -> bool:
        return self.use_chunk_boundary_critic

    def _prepare_chunk_boundary_critic_batch(
        self,
        batch: DataProto,
        rollout_targets: torch.Tensor,
    ) -> tuple[DataProto, dict[str, Any]]:
        response_mask_for_boundaries = batch.batch.get("raw_response_mask", batch.batch["response_mask"])
        sample_weight = batch.batch.get("sample_weight")
        chunk_boundary_cfg = self._get_chunk_boundary_critic_cfg()
        boundary_mask, boundary_targets, boundary_weights, metrics = build_chunk_boundary_training_tensors(
            response_mask=response_mask_for_boundaries,
            rollout_targets=rollout_targets,
            chunk_size=int(chunk_boundary_cfg.chunk_size),
            sample_weight=sample_weight,
            uniform_per_state_weight=bool(chunk_boundary_cfg.uniform_per_state_weight),
        )
        batch.batch["chunk_boundary_mask"] = boundary_mask
        batch.batch["chunk_boundary_targets"] = boundary_targets
        batch.batch["chunk_boundary_weights"] = boundary_weights
        rollout_targets = rollout_targets.float()
        batch.batch["rollout_returns"] = rollout_targets
        batch.batch["returns"] = core_algos.broadcast_outcome_rollout_returns(
            rollout_returns=rollout_targets,
            response_mask=batch.batch["response_mask"],
        )
        batch.batch["advantages"] = torch.zeros_like(batch.batch["returns"])
        return batch, metrics

    def _get_actor_update_interval(self) -> int:
        interval = int(self.config.trainer.get("actor_update_interval", 1))
        if interval < 1:
            raise ValueError(f"trainer.actor_update_interval must be >= 1, got {interval}")
        return interval

    def _get_effective_actor_update_interval(self) -> int:
        if self._freeze_actor_updates():
            return 0
        return 1 if self.use_zero_critic else self._get_actor_update_interval()

    def _get_first_actor_update_step(self) -> int:
        critic_warmup = int(self.config.trainer.critic_warmup)
        if critic_warmup < 0:
            raise ValueError(f"trainer.critic_warmup must be >= 0, got {critic_warmup}")
        return critic_warmup if critic_warmup > 0 else self._get_actor_update_interval()

    def _count_actor_updates_through_step(self, global_step: int) -> int:
        if self._freeze_actor_updates():
            return 0
        if self.use_zero_critic:
            return global_step

        first_actor_update_step = self._get_first_actor_update_step()
        if global_step < first_actor_update_step:
            return 0

        actor_update_interval = self._get_actor_update_interval()
        return 1 + (global_step - first_actor_update_step) // actor_update_interval

    def _should_update_actor(self) -> bool:
        if self._freeze_actor_updates():
            return False
        if self.use_zero_critic:
            return True

        first_actor_update_step = self._get_first_actor_update_step()
        if self.global_steps < first_actor_update_step:
            return False

        actor_update_interval = self._get_actor_update_interval()
        return (self.global_steps - first_actor_update_step) % actor_update_interval == 0

    def _should_refresh_rollout_after_critic_only_step(self) -> bool:
        rollout_config = self.config.actor_rollout_ref.rollout
        return rollout_config.checkpoint_engine.backend == "naive" and rollout_config.free_cache_engine

    def _maybe_add_zero_critic_metrics(self, metrics: dict[str, Any]) -> None:
        if self.use_zero_critic:
            metrics.update(compute_zero_critic_metrics())

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        self.total_actor_training_steps = self._count_actor_updates_through_step(total_training_steps)
        self.actor_scheduler_total_training_steps = max(1, self.total_actor_training_steps)
        print(f"Total training steps: {self.total_training_steps}")
        print(f"Total actor update steps: {self.total_actor_training_steps}")
        if self.total_actor_training_steps == 0:
            print("Warning: actor update schedule yields zero actor optimizer steps; using 1 scheduler step.")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        self.actor_scheduler_total_training_steps
                    )
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        chunk_boundary_metric_sums: dict[str, torch.Tensor] | None = None

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_response_lengths = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Compute response lengths (number of non-pad tokens per response)
            pad_token_id = self.tokenizer.pad_token_id
            response_lengths = (output_ids != pad_token_id).sum(dim=-1).cpu().tolist()
            sample_response_lengths.extend(response_lengths)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            if "response_mask" not in test_batch.batch.keys():
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            if self.use_chunk_boundary_critic and self.use_critic:
                rollout_targets = reward_tensor.sum(dim=-1).float()
                test_batch, _ = self._prepare_chunk_boundary_critic_batch(
                    batch=test_batch,
                    rollout_targets=rollout_targets,
                )
                values = self._compute_values(test_batch)
                test_batch = test_batch.union(values)
                batch_chunk_boundary_metric_sums = compute_chunk_boundary_metric_sums(
                    predictions=test_batch.batch["chunk_boundary_values"],
                    targets=test_batch.batch["chunk_boundary_targets"],
                    boundary_mask=test_batch.batch["chunk_boundary_mask"],
                    boundary_weights=test_batch.batch["chunk_boundary_weights"],
                    loss_type=self._chunk_boundary_loss_type(),
                )
                if chunk_boundary_metric_sums is None:
                    chunk_boundary_metric_sums = {
                        key: value.detach().cpu() for key, value in batch_chunk_boundary_metric_sums.items()
                    }
                else:
                    for key, value in batch_chunk_boundary_metric_sums.items():
                        chunk_boundary_metric_sums[key] = chunk_boundary_metric_sums[key] + value.detach().cpu()

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            merged_result = {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
                "sample_response_lengths": sample_response_lengths,
            }
            if chunk_boundary_metric_sums is not None:
                merged_result["chunk_boundary_metric_sums"] = chunk_boundary_metric_sums
            return merged_result
        data_sources = np.concatenate(data_source_lst, axis=0)
        metric_dict = self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            sample_response_lengths,
        )
        if chunk_boundary_metric_sums is not None:
            metric_dict.update(
                finalize_chunk_boundary_metric_sums(
                    chunk_boundary_metric_sums,
                    prefix="val-aux/chunk_boundary",
                )
            )
        return metric_dict

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns, sample_response_lengths=None):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if sample_response_lengths and len(sample_response_lengths) > 0:
            response_lengths = np.array(sample_response_lengths)
            metric_dict["val-aux/response_length/min"] = response_lengths.min().item()
            metric_dict["val-aux/response_length/max"] = response_lengths.max().item()
            metric_dict["val-aux/response_length/mean"] = response_lengths.mean().item()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "reward_extra_infos_dict": {},
                "sample_response_lengths": [],
            }
        if result_b is None:
            result_b = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "reward_extra_infos_dict": {},
                "sample_response_lengths": [],
            }

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]
        sample_response_lengths = result_a.get("sample_response_lengths", []) + result_b.get("sample_response_lengths", [])

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        metric_dict = self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            sample_response_lengths,
        )
        chunk_boundary_metric_sums_a = result_a.get("chunk_boundary_metric_sums")
        chunk_boundary_metric_sums_b = result_b.get("chunk_boundary_metric_sums")
        if chunk_boundary_metric_sums_a is not None or chunk_boundary_metric_sums_b is not None:
            merged_chunk_boundary_metric_sums = {}
            for source_metric_sums in (chunk_boundary_metric_sums_a, chunk_boundary_metric_sums_b):
                if source_metric_sums is None:
                    continue
                for key, value in source_metric_sums.items():
                    merged_chunk_boundary_metric_sums[key] = merged_chunk_boundary_metric_sums.get(key, 0) + value
            metric_dict.update(
                finalize_chunk_boundary_metric_sums(
                    merged_chunk_boundary_metric_sums,
                    prefix="val-aux/chunk_boundary",
                )
            )
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            if self.use_chunk_boundary_critic:
                critic_cfg.value_loss_mode = f"chunk_boundary_{self._chunk_boundary_loss_type()}"
            elif self.config.algorithm.adv_estimator == AdvantageEstimator.PROMPT_BASELINE_REGRESSION:
                critic_cfg.value_loss_mode = "prompt_baseline_regression"
            elif self.config.algorithm.adv_estimator == AdvantageEstimator.PROMPT_BASELINE_BCE:
                critic_cfg.value_loss_mode = "prompt_baseline_bce"
            elif self.config.algorithm.adv_estimator in (
                AdvantageEstimator.PROMPT_RESIDUAL_BASELINE,
                AdvantageEstimator.PROMPT_RESIDUAL_BASELINE_RAMP,
            ):
                critic_cfg.value_loss_mode = "prompt_residual_regression"

            if self.use_legacy_worker_impl == "disable":
                if critic_cfg.value_loss_mode in {"prompt_residual_regression", "chunk_boundary_bce", "chunk_boundary_mse"}:
                    raise ValueError(
                        f"critic.value_loss_mode={critic_cfg.value_loss_mode} is currently supported only "
                        "with the legacy critic worker. Set trainer.use_legacy_worker_impl to 'auto' or 'enable'."
                    )
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        if self.config.algorithm.adv_estimator == AdvantageEstimator.PROMPT_BASELINE_BCE:
            response_mask = batch.batch.get("response_mask")
            if response_mask is None:
                response_mask = compute_response_mask(batch)
            values.batch["values"] = torch.sigmoid(values.batch["values"].float()) * response_mask.to(torch.float32)
        if self.use_chunk_boundary_critic and self._chunk_boundary_loss_type() == "bce":
            response_mask = batch.batch.get("response_mask")
            if response_mask is None:
                response_mask = compute_response_mask(batch)
            state_mask = build_response_state_mask(batch.batch.get("raw_response_mask", response_mask))
            values.batch["values"] = torch.sigmoid(values.batch["values"].float()) * response_mask.to(torch.float32)
            if "chunk_boundary_values" in values.batch:
                values.batch["chunk_boundary_values"] = (
                    torch.sigmoid(values.batch["chunk_boundary_values"].float()) * state_mask.to(torch.float32)
                )
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.get("overlong_filtering", False):
                        eos_token_id = batch.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
                        batch = apply_overlong_filtering(batch, eos_token_id=eos_token_id)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    all_training_tokens_masked = False
                    if self.use_chunk_boundary_critic:
                        with marked_timer("chunk_boundary_prepare", timing_raw, color="brown"):
                            reward_extra_infos_dict: dict[str, list]
                            batch.batch["token_level_scores"] = reward_tensor
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            rollout_targets = reward_tensor.sum(dim=-1).float()
                            batch, chunk_boundary_metrics = self._prepare_chunk_boundary_critic_batch(
                                batch=batch,
                                rollout_targets=rollout_targets,
                            )
                            metrics.update(chunk_boundary_metrics)
                            all_training_tokens_masked = batch.batch["chunk_boundary_weights"].sum().item() == 0
                            metrics["chunk_boundary/all_training_states_masked"] = int(all_training_tokens_masked)
                            if self.config.trainer.get("overlong_filtering", False):
                                metrics["overlong_filtering/all_training_tokens_masked"] = int(all_training_tokens_masked)
                    else:
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                    router_mode = getattr(
                                        self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                    )
                                    if router_mode == "R2":
                                        batch.batch.pop("routed_experts")
                                    else:
                                        old_log_prob.batch.pop("routed_experts")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            batch.batch["token_level_scores"] = reward_tensor
                            has_training_tokens = batch.batch["response_mask"].sum().item() > 0

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                                and has_training_tokens
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                                global_step=self.global_steps,
                            )
                            advantage_metrics = batch.meta_info.pop("advantage_metrics", None)
                            if advantage_metrics is not None:
                                metrics.update(advantage_metrics)
                            all_training_tokens_masked = batch.batch["response_mask"].sum().item() == 0
                            if self.config.trainer.get("overlong_filtering", False):
                                metrics["overlong_filtering/all_training_tokens_masked"] = int(all_training_tokens_masked)

                    # update critic
                    refresh_rollout_after_skip = False
                    if self.use_critic:
                        if all_training_tokens_masked:
                            refresh_rollout_after_skip = True
                            skipped_critic_metrics = {
                                "critic/vf_loss": 0.0,
                                "critic/vf_clipfrac": 0.0,
                                "critic/vpred_mean": 0.0,
                                "critic/grad_norm": 0.0,
                                "training/critic_update_skipped_no_valid_samples": 1,
                            }
                            if self.use_chunk_boundary_critic:
                                skipped_critic_metrics.update(
                                    {
                                        "critic/chunk_boundary/prediction_mean": 0.0,
                                        "critic/chunk_boundary/prediction_mean_positive": 0.0,
                                        "critic/chunk_boundary/prediction_mean_negative": 0.0,
                                        "critic/chunk_boundary/mse": 0.0,
                                        "critic/chunk_boundary/state_count": 0.0,
                                    }
                                )
                                if self._chunk_boundary_loss_type() == "bce":
                                    skipped_critic_metrics["critic/chunk_boundary/bce"] = 0.0
                            metrics.update(skipped_critic_metrics)
                        else:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)
                            metrics["training/critic_update_skipped_no_valid_samples"] = 0

                    # implement the actor update schedule
                    scheduled_actor_update = self._should_update_actor()
                    metrics["training/actor_update_interval"] = self._get_effective_actor_update_interval()
                    metrics["training/is_actor_update_step"] = int(scheduled_actor_update)
                    should_update_actor = scheduled_actor_update and not all_training_tokens_masked
                    metrics["training/actor_update_skipped_no_valid_samples"] = int(
                        scheduled_actor_update and all_training_tokens_masked
                    )
                    if should_update_actor:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                    elif scheduled_actor_update and all_training_tokens_masked:
                        refresh_rollout_after_skip = True
                        metrics.update(
                            {
                                "actor/pg_loss": 0.0,
                                "actor/grad_norm": 0.0,
                            }
                        )
                        if self.config.algorithm.get("adv_mode", "token") == "chunk":
                            metrics["actor_chunk/clipfrac"] = 0.0
                            metrics["actor_chunk/approx_kl"] = 0.0
                        if self.config.actor_rollout_ref.actor.use_kl_loss:
                            metrics["actor/kl_loss"] = 0.0

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    if should_update_actor or refresh_rollout_after_skip or self._should_refresh_rollout_after_critic_only_step():
                        # In hybrid mode with the naive checkpoint engine, rollout replicas are
                        # put to sleep after generation. On any critic-only step (critic warmup
                        # or a sparse actor update schedule), refresh the sleeping rollout so the
                        # next generation step does not run against freed GPU state.
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                    if should_update_actor:
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if self.config.algorithm.get("adv_mode", "token") == "chunk":
                            metrics["actor_chunk/clipfrac"] = actor_output_metrics.get("actor/pg_clipfrac", 0.0)
                            metrics["actor_chunk/approx_kl"] = actor_output_metrics.get("actor/ppo_kl", 0.0)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/actor_frozen": int(self._freeze_actor_updates()),
                        "training/chunk_boundary_critic_enabled": int(self.use_chunk_boundary_critic),
                    }
                )
                # collect metrics
                metrics.update(
                    compute_data_metrics(
                        batch=batch,
                        use_critic=self.use_critic and not self.use_chunk_boundary_critic,
                    )
                )
                if self.config.trainer.get("overlong_filtering", False):
                    metrics.update(compute_overlong_filtering_metrics(batch=batch))
                if self.config.algorithm.adv_estimator in {
                    AdvantageEstimator.GRPO,
                    AdvantageEstimator.GRPO_VECTORIZED,
                }:
                    metrics.update(compute_grpo_baseline_metrics(batch=batch))
                self._maybe_add_zero_critic_metrics(metrics)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                if self.config.trainer.get("overlong_filtering", False):
                    if "actor/pg_loss" in metrics:
                        metrics["overlong_filtering/actor_pg_loss_valid_only"] = metrics["actor/pg_loss"]
                    if "critic/vf_loss" in metrics:
                        metrics["overlong_filtering/critic_vf_loss_valid_only"] = metrics["critic/vf_loss"]
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
