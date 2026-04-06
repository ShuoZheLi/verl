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

from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    from verl.trainer.ppo.core_algos import AdvantageEstimator, resolve_advantage_lambdas

    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    adv_mode = config.algorithm.get("adv_mode", "token")
    if adv_mode == "chunk":
        if config.algorithm.adv_estimator != AdvantageEstimator.GAE:
            raise ValueError(
                "algorithm.adv_mode=chunk currently requires algorithm.adv_estimator=gae so the critic "
                "continues to use the standard PPO value path while only the actor advantage tensor changes."
            )
        if not use_critic:
            raise ValueError(
                "algorithm.adv_mode=chunk requires a learned critic because it builds chunk baselines from "
                "token-level critic predictions."
            )

    actor_lam, critic_lam = resolve_advantage_lambdas(
        lam=config.algorithm.lam,
        config=config.algorithm,
    )
    if (
        actor_lam != critic_lam
        and config.algorithm.adv_estimator
        not in (AdvantageEstimator.GAE, AdvantageEstimator.REVERSE_DECAY_GAE)
    ):
        raise ValueError(
            "Different algorithm.actor_lam and algorithm.critic_lam are currently supported only for "
            "algorithm.adv_estimator in {'gae', 'reverse_decay_gae'}."
        )

    if config.algorithm.adv_estimator == AdvantageEstimator.ZERO_CRITIC and (
        actor_lam != 1.0 or critic_lam != 1.0
    ):
        raise ValueError(
            "algorithm.adv_estimator=zero_critic requires effective actor and critic lambdas to be 1.0. "
            "Set algorithm.lam=1.0 or override both algorithm.actor_lam and algorithm.critic_lam to 1.0."
        )
    if config.algorithm.adv_estimator == AdvantageEstimator.REVERSE_DECAY_GAE and (
        actor_lam <= 0.0 or critic_lam <= 0.0
    ):
        raise ValueError(
            "algorithm.adv_estimator=reverse_decay_gae requires effective actor and critic lambdas > 0 "
            "because the reverse-decay kernel is undefined at lam <= 0."
        )
    if config.algorithm.adv_estimator == AdvantageEstimator.PROMPT_BASELINE_REGRESSION:
        if actor_lam != 1.0 or critic_lam != 1.0:
            raise ValueError(
                "algorithm.adv_estimator=prompt_baseline_regression requires effective actor and critic lambdas "
                "= 1.0 because it reuses the prompt-baseline reward-to-go estimator."
            )
        if config.critic.value_head_type != "scalar":
            raise ValueError(
                "algorithm.adv_estimator=prompt_baseline_regression requires critic.value_head_type=scalar "
                "because the prompt-only regression loss is currently implemented for scalar heads."
            )
    if config.algorithm.adv_estimator == AdvantageEstimator.PROMPT_BASELINE_BCE:
        if actor_lam != 1.0 or critic_lam != 1.0:
            raise ValueError(
                "algorithm.adv_estimator=prompt_baseline_bce requires effective actor and critic lambdas = 1.0 "
                "because it reuses the prompt-baseline reward-to-go estimator."
            )
        if config.algorithm.gamma != 1.0:
            raise ValueError(
                "algorithm.adv_estimator=prompt_baseline_bce requires algorithm.gamma=1.0 "
                "so the critic target matches an undiscounted success probability."
            )
        if config.algorithm.use_kl_in_reward:
            raise ValueError(
                "algorithm.adv_estimator=prompt_baseline_bce does not support algorithm.use_kl_in_reward=True "
                "because the critic target must stay in [0, 1]."
            )
        if config.critic.value_head_type != "scalar":
            raise ValueError(
                "algorithm.adv_estimator=prompt_baseline_bce requires critic.value_head_type=scalar "
                "because it interprets the critic head as a single Bernoulli logit."
            )
    if config.algorithm.adv_estimator in (
        AdvantageEstimator.PROMPT_RESIDUAL_BASELINE,
        AdvantageEstimator.PROMPT_RESIDUAL_BASELINE_RAMP,
    ):
        if actor_lam != 1.0 or critic_lam != 1.0:
            raise ValueError(
                f"algorithm.adv_estimator={config.algorithm.adv_estimator} requires effective actor and critic "
                "lambdas = 1.0 "
                "because it uses a rollout-return minus combined baseline estimator."
            )
        if config.critic.value_head_type != "scalar":
            raise ValueError(
                f"algorithm.adv_estimator={config.algorithm.adv_estimator} requires critic.value_head_type=scalar "
                "because the decomposed prompt/residual critic is currently scalar-only."
            )
        if config.critic.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(
                f"algorithm.adv_estimator={config.algorithm.adv_estimator} currently supports only "
                "critic.strategy in {'fsdp', 'fsdp2'}."
            )
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            raise ValueError(
                f"algorithm.adv_estimator={config.algorithm.adv_estimator} is currently supported only "
                "with the legacy FSDP critic worker. Set trainer.use_legacy_worker_impl to 'auto' or 'enable'."
            )
        if not 0.0 <= float(config.algorithm.prompt_residual_alpha) <= 1.0:
            raise ValueError(
                "algorithm.prompt_residual_alpha must be in [0, 1] for prompt-residual baselines, "
                f"got {config.algorithm.prompt_residual_alpha}."
            )
        if (
            config.algorithm.adv_estimator == AdvantageEstimator.PROMPT_RESIDUAL_BASELINE_RAMP
            and int(config.algorithm.prompt_residual_alpha_ramp_steps) <= 0
        ):
            raise ValueError(
                "algorithm.adv_estimator=prompt_residual_baseline_ramp requires "
                "algorithm.prompt_residual_alpha_ramp_steps > 0."
            )

    actor_update_interval = int(config.trainer.get("actor_update_interval", 1))
    if actor_update_interval < 1:
        raise ValueError(
            f"trainer.actor_update_interval must be >= 1, got {actor_update_interval}."
        )

    if config.trainer.critic_warmup < 0:
        raise ValueError(f"trainer.critic_warmup must be >= 0, got {config.trainer.critic_warmup}.")

    if config.algorithm.adv_estimator == AdvantageEstimator.ZERO_CRITIC and actor_update_interval != 1:
        print("WARNING: trainer.actor_update_interval is ignored when algorithm.adv_estimator=zero_critic.")

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
