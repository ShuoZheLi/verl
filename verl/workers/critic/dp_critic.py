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
"""
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.chunk_boundary_value import build_response_state_mask
from verl.trainer.ppo.value_categorical import extract_value_head_spec, value_logits_to_scalar_expectation
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.recurrent_value_head import StatefulValueHead
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.value_spec = extract_value_head_spec(self.config)
        if self.value_spec.is_categorical() and hasattr(self.critic_module, "v_head"):
            raise ValueError(
                "Categorical critic requires token-classification logits. "
                "TRL AutoModelForCausalLMWithValueHead is scalar-only."
            )
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

    def _collect_stateful_value_head_metrics(self) -> dict[str, float]:
        for module in self.critic_module.modules():
            if isinstance(module, StatefulValueHead):
                metrics = module.get_debug_metrics()
                module.clear_debug_metrics()
                return metrics
        return {}

    def _use_prompt_residual_regression(self) -> bool:
        return self.config.value_loss_mode == "prompt_residual_regression"

    def _use_chunk_boundary_value_loss(self) -> bool:
        return self.config.value_loss_mode in {"chunk_boundary_bce", "chunk_boundary_mse"}

    def _chunk_boundary_loss_type(self) -> str:
        if self.config.value_loss_mode == "chunk_boundary_bce":
            return "bce"
        if self.config.value_loss_mode == "chunk_boundary_mse":
            return "mse"
        raise ValueError(f"Unsupported chunk-boundary value_loss_mode: {self.config.value_loss_mode}.")

    def _value_output_slice(self, response_length: int) -> slice:
        if self._use_chunk_boundary_value_loss():
            return slice(-response_length - 1, None)
        return slice(-response_length - 1, -1)

    def _prompt_prior_head(self) -> nn.Module:
        prompt_prior_head = getattr(self.critic_module, "prompt_prior_head", None)
        if not isinstance(prompt_prior_head, nn.Module):
            raise RuntimeError(
                "Prompt-residual critic expected `critic_module.prompt_prior_head`, but it was not found."
            )
        return prompt_prior_head

    def _compute_prompt_prior_values(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        response_length: int,
    ) -> torch.Tensor:
        prompt_mask = attention_mask[:, :-response_length]
        prompt_lengths = prompt_mask.sum(dim=-1)
        if torch.any(prompt_lengths <= 0):
            raise ValueError(
                "Prompt-residual critic requires at least one prompt token per sample so the prompt prior "
                "can read the prompt-end hidden state."
            )
        prompt_end_indices = (prompt_lengths - 1).to(dtype=torch.long)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        prompt_end_hidden = hidden_states[batch_indices, prompt_end_indices]
        prompt_prior_values = self._prompt_prior_head()(prompt_end_hidden)
        return prompt_prior_values.squeeze(-1)

    def _forward_micro_batch(self, micro_batch):
        is_categorical_value = self.value_spec.is_categorical()
        use_prompt_residual = self._use_prompt_residual_regression()
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    output_hidden_states=use_prompt_residual,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    if is_categorical_value:
                        raise ValueError(
                            "Categorical critic requires logits over bins from a token-classification head. "
                            "TRL value heads are scalar-only."
                        )
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)
                    if not is_categorical_value:
                        values_rmpad = values_rmpad.squeeze(-1)

                hidden_states_rmpad = None
                if use_prompt_residual:
                    if hasattr(self.critic_module, "v_head"):
                        raise RuntimeError(
                            "Prompt-residual critic requires a HuggingFace token-classification model. "
                            "TRL value-head critics are not supported."
                        )
                    hidden_states_rmpad = output.hidden_states[-1].squeeze(0)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    if hidden_states_rmpad is not None:
                        hidden_states_rmpad = gather_outputs_and_unpad(
                            hidden_states_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                if not is_categorical_value:
                    values = values.squeeze(-1)
                values = values[:, self._value_output_slice(response_length)]
                hidden_states = None
                if hidden_states_rmpad is not None:
                    hidden_states = pad_input(hidden_states_rmpad, indices=indices, batch=batch, seqlen=seqlen)
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    output_hidden_states=use_prompt_residual,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    if is_categorical_value:
                        raise ValueError(
                            "Categorical critic requires logits over bins from a token-classification head. "
                            "TRL value heads are scalar-only."
                        )
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, self._value_output_slice(response_length)]
                if not is_categorical_value:
                    values = values.squeeze(-1)
                hidden_states = output.hidden_states[-1] if use_prompt_residual else None

            if not use_prompt_residual:
                return values

            if is_categorical_value:
                raise ValueError("Prompt-residual critic currently requires critic.value_head_type=scalar.")
            if hidden_states is None:
                raise RuntimeError("Prompt-residual critic expected final hidden states, but none were returned.")

            prompt_prior_values = self._compute_prompt_prior_values(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                response_length=response_length,
            )
            residual_values = values
            combined_values = prompt_prior_values.unsqueeze(-1) + residual_values
            return {
                "values": combined_values,
                "prompt_prior_values": prompt_prior_values,
                "residual_values": residual_values,
            }

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor | dict[str, torch.Tensor]:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        output_lists: dict[str, list[torch.Tensor]] = {"values": []}
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                model_output = self._forward_micro_batch(model_inputs)
            if isinstance(model_output, dict):
                for key, value in model_output.items():
                    output_lists.setdefault(key, []).append(value)
            else:
                output_lists["values"].append(model_output)

        critic_outputs = {key: torch.concat(value_list, dim=0) for key, value_list in output_lists.items()}

        if use_dynamic_bsz:
            critic_outputs = {
                key: restore_dynamic_batch(value, batch_idx_list) for key, value in critic_outputs.items()
            }

        if self.value_spec.is_categorical():
            critic_outputs["values"], _, _ = value_logits_to_scalar_expectation(critic_outputs["values"], self.value_spec)

        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"].to(critic_outputs["values"].device)
            if self._use_chunk_boundary_value_loss():
                boundary_values = critic_outputs["values"]
                state_mask = build_response_state_mask(response_mask)
                critic_outputs["chunk_boundary_values"] = boundary_values * state_mask.to(dtype=boundary_values.dtype)
                critic_outputs["values"] = boundary_values[:, :-1] * response_mask.to(dtype=boundary_values.dtype)
            else:
                critic_outputs["values"] = critic_outputs["values"] * response_mask  # Only action tokens have values
            if "residual_values" in critic_outputs:
                critic_outputs["residual_values"] = critic_outputs["residual_values"] * response_mask

        if len(critic_outputs) == 1:
            return critic_outputs["values"]
        return critic_outputs

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {
            "critic/vf_loss": 0.0,
        }

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids"]
        if self._use_prompt_residual_regression():
            select_keys.append("rollout_returns")
        elif self._use_chunk_boundary_value_loss():
            select_keys.extend(["chunk_boundary_mask", "chunk_boundary_targets", "chunk_boundary_weights"])
        else:
            select_keys.extend(["values", "returns"])
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]

                    model_output = self._forward_micro_batch(model_inputs)
                    if self.config.value_loss_mode == "prompt_residual_regression":
                        if self.value_spec.is_categorical():
                            raise ValueError(
                                "critic.value_loss_mode=prompt_residual_regression requires "
                                "critic.value_head_type=scalar."
                            )
                        if not isinstance(model_output, dict):
                            raise RuntimeError(
                                "Prompt-residual critic expected structured forward outputs with prompt and residual "
                                "values, but received a tensor."
                            )
                        rollout_returns = model_inputs["rollout_returns"]
                        vf_loss, vf_clipfrac, vpreds, categorical_metrics = (
                            core_algos.compute_prompt_residual_regression_value_loss(
                                prompt_prior_vpreds=model_output["prompt_prior_values"],
                                residual_vpreds=model_output["residual_values"],
                                rollout_returns=rollout_returns,
                                response_mask=response_mask,
                                prompt_loss_weight=self.config.prompt_residual_prompt_loss_weight,
                                residual_loss_weight=self.config.prompt_residual_residual_loss_weight,
                                cliprange_value=self.config.cliprange_value,
                                loss_agg_mode=self.config.loss_agg_mode,
                            )
                        )
                    elif self.config.value_loss_mode == "chunk_boundary_bce":
                        if self.value_spec.is_categorical():
                            raise ValueError(
                                "critic.value_loss_mode=chunk_boundary_bce requires critic.value_head_type=scalar."
                            )
                        vf_loss, vf_clipfrac, vpreds, categorical_metrics = core_algos.compute_chunk_boundary_bce_value_loss(
                            vpred_logits=model_output,
                            boundary_targets=model_inputs["chunk_boundary_targets"],
                            boundary_mask=model_inputs["chunk_boundary_mask"],
                            boundary_weights=model_inputs["chunk_boundary_weights"],
                        )
                    elif self.config.value_loss_mode == "chunk_boundary_mse":
                        if self.value_spec.is_categorical():
                            raise ValueError(
                                "critic.value_loss_mode=chunk_boundary_mse requires critic.value_head_type=scalar."
                            )
                        vf_loss, vf_clipfrac, vpreds, categorical_metrics = core_algos.compute_chunk_boundary_mse_value_loss(
                            vpreds=model_output,
                            boundary_targets=model_inputs["chunk_boundary_targets"],
                            boundary_mask=model_inputs["chunk_boundary_mask"],
                            boundary_weights=model_inputs["chunk_boundary_weights"],
                        )
                    elif self.config.value_loss_mode == "prompt_baseline_regression":
                        if self.value_spec.is_categorical():
                            raise ValueError(
                                "critic.value_loss_mode=prompt_baseline_regression requires "
                                "critic.value_head_type=scalar."
                            )
                        values = model_inputs["values"]
                        returns = model_inputs["returns"]
                        vpreds = model_output
                        vf_loss, vf_clipfrac, vpreds, categorical_metrics = (
                            core_algos.compute_prompt_baseline_regression_value_loss(
                                vpreds=vpreds,
                                values=values,
                                returns=returns,
                                response_mask=response_mask,
                                cliprange_value=self.config.cliprange_value,
                                loss_agg_mode=self.config.loss_agg_mode,
                            )
                        )
                    elif self.config.value_loss_mode == "prompt_baseline_bce":
                        if self.value_spec.is_categorical():
                            raise ValueError(
                                "critic.value_loss_mode=prompt_baseline_bce requires critic.value_head_type=scalar."
                            )
                        values = model_inputs["values"]
                        returns = model_inputs["returns"]
                        vpred_logits = model_output
                        vf_loss, vf_clipfrac, vpreds, categorical_metrics = (
                            core_algos.compute_prompt_baseline_bce_value_loss(
                                vpred_logits=vpred_logits,
                                values=values,
                                returns=returns,
                                response_mask=response_mask,
                                cliprange_value=self.config.cliprange_value,
                                loss_agg_mode=self.config.loss_agg_mode,
                            )
                        )
                    elif self.value_spec.is_categorical():
                        values = model_inputs["values"]
                        returns = model_inputs["returns"]
                        vf_loss, vf_clipfrac, vpreds, categorical_metrics = core_algos.compute_categorical_value_loss(
                            value_logits=model_output,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                            value_spec=self.value_spec,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                    else:
                        values = model_inputs["values"]
                        returns = model_inputs["returns"]
                        vpreds = model_output
                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        categorical_metrics = {}
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        loss = vf_loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = vf_loss * loss_scale_factor

                    loss.backward()

                    if self._use_chunk_boundary_value_loss():
                        boundary_mask = model_inputs["chunk_boundary_mask"].to(dtype=torch.float32)
                        micro_batch_metrics.update(
                            {
                                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                                "critic/vpred_mean": masked_mean(vpreds, boundary_mask).detach().item(),
                            }
                        )
                    else:
                        micro_batch_metrics.update(
                            {
                                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                            }
                        )
                    micro_batch_metrics.update(self._collect_stateful_value_head_metrics())
                    for metric_name, metric_value in categorical_metrics.items():
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.detach().item()
                        micro_batch_metrics[metric_name] = metric_value

                    metrics["critic/vf_loss"] += vf_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                learning_rate = self.critic_optimizer.param_groups[-1]["lr"]
                mini_batch_metrics = {
                    "critic/grad_norm": grad_norm.detach().item(),
                    "critic/lr": learning_rate,
                }
                append_to_dict(metrics, mini_batch_metrics)
        self.critic_optimizer.zero_grad()
        return metrics
