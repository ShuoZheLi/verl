from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.value_categorical import (
    extract_value_head_spec,
    unscale_scalar_values,
    value_logits_to_probs,
    value_probs_to_scaled_scalar,
)

from value_decoding.data import ExampleRecord, score_response


class DecodingMode(str, Enum):
    ACTOR_ONLY = "actor_only"
    CRITIC_ONLY_RERANK = "critic_only_rerank"
    ACTOR_CRITIC_RERANK = "actor_critic_rerank"
    ACTOR_CRITIC_SOFT_RERANK = "actor_critic_soft_rerank"


class CandidateBuilder(str, Enum):
    TOP_K = "top_k"
    SAMPLED = "sampled"


class ActorSamplingMode(str, Enum):
    GREEDY = "greedy"
    SAMPLE = "sample"


class NormalizationType(str, Enum):
    NONE = "none"
    ZSCORE = "zscore"
    MINMAX = "minmax"


@dataclass(frozen=True)
class RunSpec:
    config_id: str
    mode: str
    candidate_builder: str | None = None
    candidate_size: int | None = None
    beta: float | None = None
    normalization: str = NormalizationType.NONE.value
    rank_temperature: float | None = None
    actor_sampling_mode: str = ActorSamplingMode.GREEDY.value
    actor_temperature: float = 1.0
    actor_top_p: float = 1.0
    actor_top_k: int = 0


@dataclass
class DecodeArtifacts:
    example_result: dict[str, Any]
    step_results: list[dict[str, Any]]


def set_decode_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _token_texts(tokenizer, token_ids: list[int]) -> list[str]:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if tokens is None:
        return [str(token_id) for token_id in token_ids]
    return [str(token) for token in tokens]


def _filter_logits(logits: torch.Tensor, *, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    filtered = logits.clone()

    if top_k > 0:
        top_k = min(top_k, filtered.shape[-1])
        kth_values = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth_values, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(filtered, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        filtered = filtered.masked_fill(indices_to_remove, float("-inf"))

    return filtered


def sample_token_from_actor(
    logits: torch.Tensor,
    *,
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    if sampling_mode == ActorSamplingMode.GREEDY.value or temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled_logits = logits.float() / temperature
    filtered_logits = _filter_logits(scaled_logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered_logits, dim=-1)
    if not torch.isfinite(probs).all() or torch.sum(probs) <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def build_candidate_ids(actor_log_probs: torch.Tensor, *, builder: str, candidate_size: int) -> torch.Tensor:
    if candidate_size <= 0:
        raise ValueError(f"candidate_size must be > 0, got {candidate_size}")

    vocab_size = actor_log_probs.shape[-1]
    candidate_size = min(candidate_size, vocab_size)

    if builder == CandidateBuilder.TOP_K.value:
        return torch.topk(actor_log_probs, k=candidate_size, dim=-1).indices.squeeze(0)

    if builder == CandidateBuilder.SAMPLED.value:
        probs = actor_log_probs.squeeze(0).exp()
        sampled = torch.multinomial(probs, num_samples=candidate_size, replacement=False)
        sampled_scores = actor_log_probs[0, sampled]
        sort_order = torch.argsort(sampled_scores, descending=True)
        return sampled[sort_order]

    raise ValueError(f"Unsupported candidate builder: {builder}")


def normalize_values(values: torch.Tensor, *, normalization: str, eps: float) -> torch.Tensor:
    if normalization == NormalizationType.NONE.value:
        return values

    if normalization == NormalizationType.ZSCORE.value:
        mean = values.mean()
        std = values.std(unbiased=False)
        return (values - mean) / (std + eps)

    if normalization == NormalizationType.MINMAX.value:
        vmin = values.min()
        vmax = values.max()
        return (values - vmin) / (vmax - vmin + eps)

    raise ValueError(f"Unsupported normalization: {normalization}")


def _value_logits_to_scalar_values(critic, values: torch.Tensor) -> torch.Tensor:
    if values.dim() == 2:
        return values.float()

    if values.dim() != 3:
        raise ValueError(f"Unexpected critic value tensor shape: {tuple(values.shape)}")

    if values.shape[-1] == 1:
        return values.squeeze(-1).float()

    spec = extract_value_head_spec(getattr(critic, "config", {}))
    probs = value_logits_to_probs(values.float())
    support = spec.support(device=probs.device, dtype=probs.dtype)
    scaled_values = value_probs_to_scaled_scalar(probs, support)
    return unscale_scalar_values(scaled_values, spec).float()


def _extract_scalar_values_from_critic_outputs(critic, outputs) -> torch.Tensor:
    if hasattr(critic, "v_head"):
        values = outputs[2]
    elif hasattr(outputs, "logits"):
        values = outputs.logits
    elif isinstance(outputs, tuple):
        values = outputs[0]
    else:
        raise TypeError(f"Unsupported critic output type: {type(outputs).__name__}")
    return _value_logits_to_scalar_values(critic, values)


@torch.inference_mode()
def critic_last_token_values(critic, input_ids: torch.Tensor) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    outputs = critic(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    values = _extract_scalar_values_from_critic_outputs(critic, outputs)
    return values[:, -1]


@torch.inference_mode()
def critic_child_values(critic, prefix_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    expanded_prefix = prefix_ids.expand(candidate_ids.shape[0], -1)
    child_ids = torch.cat([expanded_prefix, candidate_ids[:, None]], dim=1)
    return critic_last_token_values(critic, child_ids)


class ActorStepper:
    def __init__(self, model, prompt_ids: torch.Tensor, *, use_cache: bool = True):
        self.model = model
        self.sequence_ids = prompt_ids
        self.attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        self.request_cache = bool(use_cache)
        self.use_cache = False
        self.past_key_values = None
        self.current_logits = self._initialize()

    @torch.inference_mode()
    def _initialize(self) -> torch.Tensor:
        outputs = self.model(
            input_ids=self.sequence_ids,
            attention_mask=self.attention_mask,
            use_cache=self.request_cache,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        self.use_cache = bool(self.request_cache and past_key_values is not None)
        self.past_key_values = past_key_values if self.use_cache else None
        return outputs.logits[:, -1, :]

    @torch.inference_mode()
    def append(self, token_id: int) -> None:
        token_tensor = torch.tensor([[token_id]], device=self.sequence_ids.device, dtype=self.sequence_ids.dtype)
        self.sequence_ids = torch.cat([self.sequence_ids, token_tensor], dim=1)
        self.attention_mask = torch.cat(
            [self.attention_mask, torch.ones_like(token_tensor, device=self.attention_mask.device)],
            dim=1,
        )

        if self.use_cache and self.past_key_values is not None:
            try:
                outputs = self.model(
                    input_ids=token_tensor,
                    attention_mask=self.attention_mask,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
            except Exception:
                self.use_cache = False
                self.past_key_values = None
                outputs = self.model(
                    input_ids=self.sequence_ids,
                    attention_mask=self.attention_mask,
                    use_cache=False,
                )
        else:
            outputs = self.model(
                input_ids=self.sequence_ids,
                attention_mask=self.attention_mask,
                use_cache=False,
            )

        if self.use_cache:
            self.past_key_values = getattr(outputs, "past_key_values", None)
            self.use_cache = self.past_key_values is not None
        self.current_logits = outputs.logits[:, -1, :]


def decode_example(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    run_spec: RunSpec,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    actor_device: torch.device,
    critic_device: torch.device,
    seed: int,
    normalization_eps: float = 1e-6,
    use_actor_cache: bool = True,
    debug_full_candidates: bool = False,
) -> DecodeArtifacts:
    set_decode_seed(seed)

    if example.prompt_token_ids is not None:
        prompt_ids = torch.tensor(
            [list(example.prompt_token_ids)],
            device=actor_device,
            dtype=torch.long,
        )
    else:
        tokenized = tokenizer(
            example.prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        prompt_ids = tokenized["input_ids"].to(actor_device)
    prompt_length = int(prompt_ids.shape[1])

    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    generated_token_ids: list[int] = []
    step_results: list[dict[str, Any]] = []

    sum_logp = 0.0
    sum_value = 0.0
    choice_change_count = 0
    eos_emitted = False
    last_step_value: float | None = None

    start_time = time.perf_counter()
    for step_index in range(max_new_tokens):
        logits = actor_state.current_logits
        actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
        actor_top1_id = int(torch.argmax(actor_log_probs, dim=-1).item())

        selected_token_id: int
        selected_logp: float
        selected_value: float
        selected_rank_in_candidates: int | None = None
        candidate_ids_list: list[int] | None = None
        candidate_logps_list: list[float] | None = None
        candidate_values_list: list[float] | None = None
        candidate_scores_list: list[float] | None = None
        candidate_norm_values_list: list[float] | None = None
        candidate_selection_probs_list: list[float] | None = None

        if run_spec.mode == DecodingMode.ACTOR_ONLY.value:
            selected_token_id = sample_token_from_actor(
                logits.squeeze(0),
                sampling_mode=run_spec.actor_sampling_mode,
                temperature=run_spec.actor_temperature,
                top_p=run_spec.actor_top_p,
                top_k=run_spec.actor_top_k,
            )
            selected_logp = float(actor_log_probs[0, selected_token_id].item())
            critic_prefix_ids = actor_state.sequence_ids.to(critic_device)
            selected_value = float(
                critic_child_values(
                    critic,
                    critic_prefix_ids,
                    torch.tensor([selected_token_id], device=critic_device, dtype=critic_prefix_ids.dtype),
                )[0].item()
            )
        else:
            if run_spec.candidate_builder is None or run_spec.candidate_size is None:
                raise ValueError(f"Run spec {run_spec.config_id} is missing candidate settings")

            candidate_ids = build_candidate_ids(
                actor_log_probs,
                builder=run_spec.candidate_builder,
                candidate_size=run_spec.candidate_size,
            )
            candidate_logps = actor_log_probs[0, candidate_ids].float()
            critic_candidate_ids = candidate_ids.to(critic_device)
            critic_prefix_ids = actor_state.sequence_ids.to(critic_device)
            candidate_values = critic_child_values(critic, critic_prefix_ids, critic_candidate_ids).float()
            candidate_values = candidate_values.to(candidate_logps.device)

            candidate_scores = candidate_values
            if run_spec.mode == DecodingMode.CRITIC_ONLY_RERANK.value:
                selected_rank_in_candidates = int(torch.argmax(candidate_scores).item())
            else:
                normalized_values = normalize_values(
                    candidate_values,
                    normalization=run_spec.normalization,
                    eps=normalization_eps,
                )
                candidate_scores = candidate_logps + float(run_spec.beta) * normalized_values

                if run_spec.mode == DecodingMode.ACTOR_CRITIC_RERANK.value or (
                    run_spec.rank_temperature is not None and run_spec.rank_temperature <= 0.0
                ):
                    selected_rank_in_candidates = int(torch.argmax(candidate_scores).item())
                else:
                    selection_probs = torch.softmax(candidate_scores / float(run_spec.rank_temperature), dim=-1)
                    selected_rank_in_candidates = int(torch.multinomial(selection_probs, num_samples=1).item())
                    candidate_selection_probs_list = [float(value) for value in selection_probs.tolist()]

                candidate_norm_values_list = [float(value) for value in normalized_values.tolist()]

            selected_token_id = int(candidate_ids[selected_rank_in_candidates].item())
            selected_logp = float(candidate_logps[selected_rank_in_candidates].item())
            selected_value = float(candidate_values[selected_rank_in_candidates].item())

            if debug_full_candidates:
                candidate_ids_list = [int(value) for value in candidate_ids.tolist()]
                candidate_logps_list = [float(value) for value in candidate_logps.tolist()]
                candidate_values_list = [float(value) for value in candidate_values.tolist()]
                candidate_scores_list = [float(value) for value in candidate_scores.tolist()]
                if candidate_norm_values_list is None and run_spec.mode != DecodingMode.CRITIC_ONLY_RERANK.value:
                    candidate_norm_values_list = [float(value) for value in normalized_values.tolist()]

        changed_from_actor_top1 = selected_token_id != actor_top1_id
        if changed_from_actor_top1:
            choice_change_count += 1

        generated_token_ids.append(selected_token_id)
        sum_logp += selected_logp
        sum_value += selected_value
        last_step_value = selected_value

        step_result: dict[str, Any] = {
            "config_id": run_spec.config_id,
            "example_id": example.example_id,
            "step_index": step_index,
            "selected_token_id": selected_token_id,
            "selected_token_text": _token_texts(tokenizer, [selected_token_id])[0],
            "actor_top1_token_id": actor_top1_id,
            "actor_top1_token_text": _token_texts(tokenizer, [actor_top1_id])[0],
            "different_from_actor_top1": changed_from_actor_top1,
            "selected_token_actor_rank_in_candidates": selected_rank_in_candidates,
            "selected_token_logprob": selected_logp,
            "selected_token_critic_value": selected_value,
        }

        if debug_full_candidates and candidate_ids_list is not None:
            step_result.update(
                {
                    "candidate_token_ids": candidate_ids_list,
                    "candidate_token_texts": _token_texts(tokenizer, candidate_ids_list),
                    "candidate_actor_logprobs": candidate_logps_list,
                    "candidate_critic_values": candidate_values_list,
                    "candidate_scores": candidate_scores_list,
                    "candidate_normalized_values": candidate_norm_values_list,
                    "candidate_selection_probabilities": candidate_selection_probs_list,
                }
            )

        step_results.append(step_result)
        actor_state.append(selected_token_id)

        if selected_token_id in eos_token_ids:
            eos_emitted = True
            break

    latency_sec = time.perf_counter() - start_time
    total_steps = len(generated_token_ids)
    max_length_hit = bool(max_new_tokens > 0 and not eos_emitted and total_steps >= max_new_tokens)
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    trajectory_value = float(critic_last_token_values(critic, actor_state.sequence_ids.to(critic_device))[0].item())
    trajectory_value_matches_last_step = None
    trajectory_value_last_step_abs_diff = None
    if last_step_value is not None:
        trajectory_value_last_step_abs_diff = abs(trajectory_value - last_step_value)
        trajectory_value_matches_last_step = math.isclose(
            trajectory_value,
            last_step_value,
            rel_tol=1e-2,
            abs_tol=1e-2,
        )

    mean_logp = (sum_logp / total_steps) if total_steps > 0 else None
    mean_value = (sum_value / total_steps) if total_steps > 0 else None
    task_score = score_response(example, response_text)

    example_result = {
        "config_id": run_spec.config_id,
        "mode": run_spec.mode,
        "example_id": example.example_id,
        "data_source": example.data_source,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "candidate_builder": run_spec.candidate_builder,
        "candidate_size": run_spec.candidate_size,
        "beta": run_spec.beta,
        "normalization": run_spec.normalization,
        "rank_temperature": run_spec.rank_temperature,
        "actor_sampling_mode": run_spec.actor_sampling_mode,
        "actor_temperature": run_spec.actor_temperature,
        "actor_top_p": run_spec.actor_top_p,
        "actor_top_k": run_spec.actor_top_k,
        "prompt_length": prompt_length,
        "generated_response": response_text,
        "response_length": total_steps,
        "eos_emitted": eos_emitted,
        "max_length_hit": max_length_hit,
        "task_score": task_score,
        "sum_chosen_token_actor_logprob": sum_logp,
        "sum_chosen_token_critic_value": sum_value,
        "mean_chosen_token_actor_logprob": mean_logp,
        "mean_chosen_token_critic_value": mean_value,
        "choice_change_count": choice_change_count,
        "choice_change_rate": (choice_change_count / total_steps) if total_steps > 0 else 0.0,
        "total_decoding_steps": total_steps,
        "trajectory_value": trajectory_value,
        "trajectory_value_matches_last_step": trajectory_value_matches_last_step,
        "trajectory_value_last_step_abs_diff": trajectory_value_last_step_abs_diff,
        "latency_sec": latency_sec,
        "tokens_per_second": (total_steps / latency_sec) if latency_sec > 0 else None,
    }
    return DecodeArtifacts(example_result=example_result, step_results=step_results)
