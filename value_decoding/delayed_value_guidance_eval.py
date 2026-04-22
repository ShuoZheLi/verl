from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import multiprocessing as mp
import os
from queue import Empty
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled explicitly at runtime
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

try:
    import ray
except ImportError:
    ray = None

from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
    load_actor_model,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.chunk_guidance_eval import (
    RAY_PROGRESS_POLL_INTERVAL_SEC,
    _build_ray_node_execution_specs,
    _discover_ray_nodes,
    _make_main_module_importable,
    _progress_postfix,
    _require_ray,
    _resolve_ray_address,
    _validate_visible_cuda_device,
)
from value_decoding.data import ExampleRecord, load_examples, score_response
from value_decoding.decoding import (
    ActorSamplingMode,
    ActorStepper,
    ActorStepperSnapshot,
    build_candidate_ids,
    critic_child_values,
    sample_token_from_actor,
    set_decode_seed,
)
from value_decoding.multi_worker import (
    RayNodeInfo,
    WorkerAssignment,
    build_distributed_worker_assignments,
    build_worker_assignments,
    parse_worker_pairs,
    worker_assignments_to_jsonable,
)


DEFAULT_START_FRACTIONS = (0, 25, 50, 75)
DEFAULT_CANDIDATE_SIZE = 8


@dataclass(frozen=True)
class StartSpec:
    percentage: int
    fraction: float
    label: str
    display_label: str


@dataclass(frozen=True)
class ActorTrajectory:
    full_response_token_ids: tuple[int, ...]
    full_response_text: str
    response_length: int
    continuation_length: int
    eos_emitted: bool
    max_length_hit: bool
    stop_reason: str
    task_score: float
    sum_actor_logprob: float
    mean_actor_logprob: float | None
    latency_sec: float
    tokens_per_second: float | None


@dataclass(frozen=True)
class ValueGuidedTrajectory:
    full_response_token_ids: tuple[int, ...]
    full_response_text: str
    response_length: int
    continuation_length: int
    eos_emitted: bool
    max_length_hit: bool
    stop_reason: str
    task_score: float
    sum_selected_token_actor_logprob: float
    mean_selected_token_actor_logprob: float | None
    sum_selected_token_value: float
    mean_selected_token_value: float | None
    mean_selected_token_score_margin: float | None
    choice_change_count: int
    choice_change_rate: float | None
    num_decoding_steps: int
    latency_sec: float
    tokens_per_second: float | None


@dataclass
class ExampleSeedArtifacts:
    per_start_records: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delayed-onset token-level value-guidance evaluation. For each prompt and seed, sample one "
            "reference actor rollout, define delayed start positions from the realized reference length, "
            "then compare matched actor-only continuation against deterministic token-level value guidance."
        )
    )
    parser.add_argument("--actor_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the frozen actor.")
    parser.add_argument("--critic_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the frozen critic.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for experiment artifacts.")
    parser.add_argument("--actor_merged_root", type=str, default=None, help="Optional merged HF root for actor.")
    parser.add_argument("--critic_merged_root", type=str, default=None, help="Optional merged HF root for critic.")
    parser.add_argument(
        "--actor_hf_source_dir",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face config/tokenizer directory for the actor merge. "
            "Use this when the raw actor checkpoint was copied without actor/huggingface."
        ),
    )
    parser.add_argument(
        "--critic_hf_source_dir",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face config/tokenizer directory for the critic merge. "
            "Use this when the raw critic checkpoint was copied without critic/huggingface."
        ),
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None, help="Optional response/ground-truth column key.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Fallback device for both actor and critic.")
    parser.add_argument("--actor_device", type=str, default=None, help="Optional actor device override.")
    parser.add_argument("--critic_device", type=str, default=None, help="Optional critic device override.")
    parser.add_argument(
        "--worker_pairs",
        nargs="+",
        default=None,
        help=(
            "Optional prompt-sharded worker layouts. Each entry should be 'actor_device,critic_device' "
            "or a single device to reuse for both."
        ),
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help=(
            "Optional Ray cluster address for cross-node execution. When set, --worker_pairs is treated as the "
            "node-local worker layout and is replicated across all alive Ray nodes. Use 'auto' to read $RAY_ADDRESS."
        ),
    )
    parser.add_argument(
        "--ray_num_cpus_per_worker",
        type=float,
        default=1.0,
        help="CPU resources reserved per Ray worker task when --ray_address is used.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--disable_actor_cache", action="store_true")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seed values. Each seed independently samples a reference rollout and actor-only continuations.",
    )
    parser.add_argument(
        "--actor_sampling_mode",
        type=str,
        default=ActorSamplingMode.SAMPLE.value,
        choices=[mode.value for mode in ActorSamplingMode],
    )
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--actor_top_p", type=float, default=1.0)
    parser.add_argument("--actor_top_k", type=int, default=0)
    parser.add_argument(
        "--start_fractions",
        nargs="+",
        type=int,
        default=list(DEFAULT_START_FRACTIONS),
        help="Delayed-onset positions as integer percentages of the realized reference rollout length.",
    )
    parser.add_argument(
        "--candidate_size",
        type=int,
        default=DEFAULT_CANDIDATE_SIZE,
        help="Top-K actor next tokens scored by the critic at each value-guided step.",
    )
    parser.add_argument("--skip_plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--plot_dpi", type=int, default=160)
    return parser.parse_args()


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True) + "\n"


def _serialize_start_specs(start_specs: Sequence[StartSpec]) -> list[dict[str, Any]]:
    return [asdict(spec) for spec in start_specs]


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _is_binary_scores(values: Sequence[float]) -> bool:
    if not values:
        return False
    for value in values:
        if value not in {0.0, 1.0}:
            return False
    return True


def _build_start_specs(start_percentages: Sequence[int]) -> list[StartSpec]:
    if not start_percentages:
        raise ValueError("At least one start fraction is required.")
    seen: set[int] = set()
    specs: list[StartSpec] = []
    for percentage in start_percentages:
        if percentage in seen:
            raise ValueError(f"Duplicate start fraction percentage is not allowed: {percentage}")
        if percentage < 0 or percentage > 100:
            raise ValueError(f"start fraction percentage must be between 0 and 100, got {percentage}")
        seen.add(int(percentage))
        specs.append(
            StartSpec(
                percentage=int(percentage),
                fraction=float(percentage) / 100.0,
                label=str(int(percentage)),
                display_label=f"{int(percentage)}%",
            )
        )
    return specs


def _progress_units_per_example_seed(start_specs: Sequence[StartSpec]) -> int:
    # One shared reference rollout plus two continuations (actor-only, value-guided)
    # for each delayed start position.
    return 1 + 2 * len(start_specs)


def _prompt_ids_tensor(
    *,
    example: ExampleRecord,
    tokenizer,
    max_prompt_length: int,
    device: torch.device,
) -> torch.Tensor:
    if example.prompt_token_ids is not None:
        prompt_ids = list(example.prompt_token_ids)
    else:
        tokenized = tokenizer(
            example.prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        prompt_ids = tokenized["input_ids"][0].tolist()
    return torch.tensor([prompt_ids], device=device, dtype=torch.long)


def _seed_to_int(seed_value: int) -> int:
    return int(seed_value)


def _reference_rollout_seed(base_seed: int, *, example_id: int) -> int:
    return int(_seed_to_int(base_seed) + (int(example_id) + 1) * 1_000_003 + 101)


def _actor_continuation_seed(base_seed: int, *, example_id: int, start_token_index: int) -> int:
    return int(_seed_to_int(base_seed) + (int(example_id) + 1) * 1_000_003 + int(start_token_index) * 10_007 + 211)


def _value_guidance_seed(base_seed: int, *, example_id: int, start_token_index: int) -> int:
    return int(_seed_to_int(base_seed) + (int(example_id) + 1) * 1_000_003 + int(start_token_index) * 10_007 + 307)


def _response_has_eos(response_token_ids: Sequence[int], eos_token_ids: Sequence[int]) -> bool:
    return bool(response_token_ids) and int(response_token_ids[-1]) in set(int(token_id) for token_id in eos_token_ids)


def _prefix_ids_tensor(
    *,
    prompt_ids: torch.Tensor,
    prefix_response_token_ids: Sequence[int],
) -> torch.Tensor:
    if not prefix_response_token_ids:
        return prompt_ids
    prefix_tensor = torch.tensor(
        [list(int(token_id) for token_id in prefix_response_token_ids)],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype,
    )
    return torch.cat([prompt_ids, prefix_tensor], dim=1)


def _decode_response_text(tokenizer, response_token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(int(token_id) for token_id in response_token_ids), skip_special_tokens=True)


def _build_stop_reason(*, eos_emitted: bool, max_length_hit: bool, zero_budget: bool) -> str:
    if eos_emitted:
        return "eos"
    if max_length_hit:
        return "max_length"
    if zero_budget:
        return "zero_budget"
    return "stopped"


def _build_actor_start_snapshots(
    *,
    actor,
    prompt_ids: torch.Tensor,
    reference_response_token_ids: Sequence[int],
    start_token_indices: Sequence[int],
    use_actor_cache: bool,
) -> dict[int, ActorStepperSnapshot]:
    needed_positions = sorted(set(int(position) for position in start_token_indices))
    if not needed_positions:
        return {}

    snapshots: dict[int, ActorStepperSnapshot] = {}
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    if 0 in needed_positions:
        snapshots[0] = actor_state.snapshot()

    for current_length, token_id in enumerate(reference_response_token_ids, start=1):
        actor_state.append(int(token_id))
        if current_length in needed_positions and current_length not in snapshots:
            snapshots[current_length] = actor_state.snapshot()
        if len(snapshots) == len(needed_positions):
            break

    missing_positions = [position for position in needed_positions if position not in snapshots]
    if missing_positions:
        raise RuntimeError(
            "Failed to reconstruct actor snapshots for delayed start positions: "
            + ", ".join(str(position) for position in missing_positions)
        )
    return snapshots


def sample_actor_response_from_prefix(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    prefix_ids: torch.Tensor,
    prefix_response_token_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    use_actor_cache: bool,
    actor_state_snapshot: ActorStepperSnapshot | None = None,
) -> ActorTrajectory:
    prefix_response = [int(token_id) for token_id in prefix_response_token_ids]
    prefix_length = len(prefix_response)
    remaining_budget = max(int(max_new_tokens) - prefix_length, 0)
    prefix_has_eos = _response_has_eos(prefix_response, eos_token_ids)

    continuation_token_ids: list[int] = []
    sum_actor_logprob = 0.0
    eos_emitted = bool(prefix_has_eos)
    latency_sec = 0.0

    if not prefix_has_eos and remaining_budget > 0:
        set_decode_seed(seed)
        actor_state = (
            ActorStepper.from_snapshot(actor, actor_state_snapshot)
            if actor_state_snapshot is not None
            else ActorStepper(actor, prefix_ids, use_cache=use_actor_cache)
        )
        start_time = time.perf_counter()
        for _step_index in range(remaining_budget):
            logits = actor_state.current_logits
            actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
            token_id = sample_token_from_actor(
                logits.squeeze(0),
                sampling_mode=sampling_mode,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            sum_actor_logprob += float(actor_log_probs[0, token_id].item())
            continuation_token_ids.append(int(token_id))
            actor_state.append(int(token_id))
            if int(token_id) in eos_token_ids:
                eos_emitted = True
                break
        latency_sec = time.perf_counter() - start_time

    full_response_token_ids = tuple(prefix_response + continuation_token_ids)
    response_text = _decode_response_text(tokenizer, full_response_token_ids)
    continuation_length = len(continuation_token_ids)
    response_length = len(full_response_token_ids)
    max_length_hit = bool(not eos_emitted and response_length >= int(max_new_tokens))
    zero_budget = bool(not prefix_has_eos and remaining_budget == 0)
    stop_reason = _build_stop_reason(
        eos_emitted=eos_emitted,
        max_length_hit=max_length_hit,
        zero_budget=zero_budget,
    )
    task_score = float(score_response(example, response_text))
    mean_actor_logprob = (sum_actor_logprob / continuation_length) if continuation_length > 0 else None
    tokens_per_second = (continuation_length / latency_sec) if latency_sec > 0 else None
    return ActorTrajectory(
        full_response_token_ids=full_response_token_ids,
        full_response_text=response_text,
        response_length=response_length,
        continuation_length=continuation_length,
        eos_emitted=eos_emitted,
        max_length_hit=max_length_hit,
        stop_reason=stop_reason,
        task_score=task_score,
        sum_actor_logprob=float(sum_actor_logprob),
        mean_actor_logprob=mean_actor_logprob,
        latency_sec=float(latency_sec),
        tokens_per_second=tokens_per_second,
    )


def run_value_guided_from_prefix(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    prefix_ids: torch.Tensor,
    prefix_response_token_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    candidate_size: int,
    actor_device: torch.device,
    critic_device: torch.device,
    seed: int,
    use_actor_cache: bool,
    actor_state_snapshot: ActorStepperSnapshot | None = None,
) -> ValueGuidedTrajectory:
    if candidate_size <= 0:
        raise ValueError(f"candidate_size must be > 0, got {candidate_size}")

    prefix_response = [int(token_id) for token_id in prefix_response_token_ids]
    prefix_length = len(prefix_response)
    remaining_budget = max(int(max_new_tokens) - prefix_length, 0)
    prefix_has_eos = _response_has_eos(prefix_response, eos_token_ids)

    continuation_token_ids: list[int] = []
    selected_values: list[float] = []
    selected_score_margins: list[float] = []
    selected_actor_logprobs: list[float] = []
    choice_change_count = 0
    eos_emitted = bool(prefix_has_eos)
    latency_sec = 0.0

    if not prefix_has_eos and remaining_budget > 0:
        set_decode_seed(seed)
        actor_state = (
            ActorStepper.from_snapshot(actor, actor_state_snapshot)
            if actor_state_snapshot is not None
            else ActorStepper(actor, prefix_ids, use_cache=use_actor_cache)
        )
        start_time = time.perf_counter()
        for _step_index in range(remaining_budget):
            logits = actor_state.current_logits
            actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
            actor_top1_id = int(torch.argmax(actor_log_probs, dim=-1).item())

            candidate_ids = build_candidate_ids(
                actor_log_probs,
                builder="top_k",
                candidate_size=int(candidate_size),
            )
            candidate_logprobs = actor_log_probs[0, candidate_ids].float()
            critic_prefix_ids = actor_state.sequence_ids.to(critic_device)
            critic_candidate_ids = candidate_ids.to(critic_device)
            candidate_values = critic_child_values(critic, critic_prefix_ids, critic_candidate_ids).float()
            candidate_values = candidate_values.to(candidate_logprobs.device)

            selected_rank = int(torch.argmax(candidate_values).item())
            selected_token_id = int(candidate_ids[selected_rank].item())
            selected_value = float(candidate_values[selected_rank].item())
            selected_actor_logprob = float(candidate_logprobs[selected_rank].item())

            if selected_token_id != actor_top1_id:
                choice_change_count += 1

            if candidate_values.numel() > 1:
                sorted_values = torch.sort(candidate_values, descending=True).values
                selected_score_margins.append(float(sorted_values[0].item() - sorted_values[1].item()))

            continuation_token_ids.append(selected_token_id)
            selected_values.append(selected_value)
            selected_actor_logprobs.append(selected_actor_logprob)
            actor_state.append(selected_token_id)
            if selected_token_id in eos_token_ids:
                eos_emitted = True
                break
        latency_sec = time.perf_counter() - start_time

    full_response_token_ids = tuple(prefix_response + continuation_token_ids)
    response_text = _decode_response_text(tokenizer, full_response_token_ids)
    continuation_length = len(continuation_token_ids)
    response_length = len(full_response_token_ids)
    max_length_hit = bool(not eos_emitted and response_length >= int(max_new_tokens))
    zero_budget = bool(not prefix_has_eos and remaining_budget == 0)
    stop_reason = _build_stop_reason(
        eos_emitted=eos_emitted,
        max_length_hit=max_length_hit,
        zero_budget=zero_budget,
    )
    task_score = float(score_response(example, response_text))
    sum_selected_token_actor_logprob = float(sum(selected_actor_logprobs))
    sum_selected_token_value = float(sum(selected_values))
    mean_selected_token_actor_logprob = (
        sum_selected_token_actor_logprob / continuation_length if continuation_length > 0 else None
    )
    mean_selected_token_value = (sum_selected_token_value / continuation_length) if continuation_length > 0 else None
    choice_change_rate = (choice_change_count / continuation_length) if continuation_length > 0 else None
    tokens_per_second = (continuation_length / latency_sec) if latency_sec > 0 else None
    return ValueGuidedTrajectory(
        full_response_token_ids=full_response_token_ids,
        full_response_text=response_text,
        response_length=response_length,
        continuation_length=continuation_length,
        eos_emitted=eos_emitted,
        max_length_hit=max_length_hit,
        stop_reason=stop_reason,
        task_score=task_score,
        sum_selected_token_actor_logprob=sum_selected_token_actor_logprob,
        mean_selected_token_actor_logprob=mean_selected_token_actor_logprob,
        sum_selected_token_value=sum_selected_token_value,
        mean_selected_token_value=mean_selected_token_value,
        mean_selected_token_score_margin=_mean(selected_score_margins),
        choice_change_count=int(choice_change_count),
        choice_change_rate=choice_change_rate,
        num_decoding_steps=int(continuation_length),
        latency_sec=float(latency_sec),
        tokens_per_second=tokens_per_second,
    )


def evaluate_example_seed(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    start_specs: Sequence[StartSpec],
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    candidate_size: int,
    actor_device: torch.device,
    critic_device: torch.device,
    base_seed: int,
    use_actor_cache: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ExampleSeedArtifacts:
    prompt_ids = _prompt_ids_tensor(
        example=example,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        device=actor_device,
    )
    prompt_length = int(prompt_ids.shape[1])

    reference_seed = _reference_rollout_seed(base_seed, example_id=example.example_id)
    reference = sample_actor_response_from_prefix(
        actor=actor,
        tokenizer=tokenizer,
        example=example,
        prefix_ids=prompt_ids,
        prefix_response_token_ids=(),
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        sampling_mode=actor_sampling_mode,
        temperature=actor_temperature,
        top_p=actor_top_p,
        top_k=actor_top_k,
        seed=reference_seed,
        use_actor_cache=use_actor_cache,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "reference_rollout",
                "phase_label": "reference",
                "start_fraction_label": None,
                "start_percentage": None,
            }
        )

    reference_length = int(reference.response_length)
    reference_is_correct = bool(float(reference.task_score) == 1.0)
    start_token_indices = [
        int(math.floor(start_spec.fraction * float(reference_length)))
        for start_spec in start_specs
    ]
    actor_start_snapshots = _build_actor_start_snapshots(
        actor=actor,
        prompt_ids=prompt_ids,
        reference_response_token_ids=reference.full_response_token_ids,
        start_token_indices=start_token_indices,
        use_actor_cache=use_actor_cache,
    )
    per_start_records: list[dict[str, Any]] = []
    seen_start_positions: dict[int, list[str]] = {}

    for start_spec, start_token_index in zip(start_specs, start_token_indices, strict=True):
        if start_token_index < 0:
            raise ValueError(f"Computed a negative start_token_index: {start_token_index}")
        if start_token_index > reference_length:
            raise ValueError(
                f"Computed start_token_index {start_token_index} that exceeds reference length {reference_length}"
            )

        prefix_response_token_ids = reference.full_response_token_ids[:start_token_index]
        prefix_ids = _prefix_ids_tensor(prompt_ids=prompt_ids, prefix_response_token_ids=prefix_response_token_ids)
        actor_state_snapshot = actor_start_snapshots[start_token_index]
        actor_continuation_seed = _actor_continuation_seed(
            base_seed,
            example_id=example.example_id,
            start_token_index=start_token_index,
        )
        value_guidance_seed = _value_guidance_seed(
            base_seed,
            example_id=example.example_id,
            start_token_index=start_token_index,
        )

        actor_only = sample_actor_response_from_prefix(
            actor=actor,
            tokenizer=tokenizer,
            example=example,
            prefix_ids=prefix_ids,
            prefix_response_token_ids=prefix_response_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            sampling_mode=actor_sampling_mode,
            temperature=actor_temperature,
            top_p=actor_top_p,
            top_k=actor_top_k,
            seed=actor_continuation_seed,
            use_actor_cache=use_actor_cache,
            actor_state_snapshot=actor_state_snapshot,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "actor_only_continuation",
                    "phase_label": f"actor@{start_spec.label}",
                    "start_fraction_label": start_spec.label,
                    "start_percentage": int(start_spec.percentage),
                }
            )
        value_guided = run_value_guided_from_prefix(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            prefix_ids=prefix_ids,
            prefix_response_token_ids=prefix_response_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            candidate_size=candidate_size,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=value_guidance_seed,
            use_actor_cache=use_actor_cache,
            actor_state_snapshot=actor_state_snapshot,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "value_guided_continuation",
                    "phase_label": f"value@{start_spec.label}",
                    "start_fraction_label": start_spec.label,
                    "start_percentage": int(start_spec.percentage),
                }
            )

        collapsed_labels = list(seen_start_positions.get(start_token_index, []))
        seen_start_positions.setdefault(start_token_index, []).append(start_spec.label)
        record = {
            "example_id": int(example.example_id),
            "prompt_id": int(example.example_id),
            "seed": int(base_seed),
            "reference_rollout_seed": int(reference_seed),
            "actor_only_continuation_seed": int(actor_continuation_seed),
            "value_guidance_seed": int(value_guidance_seed),
            "data_source": example.data_source,
            "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
            "prompt_length": int(prompt_length),
            "candidate_size": int(candidate_size),
            "start_fraction": float(start_spec.fraction),
            "start_percentage": int(start_spec.percentage),
            "start_fraction_label": start_spec.label,
            "start_fraction_display": start_spec.display_label,
            "start_token_index": int(start_token_index),
            "prefix_length": int(start_token_index),
            "remaining_tokens_after_start": int(reference_length - start_token_index),
            "start_position_collapsed": bool(collapsed_labels),
            "collapsed_against_start_labels": collapsed_labels,
            "reference_response": reference.full_response_text,
            "reference_response_length": int(reference.response_length),
            "reference_task_score": float(reference.task_score),
            "reference_is_correct": bool(reference_is_correct),
            "reference_eos_emitted": bool(reference.eos_emitted),
            "reference_max_length_hit": bool(reference.max_length_hit),
            "reference_stop_reason": reference.stop_reason,
            "reference_sum_response_actor_logprob": float(reference.sum_actor_logprob),
            "reference_mean_response_actor_logprob": reference.mean_actor_logprob,
            "reference_latency_sec": float(reference.latency_sec),
            "reference_tokens_per_second": reference.tokens_per_second,
            "actor_only_response": actor_only.full_response_text,
            "actor_only_response_length": int(actor_only.response_length),
            "actor_only_continuation_length": int(actor_only.continuation_length),
            "actor_only_task_score": float(actor_only.task_score),
            "actor_only_is_correct": bool(float(actor_only.task_score) == 1.0),
            "actor_only_eos_emitted": bool(actor_only.eos_emitted),
            "actor_only_max_length_hit": bool(actor_only.max_length_hit),
            "actor_only_stop_reason": actor_only.stop_reason,
            "actor_only_sum_continuation_actor_logprob": float(actor_only.sum_actor_logprob),
            "actor_only_mean_continuation_actor_logprob": actor_only.mean_actor_logprob,
            "actor_only_latency_sec": float(actor_only.latency_sec),
            "actor_only_tokens_per_second": actor_only.tokens_per_second,
            "value_guided_response": value_guided.full_response_text,
            "value_guided_response_length": int(value_guided.response_length),
            "value_guided_continuation_length": int(value_guided.continuation_length),
            "value_guided_task_score": float(value_guided.task_score),
            "value_guided_is_correct": bool(float(value_guided.task_score) == 1.0),
            "value_guided_eos_emitted": bool(value_guided.eos_emitted),
            "value_guided_max_length_hit": bool(value_guided.max_length_hit),
            "value_guided_stop_reason": value_guided.stop_reason,
            "value_guided_num_decoding_steps": int(value_guided.num_decoding_steps),
            "value_guided_choice_change_count": int(value_guided.choice_change_count),
            "value_guided_choice_change_rate": value_guided.choice_change_rate,
            "value_guided_sum_selected_token_value": float(value_guided.sum_selected_token_value),
            "value_guided_mean_selected_token_value": value_guided.mean_selected_token_value,
            "value_guided_mean_selected_token_score_margin": value_guided.mean_selected_token_score_margin,
            "value_guided_sum_selected_token_actor_logprob": float(value_guided.sum_selected_token_actor_logprob),
            "value_guided_mean_selected_token_actor_logprob": value_guided.mean_selected_token_actor_logprob,
            "value_guided_latency_sec": float(value_guided.latency_sec),
            "value_guided_tokens_per_second": value_guided.tokens_per_second,
            "value_guided_task_score_minus_actor_only": float(value_guided.task_score - actor_only.task_score),
            "value_guided_task_score_minus_reference": float(value_guided.task_score - reference.task_score),
            "actor_only_task_score_minus_reference": float(actor_only.task_score - reference.task_score),
            "value_guided_response_length_minus_actor_only": int(
                value_guided.response_length - actor_only.response_length
            ),
            "value_guided_response_length_minus_reference": int(
                value_guided.response_length - reference.response_length
            ),
            "actor_only_response_length_minus_reference": int(actor_only.response_length - reference.response_length),
        }
        per_start_records.append(record)

    return ExampleSeedArtifacts(per_start_records=per_start_records)


def _emit_progress(*, progress_queue, progress_actor, event: dict[str, Any]) -> None:
    if progress_queue is not None:
        progress_queue.put(event)
        return
    if progress_actor is not None:
        progress_actor.put.remote(event)


class _RayProgressActor:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def put(self, event: dict[str, Any]) -> None:
        self._events.append(dict(event))

    def drain(self) -> list[dict[str, Any]]:
        events = self._events
        self._events = []
        return events


def _worker_entry(
    *,
    assignment: WorkerAssignment,
    actor_hf_dir: str,
    critic_hf_dir: str,
    examples: list[ExampleRecord],
    start_specs: list[StartSpec] | list[dict[str, Any]],
    seeds: list[int],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    candidate_size: int,
    use_actor_cache: bool,
    worker_root: str,
    progress_queue=None,
    progress_actor=None,
) -> None:
    worker_dir = Path(worker_root) / f"worker_{assignment.worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    summary_path = worker_dir / "worker_summary.json"
    error_path = worker_dir / "worker_error.txt"

    if start_specs and isinstance(start_specs[0], dict):
        start_specs = [StartSpec(**payload) for payload in start_specs]

    try:
        start_time = time.perf_counter()
        local_examples = examples[assignment.example_start : assignment.example_end]
        progress_units_per_example_seed = _progress_units_per_example_seed(start_specs)
        worker_total_tasks = len(local_examples) * len(seeds) * progress_units_per_example_seed
        worker_completed_tasks = 0
        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_started",
                "worker_id": assignment.worker_id,
                "worker_total_tasks": worker_total_tasks,
                "status_label": "loading_models",
            },
        )

        actor_device = resolve_device(assignment.actor_device)
        critic_device = resolve_device(assignment.critic_device) if assignment.critic_device else actor_device
        _validate_visible_cuda_device(actor_device, label="actor_device")
        _validate_visible_cuda_device(critic_device, label="critic_device")
        dtype = resolve_dtype(dtype_name)

        tokenizer = load_tokenizer(Path(actor_hf_dir), trust_remote_code=trust_remote_code)
        actor = load_actor_model(
            Path(actor_hf_dir),
            dtype=dtype,
            device=actor_device,
            trust_remote_code=trust_remote_code,
        )
        critic = load_critic_model(
            Path(critic_hf_dir),
            dtype=dtype,
            device=critic_device,
            trust_remote_code=trust_remote_code,
        )

        per_example_path = worker_dir / "delayed_value_guidance_per_example.jsonl"
        start_wall_time_sec = time.time()
        with per_example_path.open("w", encoding="utf-8") as per_example_file:
            for example in local_examples:
                for seed in seeds:
                    def progress_callback(progress_payload: dict[str, Any]) -> None:
                        nonlocal worker_completed_tasks
                        worker_completed_tasks += 1
                        _emit_progress(
                            progress_queue=progress_queue,
                            progress_actor=progress_actor,
                            event={
                                "type": "task_done",
                                "worker_id": assignment.worker_id,
                                "seed": int(seed),
                                "example_id": int(example.example_id),
                                "phase": progress_payload.get("phase"),
                                "phase_label": progress_payload.get("phase_label"),
                                "start_fraction_label": progress_payload.get("start_fraction_label"),
                                "start_percentage": progress_payload.get("start_percentage"),
                                "worker_completed_tasks": worker_completed_tasks,
                                "worker_total_tasks": worker_total_tasks,
                            },
                        )

                    artifacts = evaluate_example_seed(
                        actor=actor,
                        critic=critic,
                        tokenizer=tokenizer,
                        example=example,
                        start_specs=start_specs,
                        max_prompt_length=max_prompt_length,
                        max_new_tokens=max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        actor_sampling_mode=actor_sampling_mode,
                        actor_temperature=actor_temperature,
                        actor_top_p=actor_top_p,
                        actor_top_k=actor_top_k,
                        candidate_size=candidate_size,
                        actor_device=actor_device,
                        critic_device=critic_device,
                        base_seed=int(seed),
                        use_actor_cache=use_actor_cache,
                        progress_callback=progress_callback,
                    )
                    for record in artifacts.per_start_records:
                        per_example_file.write(_json_line(record))
        end_wall_time_sec = time.time()

        summary_payload = {
            "worker_id": assignment.worker_id,
            "actor_device": str(actor_device),
            "critic_device": str(critic_device),
            "node_index": assignment.node_index,
            "node_ip": assignment.node_ip,
            "node_resource_key": assignment.node_resource_key,
            "local_worker_index": assignment.local_worker_index,
            "example_start": assignment.example_start,
            "example_end": assignment.example_end,
            "num_examples": assignment.num_examples,
            "num_seeds": len(seeds),
            "worker_total_tasks": worker_total_tasks,
            "start_wall_time_sec": start_wall_time_sec,
            "end_wall_time_sec": end_wall_time_sec,
            "runtime_sec": time.perf_counter() - start_time,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)
        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_done",
                "worker_id": assignment.worker_id,
                "worker_completed_tasks": worker_completed_tasks,
                "worker_total_tasks": worker_total_tasks,
            },
        )
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_error",
                "worker_id": assignment.worker_id,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _start_local_worker_processes(
    *,
    assignments: Sequence[WorkerAssignment],
    actor_hf_dir: Path,
    critic_hf_dir: Path,
    examples: list[ExampleRecord],
    start_specs: list[StartSpec],
    seeds: list[int],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    candidate_size: int,
    use_actor_cache: bool,
    worker_root: Path,
) -> tuple[Any, list[tuple[mp.Process, WorkerAssignment]]]:
    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[tuple[mp.Process, WorkerAssignment]] = []
    serialized_start_specs = _serialize_start_specs(start_specs)
    canonical_module = importlib.import_module("value_decoding.delayed_value_guidance_eval")
    worker_target = getattr(canonical_module, "_worker_entry")
    for assignment in assignments:
        process = context.Process(
            target=worker_target,
            kwargs={
                "assignment": assignment,
                "actor_hf_dir": str(actor_hf_dir),
                "critic_hf_dir": str(critic_hf_dir),
                "examples": examples,
                "start_specs": serialized_start_specs,
                "seeds": seeds,
                "dtype_name": dtype_name,
                "trust_remote_code": trust_remote_code,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "eos_token_ids": eos_token_ids,
                "actor_sampling_mode": actor_sampling_mode,
                "actor_temperature": actor_temperature,
                "actor_top_p": actor_top_p,
                "actor_top_k": actor_top_k,
                "candidate_size": candidate_size,
                "use_actor_cache": use_actor_cache,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
            },
            name=f"delayed_value_guidance_worker_{assignment.worker_id}",
        )
        process.start()
        processes.append((process, assignment))
    return progress_queue, processes


def _assert_local_processes_healthy(
    *,
    processes: Sequence[tuple[mp.Process, WorkerAssignment]],
    worker_root: Path,
) -> None:
    for process, assignment in processes:
        if process.exitcode not in (None, 0):
            error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
            if error_path.exists():
                raise RuntimeError(
                    f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n"
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")


def _join_local_processes(
    *,
    processes: Sequence[tuple[mp.Process, WorkerAssignment]],
    worker_root: Path,
) -> None:
    for process, assignment in processes:
        process.join()
        if process.exitcode != 0:
            error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
            if error_path.exists():
                raise RuntimeError(
                    f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n"
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")


def _collect_worker_outputs(
    *,
    output_dir: Path,
    worker_root: Path,
    assignments: Sequence[WorkerAssignment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_example_path = output_dir / "delayed_value_guidance_per_example.jsonl"
    records: list[dict[str, Any]] = []
    with per_example_path.open("w", encoding="utf-8") as per_example_file:
        for assignment in assignments:
            worker_dir = worker_root / f"worker_{assignment.worker_id:03d}"
            worker_example_path = worker_dir / "delayed_value_guidance_per_example.jsonl"
            with worker_example_path.open("r", encoding="utf-8") as worker_example_file:
                for line in worker_example_file:
                    if not line.strip():
                        continue
                    per_example_file.write(line)
                    records.append(json.loads(line))

    worker_summaries: list[dict[str, Any]] = []
    for assignment in assignments:
        summary_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_summary.json"
        with summary_path.open("r", encoding="utf-8") as summary_file:
            worker_summaries.append(json.load(summary_file))
    worker_summaries.sort(key=lambda item: int(item["worker_id"]))
    return records, worker_summaries


def run_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path,
    examples: list[ExampleRecord],
    start_specs: list[StartSpec],
    seeds: list[int],
    worker_pairs: list[tuple[str | None, str | None]],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    candidate_size: int,
    use_actor_cache: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)
    if not assignments:
        raise ValueError("No worker assignments were created.")

    worker_root = output_dir / "_worker_tmp"
    if worker_root.exists():
        import shutil

        shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    progress_queue, processes = _start_local_worker_processes(
        assignments=assignments,
        actor_hf_dir=actor_hf_dir,
        critic_hf_dir=critic_hf_dir,
        examples=examples,
        start_specs=start_specs,
        seeds=seeds,
        dtype_name=dtype_name,
        trust_remote_code=trust_remote_code,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        actor_sampling_mode=actor_sampling_mode,
        actor_temperature=actor_temperature,
        actor_top_p=actor_top_p,
        actor_top_k=actor_top_k,
        candidate_size=candidate_size,
        use_actor_cache=use_actor_cache,
        worker_root=worker_root,
    )

    progress_units_per_example_seed = _progress_units_per_example_seed(start_specs)
    total_tasks = len(examples) * len(seeds) * progress_units_per_example_seed
    completed_tasks = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {
            "done": 0,
            "total": assignment.num_examples * len(seeds) * progress_units_per_example_seed,
            "config_id": None,
        }
        for assignment in assignments
    }

    with tqdm(total=total_tasks, desc="delayed_value_guidance_eval", unit="task", dynamic_ncols=True) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_tasks < total_tasks or completed_workers < len(assignments):
            try:
                event = progress_queue.get(timeout=0.2)
            except Empty:
                _assert_local_processes_healthy(processes=processes, worker_root=worker_root)
                continue

            event_type = event.get("type")
            worker_id = int(event.get("worker_id", -1))
            if event_type == "worker_started":
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
                status_label = event.get("status_label")
                worker_progress[worker_id]["config_id"] = str(status_label) if status_label else None
            elif event_type == "task_done":
                completed_tasks += 1
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["done"] = int(event.get("worker_completed_tasks", 0))
                worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
                phase_label = event.get("phase_label")
                if phase_label:
                    worker_progress[worker_id]["config_id"] = (
                        f"seed={event.get('seed')} ex={event.get('example_id')} {phase_label}"
                    )
                else:
                    worker_progress[worker_id]["config_id"] = f"seed={event.get('seed')} example={event.get('example_id')}"
                progress_bar.update(1)
            elif event_type == "worker_done":
                completed_workers += 1
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["done"] = int(event.get("worker_completed_tasks", 0))
                worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
                worker_progress[worker_id]["config_id"] = "done"
            elif event_type == "worker_error":
                raise RuntimeError(
                    f"Worker {worker_id} reported an error.\n{event.get('traceback', 'No traceback provided.')}"
                )
            progress_bar.set_postfix_str(_progress_postfix(worker_progress))

    _join_local_processes(processes=processes, worker_root=worker_root)
    return _collect_worker_outputs(output_dir=output_dir, worker_root=worker_root, assignments=assignments)


def _ray_node_entry_remote(**kwargs) -> dict[str, Any]:
    assignments: list[WorkerAssignment] = kwargs["assignments"]
    progress_actor = kwargs["progress_actor"]
    progress_queue, processes = _start_local_worker_processes(
        assignments=assignments,
        actor_hf_dir=Path(kwargs["actor_hf_dir"]),
        critic_hf_dir=Path(kwargs["critic_hf_dir"]),
        examples=kwargs["examples"],
        start_specs=kwargs["start_specs"],
        seeds=kwargs["seeds"],
        dtype_name=kwargs["dtype_name"],
        trust_remote_code=kwargs["trust_remote_code"],
        max_prompt_length=kwargs["max_prompt_length"],
        max_new_tokens=kwargs["max_new_tokens"],
        eos_token_ids=kwargs["eos_token_ids"],
        actor_sampling_mode=kwargs["actor_sampling_mode"],
        actor_temperature=kwargs["actor_temperature"],
        actor_top_p=kwargs["actor_top_p"],
        actor_top_k=kwargs["actor_top_k"],
        candidate_size=kwargs["candidate_size"],
        use_actor_cache=kwargs["use_actor_cache"],
        worker_root=Path(kwargs["worker_root"]),
    )
    worker_root = Path(kwargs["worker_root"])
    completed_workers = 0
    while completed_workers < len(assignments):
        try:
            event = progress_queue.get(timeout=RAY_PROGRESS_POLL_INTERVAL_SEC)
        except Empty:
            _assert_local_processes_healthy(processes=processes, worker_root=worker_root)
            continue

        progress_actor.put.remote(event)
        if event.get("type") == "worker_done":
            completed_workers += 1
        elif event.get("type") == "worker_error":
            raise RuntimeError(
                f"Worker {event.get('worker_id')} reported an error.\n"
                f"{event.get('traceback', 'No traceback provided.')}"
            )

    _join_local_processes(processes=processes, worker_root=worker_root)
    return {
        "node_index": kwargs["node_index"],
        "node_ip": kwargs["node_ip"],
        "worker_ids": [int(assignment.worker_id) for assignment in assignments],
    }


def run_ray_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path,
    examples: list[ExampleRecord],
    start_specs: list[StartSpec],
    seeds: list[int],
    worker_assignments: list[WorkerAssignment],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    candidate_size: int,
    use_actor_cache: bool,
    ray_num_cpus_per_worker: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not worker_assignments:
        raise ValueError("No worker assignments were created.")
    if ray_num_cpus_per_worker <= 0:
        raise ValueError(f"ray_num_cpus_per_worker must be > 0, got {ray_num_cpus_per_worker}.")

    ray_module = _require_ray()
    if not ray_module.is_initialized():
        raise RuntimeError("Ray must be initialized before running cross-node delayed value-guidance workers.")

    worker_root = output_dir / "_worker_tmp"
    if worker_root.exists():
        import shutil

        shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    progress_actor = ray_module.remote(num_cpus=0)(_RayProgressActor).remote()
    node_execution_specs = _build_ray_node_execution_specs(
        worker_assignments=worker_assignments,
        critic_enabled=True,
        ray_num_cpus_per_worker=ray_num_cpus_per_worker,
    )
    node_remote = ray_module.remote(max_retries=0)(_ray_node_entry_remote)

    node_refs = []
    ref_to_node_spec: dict[Any, dict[str, Any]] = {}
    for node_spec in node_execution_specs:
        node_resource_key = node_spec["node_resource_key"]
        if node_resource_key is None:
            raise ValueError(
                f"Ray node execution spec for node {node_spec['node_ip']} is missing node_resource_key."
            )
        node_ref = node_remote.options(
            num_cpus=float(node_spec["num_cpus"]),
            num_gpus=float(node_spec["num_gpus"]),
            resources={node_resource_key: 1e-3},
        ).remote(
            node_index=node_spec["node_index"],
            node_ip=node_spec["node_ip"],
            assignments=node_spec["assignments"],
            actor_hf_dir=str(actor_hf_dir),
            critic_hf_dir=str(critic_hf_dir),
            examples=examples,
            start_specs=start_specs,
            seeds=seeds,
            dtype_name=dtype_name,
            trust_remote_code=trust_remote_code,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_sampling_mode=actor_sampling_mode,
            actor_temperature=actor_temperature,
            actor_top_p=actor_top_p,
            actor_top_k=actor_top_k,
            candidate_size=candidate_size,
            use_actor_cache=use_actor_cache,
            worker_root=str(worker_root),
            progress_actor=progress_actor,
        )
        node_refs.append(node_ref)
        ref_to_node_spec[node_ref] = node_spec

    progress_units_per_example_seed = _progress_units_per_example_seed(start_specs)
    total_tasks = len(examples) * len(seeds) * progress_units_per_example_seed
    completed_tasks = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {
            "done": 0,
            "total": assignment.num_examples * len(seeds) * progress_units_per_example_seed,
            "config_id": None,
        }
        for assignment in worker_assignments
    }

    pending_refs = list(node_refs)
    with tqdm(total=total_tasks, desc="delayed_value_guidance_eval", unit="task", dynamic_ncols=True) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_tasks < total_tasks or completed_workers < len(worker_assignments):
            events = ray_module.get(progress_actor.drain.remote())
            for event in events:
                event_type = event.get("type")
                worker_id = int(event.get("worker_id", -1))
                if event_type == "worker_started":
                    worker_progress.setdefault(worker_id, {})
                    worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
                    status_label = event.get("status_label")
                    worker_progress[worker_id]["config_id"] = str(status_label) if status_label else None
                elif event_type == "task_done":
                    completed_tasks += 1
                    worker_progress.setdefault(worker_id, {})
                    worker_progress[worker_id]["done"] = int(event.get("worker_completed_tasks", 0))
                    worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
                    phase_label = event.get("phase_label")
                    if phase_label:
                        worker_progress[worker_id]["config_id"] = (
                            f"seed={event.get('seed')} ex={event.get('example_id')} {phase_label}"
                        )
                    else:
                        worker_progress[worker_id]["config_id"] = (
                            f"seed={event.get('seed')} example={event.get('example_id')}"
                        )
                    progress_bar.update(1)
                elif event_type == "worker_done":
                    completed_workers += 1
                    worker_progress.setdefault(worker_id, {})
                    worker_progress[worker_id]["done"] = int(event.get("worker_completed_tasks", 0))
                    worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
                    worker_progress[worker_id]["config_id"] = "done"
                elif event_type == "worker_error":
                    raise RuntimeError(
                        f"Worker {worker_id} reported an error.\n{event.get('traceback', 'No traceback provided.')}"
                    )
                progress_bar.set_postfix_str(_progress_postfix(worker_progress))

            if pending_refs:
                done_refs, pending_refs = ray_module.wait(
                    pending_refs,
                    num_returns=1,
                    timeout=RAY_PROGRESS_POLL_INTERVAL_SEC,
                )
                for done_ref in done_refs:
                    node_spec = ref_to_node_spec[done_ref]
                    try:
                        ray_module.get(done_ref)
                    except Exception as exc:
                        for assignment in node_spec["assignments"]:
                            error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
                            if error_path.exists():
                                raise RuntimeError(
                                    f"Worker {assignment.worker_id} failed on Ray node {assignment.node_ip}.\n"
                                    f"{error_path.read_text(encoding='utf-8')}"
                                ) from exc
                        raise RuntimeError(
                            f"Ray node task failed on node {node_spec['node_ip']} "
                            f"for workers {[assignment.worker_id for assignment in node_spec['assignments']]}."
                        ) from exc
            else:
                time.sleep(RAY_PROGRESS_POLL_INTERVAL_SEC)

    if node_refs:
        try:
            ray_module.get(node_refs)
        except Exception:
            for assignment in worker_assignments:
                error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
                if error_path.exists():
                    raise RuntimeError(
                        f"Worker {assignment.worker_id} failed on Ray node {assignment.node_ip}.\n"
                        f"{error_path.read_text(encoding='utf-8')}"
                    )
            raise

    return _collect_worker_outputs(output_dir=output_dir, worker_root=worker_root, assignments=worker_assignments)


def _reference_records(start_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in start_rows:
        key = (int(row["example_id"]), int(row["seed"]))
        reference_row = {
            "example_id": int(row["example_id"]),
            "prompt_id": int(row["prompt_id"]),
            "seed": int(row["seed"]),
            "reference_rollout_seed": int(row["reference_rollout_seed"]),
            "data_source": row["data_source"],
            "ground_truth": row["ground_truth"],
            "prompt_length": int(row["prompt_length"]),
            "reference_response": row["reference_response"],
            "reference_response_length": int(row["reference_response_length"]),
            "reference_task_score": float(row["reference_task_score"]),
            "reference_is_correct": bool(row["reference_is_correct"]),
            "reference_eos_emitted": bool(row["reference_eos_emitted"]),
            "reference_max_length_hit": bool(row["reference_max_length_hit"]),
            "reference_stop_reason": row["reference_stop_reason"],
            "reference_sum_response_actor_logprob": float(row["reference_sum_response_actor_logprob"]),
            "reference_mean_response_actor_logprob": row["reference_mean_response_actor_logprob"],
            "reference_latency_sec": float(row["reference_latency_sec"]),
            "reference_tokens_per_second": row["reference_tokens_per_second"],
        }
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = reference_row
            continue
        if reference_row != existing:
            raise ValueError(f"Reference rollout mismatch encountered for prompt-seed key {key}.")
    return [by_key[key] for key in sorted(by_key)]


def _aggregate_reference_rollout_rows(reference_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    task_scores = [float(row["reference_task_score"]) for row in reference_rows]
    response_lengths = [int(row["reference_response_length"]) for row in reference_rows]
    latencies = [float(row["reference_latency_sec"]) for row in reference_rows]
    tokens_per_second = [
        float(row["reference_tokens_per_second"])
        for row in reference_rows
        if row["reference_tokens_per_second"] is not None
    ]
    binary_scores = _is_binary_scores(task_scores)
    return {
        "method_id": "reference_rollout",
        "method_family": "reference_rollout",
        "method_name": "reference_rollout",
        "start_fraction": None,
        "start_percentage": None,
        "start_fraction_label": None,
        "start_fraction_display": None,
        "candidate_size": None,
        "num_prompt_seed_pairs": int(len(reference_rows)),
        "num_examples": int(len({int(row['example_id']) for row in reference_rows})),
        "num_seeds": int(len({int(row['seed']) for row in reference_rows})),
        "mean_task_score": _mean(task_scores),
        "mean_accuracy": _mean(task_scores) if binary_scores else None,
        "mean_response_length": _mean(response_lengths),
        "eos_rate": _mean([1.0 if bool(row["reference_eos_emitted"]) else 0.0 for row in reference_rows]),
        "max_length_hit_rate": _mean(
            [1.0 if bool(row["reference_max_length_hit"]) else 0.0 for row in reference_rows]
        ),
        "mean_prefix_length": None,
        "mean_remaining_tokens_after_start": None,
        "mean_continuation_length": None,
        "mean_fraction_token_decisions_different_from_actor_top1": None,
        "mean_selected_token_value": None,
        "mean_selected_token_score_margin": None,
        "sum_example_latency_sec": float(sum(latencies)),
        "mean_tokens_per_second": _mean(tokens_per_second),
        "collapsed_start_rate": None,
    }


def _aggregate_method_rows(
    *,
    rows: Sequence[dict[str, Any]],
    method_family: str,
    candidate_size: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No rows provided for method family {method_family}.")

    first_row = rows[0]
    if method_family == "actor_only":
        score_key = "actor_only_task_score"
        response_length_key = "actor_only_response_length"
        continuation_length_key = "actor_only_continuation_length"
        eos_key = "actor_only_eos_emitted"
        max_length_key = "actor_only_max_length_hit"
        latency_key = "actor_only_latency_sec"
        tps_key = "actor_only_tokens_per_second"
        overwrite_values: list[float] = []
        selected_value_values: list[float] = []
        margin_values: list[float] = []
        method_name = "actor_only_continuation"
    elif method_family == "value_guided":
        score_key = "value_guided_task_score"
        response_length_key = "value_guided_response_length"
        continuation_length_key = "value_guided_continuation_length"
        eos_key = "value_guided_eos_emitted"
        max_length_key = "value_guided_max_length_hit"
        latency_key = "value_guided_latency_sec"
        tps_key = "value_guided_tokens_per_second"
        overwrite_values = [
            float(row["value_guided_choice_change_rate"])
            for row in rows
            if row["value_guided_choice_change_rate"] is not None
        ]
        selected_value_values = [
            float(row["value_guided_mean_selected_token_value"])
            for row in rows
            if row["value_guided_mean_selected_token_value"] is not None
        ]
        margin_values = [
            float(row["value_guided_mean_selected_token_score_margin"])
            for row in rows
            if row["value_guided_mean_selected_token_score_margin"] is not None
        ]
        method_name = "delayed_value_guided_continuation"
    else:
        raise ValueError(f"Unsupported method family: {method_family}")

    task_scores = [float(row[score_key]) for row in rows]
    response_lengths = [int(row[response_length_key]) for row in rows]
    continuation_lengths = [int(row[continuation_length_key]) for row in rows]
    latencies = [float(row[latency_key]) for row in rows]
    tokens_per_second = [float(row[tps_key]) for row in rows if row[tps_key] is not None]
    binary_scores = _is_binary_scores(task_scores)
    start_fraction_label = str(first_row["start_fraction_label"])
    return {
        "method_id": f"{method_family}_from_{start_fraction_label}",
        "method_family": method_family,
        "method_name": method_name,
        "start_fraction": float(first_row["start_fraction"]),
        "start_percentage": int(first_row["start_percentage"]),
        "start_fraction_label": start_fraction_label,
        "start_fraction_display": first_row["start_fraction_display"],
        "candidate_size": int(candidate_size) if method_family == "value_guided" else None,
        "num_prompt_seed_pairs": int(len(rows)),
        "num_examples": int(len({int(row['example_id']) for row in rows})),
        "num_seeds": int(len({int(row['seed']) for row in rows})),
        "mean_task_score": _mean(task_scores),
        "mean_accuracy": _mean(task_scores) if binary_scores else None,
        "mean_response_length": _mean(response_lengths),
        "eos_rate": _mean([1.0 if bool(row[eos_key]) else 0.0 for row in rows]),
        "max_length_hit_rate": _mean([1.0 if bool(row[max_length_key]) else 0.0 for row in rows]),
        "mean_prefix_length": _mean([float(row["prefix_length"]) for row in rows]),
        "mean_remaining_tokens_after_start": _mean(
            [float(row["remaining_tokens_after_start"]) for row in rows]
        ),
        "mean_continuation_length": _mean(continuation_lengths),
        "mean_fraction_token_decisions_different_from_actor_top1": (
            _mean(overwrite_values) if method_family == "value_guided" else None
        ),
        "mean_selected_token_value": _mean(selected_value_values) if method_family == "value_guided" else None,
        "mean_selected_token_score_margin": _mean(margin_values) if method_family == "value_guided" else None,
        "sum_example_latency_sec": float(sum(latencies)),
        "mean_tokens_per_second": _mean(tokens_per_second),
        "collapsed_start_rate": _mean([1.0 if bool(row["start_position_collapsed"]) else 0.0 for row in rows]),
    }


def _aggregate_main_results(
    *,
    start_rows: Sequence[dict[str, Any]],
    start_specs: Sequence[StartSpec],
    candidate_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reference_rows = _reference_records(start_rows)
    main_rows: list[dict[str, Any]] = [_aggregate_reference_rollout_rows(reference_rows)]
    actor_rows: list[dict[str, Any]] = []
    value_guided_rows: list[dict[str, Any]] = []

    for start_spec in start_specs:
        rows = [
            row
            for row in start_rows
            if int(row["start_percentage"]) == int(start_spec.percentage)
        ]
        if not rows:
            continue
        actor_row = _aggregate_method_rows(rows=rows, method_family="actor_only", candidate_size=candidate_size)
        value_row = _aggregate_method_rows(rows=rows, method_family="value_guided", candidate_size=candidate_size)
        actor_rows.append(actor_row)
        value_guided_rows.append(value_row)
        main_rows.append(actor_row)
        main_rows.append(value_row)
    return main_rows, actor_rows, value_guided_rows


def _paired_summary_for_start(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute paired summary for an empty row set.")
    value_scores = [float(row["value_guided_task_score"]) for row in rows]
    actor_scores = [float(row["actor_only_task_score"]) for row in rows]
    reference_scores = [float(row["reference_task_score"]) for row in rows]
    binary_scores = _is_binary_scores(value_scores) and _is_binary_scores(actor_scores) and _is_binary_scores(
        reference_scores
    )
    return {
        "start_fraction": float(rows[0]["start_fraction"]),
        "start_percentage": int(rows[0]["start_percentage"]),
        "start_fraction_label": str(rows[0]["start_fraction_label"]),
        "start_fraction_display": rows[0]["start_fraction_display"],
        "num_prompt_seed_pairs": int(len(rows)),
        "mean_task_score_delta_value_guided_minus_actor_only": _mean(
            [float(row["value_guided_task_score_minus_actor_only"]) for row in rows]
        ),
        "mean_accuracy_delta_value_guided_minus_actor_only": (
            _mean([float(row["value_guided_task_score_minus_actor_only"]) for row in rows]) if binary_scores else None
        ),
        "mean_task_score_delta_value_guided_minus_reference": _mean(
            [float(row["value_guided_task_score_minus_reference"]) for row in rows]
        ),
        "mean_accuracy_delta_value_guided_minus_reference": (
            _mean([float(row["value_guided_task_score_minus_reference"]) for row in rows]) if binary_scores else None
        ),
        "mean_task_score_delta_actor_only_minus_reference": _mean(
            [float(row["actor_only_task_score_minus_reference"]) for row in rows]
        ),
        "mean_accuracy_delta_actor_only_minus_reference": (
            _mean([float(row["actor_only_task_score_minus_reference"]) for row in rows]) if binary_scores else None
        ),
        "mean_response_length_delta_value_guided_minus_actor_only": _mean(
            [float(row["value_guided_response_length_minus_actor_only"]) for row in rows]
        ),
        "mean_response_length_delta_value_guided_minus_reference": _mean(
            [float(row["value_guided_response_length_minus_reference"]) for row in rows]
        ),
        "mean_response_length_delta_actor_only_minus_reference": _mean(
            [float(row["actor_only_response_length_minus_reference"]) for row in rows]
        ),
    }


def _build_paired_differences(start_rows: Sequence[dict[str, Any]], start_specs: Sequence[StartSpec]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for start_spec in start_specs:
        rows = [
            row
            for row in start_rows
            if int(row["start_percentage"]) == int(start_spec.percentage)
        ]
        if not rows:
            continue
        summaries.append(_paired_summary_for_start(rows))
    return summaries


def _attach_paired_metrics(
    *,
    main_rows: list[dict[str, Any]],
    paired_differences: Sequence[dict[str, Any]],
) -> None:
    paired_by_label = {str(row["start_fraction_label"]): row for row in paired_differences}
    for row in main_rows:
        if row["method_family"] != "value_guided":
            row["mean_accuracy_delta_vs_actor_only"] = None
            row["mean_task_score_delta_vs_actor_only"] = None
            row["mean_accuracy_delta_vs_reference"] = None
            row["mean_task_score_delta_vs_reference"] = None
            continue
        paired = paired_by_label.get(str(row["start_fraction_label"]))
        if paired is None:
            row["mean_accuracy_delta_vs_actor_only"] = None
            row["mean_task_score_delta_vs_actor_only"] = None
            row["mean_accuracy_delta_vs_reference"] = None
            row["mean_task_score_delta_vs_reference"] = None
            continue
        row["mean_accuracy_delta_vs_actor_only"] = paired["mean_accuracy_delta_value_guided_minus_actor_only"]
        row["mean_task_score_delta_vs_actor_only"] = paired["mean_task_score_delta_value_guided_minus_actor_only"]
        row["mean_accuracy_delta_vs_reference"] = paired["mean_accuracy_delta_value_guided_minus_reference"]
        row["mean_task_score_delta_vs_reference"] = paired["mean_task_score_delta_value_guided_minus_reference"]


def _reference_correctness_splits(
    *,
    start_rows: Sequence[dict[str, Any]],
    start_specs: Sequence[StartSpec],
    candidate_size: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for start_spec in start_specs:
        rows = [
            row
            for row in start_rows
            if int(row["start_percentage"]) == int(start_spec.percentage)
        ]
        if not rows:
            continue

        split_payload: dict[str, Any] = {}
        for split_name, split_value in (("reference_correct", True), ("reference_incorrect", False)):
            split_rows = [row for row in rows if bool(row["reference_is_correct"]) == split_value]
            if not split_rows:
                split_payload[split_name] = {
                    "num_prompt_seed_pairs": 0,
                    "actor_only": None,
                    "value_guided": None,
                    "paired_differences": None,
                }
                continue
            split_payload[split_name] = {
                "num_prompt_seed_pairs": int(len(split_rows)),
                "actor_only": _aggregate_method_rows(
                    rows=split_rows,
                    method_family="actor_only",
                    candidate_size=candidate_size,
                ),
                "value_guided": _aggregate_method_rows(
                    rows=split_rows,
                    method_family="value_guided",
                    candidate_size=candidate_size,
                ),
                "paired_differences": _paired_summary_for_start(split_rows),
            }
        summary[start_spec.label] = {
            "start_fraction": float(start_spec.fraction),
            "start_percentage": int(start_spec.percentage),
            "start_fraction_display": start_spec.display_label,
            "splits": split_payload,
        }
    return summary


def _rows_for_seed(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["seed"]) == int(seed)]


def _per_seed_main_results(
    *,
    start_rows: Sequence[dict[str, Any]],
    start_specs: Sequence[StartSpec],
    candidate_size: int,
    seeds: Sequence[int],
) -> dict[str, list[dict[str, Any]]]:
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        seed_rows = _rows_for_seed(start_rows, int(seed))
        if not seed_rows:
            per_seed[str(int(seed))] = []
            continue
        main_rows, _actor_rows, _value_rows = _aggregate_main_results(
            start_rows=seed_rows,
            start_specs=start_specs,
            candidate_size=candidate_size,
        )
        paired = _build_paired_differences(seed_rows, start_specs)
        _attach_paired_metrics(main_rows=main_rows, paired_differences=paired)
        per_seed[str(int(seed))] = main_rows
    return per_seed


def _plot_metric_by_start(
    *,
    output_path: Path,
    start_specs: Sequence[StartSpec],
    actor_rows: Sequence[dict[str, Any]],
    value_rows: Sequence[dict[str, Any]],
    reference_row: dict[str, Any] | None,
    metric_key: str,
    ylabel: str,
    title: str,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib is unavailable: {MATPLOTLIB_IMPORT_ERROR}")

    actor_by_label = {str(row["start_fraction_label"]): row for row in actor_rows}
    value_by_label = {str(row["start_fraction_label"]): row for row in value_rows}
    x_positions = [int(spec.percentage) for spec in start_specs]
    x_labels = [spec.display_label for spec in start_specs]
    actor_values = [
        actor_by_label.get(spec.label, {}).get(metric_key)
        for spec in start_specs
    ]
    value_values = [
        value_by_label.get(spec.label, {}).get(metric_key)
        for spec in start_specs
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if any(value is not None for value in actor_values):
        ax.plot(x_positions, actor_values, marker="o", linewidth=2.0, label="Actor-only continuation")
    if any(value is not None for value in value_values):
        ax.plot(x_positions, value_values, marker="o", linewidth=2.0, label="Value-guided continuation")
    if reference_row is not None and reference_row.get(metric_key) is not None:
        ax.axhline(
            y=float(reference_row[metric_key]),
            linestyle="--",
            linewidth=1.6,
            color="#555555",
            label="Reference rollout",
        )
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlabel("Start Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_gain_vs_start(
    *,
    output_path: Path,
    start_specs: Sequence[StartSpec],
    paired_differences: Sequence[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib is unavailable: {MATPLOTLIB_IMPORT_ERROR}")

    paired_by_label = {str(row["start_fraction_label"]): row for row in paired_differences}
    x_positions = [int(spec.percentage) for spec in start_specs]
    x_labels = [spec.display_label for spec in start_specs]
    values = [paired_by_label.get(spec.label, {}).get(metric_key) for spec in start_specs]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(x_positions, values, marker="o", linewidth=2.0, color="#1f77b4")
    ax.axhline(y=0.0, linestyle="--", linewidth=1.4, color="#555555")
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlabel("Start Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _write_output_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    start_specs: Sequence[StartSpec],
    reference_row: dict[str, Any],
    actor_rows: Sequence[dict[str, Any]],
    value_rows: Sequence[dict[str, Any]],
    paired_differences: Sequence[dict[str, Any]],
    plot_paths: dict[str, str | None],
) -> None:
    actor_by_label = {str(row["start_fraction_label"]): row for row in actor_rows}
    value_by_label = {str(row["start_fraction_label"]): row for row in value_rows}
    paired_by_label = {str(row["start_fraction_label"]): row for row in paired_differences}

    lines = [
        "# Delayed-Onset Token-Level Value Guidance",
        "",
        "This run tests whether token-level value guidance becomes less harmful when it is turned on later in the response.",
        "",
        "## Experiment Design",
        "",
        "For each prompt and seed:",
        "",
        "1. Sample one reference rollout from the actor.",
        "2. Define delayed start positions at fixed fractions of the realized reference rollout length.",
        "3. From each prefix, continue once with actor-only sampling and once with token-level value guidance.",
        "4. Compare final accuracy, response length, and value-guidance token-decision diagnostics.",
        "",
        "The reference rollout is reported separately. It is the source trajectory used to define the delayed prefixes.",
        "",
        "## Current Run",
        "",
        f"- Seeds: {' '.join(str(int(seed)) for seed in args.seeds)}",
        f"- Start fractions: {', '.join(spec.display_label for spec in start_specs)}",
        f"- Candidate set for value guidance: actor top-{int(args.candidate_size)} next tokens",
        f"- Actor sampling mode: {args.actor_sampling_mode}",
        f"- Actor temperature / top-p / top-k: {args.actor_temperature} / {args.actor_top_p} / {args.actor_top_k}",
        f"- Max new tokens: {int(args.max_new_tokens)}",
        "",
        "## Output Files",
        "",
        "- `delayed_value_guidance_summary_metrics.json`: metadata, aggregate results, paired deltas, and split diagnostics.",
        "- `delayed_value_guidance_main_results.csv`: flat summary table with one row per method/start condition plus the reference rollout row.",
        "- `delayed_value_guidance_per_example.jsonl`: one row per prompt/seed/start fraction containing the reference rollout fields and both matched continuations.",
        "- `README.md`: this description.",
        "",
        "## Key Metrics",
        "",
        f"- Reference rollout mean accuracy: {reference_row.get('mean_accuracy')}",
        f"- Reference rollout mean response length: {reference_row.get('mean_response_length')}",
        "",
        "### By Start Fraction",
        "",
    ]

    for start_spec in start_specs:
        actor_row = actor_by_label.get(start_spec.label)
        value_row = value_by_label.get(start_spec.label)
        paired_row = paired_by_label.get(start_spec.label)
        if actor_row is None or value_row is None or paired_row is None:
            continue
        lines.extend(
            [
                f"#### {start_spec.display_label}",
                "",
                f"- Actor-only mean accuracy: {actor_row.get('mean_accuracy')}",
                f"- Value-guided mean accuracy: {value_row.get('mean_accuracy')}",
                "- Value-guided minus actor-only mean accuracy delta: "
                f"{paired_row.get('mean_accuracy_delta_value_guided_minus_actor_only')}",
                "- Value-guided overwrite rate vs actor top-1: "
                f"{value_row.get('mean_fraction_token_decisions_different_from_actor_top1')}",
                "- Value-guided mean selected-token value: "
                f"{value_row.get('mean_selected_token_value')}",
                "- Value-guided mean selected-token value margin: "
                f"{value_row.get('mean_selected_token_score_margin')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- If value-guided continuation improves as the start fraction moves later, the main failure mode is likely early value usage.",
            "- If later starts do not help, token-level control itself is probably unstable or too off-policy.",
            "- If an intermediate start is best, the value function may need enough semantic buildup before it becomes useful.",
            "",
            "## Plots",
            "",
            f"- Accuracy vs start fraction: {plot_paths.get('accuracy_vs_start_fraction')}",
            f"- Gain over actor-only vs start fraction: {plot_paths.get('gain_vs_actor_only')}",
            f"- Overwrite rate vs start fraction: {plot_paths.get('overwrite_rate_vs_start_fraction')}",
        ]
    )

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {args.max_new_tokens}")
    if args.candidate_size <= 0:
        raise ValueError(f"candidate_size must be > 0, got {args.candidate_size}")
    if args.ray_num_cpus_per_worker <= 0:
        raise ValueError(f"ray_num_cpus_per_worker must be > 0, got {args.ray_num_cpus_per_worker}")
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if len(set(int(seed) for seed in args.seeds)) != len(args.seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")
    if not args.skip_plots and plt is None:
        raise RuntimeError(f"matplotlib is required for plot generation, but import failed: {MATPLOTLIB_IMPORT_ERROR}")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
    critic_checkpoint_dir = Path(args.critic_checkpoint_dir).resolve()
    actor_hf_dir = ensure_merged_component_checkpoint(
        actor_checkpoint_dir,
        component="actor",
        merged_root=Path(args.actor_merged_root).resolve() if args.actor_merged_root else None,
        hf_source_dir=Path(args.actor_hf_source_dir).resolve() if args.actor_hf_source_dir else None,
        skip_merge=args.skip_merge,
    )
    critic_hf_dir = ensure_merged_component_checkpoint(
        critic_checkpoint_dir,
        component="critic",
        merged_root=Path(args.critic_merged_root).resolve() if args.critic_merged_root else None,
        hf_source_dir=Path(args.critic_hf_source_dir).resolve() if args.critic_hf_source_dir else None,
        skip_merge=args.skip_merge,
    )

    dtype = resolve_dtype(args.dtype)
    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, tokenizer)
    start_specs = _build_start_specs(args.start_fractions)
    examples = load_examples(
        args.dataset_path,
        tokenizer=tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=args.start_index,
        max_examples=args.max_examples,
        shuffle_examples=args.shuffle_examples,
        seed=int(args.seeds[0]),
        pretokenize_max_length=args.max_prompt_length,
    )
    if not examples:
        raise ValueError("No evaluation examples were loaded. Check dataset path and slicing arguments.")

    ray_address = _resolve_ray_address(args.ray_address)
    execution_backend = "ray" if ray_address is not None else "local"
    worker_pairs = parse_worker_pairs(
        args.worker_pairs,
        actor_device=args.actor_device,
        critic_device=args.critic_device,
        default_device=args.device,
    )
    ray_nodes: list[RayNodeInfo] = []
    if ray_address is not None:
        ray_module = _require_ray()
        if not ray_module.is_initialized():
            ray_module.init(address=ray_address)
        ray_nodes = _discover_ray_nodes(ray_module)
        if not ray_nodes:
            raise ValueError("Ray is connected, but no alive Ray nodes were discovered.")
        worker_assignments = build_distributed_worker_assignments(
            num_examples=len(examples),
            worker_pairs=worker_pairs,
            ray_nodes=ray_nodes,
        )
    else:
        worker_assignments = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)
    multi_worker_enabled = len(worker_assignments) > 1

    if ray_address is not None:
        start_rows, worker_summaries = run_ray_multi_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            start_specs=start_specs,
            seeds=[int(seed) for seed in args.seeds],
            worker_assignments=worker_assignments,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_sampling_mode=args.actor_sampling_mode,
            actor_temperature=args.actor_temperature,
            actor_top_p=args.actor_top_p,
            actor_top_k=args.actor_top_k,
            candidate_size=args.candidate_size,
            use_actor_cache=not args.disable_actor_cache,
            ray_num_cpus_per_worker=args.ray_num_cpus_per_worker,
        )
        actor_device = None
        critic_device = None
    elif multi_worker_enabled:
        start_rows, worker_summaries = run_multi_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            start_specs=start_specs,
            seeds=[int(seed) for seed in args.seeds],
            worker_pairs=worker_pairs,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_sampling_mode=args.actor_sampling_mode,
            actor_temperature=args.actor_temperature,
            actor_top_p=args.actor_top_p,
            actor_top_k=args.actor_top_k,
            candidate_size=args.candidate_size,
            use_actor_cache=not args.disable_actor_cache,
        )
        actor_device = None
        critic_device = None
    else:
        actor_device = resolve_device(worker_pairs[0][0])
        critic_device = resolve_device(worker_pairs[0][1]) if worker_pairs[0][1] else actor_device
        actor = load_actor_model(
            actor_hf_dir,
            dtype=dtype,
            device=actor_device,
            trust_remote_code=args.trust_remote_code,
        )
        critic = load_critic_model(
            critic_hf_dir,
            dtype=dtype,
            device=critic_device,
            trust_remote_code=args.trust_remote_code,
        )
        start_rows = []
        progress_units_per_example_seed = _progress_units_per_example_seed(start_specs)
        with tqdm(
            total=len(examples) * len(args.seeds) * progress_units_per_example_seed,
            desc="delayed_value_guidance_eval",
            unit="task",
            dynamic_ncols=True,
        ) as progress_bar:
            for example in examples:
                for seed in args.seeds:
                    progress_bar.set_postfix_str(f"seed={seed} ex={example.example_id} starting")

                    def progress_callback(progress_payload: dict[str, Any]) -> None:
                        phase_label = progress_payload.get("phase_label") or "working"
                        progress_bar.set_postfix_str(f"seed={seed} ex={example.example_id} {phase_label}")
                        progress_bar.update(1)

                    artifacts = evaluate_example_seed(
                        actor=actor,
                        critic=critic,
                        tokenizer=tokenizer,
                        example=example,
                        start_specs=start_specs,
                        max_prompt_length=args.max_prompt_length,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        actor_sampling_mode=args.actor_sampling_mode,
                        actor_temperature=args.actor_temperature,
                        actor_top_p=args.actor_top_p,
                        actor_top_k=args.actor_top_k,
                        candidate_size=args.candidate_size,
                        actor_device=actor_device,
                        critic_device=critic_device,
                        base_seed=int(seed),
                        use_actor_cache=not args.disable_actor_cache,
                        progress_callback=progress_callback,
                    )
                    start_rows.extend(artifacts.per_start_records)
        per_example_path = output_dir / "delayed_value_guidance_per_example.jsonl"
        with per_example_path.open("w", encoding="utf-8") as per_example_file:
            for row in start_rows:
                per_example_file.write(_json_line(row))
        worker_summaries = [
            {
                "worker_id": 0,
                "actor_device": str(actor_device),
                "critic_device": str(critic_device),
                "example_start": 0,
                "example_end": len(examples),
                "num_examples": len(examples),
                "num_seeds": len(args.seeds),
                "worker_total_tasks": len(examples) * len(args.seeds) * progress_units_per_example_seed,
            }
        ]

    main_rows, actor_rows, value_rows = _aggregate_main_results(
        start_rows=start_rows,
        start_specs=start_specs,
        candidate_size=args.candidate_size,
    )
    paired_differences = _build_paired_differences(start_rows, start_specs)
    _attach_paired_metrics(main_rows=main_rows, paired_differences=paired_differences)
    reference_row = next(row for row in main_rows if row["method_family"] == "reference_rollout")
    reference_correctness_splits = _reference_correctness_splits(
        start_rows=start_rows,
        start_specs=start_specs,
        candidate_size=args.candidate_size,
    )
    per_seed_main_results = _per_seed_main_results(
        start_rows=start_rows,
        start_specs=start_specs,
        candidate_size=args.candidate_size,
        seeds=[int(seed) for seed in args.seeds],
    )

    csv_path = output_dir / "delayed_value_guidance_main_results.csv"
    base_fieldnames = [
        "method_id",
        "method_family",
        "method_name",
        "start_fraction",
        "start_percentage",
        "start_fraction_label",
        "start_fraction_display",
        "candidate_size",
        "num_prompt_seed_pairs",
        "num_examples",
        "num_seeds",
        "mean_task_score",
        "mean_accuracy",
        "mean_response_length",
        "eos_rate",
        "max_length_hit_rate",
        "mean_prefix_length",
        "mean_remaining_tokens_after_start",
        "mean_continuation_length",
        "mean_fraction_token_decisions_different_from_actor_top1",
        "mean_selected_token_value",
        "mean_selected_token_score_margin",
        "sum_example_latency_sec",
        "mean_tokens_per_second",
        "collapsed_start_rate",
        "mean_accuracy_delta_vs_actor_only",
        "mean_task_score_delta_vs_actor_only",
        "mean_accuracy_delta_vs_reference",
        "mean_task_score_delta_vs_reference",
    ]
    dynamic_fieldnames = sorted({key for row in main_rows for key in row.keys()} - set(base_fieldnames))
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=base_fieldnames + dynamic_fieldnames)
        writer.writeheader()
        for row in main_rows:
            writer.writerow(row)

    plot_paths: dict[str, str | None] = {
        "accuracy_vs_start_fraction": None,
        "gain_vs_actor_only": None,
        "overwrite_rate_vs_start_fraction": None,
    }
    if not args.skip_plots:
        accuracy_metric_key = "mean_accuracy" if reference_row.get("mean_accuracy") is not None else "mean_task_score"
        accuracy_plot_path = output_dir / "accuracy_vs_start_fraction.png"
        gain_plot_path = output_dir / "gain_over_actor_only_vs_start_fraction.png"
        overwrite_plot_path = output_dir / "overwrite_rate_vs_start_fraction.png"

        _plot_metric_by_start(
            output_path=accuracy_plot_path,
            start_specs=start_specs,
            actor_rows=actor_rows,
            value_rows=value_rows,
            reference_row=reference_row,
            metric_key=accuracy_metric_key,
            ylabel="Accuracy" if accuracy_metric_key == "mean_accuracy" else "Mean Task Score",
            title="Accuracy vs Start Fraction",
            dpi=args.plot_dpi,
        )
        _plot_gain_vs_start(
            output_path=gain_plot_path,
            start_specs=start_specs,
            paired_differences=paired_differences,
            metric_key=(
                "mean_accuracy_delta_value_guided_minus_actor_only"
                if accuracy_metric_key == "mean_accuracy"
                else "mean_task_score_delta_value_guided_minus_actor_only"
            ),
            ylabel=(
                "Accuracy Gain over Actor-only"
                if accuracy_metric_key == "mean_accuracy"
                else "Task Score Gain over Actor-only"
            ),
            title="Value-guided Gain over Actor-only vs Start Fraction",
            dpi=args.plot_dpi,
        )
        _plot_metric_by_start(
            output_path=overwrite_plot_path,
            start_specs=start_specs,
            actor_rows=[],
            value_rows=value_rows,
            reference_row=None,
            metric_key="mean_fraction_token_decisions_different_from_actor_top1",
            ylabel="Overwrite Rate",
            title="Overwrite Rate vs Start Fraction",
            dpi=args.plot_dpi,
        )
        plot_paths = {
            "accuracy_vs_start_fraction": str(accuracy_plot_path),
            "gain_vs_actor_only": str(gain_plot_path),
            "overwrite_rate_vs_start_fraction": str(overwrite_plot_path),
        }

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "execution_backend": execution_backend,
        "actor_checkpoint_dir": str(actor_checkpoint_dir),
        "critic_checkpoint_dir": str(critic_checkpoint_dir),
        "merged_actor_dir": str(actor_hf_dir),
        "merged_critic_dir": str(critic_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "output_dir": str(output_dir),
        "multi_worker_enabled": multi_worker_enabled,
        "actor_device": None if actor_device is None else str(actor_device),
        "critic_device": None if critic_device is None else str(critic_device),
        "ray_address": ray_address,
        "ray_nodes": [asdict(node) for node in ray_nodes],
        "worker_pairs": [[actor, critic] for actor, critic in worker_pairs],
        "worker_assignments": worker_assignments_to_jsonable(worker_assignments),
        "worker_summaries": worker_summaries,
        "dtype": args.dtype,
        "eos_token_ids": list(eos_token_ids),
        "start_specs": [asdict(spec) for spec in start_specs],
        "run_args": vars(args),
        "aggregate_metrics": main_rows,
        "reference_rollout_metrics": reference_row,
        "paired_differences": paired_differences,
        "reference_correctness_splits": reference_correctness_splits,
        "per_seed_main_results": per_seed_main_results,
        "plot_paths": plot_paths,
    }
    summary_path = output_dir / "delayed_value_guidance_summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    _write_output_readme(
        output_dir=output_dir,
        args=args,
        start_specs=start_specs,
        reference_row=reference_row,
        actor_rows=actor_rows,
        value_rows=value_rows,
        paired_differences=paired_differences,
        plot_paths=plot_paths,
    )
    return 0


_make_main_module_importable()

if __name__ == "__main__":
    raise SystemExit(main())
