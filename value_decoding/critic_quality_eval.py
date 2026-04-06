from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
    load_actor_model,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.data import ExampleRecord, load_examples, score_response
from value_decoding.decoding import (
    ActorSamplingMode,
    ActorStepper,
    critic_sequence_last_values,
    sample_token_from_actor,
    set_decode_seed,
)


TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)


@dataclass(frozen=True)
class SampledTrajectory:
    sample_idx: int
    sample_seed: int
    prompt_length: int
    full_sequence_token_ids: tuple[int, ...]
    response_text: str
    response_length: int
    eos_emitted: bool
    max_length_hit: bool
    task_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 critic-quality comparison: sample a shared bank of actor responses once per prompt, "
            "then score the same completed trajectories with an old critic and a new critic."
        )
    )
    parser.add_argument("--actor_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the frozen actor.")
    parser.add_argument(
        "--old_critic_checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint dir for the baseline critic.",
    )
    parser.add_argument(
        "--new_critic_checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint dir for the larger/new critic.",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for all experiment artifacts.")
    parser.add_argument("--actor_merged_root", type=str, default=None, help="Optional merged HF root for actor.")
    parser.add_argument(
        "--old_critic_merged_root",
        type=str,
        default=None,
        help="Optional merged HF root for the old critic.",
    )
    parser.add_argument(
        "--new_critic_merged_root",
        type=str,
        default=None,
        help="Optional merged HF root for the new critic.",
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None, help="Optional response/ground-truth column key.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--num_samples_per_prompt", type=int, default=8)
    parser.add_argument("--critic_score_batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Fallback device for any unset model device.")
    parser.add_argument("--actor_device", type=str, default=None, help="Optional actor device override.")
    parser.add_argument("--old_critic_device", type=str, default=None, help="Optional old critic device override.")
    parser.add_argument("--new_critic_device", type=str, default=None, help="Optional new critic device override.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--disable_actor_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--actor_sampling_mode",
        type=str,
        default=ActorSamplingMode.SAMPLE.value,
        choices=[mode.value for mode in ActorSamplingMode],
    )
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--actor_top_p", type=float, default=1.0)
    parser.add_argument("--actor_top_k", type=int, default=0)
    return parser.parse_args()


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.isclose(x_arr.std(), 0.0) or np.isclose(y_arr.std(), 0.0):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_arr, kind="mergesort")
    ranks = np.empty(len(values_arr), dtype=np.float64)

    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values_arr[order[end + 1]] == values_arr[order[start]]:
            end += 1
        average_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = average_rank
        start = end + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_average_ranks(xs).tolist(), _average_ranks(ys).tolist())


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True) + "\n"


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


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _resolve_model_device(explicit_device: str | None, fallback_device: str | None) -> torch.device:
    return resolve_device(explicit_device or fallback_device)


def _tokenizer_fingerprint(model_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    included_files: list[str] = []
    for filename in TOKENIZER_FINGERPRINT_FILES:
        path = model_dir / filename
        digest.update(filename.encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        included_files.append(filename)
        digest.update(path.read_bytes())
    return {
        "sha256": digest.hexdigest(),
        "files": included_files,
    }


def _assert_shared_tokenizer(tokenizer_fingerprints: dict[str, dict[str, Any]]) -> None:
    fingerprints = {name: payload["sha256"] for name, payload in tokenizer_fingerprints.items()}
    unique_fingerprints = set(fingerprints.values())
    if len(unique_fingerprints) == 1:
        return
    fingerprint_lines = ", ".join(f"{name}={fingerprint}" for name, fingerprint in fingerprints.items())
    raise ValueError(
        "Actor/critic tokenizer files do not match, so this experiment would not score the exact same token "
        f"trajectories across critics. Fingerprints: {fingerprint_lines}"
    )


def _sample_seed(base_seed: int, *, example_id: int, sample_idx: int) -> int:
    return int(base_seed + (example_id + 1) * 1_000_003 + sample_idx * 9_973)


def _random_choice_seed(base_seed: int, *, example_id: int) -> int:
    return int(base_seed + (example_id + 1) * 2_000_029 + 17)


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


def sample_actor_trajectory(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    prompt_ids: torch.Tensor,
    sample_idx: int,
    seed: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
) -> SampledTrajectory:
    set_decode_seed(seed)
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    generated_token_ids: list[int] = []
    eos_emitted = False

    for _step_index in range(max_new_tokens):
        logits = actor_state.current_logits
        selected_token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=actor_sampling_mode,
            temperature=actor_temperature,
            top_p=actor_top_p,
            top_k=actor_top_k,
        )
        generated_token_ids.append(selected_token_id)
        actor_state.append(selected_token_id)
        if selected_token_id in eos_token_ids:
            eos_emitted = True
            break

    response_length = len(generated_token_ids)
    max_length_hit = bool(max_new_tokens > 0 and not eos_emitted and response_length >= max_new_tokens)
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = score_response(example, response_text)
    full_sequence_token_ids = tuple(int(token_id) for token_id in actor_state.sequence_ids[0].detach().cpu().tolist())

    return SampledTrajectory(
        sample_idx=sample_idx,
        sample_seed=seed,
        prompt_length=int(prompt_ids.shape[1]),
        full_sequence_token_ids=full_sequence_token_ids,
        response_text=response_text,
        response_length=response_length,
        eos_emitted=eos_emitted,
        max_length_hit=max_length_hit,
        task_score=float(task_score),
    )


def _pad_sequence_batch(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot pad an empty batch of sequences.")

    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_token_id, device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), device=device, dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError("Encountered an empty sequence while preparing critic inputs.")
        sequence_tensor = torch.tensor(sequence, device=device, dtype=torch.long)
        input_ids[row_index, : len(sequence)] = sequence_tensor
        attention_mask[row_index, : len(sequence)] = 1

    return input_ids, attention_mask


def score_sequences_with_critic(
    critic,
    *,
    sequences: Sequence[Sequence[int]],
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
) -> list[float]:
    if batch_size <= 0:
        raise ValueError(f"critic_score_batch_size must be > 0, got {batch_size}")

    values: list[float] = []
    # Score one completed trajectory at a time to avoid padded-batch numerical drift.
    for sequence in sequences:
        input_ids = torch.tensor([list(sequence)], device=device, dtype=torch.long)
        sequence_value = critic_sequence_last_values(critic, input_ids)[0]
        values.append(float(sequence_value.item()))
    return values


def _argmax_indices(values: Sequence[float]) -> list[int]:
    if not values:
        raise ValueError("Cannot take argmax of an empty list.")
    best_value = max(values)
    return [index for index, value in enumerate(values) if value == best_value]


def _pairwise_ranking_stats(task_scores: Sequence[float], critic_values: Sequence[float]) -> dict[str, Any]:
    if len(task_scores) != len(critic_values):
        raise ValueError("task_scores and critic_values must have the same length.")

    correct_pairs = 0.0
    rankable_pairs = 0
    for left_index in range(len(task_scores)):
        for right_index in range(left_index + 1, len(task_scores)):
            left_score = task_scores[left_index]
            right_score = task_scores[right_index]
            if left_score == right_score:
                continue
            rankable_pairs += 1
            left_value = critic_values[left_index]
            right_value = critic_values[right_index]
            if left_value == right_value:
                correct_pairs += 0.5
                continue
            values_are_ordered_correctly = (left_value > right_value) == (left_score > right_score)
            correct_pairs += 1.0 if values_are_ordered_correctly else 0.0

    return {
        "correct_pairs": correct_pairs,
        "rankable_pairs": rankable_pairs,
        "accuracy": (correct_pairs / rankable_pairs) if rankable_pairs > 0 else None,
    }


def build_prompt_summary(
    *,
    example: ExampleRecord,
    trajectory_rows: Sequence[dict[str, Any]],
    base_seed: int,
) -> dict[str, Any]:
    if not trajectory_rows:
        raise ValueError("Each prompt must have at least one trajectory row.")

    task_scores = [float(row["task_score"]) for row in trajectory_rows]
    old_values = [float(row["old_critic_final_trajectory_value"]) for row in trajectory_rows]
    new_values = [float(row["new_critic_final_trajectory_value"]) for row in trajectory_rows]
    response_lengths = [int(row["response_length"]) for row in trajectory_rows]
    eos_flags = [bool(row["eos_emitted"]) for row in trajectory_rows]
    max_length_flags = [bool(row["max_length_hit"]) for row in trajectory_rows]
    sample_indices = [int(row["sample_idx"]) for row in trajectory_rows]
    sample_seeds = [int(row["sample_seed"]) for row in trajectory_rows]

    old_selected_indices = _argmax_indices(old_values)
    new_selected_indices = _argmax_indices(new_values)
    oracle_best_indices = _argmax_indices(task_scores)
    random_rng = random.Random(_random_choice_seed(base_seed, example_id=example.example_id))
    random_selected_index = random_rng.randrange(len(trajectory_rows))

    old_selected_index = old_selected_indices[0]
    new_selected_index = new_selected_indices[0]
    oracle_best_index = oracle_best_indices[0]

    old_pairwise = _pairwise_ranking_stats(task_scores, old_values)
    new_pairwise = _pairwise_ranking_stats(task_scores, new_values)

    return {
        "example_id": int(example.example_id),
        "prompt_id": int(example.example_id),
        "prompt_text": example.prompt_text,
        "data_source": example.data_source,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "num_samples": len(trajectory_rows),
        "sample_indices": sample_indices,
        "sample_seeds": sample_seeds,
        "task_scores": task_scores,
        "old_critic_values": old_values,
        "new_critic_values": new_values,
        "response_lengths": response_lengths,
        "eos_emitted": eos_flags,
        "max_length_hit": max_length_flags,
        "old_selected_index": old_selected_index,
        "new_selected_index": new_selected_index,
        "random_selected_index": random_selected_index,
        "oracle_best_index": oracle_best_index,
        "oracle_best_indices": oracle_best_indices,
        "old_selected_tied_indices": old_selected_indices,
        "new_selected_tied_indices": new_selected_indices,
        "old_selected_task_score": task_scores[old_selected_index],
        "new_selected_task_score": task_scores[new_selected_index],
        "random_selected_task_score": task_scores[random_selected_index],
        "oracle_best_task_score": task_scores[oracle_best_index],
        "old_selected_value": old_values[old_selected_index],
        "new_selected_value": new_values[new_selected_index],
        "old_selected_response_length": response_lengths[old_selected_index],
        "new_selected_response_length": response_lengths[new_selected_index],
        "random_selected_response_length": response_lengths[random_selected_index],
        "oracle_best_response_length": response_lengths[oracle_best_index],
        "old_selected_is_oracle_best": old_selected_index in oracle_best_indices,
        "new_selected_is_oracle_best": new_selected_index in oracle_best_indices,
        "random_selected_is_oracle_best": random_selected_index in oracle_best_indices,
        "old_pairwise_correct_pairs": old_pairwise["correct_pairs"],
        "old_pairwise_rankable_pairs": old_pairwise["rankable_pairs"],
        "old_pairwise_accuracy": old_pairwise["accuracy"],
        "new_pairwise_correct_pairs": new_pairwise["correct_pairs"],
        "new_pairwise_rankable_pairs": new_pairwise["rankable_pairs"],
        "new_pairwise_accuracy": new_pairwise["accuracy"],
    }


def _bootstrap_mean_difference(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if len(baseline_values) != len(candidate_values):
        raise ValueError("Paired bootstrap inputs must have the same length.")
    if not baseline_values:
        return None

    baseline = np.asarray(baseline_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_differences = np.empty(bootstrap_samples, dtype=np.float64)

    for sample_index in range(bootstrap_samples):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        sample_differences[sample_index] = float(np.mean(candidate[indices] - baseline[indices]))

    observed_difference = float(np.mean(candidate - baseline))
    ci_lower, ci_upper = np.quantile(sample_differences, [0.025, 0.975]).tolist()
    return {
        "observed_difference": observed_difference,
        "bootstrap_mean_difference": float(np.mean(sample_differences)),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "num_prompts": int(len(baseline)),
        "num_bootstrap_samples": int(bootstrap_samples),
    }


def _bootstrap_pairwise_accuracy_difference(
    prompt_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not prompt_rows:
        return None

    old_correct = np.asarray([float(row["old_pairwise_correct_pairs"]) for row in prompt_rows], dtype=np.float64)
    new_correct = np.asarray([float(row["new_pairwise_correct_pairs"]) for row in prompt_rows], dtype=np.float64)
    rankable_pairs = np.asarray([int(row["old_pairwise_rankable_pairs"]) for row in prompt_rows], dtype=np.int64)
    total_rankable_pairs = int(rankable_pairs.sum())
    if total_rankable_pairs <= 0:
        return None

    rng = np.random.default_rng(seed)
    sample_differences = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        indices = rng.integers(0, len(prompt_rows), size=len(prompt_rows))
        sampled_rankable = rankable_pairs[indices].sum()
        if sampled_rankable <= 0:
            sample_differences[sample_index] = 0.0
            continue
        sampled_old_accuracy = float(old_correct[indices].sum() / sampled_rankable)
        sampled_new_accuracy = float(new_correct[indices].sum() / sampled_rankable)
        sample_differences[sample_index] = sampled_new_accuracy - sampled_old_accuracy

    observed_old_accuracy = float(old_correct.sum() / total_rankable_pairs)
    observed_new_accuracy = float(new_correct.sum() / total_rankable_pairs)
    ci_lower, ci_upper = np.quantile(sample_differences, [0.025, 0.975]).tolist()
    return {
        "observed_difference": observed_new_accuracy - observed_old_accuracy,
        "bootstrap_mean_difference": float(np.mean(sample_differences)),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "num_prompts": int(len(prompt_rows)),
        "num_bootstrap_samples": int(bootstrap_samples),
        "total_rankable_pairs": total_rankable_pairs,
    }


def _calibration_by_quantile_bin(
    values: Sequence[float],
    scores: Sequence[float],
    *,
    num_bins: int,
) -> list[dict[str, Any]]:
    if not values or num_bins <= 0:
        return []

    value_array = np.asarray(values, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    ordered_indices = np.argsort(value_array, kind="mergesort")
    split_indices = np.array_split(ordered_indices, min(num_bins, len(ordered_indices)))

    bins: list[dict[str, Any]] = []
    for bin_index, bin_members in enumerate(split_indices):
        if len(bin_members) == 0:
            continue
        bin_values = value_array[bin_members]
        bin_scores = score_array[bin_members]
        bins.append(
            {
                "bin_index": int(bin_index),
                "count": int(len(bin_members)),
                "min_value": float(bin_values.min()),
                "max_value": float(bin_values.max()),
                "mean_value": float(bin_values.mean()),
                "mean_task_score": float(bin_scores.mean()),
            }
        )
    return bins


def _aggregate_old_or_new_metrics(
    *,
    trajectory_rows: Sequence[dict[str, Any]],
    prompt_rows: Sequence[dict[str, Any]],
    value_key: str,
    selected_score_key: str,
    selected_length_key: str,
    selected_value_key: str,
    selected_oracle_hit_key: str,
    pairwise_correct_key: str,
    pairwise_rankable_key: str,
    binary_scores: bool,
) -> dict[str, Any]:
    trajectory_values = [float(row[value_key]) for row in trajectory_rows]
    task_scores = [float(row["task_score"]) for row in trajectory_rows]
    selected_scores = [float(row[selected_score_key]) for row in prompt_rows]
    selected_lengths = [int(row[selected_length_key]) for row in prompt_rows]
    selected_values = [float(row[selected_value_key]) for row in prompt_rows]
    oracle_hits = [1.0 if bool(row[selected_oracle_hit_key]) else 0.0 for row in prompt_rows]

    pairwise_correct = [float(row[pairwise_correct_key]) for row in prompt_rows]
    pairwise_rankable = [int(row[pairwise_rankable_key]) for row in prompt_rows]
    total_rankable_pairs = int(sum(pairwise_rankable))
    mean_prompt_pairwise_accuracy = _mean(
        [
            float(row_accuracy)
            for row_accuracy in (
                row[
                    pairwise_rankable_key.replace("rankable_pairs", "accuracy")
                ]
                for row in prompt_rows
            )
            if row_accuracy is not None
        ]
    )

    prompts_with_any_success = [
        row for row in prompt_rows if float(row["oracle_best_task_score"]) == 1.0
    ]
    conditional_success_recovery = None
    if binary_scores and prompts_with_any_success:
        conditional_success_recovery = float(
            np.mean(
                [
                    1.0 if float(row[selected_score_key]) == 1.0 else 0.0
                    for row in prompts_with_any_success
                ]
            )
        )

    return {
        "num_trajectories": len(trajectory_rows),
        "num_prompts": len(prompt_rows),
        "global_pearson_value_vs_task_score": _pearson(trajectory_values, task_scores),
        "global_spearman_value_vs_task_score": _spearman(trajectory_values, task_scores),
        "pairwise_ranking_accuracy": (
            float(sum(pairwise_correct) / total_rankable_pairs) if total_rankable_pairs > 0 else None
        ),
        "pairwise_rankable_pairs": total_rankable_pairs,
        "mean_prompt_pairwise_ranking_accuracy": mean_prompt_pairwise_accuracy,
        "top1_selected_mean_task_score": _mean(selected_scores),
        "top1_hit_rate_against_oracle_best": _mean(oracle_hits),
        "mean_selected_response_length": _mean(selected_lengths),
        "mean_trajectory_value": _mean(trajectory_values),
        "mean_selected_trajectory_value": _mean(selected_values),
        "conditional_success_recovery_rate": conditional_success_recovery,
    }


def _aggregate_selection_baseline_metrics(
    *,
    prompt_rows: Sequence[dict[str, Any]],
    selected_score_key: str,
    selected_length_key: str,
    selected_oracle_hit_key: str,
    binary_scores: bool,
) -> dict[str, Any]:
    selected_scores = [float(row[selected_score_key]) for row in prompt_rows]
    selected_lengths = [int(row[selected_length_key]) for row in prompt_rows]
    oracle_hits = [1.0 if bool(row[selected_oracle_hit_key]) else 0.0 for row in prompt_rows]

    prompts_with_any_success = [
        row for row in prompt_rows if float(row["oracle_best_task_score"]) == 1.0
    ]
    conditional_success_recovery = None
    if binary_scores and prompts_with_any_success:
        conditional_success_recovery = float(
            np.mean(
                [
                    1.0 if float(row[selected_score_key]) == 1.0 else 0.0
                    for row in prompts_with_any_success
                ]
            )
        )

    return {
        "num_prompts": len(prompt_rows),
        "top1_selected_mean_task_score": _mean(selected_scores),
        "top1_hit_rate_against_oracle_best": _mean(oracle_hits),
        "mean_selected_response_length": _mean(selected_lengths),
        "conditional_success_recovery_rate": conditional_success_recovery,
    }


def aggregate_metrics(
    *,
    trajectory_rows: Sequence[dict[str, Any]],
    prompt_rows: Sequence[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
) -> dict[str, Any]:
    task_scores = [float(row["task_score"]) for row in trajectory_rows]
    binary_scores = set(task_scores).issubset({0.0, 1.0})

    old_metrics = _aggregate_old_or_new_metrics(
        trajectory_rows=trajectory_rows,
        prompt_rows=prompt_rows,
        value_key="old_critic_final_trajectory_value",
        selected_score_key="old_selected_task_score",
        selected_length_key="old_selected_response_length",
        selected_value_key="old_selected_value",
        selected_oracle_hit_key="old_selected_is_oracle_best",
        pairwise_correct_key="old_pairwise_correct_pairs",
        pairwise_rankable_key="old_pairwise_rankable_pairs",
        binary_scores=binary_scores,
    )
    new_metrics = _aggregate_old_or_new_metrics(
        trajectory_rows=trajectory_rows,
        prompt_rows=prompt_rows,
        value_key="new_critic_final_trajectory_value",
        selected_score_key="new_selected_task_score",
        selected_length_key="new_selected_response_length",
        selected_value_key="new_selected_value",
        selected_oracle_hit_key="new_selected_is_oracle_best",
        pairwise_correct_key="new_pairwise_correct_pairs",
        pairwise_rankable_key="new_pairwise_rankable_pairs",
        binary_scores=binary_scores,
    )
    random_metrics = _aggregate_selection_baseline_metrics(
        prompt_rows=prompt_rows,
        selected_score_key="random_selected_task_score",
        selected_length_key="random_selected_response_length",
        selected_oracle_hit_key="random_selected_is_oracle_best",
        binary_scores=binary_scores,
    )
    oracle_metrics = _aggregate_selection_baseline_metrics(
        prompt_rows=prompt_rows,
        selected_score_key="oracle_best_task_score",
        selected_length_key="oracle_best_response_length",
        selected_oracle_hit_key="random_selected_is_oracle_best",
        binary_scores=binary_scores,
    )
    random_metrics["num_trajectories"] = len(trajectory_rows)
    oracle_metrics["num_trajectories"] = len(trajectory_rows)
    oracle_metrics["top1_hit_rate_against_oracle_best"] = 1.0
    if binary_scores:
        prompts_with_any_success = [
            row for row in prompt_rows if float(row["oracle_best_task_score"]) == 1.0
        ]
        oracle_metrics["conditional_success_recovery_rate"] = 1.0 if prompts_with_any_success else None

    old_selected_scores = [float(row["old_selected_task_score"]) for row in prompt_rows]
    new_selected_scores = [float(row["new_selected_task_score"]) for row in prompt_rows]
    old_oracle_hits = [1.0 if bool(row["old_selected_is_oracle_best"]) else 0.0 for row in prompt_rows]
    new_oracle_hits = [1.0 if bool(row["new_selected_is_oracle_best"]) else 0.0 for row in prompt_rows]

    old_values = [float(row["old_critic_final_trajectory_value"]) for row in trajectory_rows]
    new_values = [float(row["new_critic_final_trajectory_value"]) for row in trajectory_rows]

    pairwise_bootstrap = _bootstrap_pairwise_accuracy_difference(
        prompt_rows,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed + 2,
    )
    comparisons = {
        "new_minus_old": {
            "global_pearson_value_vs_task_score": (
                new_metrics["global_pearson_value_vs_task_score"] - old_metrics["global_pearson_value_vs_task_score"]
                if new_metrics["global_pearson_value_vs_task_score"] is not None
                and old_metrics["global_pearson_value_vs_task_score"] is not None
                else None
            ),
            "global_spearman_value_vs_task_score": (
                new_metrics["global_spearman_value_vs_task_score"]
                - old_metrics["global_spearman_value_vs_task_score"]
                if new_metrics["global_spearman_value_vs_task_score"] is not None
                and old_metrics["global_spearman_value_vs_task_score"] is not None
                else None
            ),
            "pairwise_ranking_accuracy": (
                new_metrics["pairwise_ranking_accuracy"] - old_metrics["pairwise_ranking_accuracy"]
                if new_metrics["pairwise_ranking_accuracy"] is not None
                and old_metrics["pairwise_ranking_accuracy"] is not None
                else None
            ),
            "top1_selected_mean_task_score": (
                new_metrics["top1_selected_mean_task_score"] - old_metrics["top1_selected_mean_task_score"]
                if new_metrics["top1_selected_mean_task_score"] is not None
                and old_metrics["top1_selected_mean_task_score"] is not None
                else None
            ),
            "top1_hit_rate_against_oracle_best": (
                new_metrics["top1_hit_rate_against_oracle_best"] - old_metrics["top1_hit_rate_against_oracle_best"]
                if new_metrics["top1_hit_rate_against_oracle_best"] is not None
                and old_metrics["top1_hit_rate_against_oracle_best"] is not None
                else None
            ),
            "mean_selected_response_length": (
                new_metrics["mean_selected_response_length"] - old_metrics["mean_selected_response_length"]
                if new_metrics["mean_selected_response_length"] is not None
                and old_metrics["mean_selected_response_length"] is not None
                else None
            ),
            "mean_trajectory_value": (
                new_metrics["mean_trajectory_value"] - old_metrics["mean_trajectory_value"]
                if new_metrics["mean_trajectory_value"] is not None
                and old_metrics["mean_trajectory_value"] is not None
                else None
            ),
            "paired_bootstrap": {
                "top1_selected_mean_task_score": _bootstrap_mean_difference(
                    old_selected_scores,
                    new_selected_scores,
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed,
                ),
                "top1_hit_rate_against_oracle_best": _bootstrap_mean_difference(
                    old_oracle_hits,
                    new_oracle_hits,
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + 1,
                ),
                "pairwise_ranking_accuracy": pairwise_bootstrap,
            },
        }
    }

    return {
        "binary_task_scores": binary_scores,
        "old_critic": old_metrics,
        "new_critic": new_metrics,
        "random_single_sample": random_metrics,
        "oracle_best_in_bank": oracle_metrics,
        "comparisons": comparisons,
        "calibration_by_bin": {
            "old_critic": _calibration_by_quantile_bin(old_values, task_scores, num_bins=calibration_bins),
            "new_critic": _calibration_by_quantile_bin(new_values, task_scores, num_bins=calibration_bins),
        },
    }


def _main_results_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in ("old_critic", "new_critic", "random_single_sample", "oracle_best_in_bank"):
        payload = dict(metrics[method_name])
        payload["method"] = method_name
        rows.append(payload)
    return rows


def _write_output_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    summary_metrics: dict[str, Any],
) -> None:
    old_metrics = summary_metrics["metrics"]["old_critic"]
    new_metrics = summary_metrics["metrics"]["new_critic"]
    random_metrics = summary_metrics["metrics"]["random_single_sample"]
    oracle_metrics = summary_metrics["metrics"]["oracle_best_in_bank"]
    comparison_metrics = summary_metrics["metrics"]["comparisons"]["new_minus_old"]

    readme_text = "\n".join(
        [
            "# Stage 1 Critic Quality Comparison",
            "",
            "This run uses a shared response bank design:",
            f"- For each prompt, the frozen actor from `{args.actor_checkpoint_dir}` samples `{args.num_samples_per_prompt}` full responses exactly once.",
            "- The old critic and the new critic both score the exact same completed trajectories.",
            "- The critic value is always read at the last token position of the full scored sequence.",
            "- No training is done here, and neither critic steers generation online.",
            "",
            "## Main Metrics",
            "- `pairwise_ranking_accuracy`: within-prompt accuracy on rankable response pairs, with exact critic ties counted as 0.5.",
            "- `top1_selected_mean_task_score`: mean task score of the single response chosen by each critic from the shared bank.",
            "- `top1_hit_rate_against_oracle_best`: fraction of prompts where the critic selected one of the highest-scoring sampled responses.",
            "",
            "## Interpretation",
            "- The primary success criterion is whether the new critic improves within-prompt ranking and prompt-level top-1 selection over the old critic.",
            "- Lower training/value loss alone is not enough for this stage; the question is whether the new critic is a better trajectory evaluator on the same sampled bank.",
            "",
            "## Run Config",
            f"- Dataset: `{args.dataset_path}`",
            f"- Sampling mode: `{args.actor_sampling_mode}`",
            f"- Temperature / top-p / top-k: `{args.actor_temperature}` / `{args.actor_top_p}` / `{args.actor_top_k}`",
            f"- Max prompt length: `{args.max_prompt_length}`",
            f"- Max new tokens: `{args.max_new_tokens}`",
            f"- Seed: `{args.seed}`",
            "",
            "## Quick Read",
            f"- Old critic pairwise ranking accuracy: `{_format_metric(old_metrics['pairwise_ranking_accuracy'])}`",
            f"- New critic pairwise ranking accuracy: `{_format_metric(new_metrics['pairwise_ranking_accuracy'])}`",
            f"- New minus old ranking delta: `{_format_metric(comparison_metrics['pairwise_ranking_accuracy'])}`",
            f"- Old critic top-1 selected mean task score: `{_format_metric(old_metrics['top1_selected_mean_task_score'])}`",
            f"- New critic top-1 selected mean task score: `{_format_metric(new_metrics['top1_selected_mean_task_score'])}`",
            f"- Random single-sample mean task score: `{_format_metric(random_metrics['top1_selected_mean_task_score'])}`",
            f"- Oracle-best-in-bank mean task score: `{_format_metric(oracle_metrics['top1_selected_mean_task_score'])}`",
            "",
            "## Files",
            "- `trajectory_bank.jsonl`: one row per sampled trajectory with task score and both critic values.",
            "- `prompt_level_summary.jsonl`: one row per prompt with shared-bank scores, selection indices, and prompt-level metrics.",
            "- `summary_metrics.json`: aggregate metrics, calibration bins, and paired comparisons.",
            "- `main_results.csv`: compact summary table.",
        ]
    )
    (output_dir / "README.md").write_text(readme_text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.num_samples_per_prompt <= 0:
        raise ValueError(f"--num_samples_per_prompt must be > 0, got {args.num_samples_per_prompt}")
    if args.bootstrap_samples <= 0:
        raise ValueError(f"--bootstrap_samples must be > 0, got {args.bootstrap_samples}")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
    old_critic_checkpoint_dir = Path(args.old_critic_checkpoint_dir).resolve()
    new_critic_checkpoint_dir = Path(args.new_critic_checkpoint_dir).resolve()

    actor_hf_dir = ensure_merged_component_checkpoint(
        actor_checkpoint_dir,
        component="actor",
        merged_root=Path(args.actor_merged_root).resolve() if args.actor_merged_root else None,
        skip_merge=args.skip_merge,
    )
    old_critic_hf_dir = ensure_merged_component_checkpoint(
        old_critic_checkpoint_dir,
        component="critic",
        merged_root=Path(args.old_critic_merged_root).resolve() if args.old_critic_merged_root else None,
        skip_merge=args.skip_merge,
    )
    new_critic_hf_dir = ensure_merged_component_checkpoint(
        new_critic_checkpoint_dir,
        component="critic",
        merged_root=Path(args.new_critic_merged_root).resolve() if args.new_critic_merged_root else None,
        skip_merge=args.skip_merge,
    )

    dtype = resolve_dtype(args.dtype)

    actor_tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    old_critic_tokenizer = load_tokenizer(old_critic_hf_dir, trust_remote_code=args.trust_remote_code)
    new_critic_tokenizer = load_tokenizer(new_critic_hf_dir, trust_remote_code=args.trust_remote_code)
    tokenizer_fingerprints = {
        "actor": _tokenizer_fingerprint(actor_hf_dir),
        "old_critic": _tokenizer_fingerprint(old_critic_hf_dir),
        "new_critic": _tokenizer_fingerprint(new_critic_hf_dir),
    }
    _assert_shared_tokenizer(tokenizer_fingerprints)

    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, actor_tokenizer)
    examples = load_examples(
        args.dataset_path,
        tokenizer=actor_tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=args.start_index,
        max_examples=args.max_examples,
        shuffle_examples=args.shuffle_examples,
        seed=args.seed,
        pretokenize_max_length=args.max_prompt_length,
    )
    if not examples:
        raise ValueError("No evaluation examples were loaded. Check --dataset_path and slicing arguments.")

    actor_device = _resolve_model_device(args.actor_device, args.device)
    old_critic_device = _resolve_model_device(args.old_critic_device, args.device or args.actor_device)
    new_critic_device = _resolve_model_device(
        args.new_critic_device,
        args.device or args.old_critic_device or args.actor_device,
    )

    actor = load_actor_model(
        actor_hf_dir,
        dtype=dtype,
        device=actor_device,
        trust_remote_code=args.trust_remote_code,
    )
    old_critic = load_critic_model(
        old_critic_hf_dir,
        dtype=dtype,
        device=old_critic_device,
        trust_remote_code=args.trust_remote_code,
    )
    new_critic = load_critic_model(
        new_critic_hf_dir,
        dtype=dtype,
        device=new_critic_device,
        trust_remote_code=args.trust_remote_code,
    )

    trajectory_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []

    old_pad_token_id = int(old_critic_tokenizer.pad_token_id)
    new_pad_token_id = int(new_critic_tokenizer.pad_token_id)

    progress_bar = tqdm(examples, desc="critic_quality_eval", unit="prompt", dynamic_ncols=True)
    for example in progress_bar:
        progress_bar.set_postfix_str(f"example_id={example.example_id}")
        prompt_ids = _prompt_ids_tensor(
            example=example,
            tokenizer=actor_tokenizer,
            max_prompt_length=args.max_prompt_length,
            device=actor_device,
        )
        sampled_trajectories: list[SampledTrajectory] = []
        for sample_idx in range(args.num_samples_per_prompt):
            sampled_trajectory = sample_actor_trajectory(
                actor=actor,
                tokenizer=actor_tokenizer,
                example=example,
                prompt_ids=prompt_ids,
                sample_idx=sample_idx,
                seed=_sample_seed(args.seed, example_id=example.example_id, sample_idx=sample_idx),
                actor_sampling_mode=args.actor_sampling_mode,
                actor_temperature=args.actor_temperature,
                actor_top_p=args.actor_top_p,
                actor_top_k=args.actor_top_k,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_token_ids,
                use_actor_cache=not args.disable_actor_cache,
            )
            sampled_trajectories.append(sampled_trajectory)

        full_sequences = [trajectory.full_sequence_token_ids for trajectory in sampled_trajectories]
        old_values = score_sequences_with_critic(
            old_critic,
            sequences=full_sequences,
            device=old_critic_device,
            pad_token_id=old_pad_token_id,
            batch_size=args.critic_score_batch_size,
        )
        new_values = score_sequences_with_critic(
            new_critic,
            sequences=full_sequences,
            device=new_critic_device,
            pad_token_id=new_pad_token_id,
            batch_size=args.critic_score_batch_size,
        )

        prompt_trajectory_rows: list[dict[str, Any]] = []
        for sampled_trajectory, old_value, new_value in zip(sampled_trajectories, old_values, new_values, strict=True):
            trajectory_row = {
                "example_id": int(example.example_id),
                "prompt_id": int(example.example_id),
                "sample_idx": int(sampled_trajectory.sample_idx),
                "sample_seed": int(sampled_trajectory.sample_seed),
                "data_source": example.data_source,
                "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
                "prompt_text": example.prompt_text,
                "generated_response": sampled_trajectory.response_text,
                "prompt_length": int(sampled_trajectory.prompt_length),
                "response_length": int(sampled_trajectory.response_length),
                "full_sequence_length": int(len(sampled_trajectory.full_sequence_token_ids)),
                "eos_emitted": bool(sampled_trajectory.eos_emitted),
                "max_length_hit": bool(sampled_trajectory.max_length_hit),
                "task_score": float(sampled_trajectory.task_score),
                "old_critic_final_trajectory_value": float(old_value),
                "new_critic_final_trajectory_value": float(new_value),
            }
            trajectory_rows.append(trajectory_row)
            prompt_trajectory_rows.append(trajectory_row)

        prompt_rows.append(
            build_prompt_summary(
                example=example,
                trajectory_rows=prompt_trajectory_rows,
                base_seed=args.seed,
            )
        )

    trajectory_bank_path = output_dir / "trajectory_bank.jsonl"
    prompt_summary_path = output_dir / "prompt_level_summary.jsonl"
    summary_metrics_path = output_dir / "summary_metrics.json"
    main_results_path = output_dir / "main_results.csv"

    with trajectory_bank_path.open("w", encoding="utf-8") as trajectory_bank_file:
        for trajectory_row in trajectory_rows:
            trajectory_bank_file.write(_json_line(trajectory_row))

    with prompt_summary_path.open("w", encoding="utf-8") as prompt_summary_file:
        for prompt_row in prompt_rows:
            prompt_summary_file.write(_json_line(prompt_row))

    metrics = aggregate_metrics(
        trajectory_rows=trajectory_rows,
        prompt_rows=prompt_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed + 50_021,
        calibration_bins=args.calibration_bins,
    )

    csv_rows = _main_results_rows(metrics)
    csv_fieldnames = [
        "method",
        "num_trajectories",
        "num_prompts",
        "global_pearson_value_vs_task_score",
        "global_spearman_value_vs_task_score",
        "pairwise_ranking_accuracy",
        "pairwise_rankable_pairs",
        "mean_prompt_pairwise_ranking_accuracy",
        "top1_selected_mean_task_score",
        "top1_hit_rate_against_oracle_best",
        "mean_selected_response_length",
        "mean_trajectory_value",
        "mean_selected_trajectory_value",
        "conditional_success_recovery_rate",
    ]
    with main_results_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "actor_checkpoint_dir": str(actor_checkpoint_dir),
        "old_critic_checkpoint_dir": str(old_critic_checkpoint_dir),
        "new_critic_checkpoint_dir": str(new_critic_checkpoint_dir),
        "merged_actor_dir": str(actor_hf_dir),
        "merged_old_critic_dir": str(old_critic_hf_dir),
        "merged_new_critic_dir": str(new_critic_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "output_dir": str(output_dir),
        "trajectory_bank_path": str(trajectory_bank_path),
        "prompt_summary_path": str(prompt_summary_path),
        "main_results_path": str(main_results_path),
        "num_prompts": len(prompt_rows),
        "num_trajectories": len(trajectory_rows),
        "dtype": args.dtype,
        "devices": {
            "actor": str(actor_device),
            "old_critic": str(old_critic_device),
            "new_critic": str(new_critic_device),
        },
        "eos_token_ids": list(eos_token_ids),
        "tokenizer_fingerprints": tokenizer_fingerprints,
        "shared_response_bank": True,
        "run_args": vars(args),
        "metrics": metrics,
    }
    with summary_metrics_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    _write_output_readme(
        output_dir=output_dir,
        args=args,
        summary_metrics=summary_payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
