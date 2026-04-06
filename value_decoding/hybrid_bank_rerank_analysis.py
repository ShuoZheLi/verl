from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled at runtime
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


BASELINE_METHODS = (
    "random_single_sample",
    "best_of_n_old_critic",
    "best_of_n_new_critic",
    "best_of_n_actor_logprob",
    "best_of_n_actor_avg_logprob",
    "oracle_best_in_bank",
)
DEFAULT_LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
OPTIONAL_NEGATIVE_LAMBDAS = (-1.0, -0.5, -0.25)
DEFAULT_NORMALIZATION = "zscore"
DEFAULT_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc hybrid trajectory reranking analysis over an existing Stage 2 trajectory bank. "
            "No new actor generation or critic scoring is performed."
        )
    )
    parser.add_argument(
        "--trajectory_bank_path",
        type=str,
        required=True,
        help="Existing Stage 2 trajectory_bank.jsonl path.",
    )
    parser.add_argument(
        "--prompt_summary_path",
        type=str,
        default=None,
        help="Optional existing Stage 2 prompt_level_summary.jsonl for prompt-level baseline validation.",
    )
    parser.add_argument(
        "--summary_metrics_path",
        type=str,
        default=None,
        help="Optional existing Stage 2 summary_metrics.json for aggregate baseline validation and metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for the hybrid analysis artifacts.",
    )
    parser.add_argument(
        "--bank_size",
        type=int,
        default=None,
        help="Optional number of samples per prompt to use from the saved bank. Defaults to the minimum prompt bank size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random baseline. Defaults to the Stage 2 seed if summary_metrics.json is provided.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=DEFAULT_NORMALIZATION,
        choices=["zscore", "rank", "minmax"],
        help="Within-prompt normalization used before building the hybrid score.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help="Numerical epsilon for normalization.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
        help="Lambda values for the main new-critic-plus-actor-logprob hybrid sweep.",
    )
    parser.add_argument(
        "--include_negative_lambdas",
        action="store_true",
        help="Append the optional negative lambda sweep to --lambdas.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=2000,
        help="Number of paired bootstrap resamples over prompts.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=160,
        help="Plot DPI for PNG outputs.",
    )
    return parser.parse_args()


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def _format_lambda(value: float) -> str:
    return f"{value:g}"


def _lambda_method_name(value: float) -> str:
    return f"best_of_n_hybrid_newcritic_plus_actor_logprob__lambda_{_format_lambda(value)}"


def _random_selector_seed(base_seed: int, *, example_id: int, bank_size: int) -> int:
    return int(base_seed + (example_id + 1) * 2_000_029 + bank_size * 97 + 17)


def _argmax_indices(values: Sequence[float]) -> list[int]:
    if not values:
        raise ValueError("Cannot take argmax of an empty list.")
    best_value = max(values)
    return [index for index, value in enumerate(values) if value == best_value]


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


def _normalize_within_prompt(values: Sequence[float], *, normalization: str, eps: float) -> list[float]:
    value_array = np.asarray(values, dtype=np.float64)
    if len(value_array) == 0:
        return []

    if normalization == "zscore":
        std = float(value_array.std())
        if std <= eps:
            return [0.0] * len(value_array)
        mean = float(value_array.mean())
        return [float((value - mean) / (std + eps)) for value in value_array]

    if normalization == "minmax":
        value_min = float(value_array.min())
        value_max = float(value_array.max())
        if value_max - value_min <= eps:
            return [0.0] * len(value_array)
        return [float((value - value_min) / (value_max - value_min + eps)) for value in value_array]

    if normalization == "rank":
        ranks = _average_ranks(value_array.tolist())
        centered = ranks - ranks.mean()
        scale = float(ranks.std())
        if scale <= eps:
            return [0.0] * len(ranks)
        return [float(value / (scale + eps)) for value in centered]

    raise ValueError(f"Unsupported normalization: {normalization}")


def _select_index(values: Sequence[float]) -> tuple[int, list[int], float]:
    tied_positions = _argmax_indices(values)
    selected_position = tied_positions[0]
    return selected_position, tied_positions, float(values[selected_position])


def load_trajectory_bank(trajectory_bank_path: Path) -> dict[int, list[dict[str, Any]]]:
    prompts: dict[int, list[dict[str, Any]]] = {}
    with trajectory_bank_path.open("r", encoding="utf-8") as trajectory_file:
        for line in trajectory_file:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = int(row["example_id"])
            prompts.setdefault(example_id, []).append(row)

    for rows in prompts.values():
        rows.sort(key=lambda row: int(row["sample_idx"]))
    return prompts


def infer_bank_size(grouped_rows: dict[int, list[dict[str, Any]]], requested_bank_size: int | None) -> int:
    if not grouped_rows:
        raise ValueError("The trajectory bank is empty.")
    prompt_sizes = [len(rows) for rows in grouped_rows.values()]
    inferred_bank_size = min(prompt_sizes)
    if requested_bank_size is None:
        return inferred_bank_size
    if requested_bank_size <= 0:
        raise ValueError(f"--bank_size must be > 0, got {requested_bank_size}")
    if requested_bank_size > inferred_bank_size:
        raise ValueError(
            f"--bank_size ({requested_bank_size}) exceeds the smallest saved prompt bank size ({inferred_bank_size})."
        )
    return requested_bank_size


def _build_baseline_method_record(
    *,
    method_name: str,
    selected_row: dict[str, Any],
    tied_positions: Sequence[int],
    bank_rows: Sequence[dict[str, Any]],
    selection_score_field: str | None,
    selection_score: float | None,
    oracle_best_sample_indices: Sequence[int],
) -> dict[str, Any]:
    selected_sample_idx = int(selected_row["sample_idx"])
    return {
        "method": method_name,
        "selected_index": selected_sample_idx,
        "selected_tied_indices": [int(bank_rows[position]["sample_idx"]) for position in tied_positions],
        "selected_task_score": float(selected_row["task_score"]),
        "selected_response_length": int(selected_row["response_length"]),
        "selected_is_correct": bool(float(selected_row["task_score"]) == 1.0),
        "selected_is_oracle_best": selected_sample_idx in oracle_best_sample_indices,
        "selected_actor_response_logprob": float(selected_row["actor_response_logprob"]),
        "selected_actor_response_avg_logprob": float(selected_row["actor_response_avg_logprob"]),
        "selected_old_critic_final_trajectory_value": float(selected_row["old_critic_final_trajectory_value"]),
        "selected_new_critic_final_trajectory_value": float(selected_row["new_critic_final_trajectory_value"]),
        "selection_score_field": selection_score_field,
        "selected_selection_score": selection_score,
    }


def build_prompt_summary(
    *,
    example_id: int,
    bank_rows: Sequence[dict[str, Any]],
    bank_size: int,
    base_seed: int,
    normalization: str,
    eps: float,
    lambdas: Sequence[float],
) -> dict[str, Any]:
    if len(bank_rows) < bank_size:
        raise ValueError(
            f"Prompt {example_id} has only {len(bank_rows)} saved rows but bank_size={bank_size} was requested."
        )

    bank_rows = list(bank_rows[:bank_size])
    task_scores = [float(row["task_score"]) for row in bank_rows]
    response_lengths = [int(row["response_length"]) for row in bank_rows]
    new_values = [float(row["new_critic_final_trajectory_value"]) for row in bank_rows]
    old_values = [float(row["old_critic_final_trajectory_value"]) for row in bank_rows]
    actor_logprobs = [float(row["actor_response_logprob"]) for row in bank_rows]
    actor_avg_logprobs = [float(row["actor_response_avg_logprob"]) for row in bank_rows]

    z_new = _normalize_within_prompt(new_values, normalization=normalization, eps=eps)
    z_logp = _normalize_within_prompt(actor_logprobs, normalization=normalization, eps=eps)

    oracle_best_positions = _argmax_indices(task_scores)
    oracle_best_sample_indices = [int(bank_rows[position]["sample_idx"]) for position in oracle_best_positions]
    oracle_best_task_score = float(bank_rows[oracle_best_positions[0]]["task_score"])

    methods: dict[str, Any] = {}

    random_rng = random.Random(_random_selector_seed(base_seed, example_id=example_id, bank_size=bank_size))
    random_position = int(random_rng.randrange(len(bank_rows)))
    methods["random_single_sample"] = _build_baseline_method_record(
        method_name="random_single_sample",
        selected_row=bank_rows[random_position],
        tied_positions=[random_position],
        bank_rows=bank_rows,
        selection_score_field=None,
        selection_score=None,
        oracle_best_sample_indices=oracle_best_sample_indices,
    )

    baseline_specs = (
        ("best_of_n_old_critic", old_values, "old_critic_final_trajectory_value"),
        ("best_of_n_new_critic", new_values, "new_critic_final_trajectory_value"),
        ("best_of_n_actor_logprob", actor_logprobs, "actor_response_logprob"),
        ("best_of_n_actor_avg_logprob", actor_avg_logprobs, "actor_response_avg_logprob"),
        ("oracle_best_in_bank", task_scores, "task_score"),
    )
    for method_name, values, selection_score_field in baseline_specs:
        selected_position, tied_positions, selection_score = _select_index(values)
        methods[method_name] = _build_baseline_method_record(
            method_name=method_name,
            selected_row=bank_rows[selected_position],
            tied_positions=tied_positions,
            bank_rows=bank_rows,
            selection_score_field=selection_score_field,
            selection_score=selection_score,
            oracle_best_sample_indices=oracle_best_sample_indices,
        )

    hybrid_methods: dict[str, Any] = {}
    for lambda_value in lambdas:
        hybrid_scores = [float(z_new[index] + lambda_value * z_logp[index]) for index in range(len(bank_rows))]
        selected_position, tied_positions, selection_score = _select_index(hybrid_scores)
        selected_row = bank_rows[selected_position]
        method_name = _lambda_method_name(lambda_value)
        selected_sample_idx = int(selected_row["sample_idx"])
        hybrid_methods[method_name] = {
            "method": method_name,
            "lambda": float(lambda_value),
            "selected_index": selected_sample_idx,
            "selected_tied_indices": [int(bank_rows[position]["sample_idx"]) for position in tied_positions],
            "selected_task_score": float(selected_row["task_score"]),
            "selected_response_length": int(selected_row["response_length"]),
            "selected_is_correct": bool(float(selected_row["task_score"]) == 1.0),
            "selected_is_oracle_best": selected_sample_idx in oracle_best_sample_indices,
            "selected_new_critic_final_trajectory_value": float(selected_row["new_critic_final_trajectory_value"]),
            "selected_actor_response_logprob": float(selected_row["actor_response_logprob"]),
            "selected_actor_response_avg_logprob": float(selected_row["actor_response_avg_logprob"]),
            "selected_normalized_new_critic_value": float(z_new[selected_position]),
            "selected_normalized_actor_logprob": float(z_logp[selected_position]),
            "selected_hybrid_score": selection_score,
            "selection_score_field": "normalized_new_critic_plus_lambda_times_normalized_actor_logprob",
            "changed_vs_best_of_n_new_critic": selected_sample_idx != methods["best_of_n_new_critic"]["selected_index"],
        }

    prompt_summary = {
        "example_id": int(example_id),
        "prompt_id": int(example_id),
        "prompt_text": bank_rows[0]["prompt_text"],
        "data_source": bank_rows[0]["data_source"],
        "ground_truth": bank_rows[0]["ground_truth"],
        "bank_size": int(bank_size),
        "sample_indices": [int(row["sample_idx"]) for row in bank_rows],
        "sample_seeds": [int(row["sample_seed"]) for row in bank_rows],
        "task_scores": task_scores,
        "response_lengths": response_lengths,
        "eos_emitted": [bool(row["eos_emitted"]) for row in bank_rows],
        "max_length_hit": [bool(row["max_length_hit"]) for row in bank_rows],
        "actor_response_logprobs": actor_logprobs,
        "actor_response_avg_logprobs": actor_avg_logprobs,
        "old_critic_values": old_values,
        "new_critic_values": new_values,
        "normalized_new_critic_values": z_new,
        "normalized_actor_logprobs": z_logp,
        "normalization": normalization,
        "normalization_eps": float(eps),
        "oracle_best_index": int(bank_rows[oracle_best_positions[0]]["sample_idx"]),
        "oracle_best_indices": oracle_best_sample_indices,
        "oracle_best_task_score": oracle_best_task_score,
        "baseline_methods": methods,
        "hybrid_methods": hybrid_methods,
    }
    return prompt_summary


def _metric_from_prompt_rows(
    prompt_rows: Sequence[dict[str, Any]],
    *,
    method_name: str,
    method_group: str,
    metric_name: str,
    binary_task_scores: bool,
) -> float | None:
    def _method_record(prompt_row: dict[str, Any]) -> dict[str, Any]:
        return prompt_row[method_group][method_name]

    if metric_name == "selected_mean_task_score":
        return _mean([float(_method_record(prompt_row)["selected_task_score"]) for prompt_row in prompt_rows])

    if metric_name == "top1_hit_rate_against_oracle_best":
        return _mean(
            [1.0 if bool(_method_record(prompt_row)["selected_is_oracle_best"]) else 0.0 for prompt_row in prompt_rows]
        )

    if metric_name == "conditional_success_recovery_rate":
        if not binary_task_scores:
            return None
        successful_rows = [prompt_row for prompt_row in prompt_rows if float(prompt_row["oracle_best_task_score"]) == 1.0]
        if not successful_rows:
            return None
        return _mean(
            [1.0 if bool(_method_record(prompt_row)["selected_is_correct"]) else 0.0 for prompt_row in successful_rows]
        )

    if metric_name == "mean_selected_response_length":
        return _mean([int(_method_record(prompt_row)["selected_response_length"]) for prompt_row in prompt_rows])

    raise ValueError(f"Unsupported metric name: {metric_name}")


def _bootstrap_metric_difference(
    prompt_rows: Sequence[dict[str, Any]],
    *,
    method_a: str,
    method_a_group: str,
    method_b: str,
    method_b_group: str,
    metric_name: str,
    bootstrap_samples: int,
    seed: int,
    binary_task_scores: bool,
) -> dict[str, Any] | None:
    observed_a = _metric_from_prompt_rows(
        prompt_rows,
        method_name=method_a,
        method_group=method_a_group,
        metric_name=metric_name,
        binary_task_scores=binary_task_scores,
    )
    observed_b = _metric_from_prompt_rows(
        prompt_rows,
        method_name=method_b,
        method_group=method_b_group,
        metric_name=metric_name,
        binary_task_scores=binary_task_scores,
    )
    if observed_a is None or observed_b is None:
        return None

    rng = np.random.default_rng(seed)
    sample_differences: list[float] = []
    for _sample_index in range(bootstrap_samples):
        indices = rng.integers(0, len(prompt_rows), size=len(prompt_rows))
        sampled_rows = [prompt_rows[int(index)] for index in indices]
        sample_a = _metric_from_prompt_rows(
            sampled_rows,
            method_name=method_a,
            method_group=method_a_group,
            metric_name=metric_name,
            binary_task_scores=binary_task_scores,
        )
        sample_b = _metric_from_prompt_rows(
            sampled_rows,
            method_name=method_b,
            method_group=method_b_group,
            metric_name=metric_name,
            binary_task_scores=binary_task_scores,
        )
        if sample_a is None or sample_b is None:
            continue
        sample_differences.append(float(sample_a - sample_b))

    if not sample_differences:
        return None

    sample_array = np.asarray(sample_differences, dtype=np.float64)
    ci_lower, ci_upper = np.quantile(sample_array, [0.025, 0.975]).tolist()
    return {
        "observed_difference": float(observed_a - observed_b),
        "bootstrap_mean_difference": float(sample_array.mean()),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "num_prompts": int(len(prompt_rows)),
        "num_bootstrap_samples": int(bootstrap_samples),
        "metric_name": metric_name,
    }


def aggregate_metrics(
    *,
    prompt_rows: Sequence[dict[str, Any]],
    lambdas: Sequence[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
    binary_task_scores: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_metrics: dict[str, Any] = {}
    for method_name in BASELINE_METHODS:
        selected_scores = [float(prompt_row["baseline_methods"][method_name]["selected_task_score"]) for prompt_row in prompt_rows]
        selected_lengths = [
            int(prompt_row["baseline_methods"][method_name]["selected_response_length"]) for prompt_row in prompt_rows
        ]
        oracle_hits = [
            1.0 if bool(prompt_row["baseline_methods"][method_name]["selected_is_oracle_best"]) else 0.0
            for prompt_row in prompt_rows
        ]
        method_metrics = {
            "method": method_name,
            "method_group": "baseline_methods",
            "lambda": None,
            "num_prompts": int(len(prompt_rows)),
            "num_bank_success_prompts": int(
                sum(1 for prompt_row in prompt_rows if float(prompt_row["oracle_best_task_score"]) == 1.0)
            ),
            "selected_mean_task_score": _mean(selected_scores),
            "conditional_success_recovery_rate": _metric_from_prompt_rows(
                prompt_rows,
                method_name=method_name,
                method_group="baseline_methods",
                metric_name="conditional_success_recovery_rate",
                binary_task_scores=binary_task_scores,
            ),
            "top1_hit_rate_against_oracle_best": _mean(oracle_hits),
            "mean_selected_response_length": _mean(selected_lengths),
            "mean_selected_hybrid_score": None,
            "mean_selected_new_critic_value": _mean(
                [
                    float(prompt_row["baseline_methods"][method_name]["selected_new_critic_final_trajectory_value"])
                    for prompt_row in prompt_rows
                ]
            ),
            "mean_selected_actor_logprob": _mean(
                [
                    float(prompt_row["baseline_methods"][method_name]["selected_actor_response_logprob"])
                    for prompt_row in prompt_rows
                ]
            ),
            "selection_score_field": prompt_rows[0]["baseline_methods"][method_name]["selection_score_field"],
        }
        baseline_metrics[method_name] = method_metrics

    hybrid_metrics: dict[str, Any] = {}
    for lambda_value in lambdas:
        method_name = _lambda_method_name(lambda_value)
        selected_scores = [float(prompt_row["hybrid_methods"][method_name]["selected_task_score"]) for prompt_row in prompt_rows]
        selected_lengths = [
            int(prompt_row["hybrid_methods"][method_name]["selected_response_length"]) for prompt_row in prompt_rows
        ]
        oracle_hits = [
            1.0 if bool(prompt_row["hybrid_methods"][method_name]["selected_is_oracle_best"]) else 0.0
            for prompt_row in prompt_rows
        ]
        changed_fraction = _mean(
            [1.0 if bool(prompt_row["hybrid_methods"][method_name]["changed_vs_best_of_n_new_critic"]) else 0.0 for prompt_row in prompt_rows]
        )

        hybrid_metrics[method_name] = {
            "method": method_name,
            "method_group": "hybrid_methods",
            "lambda": float(lambda_value),
            "num_prompts": int(len(prompt_rows)),
            "num_bank_success_prompts": int(
                sum(1 for prompt_row in prompt_rows if float(prompt_row["oracle_best_task_score"]) == 1.0)
            ),
            "selected_mean_task_score": _mean(selected_scores),
            "conditional_success_recovery_rate": _metric_from_prompt_rows(
                prompt_rows,
                method_name=method_name,
                method_group="hybrid_methods",
                metric_name="conditional_success_recovery_rate",
                binary_task_scores=binary_task_scores,
            ),
            "top1_hit_rate_against_oracle_best": _mean(oracle_hits),
            "mean_selected_response_length": _mean(selected_lengths),
            "mean_selected_hybrid_score": _mean(
                [float(prompt_row["hybrid_methods"][method_name]["selected_hybrid_score"]) for prompt_row in prompt_rows]
            ),
            "mean_selected_new_critic_value": _mean(
                [
                    float(prompt_row["hybrid_methods"][method_name]["selected_new_critic_final_trajectory_value"])
                    for prompt_row in prompt_rows
                ]
            ),
            "mean_selected_actor_logprob": _mean(
                [
                    float(prompt_row["hybrid_methods"][method_name]["selected_actor_response_logprob"])
                    for prompt_row in prompt_rows
                ]
            ),
            "mean_selected_normalized_new_critic_value": _mean(
                [
                    float(prompt_row["hybrid_methods"][method_name]["selected_normalized_new_critic_value"])
                    for prompt_row in prompt_rows
                ]
            ),
            "mean_selected_normalized_actor_logprob": _mean(
                [
                    float(prompt_row["hybrid_methods"][method_name]["selected_normalized_actor_logprob"])
                    for prompt_row in prompt_rows
                ]
            ),
            "fraction_prompts_changed_vs_best_of_n_new_critic": changed_fraction,
            "selection_score_field": prompt_rows[0]["hybrid_methods"][method_name]["selection_score_field"],
        }

    comparison_specs = (
        ("best_of_n_new_critic", "baseline_methods"),
        ("best_of_n_actor_logprob", "baseline_methods"),
        ("best_of_n_old_critic", "baseline_methods"),
        ("random_single_sample", "baseline_methods"),
        ("oracle_best_in_bank", "baseline_methods"),
    )
    comparisons: dict[str, Any] = {}
    for lambda_value in lambdas:
        method_name = _lambda_method_name(lambda_value)
        method_comparisons: dict[str, Any] = {}
        for baseline_method_name, baseline_group in comparison_specs:
            comparison_name = f"{method_name}_minus_{baseline_method_name}"
            score_difference = (
                float(hybrid_metrics[method_name]["selected_mean_task_score"])
                - float(baseline_metrics[baseline_method_name]["selected_mean_task_score"])
            )
            recovery_difference = None
            hybrid_recovery = hybrid_metrics[method_name]["conditional_success_recovery_rate"]
            baseline_recovery = baseline_metrics[baseline_method_name]["conditional_success_recovery_rate"]
            if hybrid_recovery is not None and baseline_recovery is not None:
                recovery_difference = float(hybrid_recovery - baseline_recovery)

            oracle_hit_difference = (
                float(hybrid_metrics[method_name]["top1_hit_rate_against_oracle_best"])
                - float(baseline_metrics[baseline_method_name]["top1_hit_rate_against_oracle_best"])
            )
            length_difference = (
                float(hybrid_metrics[method_name]["mean_selected_response_length"])
                - float(baseline_metrics[baseline_method_name]["mean_selected_response_length"])
            )
            method_comparisons[comparison_name] = {
                "method_a": method_name,
                "method_a_group": "hybrid_methods",
                "method_b": baseline_method_name,
                "method_b_group": baseline_group,
                "selected_mean_task_score_difference": score_difference,
                "conditional_success_recovery_rate_difference": recovery_difference,
                "top1_hit_rate_against_oracle_best_difference": oracle_hit_difference,
                "mean_selected_response_length_difference": length_difference,
                "paired_bootstrap": {
                    "selected_mean_task_score_difference": _bootstrap_metric_difference(
                        prompt_rows,
                        method_a=method_name,
                        method_a_group="hybrid_methods",
                        method_b=baseline_method_name,
                        method_b_group=baseline_group,
                        metric_name="selected_mean_task_score",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + int(round((lambda_value + 10.0) * 1000)) + 1,
                        binary_task_scores=binary_task_scores,
                    ),
                    "conditional_success_recovery_rate_difference": _bootstrap_metric_difference(
                        prompt_rows,
                        method_a=method_name,
                        method_a_group="hybrid_methods",
                        method_b=baseline_method_name,
                        method_b_group=baseline_group,
                        metric_name="conditional_success_recovery_rate",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + int(round((lambda_value + 10.0) * 1000)) + 2,
                        binary_task_scores=binary_task_scores,
                    ),
                    "top1_hit_rate_against_oracle_best_difference": _bootstrap_metric_difference(
                        prompt_rows,
                        method_a=method_name,
                        method_a_group="hybrid_methods",
                        method_b=baseline_method_name,
                        method_b_group=baseline_group,
                        metric_name="top1_hit_rate_against_oracle_best",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + int(round((lambda_value + 10.0) * 1000)) + 3,
                        binary_task_scores=binary_task_scores,
                    ),
                },
            }
        comparisons[method_name] = method_comparisons

    disagreement_analysis: dict[str, Any] = {}
    for lambda_value in lambdas:
        method_name = _lambda_method_name(lambda_value)
        disagreement_rows = [
            prompt_row for prompt_row in prompt_rows if bool(prompt_row["hybrid_methods"][method_name]["changed_vs_best_of_n_new_critic"])
        ]
        if disagreement_rows:
            hybrid_accuracy = _mean(
                [float(prompt_row["hybrid_methods"][method_name]["selected_task_score"]) for prompt_row in disagreement_rows]
            )
            pure_new_accuracy = _mean(
                [
                    float(prompt_row["baseline_methods"]["best_of_n_new_critic"]["selected_task_score"])
                    for prompt_row in disagreement_rows
                ]
            )
        else:
            hybrid_accuracy = None
            pure_new_accuracy = None

        disagreement_analysis[method_name] = {
            "lambda": float(lambda_value),
            "num_disagreement_prompts": int(len(disagreement_rows)),
            "fraction_disagreement_prompts": (len(disagreement_rows) / len(prompt_rows)) if prompt_rows else None,
            "hybrid_accuracy_on_disagreement_prompts": hybrid_accuracy,
            "pure_new_critic_accuracy_on_disagreement_prompts": pure_new_accuracy,
        }

    return baseline_metrics, hybrid_metrics, comparisons, disagreement_analysis


def validate_against_stage2_prompt_summary(
    *,
    prompt_rows: Sequence[dict[str, Any]],
    stage2_prompt_summary_path: Path | None,
    stage2_bank_size: int,
) -> dict[str, Any] | None:
    if stage2_prompt_summary_path is None:
        return None

    stage2_prompt_rows: dict[int, dict[str, Any]] = {}
    with stage2_prompt_summary_path.open("r", encoding="utf-8") as prompt_file:
        for line in prompt_file:
            if not line.strip():
                continue
            row = json.loads(line)
            stage2_prompt_rows[int(row["example_id"])] = row

    n_key = str(stage2_bank_size)
    if stage2_prompt_rows:
        first_row = next(iter(stage2_prompt_rows.values()))
        if n_key not in first_row.get("by_n", {}):
            return {
                "validated_prompt_count": 0,
                "stage2_prompt_summary_path": str(stage2_prompt_summary_path),
                "stage2_bank_size": int(stage2_bank_size),
                "skipped": True,
                "skip_reason": (
                    f"Stage 2 prompt summary does not contain by_n['{n_key}']; "
                    "prompt-level validation was skipped."
                ),
            }

    checked_prompts = 0
    for prompt_row in prompt_rows:
        example_id = int(prompt_row["example_id"])
        stage2_row = stage2_prompt_rows.get(example_id)
        if stage2_row is None:
            raise ValueError(f"Prompt {example_id} missing from Stage 2 prompt summary validation file.")
        stage2_methods = stage2_row["by_n"][n_key]["methods"]
        for method_name in BASELINE_METHODS:
            current_method = prompt_row["baseline_methods"][method_name]
            stage2_method = stage2_methods[method_name]
            fields = (
                "selected_index",
                "selected_task_score",
                "selected_response_length",
                "selected_is_correct",
                "selected_is_oracle_best",
            )
            for field in fields:
                if current_method[field] != stage2_method[field]:
                    raise ValueError(
                        f"Prompt-level Stage 2 validation mismatch for example_id={example_id} method={method_name} field={field}: "
                        f"current={current_method[field]!r} stage2={stage2_method[field]!r}"
                    )
        checked_prompts += 1

    return {
        "validated_prompt_count": int(checked_prompts),
        "stage2_prompt_summary_path": str(stage2_prompt_summary_path),
        "stage2_bank_size": int(stage2_bank_size),
        "skipped": False,
    }


def validate_against_stage2_summary(
    *,
    baseline_metrics: dict[str, Any],
    stage2_summary_path: Path | None,
    stage2_bank_size: int,
) -> dict[str, Any] | None:
    if stage2_summary_path is None:
        return None

    stage2_summary = json.load(stage2_summary_path.open("r", encoding="utf-8"))
    n_key = str(stage2_bank_size)
    if n_key not in stage2_summary.get("metrics_by_n", {}):
        return {
            "stage2_summary_metrics_path": str(stage2_summary_path),
            "stage2_bank_size": int(stage2_bank_size),
            "validated_methods": [],
            "validated_metric_fields": [],
            "skipped": True,
            "skip_reason": (
                f"Stage 2 aggregate summary does not contain metrics_by_n['{n_key}']; "
                "aggregate validation was skipped."
            ),
        }

    stage2_metrics = stage2_summary["metrics_by_n"][n_key]
    metric_fields = (
        "selected_mean_task_score",
        "conditional_success_recovery_rate",
        "top1_hit_rate_against_oracle_best",
        "mean_selected_response_length",
    )
    for method_name in BASELINE_METHODS:
        for field in metric_fields:
            current_value = baseline_metrics[method_name][field]
            stage2_value = stage2_metrics[method_name][field]
            if current_value is None or stage2_value is None:
                if current_value != stage2_value:
                    raise ValueError(
                        f"Aggregate Stage 2 validation mismatch for method={method_name} field={field}: "
                        f"current={current_value!r} stage2={stage2_value!r}"
                    )
                continue
            if not math.isclose(float(current_value), float(stage2_value), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"Aggregate Stage 2 validation mismatch for method={method_name} field={field}: "
                    f"current={current_value!r} stage2={stage2_value!r}"
                )

    return {
        "stage2_summary_metrics_path": str(stage2_summary_path),
        "stage2_bank_size": int(stage2_bank_size),
        "validated_methods": list(BASELINE_METHODS),
        "validated_metric_fields": list(metric_fields),
        "skipped": False,
    }


def _plot_hybrid_with_baselines(
    *,
    x_values: Sequence[float],
    y_values: Sequence[float],
    baselines: dict[str, float | None],
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plot generation.") from MATPLOTLIB_IMPORT_ERROR

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(x_values, y_values, marker="o", linewidth=2.0, label="Hybrid")
    for label, value in baselines.items():
        if value is None:
            continue
        axis.axhline(value, linestyle="--", linewidth=1.5, label=label)
    axis.set_xlabel("lambda")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_lambda_curve(
    *,
    x_values: Sequence[float],
    y_values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plot generation.") from MATPLOTLIB_IMPORT_ERROR

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(x_values, y_values, marker="o", linewidth=2.0)
    axis.set_xlabel("lambda")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _write_output_readme(
    *,
    output_dir: Path,
    trajectory_bank_path: Path,
    bank_size: int,
    normalization: str,
    eps: float,
    lambdas: Sequence[float],
    baseline_metrics: dict[str, Any],
    hybrid_metrics: dict[str, Any],
) -> None:
    best_lambda_method = max(
        hybrid_metrics.values(),
        key=lambda payload: float(payload["selected_mean_task_score"]),
    )
    lines = [
        "# Hybrid Trajectory Reranking on Existing Stage 2 Response Bank",
        "",
        "This is a post-hoc reranking analysis only:",
        f"- Input trajectory bank: `{trajectory_bank_path}`",
        f"- Bank size reused per prompt: `{bank_size}`",
        "- No new actor generation was run.",
        "- No critic rescoring was run.",
        "- All results are computed from the saved Stage 2 bank.",
        "",
        "## Hybrid Definition",
        "- Within each prompt, normalize new critic values and actor response log-probs using the configured normalization.",
        "- For each lambda, compute `hybrid_score = normalized_new_critic_value + lambda * normalized_actor_logprob`.",
        "- Select the sampled response with highest hybrid score.",
        "",
        "## Configuration",
        f"- Normalization: `{normalization}`",
        f"- Normalization epsilon: `{eps}`",
        f"- Lambdas: `{list(lambdas)}`",
        "",
        "## Quick Read",
        f"- Pure new critic selected mean task score: `{baseline_metrics['best_of_n_new_critic']['selected_mean_task_score']:.6f}`",
        f"- Pure actor log-prob selected mean task score: `{baseline_metrics['best_of_n_actor_logprob']['selected_mean_task_score']:.6f}`",
        f"- Best hybrid lambda: `{best_lambda_method['lambda']}`",
        f"- Best hybrid selected mean task score: `{best_lambda_method['selected_mean_task_score']:.6f}`",
        "",
        "## Files",
        "- `hybrid_prompt_level_summary.jsonl`",
        "- `hybrid_summary_metrics.json`",
        "- `hybrid_main_results.csv`",
        "- `hybrid_comparisons.json`",
        "- `hybrid_disagreement_analysis.json`",
        "- `accuracy_vs_lambda.png`",
        "- `conditional_recovery_vs_lambda.png`",
        "- `mean_selected_response_length_vs_lambda.png`",
        "- `fraction_changed_vs_lambda.png`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError(f"--bootstrap_samples must be > 0, got {args.bootstrap_samples}")
    if not args.skip_plots and plt is None:
        raise RuntimeError("matplotlib is required for plot generation.") from MATPLOTLIB_IMPORT_ERROR

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_bank_path = Path(args.trajectory_bank_path).resolve()
    prompt_summary_path = Path(args.prompt_summary_path).resolve() if args.prompt_summary_path else None
    summary_metrics_path = Path(args.summary_metrics_path).resolve() if args.summary_metrics_path else None

    grouped_rows = load_trajectory_bank(trajectory_bank_path)
    bank_size = infer_bank_size(grouped_rows, args.bank_size)

    stage2_summary = None
    if summary_metrics_path is not None:
        stage2_summary = json.load(summary_metrics_path.open("r", encoding="utf-8"))
        if args.seed is None:
            args.seed = int(stage2_summary["run_args"]["seed"])
    if args.seed is None:
        args.seed = 0

    lambdas = list(dict.fromkeys(float(value) for value in args.lambdas))
    if args.include_negative_lambdas:
        lambdas = list(dict.fromkeys(list(OPTIONAL_NEGATIVE_LAMBDAS) + lambdas))
    lambdas = sorted(lambdas)

    prompt_rows: list[dict[str, Any]] = []
    for example_id in sorted(grouped_rows):
        prompt_rows.append(
            build_prompt_summary(
                example_id=example_id,
                bank_rows=grouped_rows[example_id],
                bank_size=bank_size,
                base_seed=args.seed,
                normalization=args.normalization,
                eps=args.eps,
                lambdas=lambdas,
            )
        )

    binary_task_scores = set(
        float(row["task_score"])
        for rows in grouped_rows.values()
        for row in rows[:bank_size]
    ).issubset({0.0, 1.0})

    baseline_metrics, hybrid_metrics, comparisons, disagreement_analysis = aggregate_metrics(
        prompt_rows=prompt_rows,
        lambdas=lambdas,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed + 90_001,
        binary_task_scores=binary_task_scores,
    )

    prompt_validation = validate_against_stage2_prompt_summary(
        prompt_rows=prompt_rows,
        stage2_prompt_summary_path=prompt_summary_path,
        stage2_bank_size=bank_size,
    )
    aggregate_validation = validate_against_stage2_summary(
        baseline_metrics=baseline_metrics,
        stage2_summary_path=summary_metrics_path,
        stage2_bank_size=bank_size,
    )

    prompt_output_path = output_dir / "hybrid_prompt_level_summary.jsonl"
    summary_output_path = output_dir / "hybrid_summary_metrics.json"
    main_results_path = output_dir / "hybrid_main_results.csv"
    comparisons_path = output_dir / "hybrid_comparisons.json"
    disagreement_path = output_dir / "hybrid_disagreement_analysis.json"
    accuracy_plot_path = output_dir / "accuracy_vs_lambda.png"
    recovery_plot_path = output_dir / "conditional_recovery_vs_lambda.png"
    length_plot_path = output_dir / "mean_selected_response_length_vs_lambda.png"
    changed_plot_path = output_dir / "fraction_changed_vs_lambda.png"

    with prompt_output_path.open("w", encoding="utf-8") as prompt_file:
        for prompt_row in prompt_rows:
            prompt_file.write(_json_line(prompt_row))

    flat_rows = list(baseline_metrics.values()) + [hybrid_metrics[_lambda_method_name(value)] for value in lambdas]
    csv_fieldnames = [
        "method",
        "method_group",
        "lambda",
        "num_prompts",
        "num_bank_success_prompts",
        "selected_mean_task_score",
        "conditional_success_recovery_rate",
        "top1_hit_rate_against_oracle_best",
        "mean_selected_response_length",
        "mean_selected_hybrid_score",
        "mean_selected_new_critic_value",
        "mean_selected_actor_logprob",
        "mean_selected_normalized_new_critic_value",
        "mean_selected_normalized_actor_logprob",
        "fraction_prompts_changed_vs_best_of_n_new_critic",
        "selection_score_field",
    ]
    with main_results_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)

    with comparisons_path.open("w", encoding="utf-8") as comparisons_file:
        json.dump(comparisons, comparisons_file, ensure_ascii=True, indent=2)

    with disagreement_path.open("w", encoding="utf-8") as disagreement_file:
        json.dump(disagreement_analysis, disagreement_file, ensure_ascii=True, indent=2)

    if not args.skip_plots:
        hybrid_curve_methods = [_lambda_method_name(value) for value in lambdas]
        x_values = [float(value) for value in lambdas]
        _plot_hybrid_with_baselines(
            x_values=x_values,
            y_values=[float(hybrid_metrics[method_name]["selected_mean_task_score"]) for method_name in hybrid_curve_methods],
            baselines={
                "Pure new critic": baseline_metrics["best_of_n_new_critic"]["selected_mean_task_score"],
                "Actor log-prob": baseline_metrics["best_of_n_actor_logprob"]["selected_mean_task_score"],
                "Old critic": baseline_metrics["best_of_n_old_critic"]["selected_mean_task_score"],
                "Oracle": baseline_metrics["oracle_best_in_bank"]["selected_mean_task_score"],
            },
            ylabel="Selected Mean Task Score",
            title="Accuracy vs lambda",
            output_path=accuracy_plot_path,
            dpi=args.plot_dpi,
        )
        _plot_hybrid_with_baselines(
            x_values=x_values,
            y_values=[
                float(hybrid_metrics[method_name]["conditional_success_recovery_rate"]) for method_name in hybrid_curve_methods
            ],
            baselines={
                "Pure new critic": baseline_metrics["best_of_n_new_critic"]["conditional_success_recovery_rate"],
                "Actor log-prob": baseline_metrics["best_of_n_actor_logprob"]["conditional_success_recovery_rate"],
                "Old critic": baseline_metrics["best_of_n_old_critic"]["conditional_success_recovery_rate"],
                "Oracle": baseline_metrics["oracle_best_in_bank"]["conditional_success_recovery_rate"],
            },
            ylabel="Conditional Success Recovery Rate",
            title="Conditional recovery vs lambda",
            output_path=recovery_plot_path,
            dpi=args.plot_dpi,
        )
        _plot_hybrid_with_baselines(
            x_values=x_values,
            y_values=[float(hybrid_metrics[method_name]["mean_selected_response_length"]) for method_name in hybrid_curve_methods],
            baselines={
                "Pure new critic": baseline_metrics["best_of_n_new_critic"]["mean_selected_response_length"],
                "Actor log-prob": baseline_metrics["best_of_n_actor_logprob"]["mean_selected_response_length"],
                "Old critic": baseline_metrics["best_of_n_old_critic"]["mean_selected_response_length"],
                "Oracle": baseline_metrics["oracle_best_in_bank"]["mean_selected_response_length"],
            },
            ylabel="Mean Selected Response Length",
            title="Mean selected response length vs lambda",
            output_path=length_plot_path,
            dpi=args.plot_dpi,
        )
        _plot_lambda_curve(
            x_values=x_values,
            y_values=[
                float(hybrid_metrics[method_name]["fraction_prompts_changed_vs_best_of_n_new_critic"])
                for method_name in hybrid_curve_methods
            ],
            ylabel="Fraction Changed vs Pure New Critic",
            title="Selection disagreement vs lambda",
            output_path=changed_plot_path,
            dpi=args.plot_dpi,
        )

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "trajectory_bank_path": str(trajectory_bank_path),
        "stage2_prompt_summary_path": None if prompt_summary_path is None else str(prompt_summary_path),
        "stage2_summary_metrics_path": None if summary_metrics_path is None else str(summary_metrics_path),
        "output_dir": str(output_dir),
        "bank_size": int(bank_size),
        "num_prompts": len(prompt_rows),
        "num_trajectories": len(prompt_rows) * bank_size,
        "binary_task_scores": binary_task_scores,
        "normalization": args.normalization,
        "normalization_eps": float(args.eps),
        "lambdas": [float(value) for value in lambdas],
        "base_seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "baseline_recomputation_validation": {
            "prompt_level_validation": prompt_validation,
            "aggregate_validation": aggregate_validation,
        },
        "baseline_metrics": baseline_metrics,
        "hybrid_metrics": hybrid_metrics,
        "comparisons": comparisons,
        "disagreement_analysis": disagreement_analysis,
        "plots": {
            "accuracy_vs_lambda": None if args.skip_plots else str(accuracy_plot_path),
            "conditional_recovery_vs_lambda": None if args.skip_plots else str(recovery_plot_path),
            "mean_selected_response_length_vs_lambda": None if args.skip_plots else str(length_plot_path),
            "fraction_changed_vs_lambda": None if args.skip_plots else str(changed_plot_path),
        },
        "run_args": vars(args),
    }
    with summary_output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    _write_output_readme(
        output_dir=output_dir,
        trajectory_bank_path=trajectory_bank_path,
        bank_size=bank_size,
        normalization=args.normalization,
        eps=args.eps,
        lambdas=lambdas,
        baseline_metrics=baseline_metrics,
        hybrid_metrics=hybrid_metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
