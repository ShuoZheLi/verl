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

from value_decoding.hybrid_bank_rerank_analysis import (
    BASELINE_METHODS,
    DEFAULT_EPS,
    DEFAULT_NORMALIZATION,
    _argmax_indices,
    _bootstrap_metric_difference,
    _build_baseline_method_record,
    _git_commit,
    _json_line,
    _mean,
    _normalize_within_prompt,
    _plot_hybrid_with_baselines,
    _plot_lambda_curve,
    _random_selector_seed,
    infer_bank_size,
    load_trajectory_bank,
    validate_against_stage2_prompt_summary,
    validate_against_stage2_summary,
)


DEFAULT_TAUS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
DEFAULT_LOCAL_LAMBDAS = (0.1, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc margin-gated reranking analysis over an existing Stage 2 response bank. "
            "No new actor generation or critic scoring is performed."
        )
    )
    parser.add_argument("--trajectory_bank_path", type=str, required=True, help="Existing Stage 2 trajectory_bank.jsonl path.")
    parser.add_argument(
        "--prompt_summary_path",
        type=str,
        default=None,
        help="Optional existing Stage 2 prompt_level_summary.jsonl for baseline validation.",
    )
    parser.add_argument(
        "--summary_metrics_path",
        type=str,
        default=None,
        help="Optional existing Stage 2 summary_metrics.json for aggregate baseline validation and metadata.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for the gated analysis artifacts.")
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
        help="Within-prompt normalization used for the critic margin and optional local hybrid.",
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Numerical epsilon for normalization.")
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=list(DEFAULT_TAUS),
        help="Margin thresholds for the gated selectors.",
    )
    parser.add_argument(
        "--local_hybrid_lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LOCAL_LAMBDAS),
        help="Lambda values for the gated local-hybrid family.",
    )
    parser.add_argument(
        "--skip_local_hybrid_family",
        action="store_true",
        help="Skip the gated top-2 local-hybrid family.",
    )
    parser.add_argument(
        "--topk_actor_tiebreak_ks",
        nargs="+",
        type=int,
        default=[],
        help="Optional additional top-K actor tiebreak candidate sizes, e.g. 3 4.",
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


def _format_value(value: float) -> str:
    return f"{value:g}"


def _primary_method_name(tau: float) -> str:
    return f"gated_top2_actor_tiebreak__tau_{_format_value(tau)}"


def _local_hybrid_method_name(tau: float, lambda_value: float) -> str:
    return f"gated_top2_local_hybrid__tau_{_format_value(tau)}__lambda_{_format_value(lambda_value)}"


def _topk_actor_method_name(top_k: int, tau: float) -> str:
    return f"gated_top{top_k}_actor_tiebreak__tau_{_format_value(tau)}"


def _rank_positions_by_normalized_new(
    normalized_new_values: Sequence[float],
    bank_rows: Sequence[dict[str, Any]],
) -> list[int]:
    return sorted(
        range(len(bank_rows)),
        key=lambda index: (-float(normalized_new_values[index]), int(bank_rows[index]["sample_idx"])),
    )


def _select_over_candidates(
    *,
    candidate_positions: Sequence[int],
    values: Sequence[float],
) -> tuple[int, list[int], float]:
    candidate_values = [float(values[position]) for position in candidate_positions]
    tied_candidate_offsets = _argmax_indices(candidate_values)
    selected_candidate_offset = tied_candidate_offsets[0]
    selected_position = int(candidate_positions[selected_candidate_offset])
    tied_positions = [int(candidate_positions[offset]) for offset in tied_candidate_offsets]
    return selected_position, tied_positions, float(candidate_values[selected_candidate_offset])


def _build_gated_method_record(
    *,
    method_name: str,
    method_family: str,
    tau: float,
    lambda_value: float | None,
    top_k: int,
    gate_on: bool,
    margin: float,
    candidate_positions: Sequence[int],
    selected_position: int,
    tied_positions: Sequence[int],
    selection_score: float,
    selection_score_field: str,
    bank_rows: Sequence[dict[str, Any]],
    oracle_best_sample_indices: Sequence[int],
    normalized_new_values: Sequence[float],
    normalized_actor_logprobs: Sequence[float],
) -> dict[str, Any]:
    selected_row = bank_rows[selected_position]
    selected_sample_idx = int(selected_row["sample_idx"])
    return {
        "method": method_name,
        "method_family": method_family,
        "tau": float(tau),
        "lambda": None if lambda_value is None else float(lambda_value),
        "top_k": int(top_k),
        "gate_on": bool(gate_on),
        "normalized_new_margin_top1_vs_top2": float(margin),
        "gate_candidate_indices": [int(bank_rows[position]["sample_idx"]) for position in candidate_positions],
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
        "selected_normalized_new_critic_value": float(normalized_new_values[selected_position]),
        "selected_normalized_actor_logprob": float(normalized_actor_logprobs[selected_position]),
        "selection_score_field": selection_score_field,
        "selected_selection_score": float(selection_score),
    }


def build_prompt_summary(
    *,
    example_id: int,
    bank_rows: Sequence[dict[str, Any]],
    bank_size: int,
    base_seed: int,
    normalization: str,
    eps: float,
    taus: Sequence[float],
    local_hybrid_lambdas: Sequence[float],
    include_local_hybrid_family: bool,
    topk_actor_tiebreak_ks: Sequence[int],
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

    normalized_new_values = _normalize_within_prompt(new_values, normalization=normalization, eps=eps)
    normalized_actor_logprobs = _normalize_within_prompt(actor_logprobs, normalization=normalization, eps=eps)
    sorted_positions = _rank_positions_by_normalized_new(normalized_new_values, bank_rows)
    if len(sorted_positions) < 2:
        raise ValueError(f"Prompt {example_id} needs at least two bank items for margin-gated analysis.")

    top1_position = int(sorted_positions[0])
    top2_position = int(sorted_positions[1])
    margin = float(normalized_new_values[top1_position] - normalized_new_values[top2_position])

    oracle_best_positions = _argmax_indices(task_scores)
    oracle_best_sample_indices = [int(bank_rows[position]["sample_idx"]) for position in oracle_best_positions]
    oracle_best_task_score = float(bank_rows[oracle_best_positions[0]]["task_score"])

    baseline_methods: dict[str, Any] = {}
    random_rng = random.Random(_random_selector_seed(base_seed, example_id=example_id, bank_size=bank_size))
    random_position = int(random_rng.randrange(len(bank_rows)))
    baseline_methods["random_single_sample"] = _build_baseline_method_record(
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
        selected_position, tied_positions, selection_score = _select_over_candidates(
            candidate_positions=list(range(len(bank_rows))),
            values=values,
        )
        baseline_methods[method_name] = _build_baseline_method_record(
            method_name=method_name,
            selected_row=bank_rows[selected_position],
            tied_positions=tied_positions,
            bank_rows=bank_rows,
            selection_score_field=selection_score_field,
            selection_score=selection_score,
            oracle_best_sample_indices=oracle_best_sample_indices,
        )

    gated_methods: dict[str, Any] = {}
    top2_positions = [top1_position, top2_position]
    for tau in taus:
        gate_on = bool(margin <= float(tau))

        if gate_on:
            selected_position, tied_positions, selection_score = _select_over_candidates(
                candidate_positions=top2_positions,
                values=actor_logprobs,
            )
        else:
            selected_position = top1_position
            tied_positions = [top1_position]
            selection_score = float(normalized_new_values[top1_position])
        method_name = _primary_method_name(tau)
        gated_methods[method_name] = _build_gated_method_record(
            method_name=method_name,
            method_family="gated_top2_actor_tiebreak",
            tau=float(tau),
            lambda_value=None,
            top_k=2,
            gate_on=gate_on,
            margin=margin,
            candidate_positions=top2_positions,
            selected_position=selected_position,
            tied_positions=tied_positions,
            selection_score=selection_score,
            selection_score_field=(
                "actor_response_logprob_within_top2_if_margin_le_tau_else_normalized_new_critic"
            ),
            bank_rows=bank_rows,
            oracle_best_sample_indices=oracle_best_sample_indices,
            normalized_new_values=normalized_new_values,
            normalized_actor_logprobs=normalized_actor_logprobs,
        )

        if include_local_hybrid_family:
            for lambda_value in local_hybrid_lambdas:
                if gate_on:
                    local_hybrid_scores = [
                        float(normalized_new_values[position] + float(lambda_value) * normalized_actor_logprobs[position])
                        for position in top2_positions
                    ]
                    selected_local_offset, tied_local_offsets, selection_score = _select_over_candidates(
                        candidate_positions=list(range(len(top2_positions))),
                        values=local_hybrid_scores,
                    )
                    selected_position = top2_positions[selected_local_offset]
                    tied_positions = [top2_positions[offset] for offset in tied_local_offsets]
                else:
                    selected_position = top1_position
                    tied_positions = [top1_position]
                    selection_score = float(normalized_new_values[top1_position])
                method_name = _local_hybrid_method_name(tau, lambda_value)
                gated_methods[method_name] = _build_gated_method_record(
                    method_name=method_name,
                    method_family="gated_top2_local_hybrid",
                    tau=float(tau),
                    lambda_value=float(lambda_value),
                    top_k=2,
                    gate_on=gate_on,
                    margin=margin,
                    candidate_positions=top2_positions,
                    selected_position=selected_position,
                    tied_positions=tied_positions,
                    selection_score=selection_score,
                    selection_score_field=(
                        "normalized_new_critic_plus_lambda_times_normalized_actor_logprob_within_top2_if_margin_le_tau_else_normalized_new_critic"
                    ),
                    bank_rows=bank_rows,
                    oracle_best_sample_indices=oracle_best_sample_indices,
                    normalized_new_values=normalized_new_values,
                    normalized_actor_logprobs=normalized_actor_logprobs,
                )

        for top_k in topk_actor_tiebreak_ks:
            candidate_positions = sorted_positions[:top_k]
            if gate_on:
                selected_position, tied_positions, selection_score = _select_over_candidates(
                    candidate_positions=candidate_positions,
                    values=actor_logprobs,
                )
            else:
                selected_position = top1_position
                tied_positions = [top1_position]
                selection_score = float(normalized_new_values[top1_position])
            method_name = _topk_actor_method_name(top_k, tau)
            gated_methods[method_name] = _build_gated_method_record(
                method_name=method_name,
                method_family=f"gated_top{top_k}_actor_tiebreak",
                tau=float(tau),
                lambda_value=None,
                top_k=int(top_k),
                gate_on=gate_on,
                margin=margin,
                candidate_positions=candidate_positions,
                selected_position=selected_position,
                tied_positions=tied_positions,
                selection_score=selection_score,
                selection_score_field=(
                    f"actor_response_logprob_within_top{top_k}_if_margin_le_tau_else_normalized_new_critic"
                ),
                bank_rows=bank_rows,
                oracle_best_sample_indices=oracle_best_sample_indices,
                normalized_new_values=normalized_new_values,
                normalized_actor_logprobs=normalized_actor_logprobs,
            )

    return {
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
        "normalized_new_critic_values": normalized_new_values,
        "normalized_actor_logprobs": normalized_actor_logprobs,
        "normalization": normalization,
        "normalization_eps": float(eps),
        "normalized_new_critic_sorted_indices": [int(bank_rows[position]["sample_idx"]) for position in sorted_positions],
        "normalized_new_critic_top1_index": int(bank_rows[top1_position]["sample_idx"]),
        "normalized_new_critic_top2_index": int(bank_rows[top2_position]["sample_idx"]),
        "normalized_new_critic_top1_top2_margin": float(margin),
        "oracle_best_index": int(bank_rows[oracle_best_positions[0]]["sample_idx"]),
        "oracle_best_indices": oracle_best_sample_indices,
        "oracle_best_task_score": oracle_best_task_score,
        "baseline_methods": baseline_methods,
        "gated_methods": gated_methods,
    }


def _method_record(prompt_row: dict[str, Any], *, method_name: str, method_group: str) -> dict[str, Any]:
    return prompt_row[method_group][method_name]


def _subset_accuracy(prompt_rows: Sequence[dict[str, Any]], *, method_name: str, method_group: str) -> float | None:
    if not prompt_rows:
        return None
    return _mean([float(_method_record(prompt_row, method_name=method_name, method_group=method_group)["selected_task_score"]) for prompt_row in prompt_rows])


def aggregate_metrics(
    *,
    prompt_rows: Sequence[dict[str, Any]],
    taus: Sequence[float],
    local_hybrid_lambdas: Sequence[float],
    include_local_hybrid_family: bool,
    topk_actor_tiebreak_ks: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    binary_task_scores: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    num_bank_success_prompts = int(sum(1 for prompt_row in prompt_rows if float(prompt_row["oracle_best_task_score"]) == 1.0))

    baseline_metrics: dict[str, Any] = {}
    for method_name in BASELINE_METHODS:
        selected_scores = [float(prompt_row["baseline_methods"][method_name]["selected_task_score"]) for prompt_row in prompt_rows]
        selected_lengths = [int(prompt_row["baseline_methods"][method_name]["selected_response_length"]) for prompt_row in prompt_rows]
        oracle_hits = [1.0 if bool(prompt_row["baseline_methods"][method_name]["selected_is_oracle_best"]) else 0.0 for prompt_row in prompt_rows]
        baseline_metrics[method_name] = {
            "method": method_name,
            "method_group": "baseline_methods",
            "method_family": "baseline",
            "tau": None,
            "lambda": None,
            "top_k": None,
            "num_prompts": int(len(prompt_rows)),
            "num_bank_success_prompts": num_bank_success_prompts,
            "selected_mean_task_score": _mean(selected_scores),
            "conditional_success_recovery_rate": (
                _mean(
                    [
                        1.0 if bool(prompt_row["baseline_methods"][method_name]["selected_is_correct"]) else 0.0
                        for prompt_row in prompt_rows
                        if float(prompt_row["oracle_best_task_score"]) == 1.0
                    ]
                )
                if binary_task_scores and num_bank_success_prompts > 0
                else None
            ),
            "top1_hit_rate_against_oracle_best": _mean(oracle_hits),
            "mean_selected_response_length": _mean(selected_lengths),
            "fraction_gated_prompts": None,
            "num_gated_prompts": None,
            "gated_subset_accuracy": None,
            "pure_new_critic_accuracy_on_same_gated_subset": None,
            "ungated_subset_accuracy": None,
            "gated_subset_accuracy_delta_vs_pure_new_critic": None,
            "mean_selected_new_critic_value": _mean(
                [
                    float(prompt_row["baseline_methods"][method_name]["selected_new_critic_final_trajectory_value"])
                    for prompt_row in prompt_rows
                ]
            ),
            "mean_selected_actor_logprob": _mean(
                [float(prompt_row["baseline_methods"][method_name]["selected_actor_response_logprob"]) for prompt_row in prompt_rows]
            ),
            "mean_selected_local_hybrid_score": None,
            "selection_score_field": prompt_rows[0]["baseline_methods"][method_name]["selection_score_field"],
        }

    gated_method_specs: list[tuple[str, str, float, float | None, int]] = []
    for tau in taus:
        gated_method_specs.append((_primary_method_name(tau), "gated_top2_actor_tiebreak", float(tau), None, 2))
        if include_local_hybrid_family:
            for lambda_value in local_hybrid_lambdas:
                gated_method_specs.append(
                    (_local_hybrid_method_name(tau, lambda_value), "gated_top2_local_hybrid", float(tau), float(lambda_value), 2)
                )
        for top_k in topk_actor_tiebreak_ks:
            gated_method_specs.append((_topk_actor_method_name(top_k, tau), f"gated_top{top_k}_actor_tiebreak", float(tau), None, int(top_k)))

    gated_metrics: dict[str, Any] = {}
    for method_name, method_family, tau, lambda_value, top_k in gated_method_specs:
        gated_rows = [prompt_row for prompt_row in prompt_rows if bool(prompt_row["gated_methods"][method_name]["gate_on"])]
        ungated_rows = [prompt_row for prompt_row in prompt_rows if not bool(prompt_row["gated_methods"][method_name]["gate_on"])]
        gated_accuracy = _subset_accuracy(gated_rows, method_name=method_name, method_group="gated_methods")
        pure_new_on_gated = _subset_accuracy(gated_rows, method_name="best_of_n_new_critic", method_group="baseline_methods")

        gated_metrics[method_name] = {
            "method": method_name,
            "method_group": "gated_methods",
            "method_family": method_family,
            "tau": tau,
            "lambda": lambda_value,
            "top_k": top_k,
            "num_prompts": int(len(prompt_rows)),
            "num_bank_success_prompts": num_bank_success_prompts,
            "selected_mean_task_score": _mean(
                [float(prompt_row["gated_methods"][method_name]["selected_task_score"]) for prompt_row in prompt_rows]
            ),
            "conditional_success_recovery_rate": _mean(
                [
                    1.0 if bool(prompt_row["gated_methods"][method_name]["selected_is_correct"]) else 0.0
                    for prompt_row in prompt_rows
                    if float(prompt_row["oracle_best_task_score"]) == 1.0
                ]
            )
            if binary_task_scores and num_bank_success_prompts > 0
            else None,
            "top1_hit_rate_against_oracle_best": _mean(
                [
                    1.0 if bool(prompt_row["gated_methods"][method_name]["selected_is_oracle_best"]) else 0.0
                    for prompt_row in prompt_rows
                ]
            ),
            "mean_selected_response_length": _mean(
                [int(prompt_row["gated_methods"][method_name]["selected_response_length"]) for prompt_row in prompt_rows]
            ),
            "num_gated_prompts": int(len(gated_rows)),
            "fraction_gated_prompts": (len(gated_rows) / len(prompt_rows)) if prompt_rows else None,
            "gated_subset_accuracy": gated_accuracy,
            "pure_new_critic_accuracy_on_same_gated_subset": pure_new_on_gated,
            "ungated_subset_accuracy": _subset_accuracy(
                ungated_rows,
                method_name=method_name,
                method_group="gated_methods",
            ),
            "gated_subset_accuracy_delta_vs_pure_new_critic": (
                float(gated_accuracy - pure_new_on_gated)
                if gated_accuracy is not None and pure_new_on_gated is not None
                else None
            ),
            "mean_selected_new_critic_value": _mean(
                [
                    float(prompt_row["gated_methods"][method_name]["selected_new_critic_final_trajectory_value"])
                    for prompt_row in prompt_rows
                ]
            ),
            "mean_selected_actor_logprob": _mean(
                [float(prompt_row["gated_methods"][method_name]["selected_actor_response_logprob"]) for prompt_row in prompt_rows]
            ),
            "mean_selected_local_hybrid_score": _mean(
                [float(prompt_row["gated_methods"][method_name]["selected_selection_score"]) for prompt_row in prompt_rows]
            )
            if method_family == "gated_top2_local_hybrid"
            else None,
            "selection_score_field": prompt_rows[0]["gated_methods"][method_name]["selection_score_field"],
        }

    comparison_specs = (
        ("best_of_n_new_critic", "baseline_methods"),
        ("best_of_n_actor_logprob", "baseline_methods"),
        ("best_of_n_old_critic", "baseline_methods"),
        ("random_single_sample", "baseline_methods"),
        ("oracle_best_in_bank", "baseline_methods"),
    )
    comparisons: dict[str, Any] = {}
    for method_name, _method_family, tau, lambda_value, top_k in gated_method_specs:
        method_comparisons: dict[str, Any] = {}
        for baseline_method_name, baseline_group in comparison_specs:
            comparison_name = f"{method_name}_minus_{baseline_method_name}"
            method_comparisons[comparison_name] = {
                "method_a": method_name,
                "method_a_group": "gated_methods",
                "method_b": baseline_method_name,
                "method_b_group": baseline_group,
                "tau": tau,
                "lambda": lambda_value,
                "top_k": top_k,
                "selected_mean_task_score_difference": (
                    float(gated_metrics[method_name]["selected_mean_task_score"])
                    - float(baseline_metrics[baseline_method_name]["selected_mean_task_score"])
                ),
                "conditional_success_recovery_rate_difference": (
                    float(gated_metrics[method_name]["conditional_success_recovery_rate"])
                    - float(baseline_metrics[baseline_method_name]["conditional_success_recovery_rate"])
                    if gated_metrics[method_name]["conditional_success_recovery_rate"] is not None
                    and baseline_metrics[baseline_method_name]["conditional_success_recovery_rate"] is not None
                    else None
                ),
                "top1_hit_rate_against_oracle_best_difference": (
                    float(gated_metrics[method_name]["top1_hit_rate_against_oracle_best"])
                    - float(baseline_metrics[baseline_method_name]["top1_hit_rate_against_oracle_best"])
                ),
                "mean_selected_response_length_difference": (
                    float(gated_metrics[method_name]["mean_selected_response_length"])
                    - float(baseline_metrics[baseline_method_name]["mean_selected_response_length"])
                ),
                "paired_bootstrap": {
                    "selected_mean_task_score_difference": _bootstrap_metric_difference(
                        prompt_rows,
                        method_a=method_name,
                        method_a_group="gated_methods",
                        method_b=baseline_method_name,
                        method_b_group=baseline_group,
                        metric_name="selected_mean_task_score",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + int(round((tau + 10.0) * 1000)) + int(top_k * 100) + 1,
                        binary_task_scores=binary_task_scores,
                    ),
                    "conditional_success_recovery_rate_difference": _bootstrap_metric_difference(
                        prompt_rows,
                        method_a=method_name,
                        method_a_group="gated_methods",
                        method_b=baseline_method_name,
                        method_b_group=baseline_group,
                        metric_name="conditional_success_recovery_rate",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + int(round((tau + 10.0) * 1000)) + int(top_k * 100) + 2,
                        binary_task_scores=binary_task_scores,
                    ),
                    "top1_hit_rate_against_oracle_best_difference": _bootstrap_metric_difference(
                        prompt_rows,
                        method_a=method_name,
                        method_a_group="gated_methods",
                        method_b=baseline_method_name,
                        method_b_group=baseline_group,
                        metric_name="top1_hit_rate_against_oracle_best",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + int(round((tau + 10.0) * 1000)) + int(top_k * 100) + 3,
                        binary_task_scores=binary_task_scores,
                    ),
                },
            }
        comparisons[method_name] = method_comparisons

    return baseline_metrics, gated_metrics, comparisons


def _plot_primary_family_with_baselines(
    *,
    gated_metrics: dict[str, Any],
    taus: Sequence[float],
    baseline_metrics: dict[str, Any],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    primary_method_names = [_primary_method_name(tau) for tau in taus]
    _plot_hybrid_with_baselines(
        x_values=[float(tau) for tau in taus],
        y_values=[float(gated_metrics[method_name][metric_name]) for method_name in primary_method_names],
        baselines={
            "Pure new critic": baseline_metrics["best_of_n_new_critic"][metric_name],
            "Actor log-prob": baseline_metrics["best_of_n_actor_logprob"][metric_name],
            "Old critic": baseline_metrics["best_of_n_old_critic"][metric_name],
            "Oracle": baseline_metrics["oracle_best_in_bank"][metric_name],
        },
        ylabel=ylabel,
        title=title,
        output_path=output_path,
        dpi=dpi,
    )


def _write_output_readme(
    *,
    output_dir: Path,
    trajectory_bank_path: Path,
    bank_size: int,
    normalization: str,
    eps: float,
    taus: Sequence[float],
    local_hybrid_lambdas: Sequence[float],
    include_local_hybrid_family: bool,
    topk_actor_tiebreak_ks: Sequence[int],
    baseline_metrics: dict[str, Any],
    gated_metrics: dict[str, Any],
) -> None:
    primary_method_names = [_primary_method_name(tau) for tau in taus]
    best_primary_method = max(primary_method_names, key=lambda method_name: float(gated_metrics[method_name]["selected_mean_task_score"]))
    lines = [
        "# Margin-Gated Hybrid Reranking on Existing Stage 2 Response Bank",
        "",
        "This is a post-hoc analysis only:",
        f"- Input trajectory bank: `{trajectory_bank_path}`",
        f"- Bank size reused per prompt: `{bank_size}`",
        "- No new actor generation was run.",
        "- No critic rescoring was run.",
        "- All results are derived from the saved Stage 2 bank.",
        "",
        "## Gate Definition",
        "- Compute normalized new-critic values within each prompt.",
        "- Let the top-1 and top-2 normalized-new-critic items be i1 and i2.",
        "- Define margin = normalized_new_critic(i1) - normalized_new_critic(i2).",
        "- If margin > tau, keep the pure new-critic top-1 decision.",
        "- If margin <= tau, activate the gated tie-break rule.",
        "",
        "## Configuration",
        f"- Normalization: `{normalization}`",
        f"- Normalization epsilon: `{eps}`",
        f"- Taus: `{list(taus)}`",
        f"- Include local hybrid family: `{include_local_hybrid_family}`",
        f"- Local hybrid lambdas: `{list(local_hybrid_lambdas)}`",
        f"- Optional top-K actor-tiebreak Ks: `{list(topk_actor_tiebreak_ks)}`",
        "",
        "## Quick Read",
        f"- Pure new critic selected mean task score: `{baseline_metrics['best_of_n_new_critic']['selected_mean_task_score']:.6f}`",
        f"- Best primary gated method: `{best_primary_method}`",
        f"- Best primary selected mean task score: `{gated_metrics[best_primary_method]['selected_mean_task_score']:.6f}`",
        f"- Best primary fraction gated prompts: `{gated_metrics[best_primary_method]['fraction_gated_prompts']:.6f}`",
        "",
        "## Files",
        "- `gated_hybrid_prompt_level_summary.jsonl`",
        "- `gated_hybrid_summary_metrics.json`",
        "- `gated_hybrid_main_results.csv`",
        "- `README.md`",
        "- `accuracy_vs_tau.png`",
        "- `conditional_recovery_vs_tau.png`",
        "- `fraction_gated_prompts_vs_tau.png`",
        "- `gated_subset_accuracy_delta_vs_tau.png`",
        "- `mean_selected_response_length_vs_tau.png`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError(f"--bootstrap_samples must be > 0, got {args.bootstrap_samples}")

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

    taus = sorted(dict.fromkeys(float(value) for value in args.taus))
    local_hybrid_lambdas = sorted(dict.fromkeys(float(value) for value in args.local_hybrid_lambdas))
    include_local_hybrid_family = not args.skip_local_hybrid_family
    topk_actor_tiebreak_ks = sorted(dict.fromkeys(int(value) for value in args.topk_actor_tiebreak_ks))
    invalid_ks = [value for value in topk_actor_tiebreak_ks if value < 2 or value > bank_size]
    if invalid_ks:
        raise ValueError(f"--topk_actor_tiebreak_ks must be between 2 and bank_size={bank_size}, got {invalid_ks}")

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
                taus=taus,
                local_hybrid_lambdas=local_hybrid_lambdas,
                include_local_hybrid_family=include_local_hybrid_family,
                topk_actor_tiebreak_ks=topk_actor_tiebreak_ks,
            )
        )

    binary_task_scores = set(
        float(row["task_score"])
        for rows in grouped_rows.values()
        for row in rows[:bank_size]
    ).issubset({0.0, 1.0})

    baseline_metrics, gated_metrics, comparisons = aggregate_metrics(
        prompt_rows=prompt_rows,
        taus=taus,
        local_hybrid_lambdas=local_hybrid_lambdas,
        include_local_hybrid_family=include_local_hybrid_family,
        topk_actor_tiebreak_ks=topk_actor_tiebreak_ks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed + 95_001,
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

    prompt_output_path = output_dir / "gated_hybrid_prompt_level_summary.jsonl"
    summary_output_path = output_dir / "gated_hybrid_summary_metrics.json"
    main_results_path = output_dir / "gated_hybrid_main_results.csv"
    accuracy_plot_path = output_dir / "accuracy_vs_tau.png"
    recovery_plot_path = output_dir / "conditional_recovery_vs_tau.png"
    fraction_gated_plot_path = output_dir / "fraction_gated_prompts_vs_tau.png"
    gated_delta_plot_path = output_dir / "gated_subset_accuracy_delta_vs_tau.png"
    length_plot_path = output_dir / "mean_selected_response_length_vs_tau.png"

    with prompt_output_path.open("w", encoding="utf-8") as prompt_file:
        for prompt_row in prompt_rows:
            prompt_file.write(_json_line(prompt_row))

    flat_rows = list(baseline_metrics.values()) + list(gated_metrics.values())
    csv_fieldnames = [
        "method",
        "method_group",
        "method_family",
        "tau",
        "lambda",
        "top_k",
        "num_prompts",
        "num_bank_success_prompts",
        "selected_mean_task_score",
        "conditional_success_recovery_rate",
        "top1_hit_rate_against_oracle_best",
        "mean_selected_response_length",
        "num_gated_prompts",
        "fraction_gated_prompts",
        "gated_subset_accuracy",
        "pure_new_critic_accuracy_on_same_gated_subset",
        "ungated_subset_accuracy",
        "gated_subset_accuracy_delta_vs_pure_new_critic",
        "mean_selected_new_critic_value",
        "mean_selected_actor_logprob",
        "mean_selected_local_hybrid_score",
        "selection_score_field",
    ]
    with main_results_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)

    primary_method_names = [_primary_method_name(tau) for tau in taus]
    if not args.skip_plots:
        _plot_primary_family_with_baselines(
            gated_metrics=gated_metrics,
            taus=taus,
            baseline_metrics=baseline_metrics,
            metric_name="selected_mean_task_score",
            ylabel="Selected Mean Task Score",
            title="Accuracy vs tau",
            output_path=accuracy_plot_path,
            dpi=args.plot_dpi,
        )
        _plot_primary_family_with_baselines(
            gated_metrics=gated_metrics,
            taus=taus,
            baseline_metrics=baseline_metrics,
            metric_name="conditional_success_recovery_rate",
            ylabel="Conditional Success Recovery Rate",
            title="Conditional recovery vs tau",
            output_path=recovery_plot_path,
            dpi=args.plot_dpi,
        )
        _plot_primary_family_with_baselines(
            gated_metrics=gated_metrics,
            taus=taus,
            baseline_metrics=baseline_metrics,
            metric_name="mean_selected_response_length",
            ylabel="Mean Selected Response Length",
            title="Mean selected response length vs tau",
            output_path=length_plot_path,
            dpi=args.plot_dpi,
        )
        _plot_lambda_curve(
            x_values=[float(tau) for tau in taus],
            y_values=[float(gated_metrics[method_name]["fraction_gated_prompts"]) for method_name in primary_method_names],
            ylabel="Fraction Gated Prompts",
            title="Fraction gated prompts vs tau",
            output_path=fraction_gated_plot_path,
            dpi=args.plot_dpi,
        )
        _plot_lambda_curve(
            x_values=[float(tau) for tau in taus],
            y_values=[
                None if gated_metrics[method_name]["gated_subset_accuracy_delta_vs_pure_new_critic"] is None
                else float(gated_metrics[method_name]["gated_subset_accuracy_delta_vs_pure_new_critic"])
                for method_name in primary_method_names
            ],
            ylabel="Gated-Subset Accuracy Delta vs Pure New Critic",
            title="Gated subset accuracy delta vs tau",
            output_path=gated_delta_plot_path,
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
        "taus": [float(value) for value in taus],
        "include_local_hybrid_family": include_local_hybrid_family,
        "local_hybrid_lambdas": [float(value) for value in local_hybrid_lambdas],
        "topk_actor_tiebreak_ks": [int(value) for value in topk_actor_tiebreak_ks],
        "base_seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "baseline_recomputation_validation": {
            "prompt_level_validation": prompt_validation,
            "aggregate_validation": aggregate_validation,
        },
        "baseline_metrics": baseline_metrics,
        "gated_method_metrics": gated_metrics,
        "comparisons": comparisons,
        "plots": {
            "accuracy_vs_tau": None if args.skip_plots else str(accuracy_plot_path),
            "conditional_recovery_vs_tau": None if args.skip_plots else str(recovery_plot_path),
            "fraction_gated_prompts_vs_tau": None if args.skip_plots else str(fraction_gated_plot_path),
            "gated_subset_accuracy_delta_vs_tau": None if args.skip_plots else str(gated_delta_plot_path),
            "mean_selected_response_length_vs_tau": None if args.skip_plots else str(length_plot_path),
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
        taus=taus,
        local_hybrid_lambdas=local_hybrid_lambdas,
        include_local_hybrid_family=include_local_hybrid_family,
        topk_actor_tiebreak_ks=topk_actor_tiebreak_ks,
        baseline_metrics=baseline_metrics,
        gated_metrics=gated_metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
