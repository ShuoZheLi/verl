from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from value_decoding.checkpointing import (
    ensure_merged_checkpoints,
    load_actor_model,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.data import load_examples
from value_decoding.decoding import (
    ActorSamplingMode,
    CandidateBuilder,
    DecodingMode,
    NormalizationType,
    RunSpec,
    decode_example,
)
from value_decoding.multi_worker import (
    build_worker_assignments,
    parse_worker_pairs,
    run_multi_worker,
    worker_assignments_to_jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run value-guided decoding experiments on VERL checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="VERL checkpoint directory with actor/critic.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for experiment outputs.")
    parser.add_argument("--merged_root", type=str, default=None, help="Optional directory for merged HF weights.")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None, help="Optional response/ground-truth column key.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Shared device for both actor and critic.")
    parser.add_argument("--actor_device", type=str, default=None, help="Optional actor device override.")
    parser.add_argument("--critic_device", type=str, default=None, help="Optional critic device override.")
    parser.add_argument(
        "--worker_pairs",
        nargs="+",
        default=None,
        help=(
            "Optional multi-worker device layout. Each entry should be 'actor_device,critic_device' "
            "or a single device name to reuse for both."
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--disable_actor_cache", action="store_true")
    parser.add_argument("--normalization_eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_full_candidates", action="store_true")

    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            DecodingMode.ACTOR_ONLY.value,
            DecodingMode.CRITIC_ONLY_RERANK.value,
            DecodingMode.ACTOR_CRITIC_RERANK.value,
            DecodingMode.ACTOR_CRITIC_SOFT_RERANK.value,
        ],
        help="Modes to run.",
    )
    parser.add_argument(
        "--candidate_builders",
        nargs="+",
        default=[CandidateBuilder.TOP_K.value],
        choices=[builder.value for builder in CandidateBuilder],
    )
    parser.add_argument("--candidate_sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--betas", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--normalizations",
        nargs="+",
        default=[NormalizationType.ZSCORE.value],
        choices=[normalization.value for normalization in NormalizationType],
    )
    parser.add_argument("--rank_temperatures", nargs="+", type=float, default=[1.0])

    parser.add_argument(
        "--actor_sampling_mode",
        type=str,
        default=ActorSamplingMode.GREEDY.value,
        choices=[mode.value for mode in ActorSamplingMode],
    )
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--actor_top_p", type=float, default=1.0)
    parser.add_argument("--actor_top_k", type=int, default=0)
    return parser.parse_args()


def _normalize_mode_name(name: str) -> str:
    aliases = {
        "critic_only": DecodingMode.CRITIC_ONLY_RERANK.value,
        "actor_critic": DecodingMode.ACTOR_CRITIC_RERANK.value,
        "actor_critic_soft": DecodingMode.ACTOR_CRITIC_SOFT_RERANK.value,
    }
    lowered = name.lower()
    return aliases.get(lowered, lowered)


def _format_float_for_id(value: float) -> str:
    formatted = f"{value:g}"
    formatted = formatted.replace("-", "m").replace(".", "p")
    return formatted


def _build_config_id(spec: RunSpec) -> str:
    parts = [spec.mode]
    if spec.mode == DecodingMode.ACTOR_ONLY.value:
        parts.append(spec.actor_sampling_mode)
        if spec.actor_sampling_mode == ActorSamplingMode.SAMPLE.value:
            parts.append(f"temp{_format_float_for_id(spec.actor_temperature)}")
            parts.append(f"top_p{_format_float_for_id(spec.actor_top_p)}")
            parts.append(f"top_k{spec.actor_top_k}")
        return "__".join(parts)

    if spec.candidate_builder is not None:
        parts.append(spec.candidate_builder)
    if spec.candidate_size is not None:
        parts.append(f"k{spec.candidate_size}")
    if spec.beta is not None:
        parts.append(f"beta{_format_float_for_id(spec.beta)}")
    if spec.normalization != NormalizationType.NONE.value:
        parts.append(f"norm_{spec.normalization}")
    if spec.rank_temperature is not None:
        parts.append(f"tau{_format_float_for_id(spec.rank_temperature)}")
    return "__".join(parts)


def build_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seen_config_ids: set[str] = set()

    for raw_mode in args.modes:
        mode = _normalize_mode_name(raw_mode)
        if mode == DecodingMode.ACTOR_ONLY.value:
            spec = RunSpec(
                config_id="",
                mode=mode,
                normalization=NormalizationType.NONE.value,
                actor_sampling_mode=args.actor_sampling_mode,
                actor_temperature=args.actor_temperature,
                actor_top_p=args.actor_top_p,
                actor_top_k=args.actor_top_k,
            )
            spec = RunSpec(**{**asdict(spec), "config_id": _build_config_id(spec)})
            if spec.config_id not in seen_config_ids:
                specs.append(spec)
                seen_config_ids.add(spec.config_id)
            continue

        if mode == DecodingMode.CRITIC_ONLY_RERANK.value:
            for builder in args.candidate_builders:
                for candidate_size in args.candidate_sizes:
                    spec = RunSpec(
                        config_id="",
                        mode=mode,
                        candidate_builder=builder,
                        candidate_size=candidate_size,
                        normalization=NormalizationType.NONE.value,
                        actor_sampling_mode=args.actor_sampling_mode,
                        actor_temperature=args.actor_temperature,
                        actor_top_p=args.actor_top_p,
                        actor_top_k=args.actor_top_k,
                    )
                    spec = RunSpec(**{**asdict(spec), "config_id": _build_config_id(spec)})
                    if spec.config_id not in seen_config_ids:
                        specs.append(spec)
                        seen_config_ids.add(spec.config_id)
            continue

        if mode == DecodingMode.ACTOR_CRITIC_RERANK.value:
            for builder in args.candidate_builders:
                for candidate_size in args.candidate_sizes:
                    for beta in args.betas:
                        for normalization in args.normalizations:
                            spec = RunSpec(
                                config_id="",
                                mode=mode,
                                candidate_builder=builder,
                                candidate_size=candidate_size,
                                beta=beta,
                                normalization=normalization,
                                actor_sampling_mode=args.actor_sampling_mode,
                                actor_temperature=args.actor_temperature,
                                actor_top_p=args.actor_top_p,
                                actor_top_k=args.actor_top_k,
                            )
                            spec = RunSpec(**{**asdict(spec), "config_id": _build_config_id(spec)})
                            if spec.config_id not in seen_config_ids:
                                specs.append(spec)
                                seen_config_ids.add(spec.config_id)
            continue

        if mode == DecodingMode.ACTOR_CRITIC_SOFT_RERANK.value:
            for builder in args.candidate_builders:
                for candidate_size in args.candidate_sizes:
                    for beta in args.betas:
                        for normalization in args.normalizations:
                            for rank_temperature in args.rank_temperatures:
                                spec = RunSpec(
                                    config_id="",
                                    mode=mode,
                                    candidate_builder=builder,
                                    candidate_size=candidate_size,
                                    beta=beta,
                                    normalization=normalization,
                                    rank_temperature=rank_temperature,
                                    actor_sampling_mode=args.actor_sampling_mode,
                                    actor_temperature=args.actor_temperature,
                                    actor_top_p=args.actor_top_p,
                                    actor_top_k=args.actor_top_k,
                                )
                                spec = RunSpec(**{**asdict(spec), "config_id": _build_config_id(spec)})
                                if spec.config_id not in seen_config_ids:
                                    specs.append(spec)
                                    seen_config_ids.add(spec.config_id)
            continue

        raise ValueError(f"Unsupported mode: {raw_mode}")

    return specs


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.isclose(x_arr.std(), 0.0) or np.isclose(y_arr.std(), 0.0):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _average_ranks(values: list[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_arr, kind="mergesort")
    ranks = np.empty(len(values_arr), dtype=np.float64)

    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values_arr[order[end + 1]] == values_arr[order[start]]:
            end += 1
        avg_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = avg_rank
        start = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_average_ranks(xs).tolist(), _average_ranks(ys).tolist())


def aggregate_results(
    spec: RunSpec,
    example_results: list[dict[str, Any]],
    *,
    wall_time_sec: float | None = None,
) -> dict[str, Any]:
    task_scores = [float(result["task_score"]) for result in example_results]
    response_lengths = [int(result["response_length"]) for result in example_results]
    eos_flags = [bool(result["eos_emitted"]) for result in example_results]
    max_length_flags = [bool(result["max_length_hit"]) for result in example_results]
    change_rates = [float(result["choice_change_rate"]) for result in example_results]
    latencies = [float(result["latency_sec"]) for result in example_results]
    tokens_per_second = [
        float(result["tokens_per_second"])
        for result in example_results
        if result["tokens_per_second"] is not None
    ]
    trajectory_values = [
        float(result["trajectory_value"])
        for result in example_results
        if result["trajectory_value"] is not None
    ]
    consistency_flags = [
        bool(result["trajectory_value_matches_last_step"])
        for result in example_results
        if result["trajectory_value_matches_last_step"] is not None
    ]
    consistency_abs_diffs = [
        float(result["trajectory_value_last_step_abs_diff"])
        for result in example_results
        if result.get("trajectory_value_last_step_abs_diff") is not None
    ]

    total_steps = sum(int(result["total_decoding_steps"]) for result in example_results)
    total_choice_changes = sum(int(result["choice_change_count"]) for result in example_results)
    total_latency = sum(latencies)
    total_logp = sum(float(result["sum_chosen_token_actor_logprob"]) for result in example_results)
    total_value = sum(float(result["sum_chosen_token_critic_value"]) for result in example_results)

    paired_traj = [
        (float(result["trajectory_value"]), float(result["task_score"]))
        for result in example_results
        if result["trajectory_value"] is not None and result["task_score"] is not None
    ]
    paired_traj_values = [item[0] for item in paired_traj]
    paired_scores = [item[1] for item in paired_traj]

    binary_scores = set(task_scores).issubset({0.0, 1.0})

    return {
        "config_id": spec.config_id,
        "mode": spec.mode,
        "candidate_builder": spec.candidate_builder,
        "candidate_size": spec.candidate_size,
        "beta": spec.beta,
        "normalization": spec.normalization,
        "rank_temperature": spec.rank_temperature,
        "actor_sampling_mode": spec.actor_sampling_mode,
        "actor_temperature": spec.actor_temperature,
        "actor_top_p": spec.actor_top_p,
        "actor_top_k": spec.actor_top_k,
        "num_examples": len(example_results),
        "mean_task_score": _mean(task_scores),
        "mean_accuracy": _mean(task_scores) if binary_scores else None,
        "mean_response_length": _mean(response_lengths),
        "eos_rate": _mean([float(flag) for flag in eos_flags]),
        "max_length_hit_rate": _mean([float(flag) for flag in max_length_flags]),
        "mean_choice_change_rate": _mean(change_rates),
        "global_choice_change_rate": (total_choice_changes / total_steps) if total_steps > 0 else None,
        "mean_chosen_token_actor_logprob": (total_logp / total_steps) if total_steps > 0 else None,
        "mean_chosen_token_critic_value": (total_value / total_steps) if total_steps > 0 else None,
        "mean_trajectory_value": _mean(trajectory_values),
        "trajectory_value_vs_score_pearson": _pearson(paired_traj_values, paired_scores),
        "trajectory_value_vs_score_spearman": _spearman(paired_traj_values, paired_scores),
        "trajectory_value_last_step_consistency_rate": _mean([float(flag) for flag in consistency_flags]),
        "mean_trajectory_value_last_step_abs_diff": _mean(consistency_abs_diffs),
        "total_decoding_steps": total_steps,
        "total_choice_changes": total_choice_changes,
        "sum_example_latency_sec": total_latency,
        "wall_time_sec": wall_time_sec,
        "total_latency_sec": wall_time_sec if wall_time_sec is not None else total_latency,
        "overall_tokens_per_second": (
            total_steps / wall_time_sec if wall_time_sec is not None and wall_time_sec > 0 else (
                total_steps / total_latency if total_latency > 0 else None
            )
        ),
        "mean_tokens_per_second": _mean(tokens_per_second),
    }


def _to_jsonable_line(record: dict[str, Any]) -> str:
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


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    merged_root = Path(args.merged_root).resolve() if args.merged_root else None

    actor_hf_dir, critic_hf_dir = ensure_merged_checkpoints(
        checkpoint_dir,
        merged_root=merged_root,
        skip_merge=args.skip_merge,
    )

    dtype = resolve_dtype(args.dtype)
    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, tokenizer)

    examples = load_examples(
        args.dataset_path,
        tokenizer=tokenizer,
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

    run_specs = build_run_specs(args)
    if not run_specs:
        raise ValueError("No run specifications were produced from the provided CLI arguments.")

    per_example_path = output_dir / "per_example_results.jsonl"
    step_level_path = output_dir / "step_level_minimal.jsonl"
    summary_path = output_dir / "summary_metrics.json"
    csv_path = output_dir / "main_results.csv"
    total_tasks = len(examples) * len(run_specs)

    worker_pairs = parse_worker_pairs(
        args.worker_pairs,
        actor_device=args.actor_device,
        critic_device=args.critic_device,
        default_device=args.device,
    )
    worker_assignments = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)
    multi_worker_enabled = len(worker_assignments) > 1

    if multi_worker_enabled:
        example_results_by_config, worker_summaries = run_multi_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            run_specs=run_specs,
            worker_pairs=worker_pairs,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            normalization_eps=args.normalization_eps,
            use_actor_cache=not args.disable_actor_cache,
            debug_full_candidates=args.debug_full_candidates,
            seed=args.seed,
        )
        actor_device = None
        critic_device = None
        per_config_wall_times: dict[str, float] = {}
        for spec in run_specs:
            start_times = []
            end_times = []
            for summary in worker_summaries:
                start_time_sec = summary.get("per_config_start_wall_time_sec", {}).get(spec.config_id)
                end_time_sec = summary.get("per_config_end_wall_time_sec", {}).get(spec.config_id)
                if start_time_sec is None or end_time_sec is None:
                    # Backward-compatible fallback for older worker summaries.
                    start_offset = float(summary.get("per_config_start_sec", {}).get(spec.config_id, 0.0))
                    runtime_sec = float(summary.get("per_config_runtime_sec", {}).get(spec.config_id, 0.0))
                    start_times.append(start_offset)
                    end_times.append(start_offset + runtime_sec)
                else:
                    start_times.append(float(start_time_sec))
                    end_times.append(float(end_time_sec))
            per_config_wall_times[spec.config_id] = (
                (max(end_times) - min(start_times)) if start_times and end_times else 0.0
            )
    else:
        actor_device = resolve_device(worker_pairs[0][0])
        critic_device = resolve_device(worker_pairs[0][1]) if worker_pairs[0][1] else actor_device
        actor = load_actor_model(actor_hf_dir, dtype=dtype, device=actor_device, trust_remote_code=args.trust_remote_code)
        critic = load_critic_model(critic_hf_dir, dtype=dtype, device=critic_device, trust_remote_code=args.trust_remote_code)

        example_results_by_config: dict[str, list[dict[str, Any]]] = {spec.config_id: [] for spec in run_specs}
        per_config_wall_times: dict[str, float] = {}

        with per_example_path.open("w", encoding="utf-8") as per_example_file, step_level_path.open(
            "w",
            encoding="utf-8",
        ) as step_level_file, tqdm(
            total=total_tasks,
            desc="whole_experiment",
            unit="task",
            dynamic_ncols=True,
        ) as whole_progress:
            for spec_index, spec in enumerate(run_specs):
                config_start_time = time.perf_counter()
                for example in examples:
                    whole_progress.set_postfix_str(f"config={spec.config_id} example_id={example.example_id}")
                    decode_seed = args.seed + spec_index * 1_000_003 + example.example_id
                    artifacts = decode_example(
                        actor=actor,
                        critic=critic,
                        tokenizer=tokenizer,
                        example=example,
                        run_spec=spec,
                        max_prompt_length=args.max_prompt_length,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        actor_device=actor_device,
                        critic_device=critic_device,
                        seed=decode_seed,
                        normalization_eps=args.normalization_eps,
                        use_actor_cache=not args.disable_actor_cache,
                        debug_full_candidates=args.debug_full_candidates,
                    )
                    per_example_file.write(_to_jsonable_line(artifacts.example_result))
                    for step_result in artifacts.step_results:
                        step_level_file.write(_to_jsonable_line(step_result))
                    example_results_by_config[spec.config_id].append(artifacts.example_result)
                    whole_progress.update(1)
                per_config_wall_times[spec.config_id] = time.perf_counter() - config_start_time
        worker_summaries = [
            {
                "worker_id": 0,
                "actor_device": str(actor_device),
                "critic_device": str(critic_device),
                "example_start": 0,
                "example_end": len(examples),
                "num_examples": len(examples),
                "num_run_specs": len(run_specs),
                "per_config_counts": {spec.config_id: len(examples) for spec in run_specs},
                "per_config_start_wall_time_sec": {spec.config_id: 0.0 for spec in run_specs},
                "per_config_end_wall_time_sec": {
                    spec.config_id: per_config_wall_times[spec.config_id] for spec in run_specs
                },
                "per_config_runtime_sec": per_config_wall_times,
            }
        ]

    aggregate_rows = [
        aggregate_results(
            spec,
            example_results_by_config[spec.config_id],
            wall_time_sec=per_config_wall_times.get(spec.config_id),
        )
        for spec in run_specs
    ]

    fieldnames = [
        "config_id",
        "mode",
        "candidate_builder",
        "candidate_size",
        "beta",
        "normalization",
        "rank_temperature",
        "actor_sampling_mode",
        "actor_temperature",
        "actor_top_p",
        "actor_top_k",
        "num_examples",
        "mean_task_score",
        "mean_accuracy",
        "mean_response_length",
        "eos_rate",
        "max_length_hit_rate",
        "mean_choice_change_rate",
        "global_choice_change_rate",
        "mean_chosen_token_actor_logprob",
        "mean_chosen_token_critic_value",
        "mean_trajectory_value",
        "trajectory_value_vs_score_pearson",
        "trajectory_value_vs_score_spearman",
        "trajectory_value_last_step_consistency_rate",
        "mean_trajectory_value_last_step_abs_diff",
        "sum_example_latency_sec",
        "wall_time_sec",
        "total_decoding_steps",
        "total_choice_changes",
        "total_latency_sec",
        "overall_tokens_per_second",
        "mean_tokens_per_second",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "checkpoint_dir": str(checkpoint_dir),
        "merged_actor_dir": str(actor_hf_dir),
        "merged_critic_dir": str(critic_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "output_dir": str(output_dir),
        "multi_worker_enabled": multi_worker_enabled,
        "actor_device": None if actor_device is None else str(actor_device),
        "critic_device": None if critic_device is None else str(critic_device),
        "worker_pairs": [[actor, critic] for actor, critic in worker_pairs],
        "worker_assignments": worker_assignments_to_jsonable(worker_assignments),
        "worker_summaries": worker_summaries,
        "dtype": args.dtype,
        "eos_token_ids": list(eos_token_ids),
        "run_args": vars(args),
        "run_specs": [asdict(spec) for spec in run_specs],
        "aggregate_metrics": aggregate_rows,
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    print(f"[saved] {summary_path}")
    print(f"[saved] {per_example_path}")
    print(f"[saved] {step_level_path}")
    print(f"[saved] {csv_path}")
    return 0
