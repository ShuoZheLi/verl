#!/usr/bin/env python3
"""Measure within-prompt critic signal by sampling multiple rollouts per prompt."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from debug_critic_values_all import (  # noqa: E402
    _dist_barrier,
    _dist_cleanup,
    _dist_setup,
    _extract_reference_from_row,
    _get_dtype,
    _get_generation_config,
    _has_hf_weights,
    _load_critic,
    _load_policy,
    _merge_fsdp_checkpoint,
    _normalize_prompt,
    _prepare_tokenizer,
)
from verl.utils.reward_score import default_compute_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample K rollouts per prompt and measure whether critic values explain within-prompt reward variance.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_850",
        help="Path to a VERL PPO checkpoint directory containing actor/critic FSDP shards.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet",
        help="Parquet dataset used for evaluation.",
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default="ground_truth")
    parser.add_argument("--data_source_key", type=str, default="data_source")
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--do_sample",
        type=str,
        default="true",
        help="Whether to sample during generation. Accepts true/false.",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Optional explicit device override for single-process runs.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--fix_mistral_regex", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_scatter_prompts", type=int, default=12)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/shuozhe/verl/critic_debug/measure_var_job_05b_vh_init_e5_metamath_step_850_test_k8",
        help="Directory for rollouts, prompt metrics, plots, and summary JSON.",
    )
    return parser.parse_args()


def _parse_bool(text: str) -> bool:
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {text}")


def _coerce_reward_to_float(value: Any) -> float:
    if isinstance(value, dict):
        raise TypeError(f"Expected scalar reward, got dict: {value}")
    return float(value)


def _compute_sequence_values(
    critic,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    value_spec=None,
) -> torch.Tensor:
    from verl.trainer.ppo.value_categorical import value_logits_to_scalar_expectation

    with torch.no_grad():
        outputs = critic(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    if hasattr(critic, "v_head"):
        values = outputs[2]
    else:
        values = outputs.logits

    is_categorical = value_spec is not None and value_spec.is_categorical()
    if is_categorical:
        values, _, _ = value_logits_to_scalar_expectation(values, value_spec)
    elif values.dim() == 3:
        values = values.squeeze(-1)

    return values


def _first_eos_in_tail(tail_ids: torch.Tensor, eos_token_id: int | None) -> int:
    if tail_ids.numel() == 0:
        return 0
    if eos_token_id is None:
        return int(tail_ids.numel())

    eos_matches = torch.nonzero(tail_ids == eos_token_id, as_tuple=False)
    if eos_matches.numel() == 0:
        return int(tail_ids.numel())
    return int(eos_matches[0].item()) + 1


def _generate_rollouts_for_prompt(
    actor,
    critic,
    tokenizer,
    prompt: str,
    rollouts_per_prompt: int,
    max_prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    generation_config,
    device: torch.device,
    value_spec=None,
) -> tuple[float, list[dict[str, Any]], float]:
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
    )
    prompt_ids = prompt_inputs["input_ids"].to(device)
    prompt_attention_mask = prompt_inputs["attention_mask"].to(device)
    prompt_len = int(prompt_attention_mask[0].sum().item())
    if prompt_len <= 0:
        raise ValueError("Prompt tokenization produced zero valid tokens.")

    prompt_values = _compute_sequence_values(
        critic=critic,
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
        value_spec=value_spec,
    )
    prompt_end_value = float(prompt_values[0, prompt_len - 1].item())

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": rollouts_per_prompt,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    if generation_config is not None:
        generate_kwargs["generation_config"] = generation_config

    with torch.inference_mode():
        output_ids = actor.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            **generate_kwargs,
        )

    if output_ids.dim() != 2:
        raise ValueError(f"Unexpected generate output shape: {tuple(output_ids.shape)}")
    if output_ids.shape[0] != rollouts_per_prompt:
        raise ValueError(
            f"Expected {rollouts_per_prompt} sequences, got {output_ids.shape[0]}. "
            "This likely indicates generate() settings were overridden unexpectedly."
        )

    generated_tails = output_ids[:, prompt_len:]
    response_lengths: list[int] = []
    attention_mask = torch.zeros_like(output_ids, dtype=torch.long)
    attention_mask[:, :prompt_len] = 1
    rollout_response_ids: list[torch.Tensor] = []

    for row_idx in range(output_ids.shape[0]):
        tail_ids = generated_tails[row_idx]
        response_len = _first_eos_in_tail(tail_ids, tokenizer.eos_token_id)
        response_lengths.append(response_len)
        if response_len > 0:
            attention_mask[row_idx, prompt_len : prompt_len + response_len] = 1
            rollout_response_ids.append(tail_ids[:response_len].detach().cpu())
        else:
            rollout_response_ids.append(torch.empty(0, dtype=tail_ids.dtype))

    full_values = _compute_sequence_values(
        critic=critic,
        input_ids=output_ids,
        attention_mask=attention_mask.to(device),
        value_spec=value_spec,
    )

    prompt_end_diffs = (
        torch.abs(full_values[:, prompt_len - 1] - prompt_end_value).to(torch.float32).detach().cpu().numpy()
    )
    max_prompt_end_diff = float(prompt_end_diffs.max()) if prompt_end_diffs.size else 0.0

    rollouts: list[dict[str, Any]] = []
    for rollout_id, response_len in enumerate(response_lengths):
        response_ids = rollout_response_ids[rollout_id]
        response_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
        response_values = full_values[rollout_id, prompt_len : prompt_len + response_len].detach().cpu().tolist()
        final_response_value = float(response_values[-1]) if response_values else None
        mean_response_value = float(np.mean(response_values)) if response_values else None
        rollouts.append(
            {
                "rollout_id": rollout_id,
                "response": response_text,
                "response_ids": response_ids.tolist(),
                "response_length": int(response_len),
                "response_values": response_values,
                "final_response_value": final_response_value,
                "mean_response_value": mean_response_value,
            }
        )

    return prompt_end_value, rollouts, max_prompt_end_diff


def _safe_corrcoef(values: np.ndarray, rewards: np.ndarray) -> float | None:
    if values.size < 2 or rewards.size < 2:
        return None
    centered_values = values - values.mean()
    centered_rewards = rewards - rewards.mean()
    value_ss = float(np.dot(centered_values, centered_values))
    reward_ss = float(np.dot(centered_rewards, centered_rewards))
    if value_ss <= 0.0 or reward_ss <= 0.0:
        return None
    return float(np.dot(centered_values, centered_rewards) / math.sqrt(value_ss * reward_ss))


def _pairwise_accuracy(values: np.ndarray, rewards: np.ndarray) -> tuple[float | None, int, int]:
    success_values = values[rewards > 0.5]
    failure_values = values[rewards <= 0.5]
    if success_values.size == 0 or failure_values.size == 0:
        return None, 0, 0

    diff = success_values[:, None] - failure_values[None, :]
    total_pairs = int(diff.size)
    tie_pairs = int(np.sum(diff == 0.0))
    correct_mass = float(np.sum(diff > 0.0) + 0.5 * tie_pairs)
    return float(correct_mass / total_pairs), total_pairs, tie_pairs


def _residual_variance_ratio(values: np.ndarray, rewards: np.ndarray) -> float | None:
    if values.size < 2 or rewards.size < 2:
        return None
    reward_var = float(np.var(rewards))
    if reward_var <= 0.0:
        return None
    residual_var = float(np.var(rewards - values))
    return float(residual_var / reward_var)


def _nanmean_or_none(items: list[float | None]) -> float | None:
    filtered = [float(x) for x in items if x is not None and np.isfinite(x)]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _nanmedian_or_none(items: list[float | None]) -> float | None:
    filtered = [float(x) for x in items if x is not None and np.isfinite(x)]
    if not filtered:
        return None
    return float(np.median(filtered))


def _population_variance(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    return float(np.var(values))


def _summarize_metric_across_prompts(prompt_groups: dict[int, list[dict[str, Any]]], value_key: str) -> dict[str, Any]:
    prompt_variances: list[float | None] = []
    prompt_correlations: list[float | None] = []
    prompt_pairwise_accs: list[float | None] = []
    prompt_residual_ratios: list[float | None] = []
    prompt_reward_vars: list[float | None] = []
    centered_values_all: list[float] = []
    centered_rewards_all: list[float] = []
    pooled_residual_ss = 0.0
    pooled_reward_ss = 0.0
    weighted_pairwise_correct_mass = 0.0
    weighted_pairwise_total_pairs = 0
    weighted_pairwise_ties = 0
    informative_corr_prompts = 0
    informative_pairwise_prompts = 0
    informative_residual_prompts = 0

    for records in prompt_groups.values():
        values = np.asarray([record[value_key] for record in records], dtype=np.float64)
        rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
        if values.size == 0 or rewards.size == 0:
            continue

        prompt_variances.append(_population_variance(values))
        reward_var = _population_variance(rewards)
        prompt_reward_vars.append(reward_var)

        corr = _safe_corrcoef(values, rewards)
        prompt_correlations.append(corr)
        if corr is not None:
            informative_corr_prompts += 1

        pairwise_acc, pairwise_pairs, tie_pairs = _pairwise_accuracy(values, rewards)
        prompt_pairwise_accs.append(pairwise_acc)
        if pairwise_acc is not None and pairwise_pairs > 0:
            informative_pairwise_prompts += 1
            weighted_pairwise_correct_mass += pairwise_acc * pairwise_pairs
            weighted_pairwise_total_pairs += pairwise_pairs
            weighted_pairwise_ties += tie_pairs

        residual_ratio = _residual_variance_ratio(values, rewards)
        prompt_residual_ratios.append(residual_ratio)
        if residual_ratio is not None:
            informative_residual_prompts += 1

        centered_values = values - values.mean()
        centered_rewards = rewards - rewards.mean()
        centered_values_all.extend(centered_values.tolist())
        centered_rewards_all.extend(centered_rewards.tolist())
        pooled_reward_ss += float(np.dot(centered_rewards, centered_rewards))
        residual = rewards - values
        residual_centered = residual - residual.mean()
        pooled_residual_ss += float(np.dot(residual_centered, residual_centered))

    pooled_corr = None
    if centered_values_all and centered_rewards_all:
        pooled_corr = _safe_corrcoef(
            np.asarray(centered_values_all, dtype=np.float64),
            np.asarray(centered_rewards_all, dtype=np.float64),
        )

    pooled_residual_ratio = None
    if pooled_reward_ss > 0.0:
        pooled_residual_ratio = float(pooled_residual_ss / pooled_reward_ss)

    weighted_pairwise_acc = None
    if weighted_pairwise_total_pairs > 0:
        weighted_pairwise_acc = float(weighted_pairwise_correct_mass / weighted_pairwise_total_pairs)

    return {
        "value_key": value_key,
        "mean_within_prompt_value_variance": _nanmean_or_none(prompt_variances),
        "median_within_prompt_value_variance": _nanmedian_or_none(prompt_variances),
        "mean_within_prompt_reward_variance": _nanmean_or_none(prompt_reward_vars),
        "mean_prompt_correlation": _nanmean_or_none(prompt_correlations),
        "median_prompt_correlation": _nanmedian_or_none(prompt_correlations),
        "pooled_within_prompt_correlation": pooled_corr,
        "mean_prompt_pairwise_accuracy": _nanmean_or_none(prompt_pairwise_accs),
        "weighted_pairwise_accuracy": weighted_pairwise_acc,
        "pairwise_total_pairs": int(weighted_pairwise_total_pairs),
        "pairwise_tie_pairs": int(weighted_pairwise_ties),
        "mean_prompt_residual_variance_ratio": _nanmean_or_none(prompt_residual_ratios),
        "pooled_residual_variance_ratio": pooled_residual_ratio,
        "informative_prompt_counts": {
            "for_correlation": int(informative_corr_prompts),
            "for_pairwise_accuracy": int(informative_pairwise_prompts),
            "for_residual_variance_ratio": int(informative_residual_prompts),
        },
    }


def _build_baseline_summary(prompt_groups: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    prompt_variances: list[float] = []
    prompt_residual_ratios: list[float | None] = []
    informative_pairwise_prompts = 0
    informative_residual_prompts = 0
    total_pairs = 0
    reward_vars: list[float | None] = []

    for records in prompt_groups.values():
        rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
        reward_var = _population_variance(rewards)
        reward_vars.append(reward_var)
        prompt_variances.append(0.0)
        if reward_var is not None and reward_var > 0.0:
            prompt_residual_ratios.append(1.0)
            informative_residual_prompts += 1
        else:
            prompt_residual_ratios.append(None)
        successes = int(np.sum(rewards > 0.5))
        failures = int(np.sum(rewards <= 0.5))
        if successes > 0 and failures > 0:
            informative_pairwise_prompts += 1
            total_pairs += successes * failures

    return {
        "value_key": "prompt_end_value",
        "mean_within_prompt_value_variance": 0.0,
        "median_within_prompt_value_variance": 0.0,
        "mean_within_prompt_reward_variance": _nanmean_or_none(reward_vars),
        "mean_prompt_correlation": None,
        "median_prompt_correlation": None,
        "pooled_within_prompt_correlation": None,
        "mean_prompt_pairwise_accuracy": 0.5 if informative_pairwise_prompts > 0 else None,
        "weighted_pairwise_accuracy": 0.5 if total_pairs > 0 else None,
        "pairwise_total_pairs": int(total_pairs),
        "pairwise_tie_pairs": int(total_pairs),
        "mean_prompt_residual_variance_ratio": _nanmean_or_none(prompt_residual_ratios),
        "pooled_residual_variance_ratio": 1.0 if informative_residual_prompts > 0 else None,
        "informative_prompt_counts": {
            "for_correlation": 0,
            "for_pairwise_accuracy": int(informative_pairwise_prompts),
            "for_residual_variance_ratio": int(informative_residual_prompts),
        },
    }


def _compute_prompt_level_rows(prompt_groups: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id, records in sorted(prompt_groups.items()):
        rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
        final_values = np.asarray([record["final_response_value"] for record in records], dtype=np.float64)
        mean_values = np.asarray([record["mean_response_value"] for record in records], dtype=np.float64)
        prompt_end_value = float(records[0]["prompt_end_value"])
        reward_rate = float(rewards.mean()) if rewards.size else None
        final_pairwise, final_pairs, final_ties = _pairwise_accuracy(final_values, rewards)
        mean_pairwise, mean_pairs, mean_ties = _pairwise_accuracy(mean_values, rewards)
        row = {
            "prompt_id": int(prompt_id),
            "num_rollouts": len(records),
            "num_success": int(np.sum(rewards > 0.5)),
            "num_failure": int(np.sum(rewards <= 0.5)),
            "reward_rate": reward_rate,
            "reward_variance": _population_variance(rewards),
            "prompt_end_value": prompt_end_value,
            "final_value_variance": _population_variance(final_values),
            "mean_value_variance": _population_variance(mean_values),
            "final_value_reward_correlation": _safe_corrcoef(final_values, rewards),
            "mean_value_reward_correlation": _safe_corrcoef(mean_values, rewards),
            "final_value_pairwise_accuracy": final_pairwise,
            "mean_value_pairwise_accuracy": mean_pairwise,
            "final_value_pairwise_pairs": int(final_pairs),
            "mean_value_pairwise_pairs": int(mean_pairs),
            "final_value_pairwise_ties": int(final_ties),
            "mean_value_pairwise_ties": int(mean_ties),
            "final_value_residual_variance_ratio": _residual_variance_ratio(final_values, rewards),
            "mean_value_residual_variance_ratio": _residual_variance_ratio(mean_values, rewards),
        }
        rows.append(row)
    return rows


def _save_prompt_metrics_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _save_histogram(path: Path, values: list[float], title: str, xlabel: str) -> None:
    if not values:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[warn] Failed to save histogram {path.name}: {exc}")
        return

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values, bins=24, color="#2a6f97", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prompt count")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_sanity_scatter(path: Path, prompt_groups: dict[int, list[dict[str, Any]]], num_prompts: int, seed: int) -> None:
    informative_ids = [
        prompt_id
        for prompt_id, records in prompt_groups.items()
        if any(record["reward"] > 0.5 for record in records) and any(record["reward"] <= 0.5 for record in records)
    ]
    if not informative_ids:
        return

    rng = random.Random(seed)
    chosen_ids = informative_ids[:]
    rng.shuffle(chosen_ids)
    chosen_ids = chosen_ids[: max(1, min(num_prompts, len(chosen_ids)))]

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[warn] Failed to save sanity scatter {path.name}: {exc}")
        return

    ncols = 3
    nrows = math.ceil(len(chosen_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for ax, prompt_id in zip(axes.flat, chosen_ids, strict=False):
        ax.axis("on")
        records = prompt_groups[prompt_id]
        x = np.asarray([record["final_response_value"] for record in records], dtype=np.float64)
        y = np.asarray([record["reward"] for record in records], dtype=np.float64)
        jitter = np.linspace(-0.035, 0.035, num=len(y)) if len(y) > 1 else np.asarray([0.0])
        ax.scatter(x, y + jitter, s=38, alpha=0.9, color="#bc4749")
        ax.set_title(f"prompt_id={prompt_id}")
        ax.set_xlabel("Final response-token value")
        ax.set_ylabel("Reward")
        ax.set_yticks([0.0, 1.0])
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_summary_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    do_sample = _parse_bool(args.do_sample)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    dist_enabled, rank, world_size, local_rank = _dist_setup()
    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    actor_ckpt = ckpt_dir / "actor"
    critic_ckpt = ckpt_dir / "critic"
    merged_root = ckpt_dir / "merged_hf"
    actor_hf = merged_root / "actor"
    critic_hf = merged_root / "critic"

    if not args.skip_merge:
        if (not dist_enabled) or rank == 0:
            if not _has_hf_weights(actor_hf):
                _merge_fsdp_checkpoint(actor_ckpt, actor_hf)
            if not _has_hf_weights(critic_hf):
                _merge_fsdp_checkpoint(critic_ckpt, critic_hf)
        _dist_barrier(dist_enabled)

    if not _has_hf_weights(actor_hf):
        raise FileNotFoundError(f"Actor HF weights not found in {actor_hf}")
    if not _has_hf_weights(critic_hf):
        raise FileNotFoundError(f"Critic HF weights not found in {critic_hf}")

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if dist_enabled else "cuda")
        else:
            device = torch.device("cpu")

    dtype = _get_dtype(args.dtype)
    tokenizer = _prepare_tokenizer(actor_hf, trust_remote_code=args.trust_remote_code, fix_mistral_regex=args.fix_mistral_regex)
    actor = _load_policy(actor_hf, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code)
    critic, critic_value_spec = _load_critic(
        critic_hf,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    generation_config = _get_generation_config(actor_hf)

    ds = load_dataset("parquet", data_files=str(dataset_path), split="train")
    dataset_total = len(ds)
    start = max(0, int(args.start_index))
    end = dataset_total if args.end_index is None else min(int(args.end_index), dataset_total)
    if args.max_prompts is not None:
        end = min(end, start + int(args.max_prompts))
    selected_indices = list(range(start, end))
    prompt_indices = selected_indices[rank::world_size] if dist_enabled else selected_indices

    if (not dist_enabled) or rank == 0:
        print(
            f"[config] prompts={len(selected_indices)} rollouts_per_prompt={args.rollouts_per_prompt} "
            f"dataset_total={dataset_total} do_sample={do_sample} temperature={args.temperature} top_p={args.top_p}"
        )

    rollouts_path = out_dir / (f"rollouts_rank{rank}.jsonl" if dist_enabled else "rollouts.jsonl")
    local_prompt_count = 0
    local_rollout_count = 0
    max_prompt_end_consistency_diff_local = 0.0

    with rollouts_path.open("w", encoding="utf-8") as out_f:
        for processed_idx, prompt_id in enumerate(prompt_indices, start=1):
            row = ds[int(prompt_id)]
            prompt_raw = row.get(args.prompt_key)
            reference_raw = _extract_reference_from_row(row, args.response_key)
            data_source = row.get(args.data_source_key) if args.data_source_key in row else None
            prompt = _normalize_prompt(prompt_raw, tokenizer)
            if reference_raw is None:
                raise ValueError(f"Missing reference for prompt_id={prompt_id}")

            prompt_end_value, rollouts, max_prompt_end_diff = _generate_rollouts_for_prompt(
                actor=actor,
                critic=critic,
                tokenizer=tokenizer,
                prompt=prompt,
                rollouts_per_prompt=args.rollouts_per_prompt,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=do_sample,
                generation_config=generation_config,
                device=device,
                value_spec=critic_value_spec,
            )
            max_prompt_end_consistency_diff_local = max(max_prompt_end_consistency_diff_local, max_prompt_end_diff)

            for rollout in rollouts:
                reward = _coerce_reward_to_float(default_compute_score(data_source, rollout["response"], reference_raw))
                record = {
                    "prompt_id": int(prompt_id),
                    "rollout_id": int(rollout["rollout_id"]),
                    "prompt": prompt,
                    "response": rollout["response"],
                    "reference": reference_raw,
                    "data_source": data_source,
                    "reward": reward,
                    "prompt_end_value": float(prompt_end_value),
                    "final_response_value": rollout["final_response_value"],
                    "mean_response_value": rollout["mean_response_value"],
                    "response_length": int(rollout["response_length"]),
                    "response_ids": rollout["response_ids"],
                }
                out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                local_rollout_count += 1

            local_prompt_count += 1
            if processed_idx % args.save_every == 0 or processed_idx == len(prompt_indices):
                print(
                    f"[rank {rank}] processed {processed_idx}/{len(prompt_indices)} prompts "
                    f"({local_rollout_count} rollouts)"
                )

    if dist_enabled:
        diff_device = device if device.type == "cuda" else torch.device("cpu")
        diff_tensor = torch.tensor(max_prompt_end_consistency_diff_local, device=diff_device, dtype=torch.float64)
        torch.distributed.all_reduce(diff_tensor, op=torch.distributed.ReduceOp.MAX)
        max_prompt_end_consistency_diff = float(diff_tensor.item())
    else:
        max_prompt_end_consistency_diff = float(max_prompt_end_consistency_diff_local)

    _dist_barrier(dist_enabled)

    if (not dist_enabled) or rank == 0:
        all_records: list[dict[str, Any]] = []
        record_paths = sorted(out_dir.glob("rollouts_rank*.jsonl"))
        if not record_paths and rollouts_path.exists():
            record_paths = [rollouts_path]
        for path in record_paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    all_records.append(record)

        prompt_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in all_records:
            prompt_groups[int(record["prompt_id"])].append(record)

        for prompt_records in prompt_groups.values():
            prompt_records.sort(key=lambda item: int(item["rollout_id"]))

        prompt_metric_rows = _compute_prompt_level_rows(prompt_groups)
        prompt_metrics_path = out_dir / "prompt_metrics.jsonl"
        _save_prompt_metrics_jsonl(prompt_metrics_path, [_to_jsonable(row) for row in prompt_metric_rows])

        final_value_summary = _summarize_metric_across_prompts(prompt_groups, "final_response_value")
        mean_value_summary = _summarize_metric_across_prompts(prompt_groups, "mean_response_value")
        prompt_end_summary = _build_baseline_summary(prompt_groups)

        reward_rates = [row["reward_rate"] for row in prompt_metric_rows if row["reward_rate"] is not None]
        final_value_variances = [
            row["final_value_variance"] for row in prompt_metric_rows if row["final_value_variance"] is not None
        ]
        mean_value_variances = [
            row["mean_value_variance"] for row in prompt_metric_rows if row["mean_value_variance"] is not None
        ]
        final_pairwise_prompt_values = [
            row["final_value_pairwise_accuracy"]
            for row in prompt_metric_rows
            if row["final_value_pairwise_accuracy"] is not None
        ]

        informative_prompts = [
            row for row in prompt_metric_rows if row["num_success"] > 0 and row["num_failure"] > 0
        ]
        summary = {
            "checkpoint_dir": str(ckpt_dir),
            "dataset_path": str(dataset_path),
            "out_dir": str(out_dir),
            "num_prompts_total": int(len(prompt_groups)),
            "num_rollouts_total": int(len(all_records)),
            "rollouts_per_prompt": int(args.rollouts_per_prompt),
            "generation": {
                "do_sample": bool(do_sample),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_prompt_len": int(args.max_prompt_len),
                "max_new_tokens": int(args.max_new_tokens),
            },
            "prompt_outcome_summary": {
                "num_prompts_with_mixed_rewards": int(len(informative_prompts)),
                "num_prompts_all_success": int(sum(row["num_success"] == row["num_rollouts"] for row in prompt_metric_rows)),
                "num_prompts_all_failure": int(sum(row["num_failure"] == row["num_rollouts"] for row in prompt_metric_rows)),
                "mean_reward_rate_per_prompt": _nanmean_or_none(reward_rates),
                "median_reward_rate_per_prompt": _nanmedian_or_none(reward_rates),
            },
            "metrics": {
                "prompt_end_baseline": prompt_end_summary,
                "final_response_value": final_value_summary,
                "mean_response_value": mean_value_summary,
            },
            "diagnostic_checks": {
                "max_prompt_end_consistency_abs_diff": float(max_prompt_end_consistency_diff),
            },
            "distribution_summaries": {
                "mean_final_value_variance_per_prompt": _nanmean_or_none(final_value_variances),
                "mean_mean_value_variance_per_prompt": _nanmean_or_none(mean_value_variances),
                "mean_prompt_pairwise_accuracy_final_value": _nanmean_or_none(final_pairwise_prompt_values),
            },
            "artifacts": {
                "rollout_records": [str(path) for path in record_paths],
                "prompt_metrics_jsonl": str(prompt_metrics_path),
            },
        }

        summary = _to_jsonable(summary)
        summary_path = out_dir / "summary.json"
        _save_summary_json(summary_path, summary)

        _save_histogram(
            out_dir / "final_value_variance_hist.png",
            [float(v) for v in final_value_variances],
            title="Within-Prompt Variance of Final Response-Token Value",
            xlabel="Variance",
        )
        _save_histogram(
            out_dir / "pairwise_accuracy_hist.png",
            [float(v) for v in final_pairwise_prompt_values],
            title="Prompt-Level Pairwise Accuracy (Final Value)",
            xlabel="Accuracy",
        )
        _save_sanity_scatter(
            out_dir / "sanity_scatter.png",
            prompt_groups=prompt_groups,
            num_prompts=args.num_scatter_prompts,
            seed=args.seed,
        )

        print(f"[saved] {summary_path}")
        print(f"[saved] {prompt_metrics_path}")
        print(f"[saved] {out_dir / 'final_value_variance_hist.png'}")
        print(f"[saved] {out_dir / 'pairwise_accuracy_hist.png'}")
        print(f"[saved] {out_dir / 'sanity_scatter.png'}")

    _dist_cleanup(dist_enabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
