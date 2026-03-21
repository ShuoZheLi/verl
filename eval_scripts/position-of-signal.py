#!/usr/bin/env python3
"""Measure where within-response critic signal appears by rescoring saved rollouts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from debug_critic_values_all import (  # noqa: E402
    _get_dtype,
    _has_hf_weights,
    _load_critic,
    _merge_fsdp_checkpoint,
    _prepare_tokenizer,
)
from measure_var import (  # noqa: E402
    _build_baseline_summary,
    _compute_sequence_values,
    _nanmean_or_none,
    _pairwise_accuracy,
    _population_variance,
    _residual_variance_ratio,
    _safe_corrcoef,
    _save_summary_json,
    _summarize_metric_across_prompts,
    _to_jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore previously saved grouped rollouts and measure how within-prompt critic signal "
            "changes across normalized response positions."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_850",
        help="Checkpoint directory containing actor/critic FSDP shards or merged_hf weights.",
    )
    parser.add_argument(
        "--source_run_dir",
        type=str,
        default="/data/shuozhe/verl/critic_debug/measure_var_job_05b_vh_init_e5_metamath_step_850_test500_k8",
        help="Existing measure_var output directory containing rollouts_rank*.jsonl.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/shuozhe/verl/critic_debug/position_of_signal_job_05b_vh_init_e5_metamath_step_850_test500_k8",
        help="Directory for rescored rollout values, prompt metrics, plots, and summary JSON.",
    )
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Explicit device override, e.g. cuda or cpu.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--fix_mistral_regex", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.25,0.5,0.75,0.9",
        help="Comma-separated normalized response positions to evaluate before adding prompt_end and final.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Start index into sorted prompt ids from source run.")
    parser.add_argument("--end_index", type=int, default=None, help="End index into sorted prompt ids from source run.")
    parser.add_argument("--max_prompts", type=int, default=None, help="Maximum number of prompt groups to rescore.")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_scatter_prompts", type=int, default=12)
    parser.add_argument(
        "--self_check_prompts",
        type=int,
        default=3,
        help="Number of prompt groups to verify by recomputing truncated-prefix values directly.",
    )
    parser.add_argument(
        "--self_check_rollouts_per_prompt",
        type=int,
        default=2,
        help="How many rollouts per checked prompt to use for the truncated-prefix correctness check.",
    )
    return parser.parse_args()


def _parse_fraction_list(text: str) -> list[float]:
    fractions: list[float] = []
    if not text.strip():
        return fractions
    for piece in text.split(","):
        raw = piece.strip()
        if not raw:
            continue
        value = float(raw)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Fractions must be strictly between 0 and 1, got {value}")
        fractions.append(value)
    deduped = sorted(set(fractions))
    return deduped


def _fraction_to_label(fraction: float) -> str:
    pct = fraction * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9:
        return f"pct{int(rounded):02d}"
    safe = str(pct).replace(".", "_")
    return f"pct{safe}"


def _fraction_to_display_name(fraction: float) -> str:
    pct = fraction * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9:
        return f"{int(rounded)}%"
    return f"{pct:.1f}%"


def _build_position_specs(fractions: list[float]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "label": "prompt_end",
            "display_name": "Prompt End",
            "fraction": None,
            "plot_x": 0.0,
            "kind": "prompt_end",
        }
    ]
    for fraction in fractions:
        specs.append(
            {
                "label": _fraction_to_label(fraction),
                "display_name": _fraction_to_display_name(fraction),
                "fraction": float(fraction),
                "plot_x": float(fraction * 100.0),
                "kind": "response_fraction",
            }
        )
    specs.append(
        {
            "label": "final",
            "display_name": "Final",
            "fraction": 1.0,
            "plot_x": 100.0,
            "kind": "response_final",
        }
    )
    return specs


def _value_key(label: str) -> str:
    return f"value_{label}"


def _index_key(label: str) -> str:
    return f"token_index_{label}"


def _actual_fraction_key(label: str) -> str:
    return f"actual_fraction_{label}"


def _load_source_summary(source_run_dir: Path) -> dict[str, Any] | None:
    summary_path = source_run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _source_rollout_paths(source_run_dir: Path) -> list[Path]:
    paths = sorted(source_run_dir.glob("rollouts_rank*.jsonl"))
    if paths:
        return paths
    single = source_run_dir / "rollouts.jsonl"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No rollouts_rank*.jsonl or rollouts.jsonl found in {source_run_dir}")


def _load_source_prompt_groups(source_run_dir: Path) -> tuple[dict[int, list[dict[str, Any]]], list[Path]]:
    prompt_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    paths = _source_rollout_paths(source_run_dir)
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                prompt_groups[int(record["prompt_id"])].append(record)
    for prompt_id, records in prompt_groups.items():
        records.sort(key=lambda item: int(item["rollout_id"]))
        _validate_source_prompt_group(prompt_id, records)
    return dict(sorted(prompt_groups.items())), paths


def _validate_source_prompt_group(prompt_id: int, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"Prompt group {prompt_id} is empty.")

    prompt = records[0]["prompt"]
    reference = records[0]["reference"]
    data_source = records[0]["data_source"]
    rollout_ids = set()
    prompt_end_values = []

    for record in records:
        if record["prompt"] != prompt:
            raise ValueError(f"Prompt text mismatch inside prompt_id={prompt_id}")
        if record["reference"] != reference:
            raise ValueError(f"Reference mismatch inside prompt_id={prompt_id}")
        if record["data_source"] != data_source:
            raise ValueError(f"data_source mismatch inside prompt_id={prompt_id}")
        rollout_id = int(record["rollout_id"])
        if rollout_id in rollout_ids:
            raise ValueError(f"Duplicate rollout_id={rollout_id} inside prompt_id={prompt_id}")
        rollout_ids.add(rollout_id)

        response_ids = list(record.get("response_ids", []))
        response_length = int(record.get("response_length", len(response_ids)))
        if response_length != len(response_ids):
            raise ValueError(
                f"response_length mismatch for prompt_id={prompt_id}, rollout_id={rollout_id}: "
                f"{response_length} vs len(response_ids)={len(response_ids)}"
            )
        prompt_end_values.append(float(record["prompt_end_value"]))

    if max(prompt_end_values) - min(prompt_end_values) > 1e-6:
        raise ValueError(f"prompt_end_value mismatch inside prompt_id={prompt_id}")


def _select_prompt_groups(
    prompt_groups: dict[int, list[dict[str, Any]]],
    start_index: int,
    end_index: int | None,
    max_prompts: int | None,
) -> dict[int, list[dict[str, Any]]]:
    prompt_ids = sorted(prompt_groups.keys())
    start = max(0, int(start_index))
    end = len(prompt_ids) if end_index is None else min(int(end_index), len(prompt_ids))
    if max_prompts is not None:
        end = min(end, start + int(max_prompts))
    selected_ids = prompt_ids[start:end]
    return {prompt_id: prompt_groups[prompt_id] for prompt_id in selected_ids}


def _response_position_index(response_len: int, fraction: float) -> int | None:
    if response_len <= 0:
        return None
    if fraction >= 1.0:
        return response_len - 1
    idx = math.ceil(fraction * response_len) - 1
    idx = max(0, min(response_len - 1, idx))
    return int(idx)


def _pad_sequences(
    prompt_ids: torch.Tensor,
    response_id_lists: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[int]]:
    prompt_len = int(prompt_ids.numel())
    full_sequences: list[torch.Tensor] = []
    full_lengths: list[int] = []

    for response_ids in response_id_lists:
        response_tensor = torch.tensor(response_ids, device=device, dtype=prompt_ids.dtype)
        if response_tensor.numel() == 0:
            full_sequence = prompt_ids.clone()
        else:
            full_sequence = torch.cat([prompt_ids, response_tensor], dim=0)
        full_sequences.append(full_sequence)
        full_lengths.append(int(full_sequence.numel()))

    max_len = max(full_lengths)
    batch_size = len(full_sequences)
    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=int(pad_token_id),
        dtype=prompt_ids.dtype,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for row_idx, sequence in enumerate(full_sequences):
        seq_len = int(sequence.numel())
        input_ids[row_idx, :seq_len] = sequence
        attention_mask[row_idx, :seq_len] = 1

    if prompt_len <= 0:
        raise ValueError("Prompt length must be positive.")
    return input_ids, attention_mask, full_sequences, full_lengths


def _filter_prompt_groups_for_value_key(
    prompt_groups: dict[int, list[dict[str, Any]]],
    value_key: str,
) -> dict[int, list[dict[str, Any]]]:
    filtered: dict[int, list[dict[str, Any]]] = {}
    for prompt_id, records in prompt_groups.items():
        subrecords = [record for record in records if record.get(value_key) is not None]
        if subrecords:
            filtered[prompt_id] = subrecords
    return filtered


def _compute_prompt_metrics_for_value_key(
    prompt_groups: dict[int, list[dict[str, Any]]],
    value_key: str,
) -> dict[int, dict[str, Any]]:
    metrics_by_prompt: dict[int, dict[str, Any]] = {}
    for prompt_id, records in prompt_groups.items():
        filtered = [record for record in records if record.get(value_key) is not None]
        rewards = np.asarray([record["reward"] for record in filtered], dtype=np.float64)
        values = np.asarray([record[value_key] for record in filtered], dtype=np.float64)
        pairwise_acc, pairwise_pairs, pairwise_ties = _pairwise_accuracy(values, rewards)
        metrics_by_prompt[prompt_id] = {
            "num_rollouts_with_value": int(len(filtered)),
            "value_variance": _population_variance(values) if len(filtered) >= 2 else None,
            "reward_correlation": _safe_corrcoef(values, rewards) if len(filtered) >= 2 else None,
            "pairwise_accuracy": pairwise_acc,
            "pairwise_pairs": int(pairwise_pairs),
            "pairwise_ties": int(pairwise_ties),
            "residual_variance_ratio": _residual_variance_ratio(values, rewards) if len(filtered) >= 2 else None,
        }
    return metrics_by_prompt


def _compute_prompt_metric_rows(
    prompt_groups: dict[int, list[dict[str, Any]]],
    position_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_maps = {
        spec["label"]: _compute_prompt_metrics_for_value_key(prompt_groups, _value_key(spec["label"]))
        for spec in position_specs
        if spec["label"] != "prompt_end"
    }

    for prompt_id, records in sorted(prompt_groups.items()):
        rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
        row: dict[str, Any] = {
            "prompt_id": int(prompt_id),
            "num_rollouts_total": int(len(records)),
            "num_success": int(np.sum(rewards > 0.5)),
            "num_failure": int(np.sum(rewards <= 0.5)),
            "reward_rate": float(rewards.mean()) if rewards.size else None,
            "reward_variance": _population_variance(rewards),
            "prompt_end_value": float(records[0]["prompt_end_value"]),
        }

        for spec in position_specs:
            label = spec["label"]
            value_key = _value_key(label)
            available = [record[value_key] for record in records if record.get(value_key) is not None]
            row[f"num_rollouts_with_{label}"] = int(len(available))
            row[f"mean_value_{label}"] = float(np.mean(available)) if available else None
            if label == "prompt_end":
                successes = row["num_success"]
                failures = row["num_failure"]
                row[f"value_variance_{label}"] = 0.0 if len(records) >= 2 else None
                row[f"reward_correlation_{label}"] = None
                row[f"pairwise_accuracy_{label}"] = 0.5 if successes > 0 and failures > 0 else None
                row[f"pairwise_pairs_{label}"] = int(successes * failures)
                row[f"pairwise_ties_{label}"] = int(successes * failures)
                row[f"residual_variance_ratio_{label}"] = 1.0 if row["reward_variance"] not in (None, 0.0) else None
            else:
                metrics = metric_maps[label][prompt_id]
                row[f"value_variance_{label}"] = metrics["value_variance"]
                row[f"reward_correlation_{label}"] = metrics["reward_correlation"]
                row[f"pairwise_accuracy_{label}"] = metrics["pairwise_accuracy"]
                row[f"pairwise_pairs_{label}"] = metrics["pairwise_pairs"]
                row[f"pairwise_ties_{label}"] = metrics["pairwise_ties"]
                row[f"residual_variance_ratio_{label}"] = metrics["residual_variance_ratio"]

        rows.append(row)
    return rows


def _save_prompt_metrics_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _save_rescored_rollouts_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _save_position_curve_plot(
    path: Path,
    position_specs: list[dict[str, Any]],
    metrics_by_position: dict[str, dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[warn] Failed to save {path.name}: {exc}")
        return

    x = [spec["plot_x"] for spec in position_specs]
    labels = [spec["display_name"] for spec in position_specs]
    pairwise = [metrics_by_position[spec["label"]]["weighted_pairwise_accuracy"] for spec in position_specs]
    corr = [metrics_by_position[spec["label"]]["pooled_within_prompt_correlation"] for spec in position_specs]
    residual = [metrics_by_position[spec["label"]]["pooled_residual_variance_ratio"] for spec in position_specs]
    variance = [metrics_by_position[spec["label"]]["mean_within_prompt_value_variance"] for spec in position_specs]

    def _series(values):
        return [float("nan") if value is None else float(value) for value in values]

    pairwise = _series(pairwise)
    corr = _series(corr)
    residual = _series(residual)
    variance = _series(variance)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)

    axes[0].plot(x, pairwise, marker="o", linewidth=2.0, color="#1d3557")
    axes[0].axhline(0.5, linestyle="--", color="#666666", linewidth=1.0)
    axes[0].set_title("Weighted Pairwise Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(x, corr, marker="o", linewidth=2.0, color="#2a9d8f")
    axes[1].axhline(0.0, linestyle="--", color="#666666", linewidth=1.0)
    axes[1].set_title("Pooled Within-Prompt Correlation")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(x, residual, marker="o", linewidth=2.0, color="#e76f51")
    axes[2].axhline(1.0, linestyle="--", color="#666666", linewidth=1.0)
    axes[2].set_title("Pooled Residual Variance Ratio")
    axes[2].set_ylabel("Var(R - V) / Var(R)")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    axes[3].plot(x, variance, marker="o", linewidth=2.0, color="#7b2cbf")
    axes[3].axhline(0.0, linestyle="--", color="#666666", linewidth=1.0)
    axes[3].set_title("Mean Within-Prompt Value Variance")
    axes[3].set_ylabel("Variance")
    axes[3].grid(True, linestyle="--", alpha=0.35)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Response Position")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_pairwise_distribution_plot(
    path: Path,
    position_specs: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[warn] Failed to save {path.name}: {exc}")
        return

    response_specs = [spec for spec in position_specs if spec["label"] != "prompt_end"]
    fig, axes = plt.subplots(1, len(response_specs), figsize=(4.0 * len(response_specs), 3.8), squeeze=False)

    for ax, spec in zip(axes[0], response_specs, strict=False):
        values = [
            row[f"pairwise_accuracy_{spec['label']}"]
            for row in prompt_rows
            if row[f"pairwise_accuracy_{spec['label']}"] is not None
        ]
        ax.hist(values, bins=20, color="#457b9d", alpha=0.8)
        ax.axvline(0.5, linestyle="--", color="#666666", linewidth=1.0)
        ax.set_title(spec["display_name"])
        ax.set_xlabel("Prompt-level pairwise accuracy")
        ax.set_ylabel("Prompt count")
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_random_prompt_scatter(
    path: Path,
    prompt_groups: dict[int, list[dict[str, Any]]],
    position_specs: list[dict[str, Any]],
    num_prompts: int,
    seed: int,
) -> None:
    response_specs = [spec for spec in position_specs if spec["label"] != "prompt_end"]
    if not response_specs:
        return

    informative_ids = [
        prompt_id
        for prompt_id, records in prompt_groups.items()
        if any(record["reward"] > 0.5 for record in records)
        and any(record["reward"] <= 0.5 for record in records)
        and all(all(record.get(_value_key(spec["label"])) is not None for record in records) for spec in response_specs)
    ]
    if not informative_ids:
        return

    rng = np.random.default_rng(seed)
    chosen_ids = list(rng.choice(informative_ids, size=min(num_prompts, len(informative_ids)), replace=False))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[warn] Failed to save {path.name}: {exc}")
        return

    ncols = len(response_specs)
    nrows = len(chosen_ids)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)

    for row_idx, prompt_id in enumerate(chosen_ids):
        records = prompt_groups[prompt_id]
        rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
        jitter = np.linspace(-0.035, 0.035, num=len(rewards)) if len(rewards) > 1 else np.asarray([0.0])
        for col_idx, spec in enumerate(response_specs):
            ax = axes[row_idx, col_idx]
            value_key = _value_key(spec["label"])
            values = np.asarray([record[value_key] for record in records], dtype=np.float64)
            ax.scatter(values, rewards + jitter, s=30, alpha=0.9, color="#bc4749")
            if row_idx == 0:
                ax.set_title(spec["display_name"])
            if col_idx == 0:
                ax.set_ylabel(f"prompt {prompt_id}\nreward")
            else:
                ax.set_ylabel("reward")
            ax.set_xlabel("critic value")
            ax.set_yticks([0.0, 1.0])
            ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    fractions = _parse_fraction_list(args.fractions)
    position_specs = _build_position_specs(fractions)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    source_run_dir = Path(args.source_run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_prompt_groups, source_record_paths = _load_source_prompt_groups(source_run_dir)
    source_summary = _load_source_summary(source_run_dir)
    prompt_groups = _select_prompt_groups(
        source_prompt_groups,
        start_index=args.start_index,
        end_index=args.end_index,
        max_prompts=args.max_prompts,
    )
    if not prompt_groups:
        raise ValueError("No prompt groups selected.")

    actor_ckpt = ckpt_dir / "actor"
    critic_ckpt = ckpt_dir / "critic"
    merged_root = ckpt_dir / "merged_hf"
    actor_hf = merged_root / "actor"
    critic_hf = merged_root / "critic"

    if not args.skip_merge:
        if not _has_hf_weights(actor_hf):
            _merge_fsdp_checkpoint(actor_ckpt, actor_hf)
        if not _has_hf_weights(critic_hf):
            _merge_fsdp_checkpoint(critic_ckpt, critic_hf)

    if not _has_hf_weights(actor_hf):
        raise FileNotFoundError(f"Actor HF weights not found in {actor_hf}")
    if not _has_hf_weights(critic_hf):
        raise FileNotFoundError(f"Critic HF weights not found in {critic_hf}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = _get_dtype(args.dtype)
    tokenizer = _prepare_tokenizer(actor_hf, trust_remote_code=args.trust_remote_code, fix_mistral_regex=args.fix_mistral_regex)
    critic, critic_value_spec = _load_critic(
        critic_hf,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    prompt_ids_sorted = sorted(prompt_groups.keys())
    print(f"[config] prompts={len(prompt_ids_sorted)} source_rollouts={sum(len(v) for v in prompt_groups.values())} device={device}")
    print(
        "[config] positions="
        + ", ".join(f"{spec['label']}({spec['display_name']})" for spec in position_specs)
    )

    rescored_rollouts: list[dict[str, Any]] = []
    rescored_prompt_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)

    max_prompt_end_source_diff = 0.0
    max_prompt_end_batch_diff = 0.0
    max_source_final_diff = 0.0
    max_source_mean_diff = 0.0
    max_prefix_self_check_diff = 0.0
    num_zero_length_responses = 0
    num_self_check_prefixes = 0
    self_checked_prompt_count = 0

    with torch.inference_mode():
        for processed_idx, prompt_id in enumerate(prompt_ids_sorted, start=1):
            records = prompt_groups[prompt_id]
            prompt = records[0]["prompt"]
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_len,
            )
            prompt_ids = prompt_inputs["input_ids"][0].to(device)
            prompt_attention_mask = prompt_inputs["attention_mask"].to(device)
            prompt_len = int(prompt_attention_mask[0].sum().item())
            if prompt_len <= 0:
                raise ValueError(f"Prompt tokenization produced zero valid tokens for prompt_id={prompt_id}")

            prompt_only_values = _compute_sequence_values(
                critic=critic,
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=prompt_attention_mask,
                value_spec=critic_value_spec,
            )
            prompt_end_value = float(prompt_only_values[0, prompt_len - 1].item())
            source_prompt_end_value = float(records[0]["prompt_end_value"])
            max_prompt_end_source_diff = max(
                max_prompt_end_source_diff,
                abs(prompt_end_value - source_prompt_end_value),
            )

            response_id_lists = [list(record["response_ids"]) for record in records]
            input_ids, attention_mask, full_sequences, _ = _pad_sequences(
                prompt_ids=prompt_ids,
                response_id_lists=response_id_lists,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            full_values = _compute_sequence_values(
                critic=critic,
                input_ids=input_ids,
                attention_mask=attention_mask,
                value_spec=critic_value_spec,
            )

            prompt_end_batch_diffs = (
                torch.abs(full_values[:, prompt_len - 1] - prompt_end_value).to(torch.float32).detach().cpu().numpy()
            )
            if prompt_end_batch_diffs.size:
                max_prompt_end_batch_diff = max(max_prompt_end_batch_diff, float(prompt_end_batch_diffs.max()))

            need_self_check = (
                args.self_check_prompts > 0
                and self_checked_prompt_count < args.self_check_prompts
            )

            for row_idx, source_record in enumerate(records):
                response_ids = list(source_record["response_ids"])
                response_len = len(response_ids)
                if response_len == 0:
                    num_zero_length_responses += 1

                response_values_tensor = full_values[row_idx, prompt_len : prompt_len + response_len]
                response_values = response_values_tensor.to(torch.float32).detach().cpu().tolist()
                recomputed_final_value = float(response_values[-1]) if response_values else None
                recomputed_mean_value = float(np.mean(response_values)) if response_values else None

                if recomputed_final_value is not None and source_record.get("final_response_value") is not None:
                    max_source_final_diff = max(
                        max_source_final_diff,
                        abs(recomputed_final_value - float(source_record["final_response_value"])),
                    )
                if recomputed_mean_value is not None and source_record.get("mean_response_value") is not None:
                    max_source_mean_diff = max(
                        max_source_mean_diff,
                        abs(recomputed_mean_value - float(source_record["mean_response_value"])),
                    )

                rescored_record: dict[str, Any] = {
                    "prompt_id": int(prompt_id),
                    "rollout_id": int(source_record["rollout_id"]),
                    "reward": float(source_record["reward"]),
                    "response_length": int(response_len),
                    "prompt_end_value": float(prompt_end_value),
                    "source_prompt_end_value": source_prompt_end_value,
                    "source_final_response_value": source_record.get("final_response_value"),
                    "source_mean_response_value": source_record.get("mean_response_value"),
                    "value_prompt_end": float(prompt_end_value),
                    "token_index_prompt_end": prompt_len - 1,
                    "actual_fraction_prompt_end": 0.0,
                    "value_final": recomputed_final_value,
                    "recomputed_final_response_value": recomputed_final_value,
                    "recomputed_mean_response_value": recomputed_mean_value,
                }

                position_indices_for_self_check: dict[str, int] = {}
                for spec in position_specs:
                    label = spec["label"]
                    if label == "prompt_end":
                        continue
                    idx = _response_position_index(response_len, float(spec["fraction"]))
                    if idx is None:
                        rescored_record[_value_key(label)] = None
                        rescored_record[_index_key(label)] = None
                        rescored_record[_actual_fraction_key(label)] = None
                        continue
                    value = float(response_values[idx])
                    actual_fraction = float((idx + 1) / response_len)
                    rescored_record[_value_key(label)] = value
                    rescored_record[_index_key(label)] = int(idx)
                    rescored_record[_actual_fraction_key(label)] = actual_fraction
                    position_indices_for_self_check[label] = idx

                rescored_rollouts.append(rescored_record)
                rescored_prompt_groups[prompt_id].append(rescored_record)

                if need_self_check and row_idx < args.self_check_rollouts_per_prompt and response_len > 0:
                    full_sequence = full_sequences[row_idx]
                    for spec in position_specs:
                        label = spec["label"]
                        if label == "prompt_end":
                            continue
                        idx = position_indices_for_self_check.get(label)
                        if idx is None:
                            continue
                        prefix_len = prompt_len + idx + 1
                        truncated_ids = full_sequence[:prefix_len].unsqueeze(0)
                        truncated_mask = torch.ones_like(truncated_ids, dtype=torch.long)
                        truncated_values = _compute_sequence_values(
                            critic=critic,
                            input_ids=truncated_ids,
                            attention_mask=truncated_mask,
                            value_spec=critic_value_spec,
                        )
                        check_value = float(truncated_values[0, prefix_len - 1].item())
                        selected_value = float(rescored_record[_value_key(label)])
                        max_prefix_self_check_diff = max(
                            max_prefix_self_check_diff,
                            abs(check_value - selected_value),
                        )
                        num_self_check_prefixes += 1

            if need_self_check:
                self_checked_prompt_count += 1

            if processed_idx % args.save_every == 0 or processed_idx == len(prompt_ids_sorted):
                print(f"[progress] processed {processed_idx}/{len(prompt_ids_sorted)} prompts")

    for prompt_id in rescored_prompt_groups:
        rescored_prompt_groups[prompt_id].sort(key=lambda item: int(item["rollout_id"]))

    rescored_rollouts_path = out_dir / "rescored_rollouts.jsonl"
    _save_rescored_rollouts_jsonl(rescored_rollouts_path, [_to_jsonable(row) for row in rescored_rollouts])

    prompt_metrics_path = out_dir / "prompt_metrics.jsonl"
    prompt_metric_rows = _compute_prompt_metric_rows(rescored_prompt_groups, position_specs)
    _save_prompt_metrics_jsonl(prompt_metrics_path, [_to_jsonable(row) for row in prompt_metric_rows])

    metrics_by_position: dict[str, dict[str, Any]] = {}
    for spec in position_specs:
        label = spec["label"]
        value_key = _value_key(label)
        filtered_prompt_groups = _filter_prompt_groups_for_value_key(rescored_prompt_groups, value_key)
        if label == "prompt_end":
            summary = _build_baseline_summary(filtered_prompt_groups)
        else:
            summary = _summarize_metric_across_prompts(filtered_prompt_groups, value_key)
        informative_prompt_count = sum(
            1
            for records in filtered_prompt_groups.values()
            if any(record["reward"] > 0.5 for record in records) and any(record["reward"] <= 0.5 for record in records)
        )
        summary["label"] = label
        summary["display_name"] = spec["display_name"]
        summary["plot_x"] = float(spec["plot_x"])
        summary["num_prompts_with_any_value"] = int(len(filtered_prompt_groups))
        summary["num_rollouts_with_any_value"] = int(sum(len(records) for records in filtered_prompt_groups.values()))
        summary["num_informative_mixed_prompts"] = int(informative_prompt_count)
        metrics_by_position[label] = summary

    source_reward_rates = [row["reward_rate"] for row in prompt_metric_rows if row["reward_rate"] is not None]
    num_all_success = sum(row["num_success"] == row["num_rollouts_total"] for row in prompt_metric_rows)
    num_all_failure = sum(row["num_failure"] == row["num_rollouts_total"] for row in prompt_metric_rows)
    num_mixed = sum((row["num_success"] > 0 and row["num_failure"] > 0) for row in prompt_metric_rows)

    response_position_note = (
        "For a rollout with L generated tokens, position q uses token index ceil(q * L) - 1, "
        "clipped to [0, L - 1]. Final uses L - 1. Stored response_ids include EOS when the model generated it."
    )
    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "source_run_dir": str(source_run_dir),
        "source_rollout_paths": [str(path) for path in source_record_paths],
        "source_summary_path": str(source_run_dir / "summary.json"),
        "source_summary_excerpt": {
            "num_prompts_total": None if source_summary is None else source_summary.get("num_prompts_total"),
            "num_rollouts_total": None if source_summary is None else source_summary.get("num_rollouts_total"),
            "rollouts_per_prompt": None if source_summary is None else source_summary.get("rollouts_per_prompt"),
            "generation": None if source_summary is None else source_summary.get("generation"),
        },
        "out_dir": str(out_dir),
        "num_prompts_total": int(len(rescored_prompt_groups)),
        "num_rollouts_total": int(len(rescored_rollouts)),
        "prompt_outcome_summary": {
            "num_prompts_with_mixed_rewards": int(num_mixed),
            "num_prompts_all_success": int(num_all_success),
            "num_prompts_all_failure": int(num_all_failure),
            "mean_reward_rate_per_prompt": _nanmean_or_none(source_reward_rates),
        },
        "positions": [
            {
                "label": spec["label"],
                "display_name": spec["display_name"],
                "fraction": spec["fraction"],
                "plot_x": spec["plot_x"],
                "kind": spec["kind"],
            }
            for spec in position_specs
        ],
        "response_position_definition": response_position_note,
        "metrics_by_position": metrics_by_position,
        "diagnostic_checks": {
            "max_source_prompt_end_abs_diff": float(max_prompt_end_source_diff),
            "max_prompt_end_full_vs_prompt_only_abs_diff": float(max_prompt_end_batch_diff),
            "max_source_final_value_abs_diff": float(max_source_final_diff),
            "max_source_mean_value_abs_diff": float(max_source_mean_diff),
            "max_prefix_self_check_abs_diff": float(max_prefix_self_check_diff),
            "num_prefix_self_checks": int(num_self_check_prefixes),
            "num_zero_length_responses": int(num_zero_length_responses),
        },
        "artifacts": {
            "rescored_rollouts_jsonl": str(rescored_rollouts_path),
            "prompt_metrics_jsonl": str(prompt_metrics_path),
        },
    }
    summary = _to_jsonable(summary)

    summary_path = out_dir / "summary.json"
    _save_summary_json(summary_path, summary)

    curve_plot_path = out_dir / "position_curves.png"
    pairwise_dist_path = out_dir / "pairwise_distribution_by_position.png"
    scatter_path = out_dir / "random_prompt_scatter.png"
    _save_position_curve_plot(curve_plot_path, position_specs, metrics_by_position)
    _save_pairwise_distribution_plot(pairwise_dist_path, position_specs, prompt_metric_rows)
    _save_random_prompt_scatter(
        scatter_path,
        prompt_groups=rescored_prompt_groups,
        position_specs=position_specs,
        num_prompts=args.num_scatter_prompts,
        seed=args.seed,
    )

    print(f"[saved] {summary_path}")
    print(f"[saved] {rescored_rollouts_path}")
    print(f"[saved] {prompt_metrics_path}")
    print(f"[saved] {curve_plot_path}")
    print(f"[saved] {pairwise_dist_path}")
    print(f"[saved] {scatter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
