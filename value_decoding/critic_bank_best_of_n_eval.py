from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
)
from value_decoding.decoding import critic_sequence_last_values


DEFAULT_N_VALUES = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score an existing response-bank JSONL with a critic checkpoint, then run response-level Best-of-N "
            "selection using the critic final trajectory value. This performs no actor generation."
        )
    )
    parser.add_argument("--critic_checkpoint_dir", type=str, required=True)
    parser.add_argument("--response_bank_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--critic_merged_root", type=str, default=None)
    parser.add_argument("--critic_hf_source_dir", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default="response")
    parser.add_argument("--response_token_ids_key", type=str, default="response_token_ids")
    parser.add_argument("--prompt_index_key", type=str, default="prompt_index")
    parser.add_argument("--sample_index_key", type=str, default="sample_index")
    parser.add_argument("--task_score_key", type=str, default="task_score")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_tokens", type=int, default=2048)
    parser.add_argument("--n_values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument(
        "--retokenize_responses",
        action="store_true",
        help=(
            "Ignore stored response_token_ids and tokenize response text with the critic tokenizer. "
            "By default, stored token ids are used to preserve the collected trajectory exactly."
        ),
    )
    parser.add_argument(
        "--allow_missing_task_score",
        action="store_true",
        help="Allow rows without task_score; selection files are still written, but accuracy metrics are null.",
    )
    parser.add_argument(
        "--keep_all_original_fields",
        action="store_true",
        help="Copy every original response-bank field into critic_scored_response_bank.jsonl.",
    )
    return parser.parse_args()


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True) + "\n"


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(fmean(float(value) for value in values))


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            row["_source_line_number"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"Response bank is empty: {path}")
    return rows


def _coerce_token_ids(value: Any, *, field_name: str, line_number: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} at source line {line_number} must be a list of token ids.")
    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if not isinstance(token_id, int):
            raise ValueError(
                f"{field_name}[{index}] at source line {line_number} must be an int, got {type(token_id).__name__}."
            )
        token_ids.append(int(token_id))
    return token_ids


def _prompt_token_ids(
    row: dict[str, Any],
    *,
    tokenizer,
    prompt_key: str,
    max_prompt_length: int,
) -> list[int]:
    if "prompt_token_ids" in row and row["prompt_token_ids"] is not None:
        prompt_ids = _coerce_token_ids(
            row["prompt_token_ids"],
            field_name="prompt_token_ids",
            line_number=int(row["_source_line_number"]),
        )
        return prompt_ids[:max_prompt_length]

    if prompt_key not in row:
        raise KeyError(f"Missing prompt field {prompt_key!r} at source line {row['_source_line_number']}.")
    tokenized = tokenizer(
        str(row[prompt_key]),
        truncation=True,
        max_length=max_prompt_length,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(token_id) for token_id in tokenized["input_ids"]]


def _response_token_ids(
    row: dict[str, Any],
    *,
    tokenizer,
    response_key: str,
    response_token_ids_key: str,
    max_response_tokens: int,
    retokenize_responses: bool,
) -> list[int]:
    if not retokenize_responses and row.get(response_token_ids_key) is not None:
        response_ids = _coerce_token_ids(
            row[response_token_ids_key],
            field_name=response_token_ids_key,
            line_number=int(row["_source_line_number"]),
        )
        return response_ids[:max_response_tokens]

    if response_key not in row:
        raise KeyError(f"Missing response field {response_key!r} at source line {row['_source_line_number']}.")
    tokenized = tokenizer(
        str(row[response_key]),
        add_special_tokens=False,
        truncation=True,
        max_length=max_response_tokens,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(token_id) for token_id in tokenized["input_ids"]]


def _score_sequence(
    *,
    critic,
    device: torch.device,
    sequence_ids: Sequence[int],
) -> float:
    if not sequence_ids:
        raise ValueError("Cannot score an empty prompt+response sequence.")
    input_ids = torch.tensor([list(sequence_ids)], device=device, dtype=torch.long)
    value = critic_sequence_last_values(critic, input_ids)[0]
    score = float(value.item())
    if not math.isfinite(score):
        raise ValueError(f"Critic produced a non-finite value: {score}")
    return score


def _copy_original_fields(row: dict[str, Any], *, keep_all_original_fields: bool) -> dict[str, Any]:
    if keep_all_original_fields:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    keys_to_keep = (
        "actor_name",
        "actor_checkpoint_dir",
        "prompt_index",
        "example_id",
        "sample_index",
        "seed",
        "response",
        "response_length",
        "ended_with_eos",
        "hit_max_length",
        "task_score",
        "accuracy",
        "prompt_length",
        "prompt",
    )
    return {key: row[key] for key in keys_to_keep if key in row}


def _ensure_field_copy(scored_row: dict[str, Any], source_row: dict[str, Any], field_name: str) -> None:
    if field_name in source_row and field_name not in scored_row:
        scored_row[field_name] = source_row[field_name]


def _group_by_prompt(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_index_key: str,
    sample_index_key: str,
) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            prompt_index = int(row[prompt_index_key])
            sample_index = int(row[sample_index_key])
        except KeyError as exc:
            raise KeyError(f"Missing grouping field at source line {row.get('_source_line_number')}: {exc}") from exc
        row["_prompt_index_int"] = prompt_index
        row["_sample_index_int"] = sample_index
        grouped[prompt_index].append(row)

    grouped_items: list[tuple[int, list[dict[str, Any]]]] = []
    for prompt_index, prompt_rows in sorted(grouped.items()):
        sorted_rows = sorted(prompt_rows, key=lambda item: int(item["_sample_index_int"]))
        sample_indices = [int(row["_sample_index_int"]) for row in sorted_rows]
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError(f"Prompt {prompt_index} has duplicate sample indices: {sample_indices}")
        grouped_items.append((prompt_index, sorted_rows))
    return grouped_items


def _selected_payload(selected_row: dict[str, Any], *, task_score_key: str) -> dict[str, Any]:
    payload = {
        "sample_index": int(selected_row["_sample_index_int"]),
        "critic_final_trajectory_value": float(selected_row["critic_final_trajectory_value"]),
        "response_length": selected_row.get("response_length"),
    }
    if task_score_key in selected_row and selected_row[task_score_key] is not None:
        payload["task_score"] = float(selected_row[task_score_key])
        payload["is_correct"] = bool(float(selected_row[task_score_key]) == 1.0)
    return payload


def _best_positions(values: Sequence[float]) -> list[int]:
    if not values:
        raise ValueError("Cannot select from an empty bank.")
    best_value = max(values)
    return [index for index, value in enumerate(values) if value == best_value]


def _build_prompt_summary(
    *,
    prompt_index: int,
    prompt_rows: Sequence[dict[str, Any]],
    n_values: Sequence[int],
    task_score_key: str,
    allow_missing_task_score: bool,
) -> dict[str, Any]:
    max_bank_size = len(prompt_rows)
    response_values = [float(row["critic_final_trajectory_value"]) for row in prompt_rows]
    task_scores: list[float] = []
    missing_task_score = False
    for row in prompt_rows:
        if task_score_key not in row or row[task_score_key] is None:
            missing_task_score = True
            continue
        task_scores.append(float(row[task_score_key]))

    if missing_task_score and not allow_missing_task_score:
        raise KeyError(
            f"Prompt {prompt_index} has rows without {task_score_key!r}. "
            "Pass --allow_missing_task_score to skip accuracy metrics."
        )

    summary: dict[str, Any] = {
        "prompt_index": int(prompt_index),
        "example_id": prompt_rows[0].get("example_id"),
        "max_bank_size": int(max_bank_size),
        "sample_indices": [int(row["_sample_index_int"]) for row in prompt_rows],
        "critic_values": response_values,
        "task_scores": task_scores if not missing_task_score else None,
        "by_n": {},
    }

    for n_value in n_values:
        if n_value > max_bank_size:
            raise ValueError(f"Requested N={n_value}, but prompt {prompt_index} only has {max_bank_size} samples.")
        bank_rows = list(prompt_rows[:n_value])
        critic_values = [float(row["critic_final_trajectory_value"]) for row in bank_rows]
        selected_positions = _best_positions(critic_values)
        selected_row = bank_rows[selected_positions[0]]

        payload: dict[str, Any] = {
            "n": int(n_value),
            "best_of_n_critic": _selected_payload(selected_row, task_score_key=task_score_key),
            "critic_tied_sample_indices": [int(bank_rows[position]["_sample_index_int"]) for position in selected_positions],
        }

        if not missing_task_score:
            current_task_scores = [float(row[task_score_key]) for row in bank_rows]
            oracle_positions = _best_positions(current_task_scores)
            oracle_row = bank_rows[oracle_positions[0]]
            payload["oracle_best_in_bank"] = _selected_payload(oracle_row, task_score_key=task_score_key)
            payload["selected_is_oracle_best"] = int(selected_row["_sample_index_int"]) in {
                int(bank_rows[position]["_sample_index_int"]) for position in oracle_positions
            }

        summary["by_n"][str(n_value)] = payload

    return summary


def _aggregate_summaries(
    *,
    prompt_summaries: Sequence[dict[str, Any]],
    n_values: Sequence[int],
) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    for n_value in n_values:
        selected_scores: list[float] = []
        oracle_scores: list[float] = []
        oracle_hit_flags: list[float] = []
        selected_lengths: list[float] = []
        selected_values: list[float] = []

        for prompt_summary in prompt_summaries:
            payload = prompt_summary["by_n"][str(n_value)]
            selected = payload["best_of_n_critic"]
            selected_values.append(float(selected["critic_final_trajectory_value"]))
            if selected.get("response_length") is not None:
                selected_lengths.append(float(selected["response_length"]))
            if "task_score" in selected:
                selected_scores.append(float(selected["task_score"]))
            if "oracle_best_in_bank" in payload:
                oracle_scores.append(float(payload["oracle_best_in_bank"]["task_score"]))
                oracle_hit_flags.append(1.0 if bool(payload["selected_is_oracle_best"]) else 0.0)

        row = {
            "n": int(n_value),
            "num_prompts": int(len(prompt_summaries)),
            "mean_selected_critic_value": _mean(selected_values),
            "mean_selected_response_length": _mean(selected_lengths),
            "mean_selected_task_score": _mean(selected_scores),
            "accuracy": _mean([1.0 if score == 1.0 else 0.0 for score in selected_scores]),
            "oracle_mean_task_score": _mean(oracle_scores),
            "oracle_accuracy": _mean([1.0 if score == 1.0 else 0.0 for score in oracle_scores]),
            "oracle_best_hit_rate": _mean(oracle_hit_flags),
        }
        aggregate_rows.append(row)
    return aggregate_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    response_bank_path = Path(args.response_bank_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_values = sorted({int(value) for value in args.n_values})
    if any(value <= 0 for value in n_values):
        raise ValueError(f"All N values must be positive, got {n_values}")

    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    critic_checkpoint_dir = Path(args.critic_checkpoint_dir)
    critic_hf_dir = ensure_merged_component_checkpoint(
        critic_checkpoint_dir,
        component="critic",
        merged_root=Path(args.critic_merged_root) if args.critic_merged_root else None,
        hf_source_dir=Path(args.critic_hf_source_dir) if args.critic_hf_source_dir else None,
        skip_merge=args.skip_merge,
    )

    tokenizer = load_tokenizer(critic_hf_dir, trust_remote_code=args.trust_remote_code)
    critic = load_critic_model(critic_hf_dir, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code)

    rows = _load_jsonl(response_bank_path)
    scored_path = output_dir / "critic_scored_response_bank.jsonl"
    scored_rows: list[dict[str, Any]] = []

    with scored_path.open("w", encoding="utf-8") as scored_file:
        for row in tqdm(rows, desc="Scoring response bank"):
            prompt_ids = _prompt_token_ids(
                row,
                tokenizer=tokenizer,
                prompt_key=args.prompt_key,
                max_prompt_length=args.max_prompt_length,
            )
            response_ids = _response_token_ids(
                row,
                tokenizer=tokenizer,
                response_key=args.response_key,
                response_token_ids_key=args.response_token_ids_key,
                max_response_tokens=args.max_response_tokens,
                retokenize_responses=args.retokenize_responses,
            )
            sequence_ids = prompt_ids + response_ids
            critic_value = _score_sequence(critic=critic, device=device, sequence_ids=sequence_ids)

            scored_row = _copy_original_fields(row, keep_all_original_fields=args.keep_all_original_fields)
            for required_field in (
                args.prompt_index_key,
                args.sample_index_key,
                args.task_score_key,
                args.prompt_key,
                args.response_key,
            ):
                _ensure_field_copy(scored_row, row, required_field)
            scored_row.update(
                {
                    "critic_checkpoint_dir": str(critic_checkpoint_dir),
                    "critic_hf_dir": str(critic_hf_dir),
                    "critic_final_trajectory_value": critic_value,
                    "critic_prompt_tokens_used": int(len(prompt_ids)),
                    "critic_response_tokens_used": int(len(response_ids)),
                    "critic_sequence_tokens_used": int(len(sequence_ids)),
                    "_source_line_number": int(row["_source_line_number"]),
                }
            )
            scored_rows.append(scored_row)
            scored_file.write(_json_line({key: value for key, value in scored_row.items() if not key.startswith("_")}))

    grouped = _group_by_prompt(
        scored_rows,
        prompt_index_key=args.prompt_index_key,
        sample_index_key=args.sample_index_key,
    )
    prompt_summaries = [
        _build_prompt_summary(
            prompt_index=prompt_index,
            prompt_rows=prompt_rows,
            n_values=n_values,
            task_score_key=args.task_score_key,
            allow_missing_task_score=args.allow_missing_task_score,
        )
        for prompt_index, prompt_rows in grouped
    ]
    aggregate_rows = _aggregate_summaries(prompt_summaries=prompt_summaries, n_values=n_values)

    prompt_summary_path = output_dir / "prompt_level_best_of_n.jsonl"
    with prompt_summary_path.open("w", encoding="utf-8") as output_file:
        for prompt_summary in prompt_summaries:
            output_file.write(_json_line(prompt_summary))

    _write_csv(output_dir / "best_of_n_summary.csv", aggregate_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "response_bank_path": str(response_bank_path),
        "critic_checkpoint_dir": str(critic_checkpoint_dir),
        "critic_hf_dir": str(critic_hf_dir),
        "output_dir": str(output_dir),
        "num_rows": int(len(scored_rows)),
        "num_prompts": int(len(prompt_summaries)),
        "bank_sizes": sorted({int(summary_row["max_bank_size"]) for summary_row in prompt_summaries}),
        "n_values": n_values,
        "dtype": args.dtype,
        "device": str(device),
        "max_prompt_length": int(args.max_prompt_length),
        "max_response_tokens": int(args.max_response_tokens),
        "retokenize_responses": bool(args.retokenize_responses),
        "metrics_by_n": aggregate_rows,
        "artifacts": {
            "critic_scored_response_bank": str(scored_path),
            "prompt_level_best_of_n": str(prompt_summary_path),
            "best_of_n_summary_csv": str(output_dir / "best_of_n_summary.csv"),
        },
    }
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(f"Wrote critic-scored bank to {scored_path}")
    print(f"Wrote prompt-level Best-of-N selections to {prompt_summary_path}")
    print(f"Wrote summary metrics to {output_dir / 'summary_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
