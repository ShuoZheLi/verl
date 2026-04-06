from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from queue import Empty
import random
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from value_decoding.data import ExampleRecord, load_examples, score_response
from value_decoding.decoding import (
    ActorSamplingMode,
    ActorStepper,
    critic_child_values,
    critic_last_token_values,
    sample_token_from_actor,
    set_decode_seed,
)
from value_decoding.multi_worker import (
    WorkerAssignment,
    build_worker_assignments,
    parse_worker_pairs,
    worker_assignments_to_jsonable,
)

BUCKET_ORDER = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight EOS-bias diagnostic on actor+critic checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="VERL checkpoint directory with actor/critic.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for diagnostic outputs.")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_final_eos_check", action="store_true")

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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


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


def _token_text(tokenizer, token_id: int) -> str:
    tokens = tokenizer.convert_ids_to_tokens([token_id])
    if tokens and tokens[0] is not None:
        return str(tokens[0])
    return str(token_id)


def _prompt_ids_for_example(
    example: ExampleRecord,
    *,
    tokenizer,
    max_prompt_length: int,
    device: torch.device,
) -> torch.Tensor:
    if example.prompt_token_ids is not None:
        prompt_ids = list(example.prompt_token_ids)
    else:
        tokenized = tokenizer(
            example.prompt_text,
            truncation=True,
            max_length=max_prompt_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        prompt_ids = tokenized["input_ids"]
    return torch.tensor([prompt_ids], dtype=torch.long, device=device)


def _bucket_step_indices(num_steps: int) -> dict[str, list[int]]:
    if num_steps <= 0:
        return {bucket: [] for bucket in BUCKET_ORDER}

    early_end = max(1, math.ceil(num_steps * 0.25))
    late_start = max(early_end, math.floor(num_steps * 0.75))
    return {
        "early": list(range(0, early_end)),
        "middle": list(range(early_end, late_start)),
        "late": list(range(late_start, num_steps)),
    }


def _sample_prefix_positions(num_steps: int, *, rng: random.Random) -> list[tuple[str, int]]:
    bucket_map = _bucket_step_indices(num_steps)
    selected: list[tuple[str, int]] = []
    for bucket in BUCKET_ORDER:
        candidates = bucket_map[bucket]
        if not candidates:
            continue
        selected.append((bucket, rng.choice(candidates)))
    return selected


def _primary_eos_token_id(tokenizer, eos_token_ids: tuple[int, ...]) -> int:
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    if eos_token_ids:
        return int(eos_token_ids[0])
    raise ValueError("No EOS token id could be resolved for the diagnostic.")


def _best_non_eos_token_id(actor_log_probs: torch.Tensor, eos_token_ids: set[int]) -> int | None:
    masked = actor_log_probs.clone()
    vocab_size = masked.shape[-1]
    valid_eos_ids = [token_id for token_id in eos_token_ids if 0 <= token_id < vocab_size]
    if valid_eos_ids:
        masked[..., valid_eos_ids] = float("-inf")
    best_id = int(torch.argmax(masked, dim=-1).item())
    best_value = float(masked[0, best_id].item())
    if not math.isfinite(best_value):
        return None
    return best_id


def _full_sequence_ids(prompt_ids: torch.Tensor, generated_token_ids: list[int], *, device: torch.device) -> torch.Tensor:
    if not generated_token_ids:
        return prompt_ids.to(device)
    generated = torch.tensor([generated_token_ids], dtype=prompt_ids.dtype, device=device)
    return torch.cat([prompt_ids.to(device), generated], dim=1)


def _generate_rollout(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    actor_device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: set[int],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    seed: int,
) -> dict[str, Any]:
    set_decode_seed(seed)
    prompt_ids = _prompt_ids_for_example(
        example,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        device=actor_device,
    )
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)

    generated_token_ids: list[int] = []
    eos_emitted = False
    start_time = time.perf_counter()
    for _step_index in range(max_new_tokens):
        selected_token_id = sample_token_from_actor(
            actor_state.current_logits.squeeze(0),
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

    latency_sec = time.perf_counter() - start_time
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = score_response(example, response_text)
    total_steps = len(generated_token_ids)

    return {
        "prompt_ids": prompt_ids.detach().clone(),
        "generated_token_ids": generated_token_ids,
        "response_text": response_text,
        "response_length": total_steps,
        "eos_emitted": eos_emitted,
        "max_length_hit": bool(max_new_tokens > 0 and not eos_emitted and total_steps >= max_new_tokens),
        "task_score": task_score,
        "latency_sec": latency_sec,
    }


def _analyze_prefixes_for_rollout(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    rollout: dict[str, Any],
    selected_positions: list[tuple[str, int]],
    actor_device: torch.device,
    critic_device: torch.device,
    max_prompt_length: int,
    eos_token_ids: tuple[int, ...],
    primary_eos_token_id: int,
    use_actor_cache: bool,
) -> tuple[list[dict[str, Any]], int]:
    generated_token_ids: list[int] = rollout["generated_token_ids"]
    if not selected_positions:
        return [], 0

    prompt_ids = _prompt_ids_for_example(
        example,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        device=actor_device,
    )
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)

    selected_positions_sorted = sorted(selected_positions, key=lambda item: item[1])
    eos_token_set = set(int(token_id) for token_id in eos_token_ids)
    rows: list[dict[str, Any]] = []
    skipped_no_non_eos = 0
    current_step = 0

    for bucket_name, prefix_step_index in selected_positions_sorted:
        while current_step < prefix_step_index:
            actor_state.append(generated_token_ids[current_step])
            current_step += 1

        prefix_ids_actor = actor_state.sequence_ids.detach().clone()
        prefix_ids_critic = prefix_ids_actor.to(critic_device)
        actor_log_probs = torch.log_softmax(actor_state.current_logits.float(), dim=-1)

        actor_top1_token_id = int(torch.argmax(actor_log_probs, dim=-1).item())
        actor_top1_is_eos = actor_top1_token_id in eos_token_set
        top1_non_eos_token_id = _best_non_eos_token_id(actor_log_probs, eos_token_set)
        if top1_non_eos_token_id is None:
            skipped_no_non_eos += 1
            continue

        eos_tensor = torch.tensor([primary_eos_token_id], dtype=prefix_ids_critic.dtype, device=critic_device)
        non_eos_tensor = torch.tensor([top1_non_eos_token_id], dtype=prefix_ids_critic.dtype, device=critic_device)

        value_prefix = float(critic_last_token_values(critic, prefix_ids_critic)[0].item())
        value_prefix_eos = float(critic_child_values(critic, prefix_ids_critic, eos_tensor)[0].item())
        value_prefix_top1_non_eos = float(critic_child_values(critic, prefix_ids_critic, non_eos_tensor)[0].item())

        rows.append(
            {
                "example_id": example.example_id,
                "data_source": example.data_source,
                "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
                "prefix_step_index": prefix_step_index,
                "prefix_length": int(prefix_ids_actor.shape[1]),
                "progress_bucket": bucket_name,
                "actor_top1_token_id": actor_top1_token_id,
                "actor_top1_token_text": _token_text(tokenizer, actor_top1_token_id),
                "actor_top1_is_eos": actor_top1_is_eos,
                "top1_non_eos_token_id": top1_non_eos_token_id,
                "top1_non_eos_token_text": _token_text(tokenizer, top1_non_eos_token_id),
                "value_prefix": value_prefix,
                "value_prefix_eos": value_prefix_eos,
                "value_prefix_top1_non_eos": value_prefix_top1_non_eos,
                "delta_eos_prefix": value_prefix_eos - value_prefix,
                "delta_eos_top1": value_prefix_eos - value_prefix_top1_non_eos,
                "final_task_score": float(rollout["task_score"]),
                "full_response_length": int(rollout["response_length"]),
                "full_eos_emitted": bool(rollout["eos_emitted"]),
                "full_max_length_hit": bool(rollout["max_length_hit"]),
            }
        )

    return rows, skipped_no_non_eos


def _compute_final_append_eos_delta(
    *,
    critic,
    prompt_ids: torch.Tensor,
    generated_token_ids: list[int],
    critic_device: torch.device,
    primary_eos_token_id: int,
) -> float:
    full_ids = _full_sequence_ids(prompt_ids, generated_token_ids, device=critic_device)
    final_value = float(critic_last_token_values(critic, full_ids)[0].item())
    eos_tensor = torch.tensor([primary_eos_token_id], dtype=full_ids.dtype, device=critic_device)
    appended_eos_value = float(critic_child_values(critic, full_ids, eos_tensor)[0].item())
    return appended_eos_value - final_value


def _worker_entry(
    *,
    assignment: WorkerAssignment,
    actor_hf_dir: str,
    critic_hf_dir: str,
    examples: list[ExampleRecord],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    primary_eos_token_id: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    enable_final_eos_check: bool,
    seed: int,
    worker_root: str,
    progress_queue,
) -> None:
    worker_dir = Path(worker_root) / f"worker_{assignment.worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = worker_dir / "prefix_eos_bias.jsonl"
    final_eos_path = worker_dir / "final_append_eos.jsonl"
    summary_path = worker_dir / "worker_summary.json"
    error_path = worker_dir / "worker_error.txt"

    try:
        start_time = time.perf_counter()
        actor_device = resolve_device(assignment.actor_device)
        critic_device = resolve_device(assignment.critic_device) if assignment.critic_device else actor_device
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

        local_examples = examples[assignment.example_start : assignment.example_end]
        worker_total_examples = len(local_examples)
        worker_completed_examples = 0
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_started",
                    "worker_id": assignment.worker_id,
                    "worker_total_examples": worker_total_examples,
                }
            )

        num_selected_prefix_slots = 0
        num_prefix_rows = 0
        num_skipped_prefixes_no_valid_non_eos = 0
        num_rollouts_with_nonempty_response = 0
        num_final_append_eos_rows = 0
        num_skipped_final_append_existing_eos = 0
        prefix_rows_per_bucket = {bucket: 0 for bucket in BUCKET_ORDER}

        with prefix_path.open("w", encoding="utf-8") as prefix_file, final_eos_path.open(
            "w",
            encoding="utf-8",
        ) as final_eos_file:
            for example in local_examples:
                rollout_seed = seed + example.example_id * 1_000_003
                rollout = _generate_rollout(
                    actor=actor,
                    tokenizer=tokenizer,
                    example=example,
                    actor_device=actor_device,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=set(eos_token_ids),
                    actor_sampling_mode=actor_sampling_mode,
                    actor_temperature=actor_temperature,
                    actor_top_p=actor_top_p,
                    actor_top_k=actor_top_k,
                    use_actor_cache=use_actor_cache,
                    seed=rollout_seed,
                )

                if rollout["response_length"] > 0:
                    num_rollouts_with_nonempty_response += 1

                selected_positions = _sample_prefix_positions(
                    rollout["response_length"],
                    rng=random.Random(rollout_seed + 17),
                )
                num_selected_prefix_slots += len(selected_positions)

                prefix_rows, skipped_count = _analyze_prefixes_for_rollout(
                    actor=actor,
                    critic=critic,
                    tokenizer=tokenizer,
                    example=example,
                    rollout=rollout,
                    selected_positions=selected_positions,
                    actor_device=actor_device,
                    critic_device=critic_device,
                    max_prompt_length=max_prompt_length,
                    eos_token_ids=eos_token_ids,
                    primary_eos_token_id=primary_eos_token_id,
                    use_actor_cache=use_actor_cache,
                )
                num_skipped_prefixes_no_valid_non_eos += skipped_count
                for row in prefix_rows:
                    prefix_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                    num_prefix_rows += 1
                    prefix_rows_per_bucket[row["progress_bucket"]] += 1

                if enable_final_eos_check and not rollout["eos_emitted"]:
                    delta_append_final_eos = _compute_final_append_eos_delta(
                        critic=critic,
                        prompt_ids=rollout["prompt_ids"],
                        generated_token_ids=rollout["generated_token_ids"],
                        critic_device=critic_device,
                        primary_eos_token_id=primary_eos_token_id,
                    )
                    final_row = {
                        "example_id": example.example_id,
                        "data_source": example.data_source,
                        "final_task_score": float(rollout["task_score"]),
                        "full_response_length": int(rollout["response_length"]),
                        "full_eos_emitted": bool(rollout["eos_emitted"]),
                        "full_max_length_hit": bool(rollout["max_length_hit"]),
                        "delta_append_final_eos": delta_append_final_eos,
                    }
                    final_eos_file.write(json.dumps(final_row, ensure_ascii=True) + "\n")
                    num_final_append_eos_rows += 1
                elif enable_final_eos_check:
                    num_skipped_final_append_existing_eos += 1

                worker_completed_examples += 1
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "type": "example_done",
                            "worker_id": assignment.worker_id,
                            "worker_completed_examples": worker_completed_examples,
                            "worker_total_examples": worker_total_examples,
                        }
                    )

        summary = {
            "worker_id": assignment.worker_id,
            "actor_device": str(actor_device),
            "critic_device": str(critic_device),
            "example_start": assignment.example_start,
            "example_end": assignment.example_end,
            "num_examples": assignment.num_examples,
            "num_rollouts_with_nonempty_response": num_rollouts_with_nonempty_response,
            "num_selected_prefix_slots": num_selected_prefix_slots,
            "num_prefix_rows": num_prefix_rows,
            "prefix_rows_per_bucket": prefix_rows_per_bucket,
            "num_skipped_prefixes_no_valid_non_eos": num_skipped_prefixes_no_valid_non_eos,
            "num_final_append_eos_rows": num_final_append_eos_rows,
            "num_skipped_final_append_existing_eos": num_skipped_final_append_existing_eos,
            "runtime_sec": time.perf_counter() - start_time,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=True, indent=2)
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_done",
                    "worker_id": assignment.worker_id,
                    "worker_completed_examples": worker_completed_examples,
                    "worker_total_examples": worker_total_examples,
                }
            )
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_error",
                    "worker_id": assignment.worker_id,
                    "traceback": traceback.format_exc(),
                }
            )
        raise


def _collect_worker_outputs(
    *,
    worker_root: Path,
    assignments: list[WorkerAssignment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prefix_rows: list[dict[str, Any]] = []
    final_eos_rows: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []

    for assignment in assignments:
        worker_dir = worker_root / f"worker_{assignment.worker_id:03d}"
        prefix_path = worker_dir / "prefix_eos_bias.jsonl"
        final_eos_path = worker_dir / "final_append_eos.jsonl"
        summary_path = worker_dir / "worker_summary.json"

        if prefix_path.exists():
            with prefix_path.open("r", encoding="utf-8") as prefix_file:
                for line in prefix_file:
                    if line.strip():
                        prefix_rows.append(json.loads(line))

        if final_eos_path.exists():
            with final_eos_path.open("r", encoding="utf-8") as final_eos_file:
                for line in final_eos_file:
                    if line.strip():
                        final_eos_rows.append(json.loads(line))

        with summary_path.open("r", encoding="utf-8") as summary_file:
            worker_summaries.append(json.load(summary_file))

    worker_summaries.sort(key=lambda item: int(item["worker_id"]))
    prefix_rows.sort(key=lambda row: (int(row["example_id"]), int(row["prefix_step_index"]), BUCKET_ORDER.index(row["progress_bucket"])))
    final_eos_rows.sort(key=lambda row: int(row["example_id"]))
    return prefix_rows, final_eos_rows, worker_summaries


def _run_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path,
    examples: list[ExampleRecord],
    worker_pairs: list[tuple[str | None, str | None]],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    primary_eos_token_id: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    enable_final_eos_check: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[WorkerAssignment]]:
    assignments = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)
    if not assignments:
        raise ValueError("No worker assignments were created.")

    worker_root = output_dir / "_worker_tmp"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[tuple[mp.Process, WorkerAssignment]] = []
    for assignment in assignments:
        process = context.Process(
            target=_worker_entry,
            kwargs={
                "assignment": assignment,
                "actor_hf_dir": str(actor_hf_dir),
                "critic_hf_dir": str(critic_hf_dir),
                "examples": examples,
                "dtype_name": dtype_name,
                "trust_remote_code": trust_remote_code,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "eos_token_ids": eos_token_ids,
                "primary_eos_token_id": primary_eos_token_id,
                "actor_sampling_mode": actor_sampling_mode,
                "actor_temperature": actor_temperature,
                "actor_top_p": actor_top_p,
                "actor_top_k": actor_top_k,
                "use_actor_cache": use_actor_cache,
                "enable_final_eos_check": enable_final_eos_check,
                "seed": seed,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
            },
            name=f"eos_bias_worker_{assignment.worker_id}",
        )
        process.start()
        processes.append((process, assignment))

    total_examples = len(examples)
    completed_examples = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, int]] = {
        assignment.worker_id: {"done": 0, "total": assignment.num_examples} for assignment in assignments
    }

    with tqdm(total=total_examples, desc="eos_bias_diag", unit="example", dynamic_ncols=True) as progress_bar:
        while completed_examples < total_examples or completed_workers < len(assignments):
            try:
                event = progress_queue.get(timeout=0.2)
            except Empty:
                for process, assignment in processes:
                    if process.exitcode not in (None, 0):
                        error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
                        if error_path.exists():
                            error_text = error_path.read_text(encoding="utf-8")
                            raise RuntimeError(
                                f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n{error_text}"
                            )
                        raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")
                continue

            event_type = event.get("type")
            worker_id = int(event.get("worker_id", -1))
            if event_type == "example_done":
                completed_examples += 1
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["done"] = int(event.get("worker_completed_examples", 0))
                worker_progress[worker_id]["total"] = int(event.get("worker_total_examples", 0))
                progress_bar.update(1)
            elif event_type == "worker_done":
                completed_workers += 1
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["done"] = int(event.get("worker_completed_examples", 0))
                worker_progress[worker_id]["total"] = int(event.get("worker_total_examples", 0))
            elif event_type == "worker_started":
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["total"] = int(event.get("worker_total_examples", 0))
            elif event_type == "worker_error":
                raise RuntimeError(
                    f"Worker {worker_id} reported an error.\n{event.get('traceback', 'No traceback provided.')}"
                )

            postfix = " | ".join(
                f"w{worker_id}:{state.get('done', 0)}/{state.get('total', 0)}"
                for worker_id, state in sorted(worker_progress.items())
            )
            progress_bar.set_postfix_str(postfix)

    for process, assignment in processes:
        process.join()
        if process.exitcode != 0:
            error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
            if error_path.exists():
                error_text = error_path.read_text(encoding="utf-8")
                raise RuntimeError(
                    f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n{error_text}"
                )
            raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")

    prefix_rows, final_eos_rows, worker_summaries = _collect_worker_outputs(worker_root=worker_root, assignments=assignments)
    return prefix_rows, final_eos_rows, worker_summaries, assignments


def _run_single_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path,
    examples: list[ExampleRecord],
    worker_pairs: list[tuple[str | None, str | None]],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    primary_eos_token_id: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    enable_final_eos_check: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[WorkerAssignment]]:
    assignment = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)[0]
    actor_device = resolve_device(worker_pairs[0][0])
    critic_device = resolve_device(worker_pairs[0][1]) if worker_pairs[0][1] else actor_device
    dtype = resolve_dtype(dtype_name)

    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=trust_remote_code)
    actor = load_actor_model(actor_hf_dir, dtype=dtype, device=actor_device, trust_remote_code=trust_remote_code)
    critic = load_critic_model(critic_hf_dir, dtype=dtype, device=critic_device, trust_remote_code=trust_remote_code)

    prefix_rows: list[dict[str, Any]] = []
    final_eos_rows: list[dict[str, Any]] = []
    prefix_rows_per_bucket = {bucket: 0 for bucket in BUCKET_ORDER}
    num_rollouts_with_nonempty_response = 0
    num_selected_prefix_slots = 0
    num_skipped_prefixes_no_valid_non_eos = 0
    num_skipped_final_append_existing_eos = 0
    start_time = time.perf_counter()

    with tqdm(total=len(examples), desc="eos_bias_diag", unit="example", dynamic_ncols=True) as progress_bar:
        for example in examples:
            rollout_seed = seed + example.example_id * 1_000_003
            rollout = _generate_rollout(
                actor=actor,
                tokenizer=tokenizer,
                example=example,
                actor_device=actor_device,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                eos_token_ids=set(eos_token_ids),
                actor_sampling_mode=actor_sampling_mode,
                actor_temperature=actor_temperature,
                actor_top_p=actor_top_p,
                actor_top_k=actor_top_k,
                use_actor_cache=use_actor_cache,
                seed=rollout_seed,
            )

            if rollout["response_length"] > 0:
                num_rollouts_with_nonempty_response += 1

            selected_positions = _sample_prefix_positions(
                rollout["response_length"],
                rng=random.Random(rollout_seed + 17),
            )
            num_selected_prefix_slots += len(selected_positions)

            example_prefix_rows, skipped_count = _analyze_prefixes_for_rollout(
                actor=actor,
                critic=critic,
                tokenizer=tokenizer,
                example=example,
                rollout=rollout,
                selected_positions=selected_positions,
                actor_device=actor_device,
                critic_device=critic_device,
                max_prompt_length=max_prompt_length,
                eos_token_ids=eos_token_ids,
                primary_eos_token_id=primary_eos_token_id,
                use_actor_cache=use_actor_cache,
            )
            num_skipped_prefixes_no_valid_non_eos += skipped_count
            prefix_rows.extend(example_prefix_rows)
            for row in example_prefix_rows:
                prefix_rows_per_bucket[row["progress_bucket"]] += 1

            if enable_final_eos_check and not rollout["eos_emitted"]:
                delta_append_final_eos = _compute_final_append_eos_delta(
                    critic=critic,
                    prompt_ids=rollout["prompt_ids"],
                    generated_token_ids=rollout["generated_token_ids"],
                    critic_device=critic_device,
                    primary_eos_token_id=primary_eos_token_id,
                )
                final_eos_rows.append(
                    {
                        "example_id": example.example_id,
                        "data_source": example.data_source,
                        "final_task_score": float(rollout["task_score"]),
                        "full_response_length": int(rollout["response_length"]),
                        "full_eos_emitted": bool(rollout["eos_emitted"]),
                        "full_max_length_hit": bool(rollout["max_length_hit"]),
                        "delta_append_final_eos": delta_append_final_eos,
                    }
                )
            elif enable_final_eos_check:
                num_skipped_final_append_existing_eos += 1
            progress_bar.update(1)

    prefix_rows.sort(key=lambda row: (int(row["example_id"]), int(row["prefix_step_index"]), BUCKET_ORDER.index(row["progress_bucket"])))
    final_eos_rows.sort(key=lambda row: int(row["example_id"]))

    worker_summary = {
        "worker_id": 0,
        "actor_device": str(actor_device),
        "critic_device": str(critic_device),
        "example_start": 0,
        "example_end": len(examples),
        "num_examples": len(examples),
        "num_rollouts_with_nonempty_response": num_rollouts_with_nonempty_response,
        "num_selected_prefix_slots": num_selected_prefix_slots,
        "num_prefix_rows": len(prefix_rows),
        "prefix_rows_per_bucket": prefix_rows_per_bucket,
        "num_skipped_prefixes_no_valid_non_eos": num_skipped_prefixes_no_valid_non_eos,
        "num_final_append_eos_rows": len(final_eos_rows),
        "num_skipped_final_append_existing_eos": num_skipped_final_append_existing_eos,
        "runtime_sec": time.perf_counter() - start_time,
    }
    return prefix_rows, final_eos_rows, [worker_summary], [assignment]


def _aggregate_subset(prefix_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(prefix_rows),
        "mean_delta_eos_prefix": _mean([float(row["delta_eos_prefix"]) for row in prefix_rows]),
        "mean_delta_eos_top1": _mean([float(row["delta_eos_top1"]) for row in prefix_rows]),
    }


def _aggregate_bucketed(prefix_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        bucket: _aggregate_subset([row for row in prefix_rows if row["progress_bucket"] == bucket])
        for bucket in BUCKET_ORDER
    }


def _build_summary_payload(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    worker_pairs: list[tuple[str | None, str | None]],
    worker_assignments: list[WorkerAssignment],
    worker_summaries: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    final_eos_rows: list[dict[str, Any]],
    eos_token_ids: tuple[int, ...],
    primary_eos_token_id: int,
) -> dict[str, Any]:
    prefix_rows_actor_top1_is_eos = [row for row in prefix_rows if bool(row["actor_top1_is_eos"])]
    prefix_rows_actor_top1_not_eos = [row for row in prefix_rows if not bool(row["actor_top1_is_eos"])]
    multi_worker_enabled = len(worker_assignments) > 1
    resolved_actor_device = None if multi_worker_enabled else worker_summaries[0].get("actor_device")
    resolved_critic_device = None if multi_worker_enabled else worker_summaries[0].get("critic_device")

    counts = {
        "num_examples_loaded": sum(int(summary["num_examples"]) for summary in worker_summaries),
        "num_rollouts_with_nonempty_response": sum(int(summary["num_rollouts_with_nonempty_response"]) for summary in worker_summaries),
        "num_selected_prefix_slots": sum(int(summary["num_selected_prefix_slots"]) for summary in worker_summaries),
        "num_prefix_rows": len(prefix_rows),
        "num_prefix_rows_per_bucket": {
            bucket: sum(int(summary["prefix_rows_per_bucket"].get(bucket, 0)) for summary in worker_summaries)
            for bucket in BUCKET_ORDER
        },
        "num_prefix_rows_actor_top1_is_eos": len(prefix_rows_actor_top1_is_eos),
        "num_prefix_rows_actor_top1_not_eos": len(prefix_rows_actor_top1_not_eos),
        "num_skipped_prefixes_no_valid_non_eos": sum(
            int(summary["num_skipped_prefixes_no_valid_non_eos"]) for summary in worker_summaries
        ),
        "num_final_append_eos_rows": len(final_eos_rows),
        "num_skipped_final_append_existing_eos": sum(
            int(summary.get("num_skipped_final_append_existing_eos", 0)) for summary in worker_summaries
        ),
    }

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
        "actor_device": resolved_actor_device,
        "critic_device": resolved_critic_device,
        "worker_pairs": [[actor, critic] for actor, critic in worker_pairs],
        "worker_assignments": worker_assignments_to_jsonable(worker_assignments),
        "worker_summaries": worker_summaries,
        "dtype": args.dtype,
        "eos_token_ids": list(eos_token_ids),
        "primary_eos_token_id": primary_eos_token_id,
        "run_args": vars(args),
        "prefix_sampling": {
            "prefix_step_index_definition": (
                "0-based decode step before choosing the next generated token; "
                "step 0 corresponds to the prompt-only prefix."
            ),
            "bucket_policy": {
                "early": "first 25% of decode steps",
                "middle": "middle 50% of decode steps",
                "late": "last 25% of decode steps",
            },
            "selection_rule": "At most one uniformly sampled prefix per non-empty bucket per example.",
        },
        "counts": counts,
        "global_means": _aggregate_subset(prefix_rows),
        "bucketed_means": _aggregate_bucketed(prefix_rows),
        "eos_top1_subsets": {
            "actor_top1_is_eos": {
                "global": _aggregate_subset(prefix_rows_actor_top1_is_eos),
                "bucketed": _aggregate_bucketed(prefix_rows_actor_top1_is_eos),
            },
            "actor_top1_is_not_eos": {
                "global": _aggregate_subset(prefix_rows_actor_top1_not_eos),
                "bucketed": _aggregate_bucketed(prefix_rows_actor_top1_not_eos),
            },
        },
        "final_append_eos_check": {
            "enabled": not args.skip_final_eos_check,
            "count": len(final_eos_rows),
            "note": "Computed only for rollouts that did not already emit EOS.",
            "mean_delta_append_final_eos": _mean(
                [float(row["delta_append_final_eos"]) for row in final_eos_rows]
            ) if final_eos_rows else None,
        },
        "interpretation_guide": [
            "If delta_eos_prefix is consistently positive even in early prefixes, that suggests EOS-token bias.",
            "If delta_eos_top1 is positive in early or middle prefixes, that is stronger evidence that the critic prefers EOS over ordinary continuation.",
            "If EOS only becomes favorable in late prefixes, that is more consistent with normal completion behavior than pathological EOS bias.",
        ],
    }
    return summary_payload


def _build_readme(summary_payload: dict[str, Any]) -> str:
    global_means = summary_payload["global_means"]
    final_eos = summary_payload["final_append_eos_check"]
    counts = summary_payload["counts"]
    return f"""# EOS Bias Diagnostic

This directory contains a lightweight analysis-only diagnostic for whether the critic appears to assign extra value to EOS termination.

## What Was Measured

For sampled prefixes `s` from normal actor-generated rollouts, the diagnostic computes:

- `V(s)`
- `V(s + EOS)`
- `V(s + top1_nonEOS)`

and reports:

- `delta_eos_prefix = V(s + EOS) - V(s)`
- `delta_eos_top1 = V(s + EOS) - V(s + top1_nonEOS)`

The critic value is always read at the last token position of the scored sequence.

## How Prefixes Were Sampled

- One normal actor rollout was generated per evaluation example.
- Decode steps were bucketed into:
  - `early`: first 25% of decode steps
  - `middle`: middle 50% of decode steps
  - `late`: last 25% of decode steps
- At most one prefix was sampled from each non-empty bucket per example.
- Prefix step index `0` corresponds to the prompt-only prefix, i.e. before the first generated token.
- Prefixes with no valid non-EOS continuation token were skipped.

## What The Files Mean

- `prefix_eos_bias.jsonl`: one analyzed prefix per row
- `final_append_eos.jsonl`: one analyzed non-EOS-ended rollout per row for the optional final-EOS append check
- `summary_metrics.json`: aggregated means, bucketed means, and EOS-top1 subset statistics
- `README.md`: this explanation

## Key Numbers From This Run

- analyzed prefixes: {counts['num_prefix_rows']}
- mean `delta_eos_prefix`: {global_means['mean_delta_eos_prefix']}
- mean `delta_eos_top1`: {global_means['mean_delta_eos_top1']}
- final-EOS append check count: {final_eos['count']}
- mean `delta_append_final_eos`: {final_eos['mean_delta_append_final_eos']}

## Interpretation Guide

- If `delta_eos_prefix` is consistently positive even in `early` prefixes, that suggests EOS-token bias.
- If `delta_eos_top1` is positive in `early` or `middle` prefixes, that is stronger evidence that the critic prefers EOS over ordinary continuation.
- If EOS only becomes favorable in `late` prefixes, that is more consistent with normal completion behavior than pathological EOS bias.
"""


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
    del dtype  # resolved here to fail fast on invalid input; workers resolve again locally.

    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, tokenizer)
    primary_eos_token_id = _primary_eos_token_id(tokenizer, eos_token_ids)

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

    worker_pairs = parse_worker_pairs(
        args.worker_pairs,
        actor_device=args.actor_device,
        critic_device=args.critic_device,
        default_device=args.device,
    )
    use_actor_cache = not args.disable_actor_cache
    enable_final_eos_check = not args.skip_final_eos_check

    if len(build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)) > 1:
        prefix_rows, final_eos_rows, worker_summaries, worker_assignments = _run_multi_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            worker_pairs=worker_pairs,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            primary_eos_token_id=primary_eos_token_id,
            actor_sampling_mode=args.actor_sampling_mode,
            actor_temperature=args.actor_temperature,
            actor_top_p=args.actor_top_p,
            actor_top_k=args.actor_top_k,
            use_actor_cache=use_actor_cache,
            enable_final_eos_check=enable_final_eos_check,
            seed=args.seed,
        )
    else:
        prefix_rows, final_eos_rows, worker_summaries, worker_assignments = _run_single_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            worker_pairs=worker_pairs,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            primary_eos_token_id=primary_eos_token_id,
            actor_sampling_mode=args.actor_sampling_mode,
            actor_temperature=args.actor_temperature,
            actor_top_p=args.actor_top_p,
            actor_top_k=args.actor_top_k,
            use_actor_cache=use_actor_cache,
            enable_final_eos_check=enable_final_eos_check,
            seed=args.seed,
        )

    prefix_jsonl_path = output_dir / "prefix_eos_bias.jsonl"
    with prefix_jsonl_path.open("w", encoding="utf-8") as prefix_file:
        for row in prefix_rows:
            prefix_file.write(json.dumps(row, ensure_ascii=True) + "\n")

    final_eos_jsonl_path = output_dir / "final_append_eos.jsonl"
    with final_eos_jsonl_path.open("w", encoding="utf-8") as final_eos_file:
        for row in final_eos_rows:
            final_eos_file.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary_payload = _build_summary_payload(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        actor_hf_dir=actor_hf_dir,
        critic_hf_dir=critic_hf_dir,
        output_dir=output_dir,
        args=args,
        worker_pairs=worker_pairs,
        worker_assignments=worker_assignments,
        worker_summaries=worker_summaries,
        prefix_rows=prefix_rows,
        final_eos_rows=final_eos_rows,
        eos_token_ids=eos_token_ids,
        primary_eos_token_id=primary_eos_token_id,
    )

    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    readme_path = output_dir / "README.md"
    readme_path.write_text(_build_readme(summary_payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
