from __future__ import annotations

import json
import multiprocessing as mp
from queue import Empty
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from value_decoding.checkpointing import (
    load_actor_model,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
)
from value_decoding.data import ExampleRecord
from value_decoding.decoding import RunSpec, decode_example


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    actor_device: str | None
    critic_device: str | None
    example_start: int
    example_end: int

    @property
    def num_examples(self) -> int:
        return max(self.example_end - self.example_start, 0)


def parse_worker_pairs(
    worker_pairs: list[str] | None,
    *,
    actor_device: str | None,
    critic_device: str | None,
    default_device: str | None,
) -> list[tuple[str | None, str | None]]:
    if worker_pairs:
        parsed: list[tuple[str | None, str | None]] = []
        for raw_pair in worker_pairs:
            value = raw_pair.strip()
            if not value:
                continue
            if "," in value:
                actor, critic = value.split(",", maxsplit=1)
                actor = actor.strip() or None
                critic = critic.strip() or None
            else:
                actor = value
                critic = value
            parsed.append((actor, critic))
        if not parsed:
            raise ValueError("--worker_pairs was provided, but no valid worker entries were parsed.")
        return parsed

    resolved_actor = actor_device or default_device
    resolved_critic = critic_device or default_device or resolved_actor
    return [(resolved_actor, resolved_critic)]


def build_worker_assignments(
    *,
    num_examples: int,
    worker_pairs: list[tuple[str | None, str | None]],
) -> list[WorkerAssignment]:
    if not worker_pairs:
        raise ValueError("At least one worker pair is required.")
    if num_examples <= 0:
        return []

    active_workers = min(len(worker_pairs), num_examples)
    assignments: list[WorkerAssignment] = []
    start = 0
    base = num_examples // active_workers
    remainder = num_examples % active_workers

    for worker_id in range(active_workers):
        shard_size = base + (1 if worker_id < remainder else 0)
        end = start + shard_size
        actor_dev, critic_dev = worker_pairs[worker_id]
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                actor_device=actor_dev,
                critic_device=critic_dev,
                example_start=start,
                example_end=end,
            )
        )
        start = end

    return assignments


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True) + "\n"


def _worker_entry(
    *,
    assignment: WorkerAssignment,
    actor_hf_dir: str,
    critic_hf_dir: str,
    examples: list[ExampleRecord],
    run_specs: list[RunSpec],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    use_actor_cache: bool,
    debug_full_candidates: bool,
    seed: int,
    worker_root: str,
    progress_queue,
) -> None:
    worker_dir = Path(worker_root) / f"worker_{assignment.worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
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
        worker_total_tasks = len(local_examples) * len(run_specs)
        worker_completed_tasks = 0
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_started",
                    "worker_id": assignment.worker_id,
                    "worker_total_tasks": worker_total_tasks,
                }
            )
        per_config_counts: dict[str, int] = {}
        per_config_start_wall_time_sec: dict[str, float] = {}
        per_config_end_wall_time_sec: dict[str, float] = {}
        per_config_runtime_sec: dict[str, float] = {}
        for spec_index, spec in enumerate(run_specs):
            config_start_perf = time.perf_counter()
            config_start_wall = time.time()
            per_example_path = worker_dir / f"per_example__{spec.config_id}.jsonl"
            step_level_path = worker_dir / f"step_level__{spec.config_id}.jsonl"
            count = 0
            with per_example_path.open("w", encoding="utf-8") as per_example_file, step_level_path.open(
                "w",
                encoding="utf-8",
            ) as step_level_file:
                for example in local_examples:
                    decode_seed = seed + spec_index * 1_000_003 + example.example_id
                    artifacts = decode_example(
                        actor=actor,
                        critic=critic,
                        tokenizer=tokenizer,
                        example=example,
                        run_spec=spec,
                        max_prompt_length=max_prompt_length,
                        max_new_tokens=max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        actor_device=actor_device,
                        critic_device=critic_device,
                        seed=decode_seed,
                        normalization_eps=normalization_eps,
                        use_actor_cache=use_actor_cache,
                        debug_full_candidates=debug_full_candidates,
                    )
                    per_example_file.write(_json_line(artifacts.example_result))
                    for step_result in artifacts.step_results:
                        step_level_file.write(_json_line(step_result))
                    count += 1
                    worker_completed_tasks += 1
                    if progress_queue is not None:
                        progress_queue.put(
                            {
                                "type": "task_done",
                                "worker_id": assignment.worker_id,
                                "config_id": spec.config_id,
                                "worker_completed_tasks": worker_completed_tasks,
                                "worker_total_tasks": worker_total_tasks,
                            }
                        )
            per_config_counts[spec.config_id] = count
            per_config_start_wall_time_sec[spec.config_id] = config_start_wall
            per_config_end_wall_time_sec[spec.config_id] = time.time()
            per_config_runtime_sec[spec.config_id] = time.perf_counter() - config_start_perf

        summary = {
            "worker_id": assignment.worker_id,
            "actor_device": str(actor_device),
            "critic_device": str(critic_device),
            "example_start": assignment.example_start,
            "example_end": assignment.example_end,
            "num_examples": assignment.num_examples,
            "num_run_specs": len(run_specs),
            "per_config_counts": per_config_counts,
            "per_config_start_wall_time_sec": per_config_start_wall_time_sec,
            "per_config_end_wall_time_sec": per_config_end_wall_time_sec,
            "per_config_runtime_sec": per_config_runtime_sec,
            "runtime_sec": time.perf_counter() - start_time,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=True, indent=2)
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_done",
                    "worker_id": assignment.worker_id,
                    "worker_completed_tasks": worker_completed_tasks,
                    "worker_total_tasks": worker_total_tasks,
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


def _progress_postfix(worker_progress: dict[int, dict[str, Any]]) -> str:
    parts: list[str] = []
    for worker_id in sorted(worker_progress):
        state = worker_progress[worker_id]
        done = int(state.get("done", 0))
        total = int(state.get("total", 0))
        config_id = state.get("config_id")
        if config_id:
            parts.append(f"w{worker_id}:{done}/{total} {config_id}")
        else:
            parts.append(f"w{worker_id}:{done}/{total}")
    return " | ".join(parts)


def run_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path,
    examples: list[ExampleRecord],
    run_specs: list[RunSpec],
    worker_pairs: list[tuple[str | None, str | None]],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    use_actor_cache: bool,
    debug_full_candidates: bool,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
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
                "run_specs": run_specs,
                "dtype_name": dtype_name,
                "trust_remote_code": trust_remote_code,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "eos_token_ids": eos_token_ids,
                "normalization_eps": normalization_eps,
                "use_actor_cache": use_actor_cache,
                "debug_full_candidates": debug_full_candidates,
                "seed": seed,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
            },
            name=f"value_decoding_worker_{assignment.worker_id}",
        )
        process.start()
        processes.append((process, assignment))

    total_tasks = len(examples) * len(run_specs)
    completed_tasks = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {
            "done": 0,
            "total": assignment.num_examples * len(run_specs),
            "config_id": None,
        }
        for assignment in assignments
    }

    with tqdm(total=total_tasks, desc="whole_experiment", unit="task", dynamic_ncols=True) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_tasks < total_tasks or completed_workers < len(assignments):
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
            if event_type == "worker_started":
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
            elif event_type == "task_done":
                completed_tasks += 1
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["done"] = int(event.get("worker_completed_tasks", 0))
                worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
                worker_progress[worker_id]["config_id"] = str(event.get("config_id"))
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

    per_example_path = output_dir / "per_example_results.jsonl"
    step_level_path = output_dir / "step_level_minimal.jsonl"
    example_results_by_config: dict[str, list[dict[str, Any]]] = {spec.config_id: [] for spec in run_specs}

    with per_example_path.open("w", encoding="utf-8") as per_example_file, step_level_path.open(
        "w",
        encoding="utf-8",
    ) as step_level_file:
        for spec in run_specs:
            for assignment in assignments:
                worker_dir = worker_root / f"worker_{assignment.worker_id:03d}"
                worker_example_path = worker_dir / f"per_example__{spec.config_id}.jsonl"
                worker_step_path = worker_dir / f"step_level__{spec.config_id}.jsonl"

                with worker_example_path.open("r", encoding="utf-8") as worker_example_file:
                    for line in worker_example_file:
                        if not line.strip():
                            continue
                        per_example_file.write(line)
                        example_results_by_config[spec.config_id].append(json.loads(line))

                with worker_step_path.open("r", encoding="utf-8") as worker_step_file:
                    shutil.copyfileobj(worker_step_file, step_level_file)

    worker_summaries: list[dict[str, Any]] = []
    for assignment in assignments:
        summary_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_summary.json"
        with summary_path.open("r", encoding="utf-8") as summary_file:
            worker_summaries.append(json.load(summary_file))

    worker_summaries.sort(key=lambda item: int(item["worker_id"]))
    return example_results_by_config, worker_summaries


def worker_assignments_to_jsonable(assignments: list[WorkerAssignment]) -> list[dict[str, Any]]:
    return [asdict(assignment) for assignment in assignments]
