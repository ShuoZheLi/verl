from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
    load_actor_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.data import ExampleRecord, load_examples, score_response
from value_decoding.decoding import ActorStepper, set_decode_seed

try:
    import ray
except ImportError:
    ray = None


RAY_PROGRESS_POLL_INTERVAL_SEC = 2.0
RAY_NODE_RESOURCE_FRACTION = 0.001


@dataclass(frozen=True)
class GreedyResult:
    example_id: int
    prompt_length: int
    response: str
    response_token_ids: tuple[int, ...]
    response_length: int
    ended_with_eos: bool
    hit_max_length: bool
    score: float
    elapsed_sec: float


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    prompt_start: int
    prompt_end: int
    device_name: str | None
    node_index: int | None = None
    node_ip: str | None = None
    node_resource_key: str | None = None

    @property
    def num_prompts(self) -> int:
        return max(0, self.prompt_end - self.prompt_start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate actor accuracy with deterministic greedy decoding.")
    parser.add_argument("--actor_checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--actor_name", type=str, default="actor")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, required=True)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--worker_devices", nargs="+", default=None)
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help="Optional Ray cluster address. Use 'auto' to read $RAY_ADDRESS.",
    )
    parser.add_argument(
        "--ray_num_cpus_per_worker",
        type=float,
        default=1.0,
        help="CPU resources reserved per Ray worker task when --ray_address is used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--actor_merged_root", type=str, default=None)
    parser.add_argument("--actor_hf_source_dir", type=str, default=None)
    parser.add_argument(
        "--omit_prompt_text",
        action="store_true",
        help="Exclude full prompt text from per-example JSONL rows.",
    )
    parser.add_argument(
        "--omit_response_token_ids",
        action="store_true",
        help="Exclude generated response token ids from per-example JSONL rows.",
    )
    parser.add_argument(
        "--disable_actor_cache",
        action="store_true",
        help="Disable KV-cache decoding. This is slower but can help unusual model implementations.",
    )
    return parser.parse_args()


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True) + "\n"


def _resolve_ray_address(ray_address: str | None) -> str | None:
    if ray_address is None:
        return None
    normalized = str(ray_address).strip()
    if not normalized:
        return None
    if normalized.lower() == "auto":
        import os

        env_address = os.environ.get("RAY_ADDRESS")
        if not env_address:
            raise ValueError("--ray_address=auto was requested, but $RAY_ADDRESS is not set.")
        return env_address
    return normalized


def _assignment_ranges(*, num_items: int, num_workers: int) -> list[tuple[int, int]]:
    if num_items <= 0:
        return []
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0 when examples are present.")

    active_workers = min(num_workers, num_items)
    ranges: list[tuple[int, int]] = []
    start = 0
    base = num_items // active_workers
    remainder = num_items % active_workers
    for worker_index in range(active_workers):
        size = base + (1 if worker_index < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _build_local_assignments(*, num_examples: int, worker_devices: Sequence[str | None]) -> list[WorkerAssignment]:
    assignments: list[WorkerAssignment] = []
    for worker_id, (start, end) in enumerate(
        _assignment_ranges(num_items=num_examples, num_workers=len(worker_devices))
    ):
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                prompt_start=start,
                prompt_end=end,
                device_name=worker_devices[worker_id],
            )
        )
    return assignments


def _resolve_ray_node_resource_key(node_payload: dict[str, Any]) -> str:
    resources = node_payload.get("Resources") or {}
    node_ip = str(node_payload.get("NodeManagerAddress") or "").strip()
    direct_key = f"node:{node_ip}" if node_ip else None
    if direct_key is not None and direct_key in resources:
        return direct_key

    candidates = sorted(str(key) for key in resources if str(key).startswith("node:"))
    if len(candidates) == 1:
        return candidates[0]

    if direct_key is not None:
        matches = [candidate for candidate in candidates if candidate == direct_key or candidate.endswith(node_ip)]
        if len(matches) == 1:
            return matches[0]

    node_id = str(node_payload.get("NodeID") or "").strip()
    if node_id:
        matches = [candidate for candidate in candidates if node_id in candidate]
        if len(matches) == 1:
            return matches[0]

    raise ValueError(
        "Unable to resolve a unique Ray node resource key for node payload with "
        f"NodeManagerAddress={node_ip!r} and resources={sorted(str(key) for key in resources)}"
    )


def _ray_alive_nodes(ray_module) -> list[dict[str, Any]]:
    nodes = []
    for node in ray_module.nodes():
        if not node.get("Alive"):
            continue
        node_ip = str(node.get("NodeManagerAddress") or "").strip()
        if not node_ip:
            raise ValueError(f"Ray reported an alive node without NodeManagerAddress: {node}")
        nodes.append(
            {
                "node_index": len(nodes),
                "node_ip": node_ip,
                "node_resource_key": _resolve_ray_node_resource_key(node),
            }
        )
    if not nodes:
        raise RuntimeError("Ray cluster has no alive nodes.")
    return nodes


def _is_cuda_device_name(device_name: str | None) -> bool:
    if device_name is None:
        return False
    return str(device_name).strip().lower().startswith("cuda")


def _ray_task_device_name(device_name: str | None) -> str | None:
    # Ray remaps each task's allocated GPU(s) into a task-local CUDA namespace.
    # A task that requests one GPU should therefore use cuda:0 even if the
    # node-local worker layout was written as cuda:1, cuda:2, etc.
    if _is_cuda_device_name(device_name):
        return "cuda:0"
    return device_name


def _build_ray_assignments(
    *,
    ray_module,
    num_examples: int,
    worker_devices: Sequence[str | None],
) -> list[WorkerAssignment]:
    nodes = _ray_alive_nodes(ray_module)
    worker_specs: list[tuple[dict[str, Any], str | None]] = []
    for node in nodes:
        for device_name in worker_devices:
            worker_specs.append((node, device_name))

    assignments: list[WorkerAssignment] = []
    for worker_id, (start, end) in enumerate(
        _assignment_ranges(num_items=num_examples, num_workers=len(worker_specs))
    ):
        node, device_name = worker_specs[worker_id]
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                prompt_start=start,
                prompt_end=end,
                device_name=_ray_task_device_name(device_name),
                node_index=int(node["node_index"]),
                node_ip=node["node_ip"],
                node_resource_key=node["node_resource_key"],
            )
        )
    return assignments


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


@torch.inference_mode()
def greedy_decode_response(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
) -> GreedyResult:
    start_time = time.perf_counter()
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    generated_token_ids: list[int] = []
    ended_with_eos = False

    for _step_index in range(max_new_tokens):
        selected_token_id = int(torch.argmax(actor_state.current_logits, dim=-1).item())
        generated_token_ids.append(selected_token_id)
        if selected_token_id in eos_token_ids:
            ended_with_eos = True
            break
        actor_state.append(selected_token_id)

    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    score = float(score_response(example, response_text))
    response_length = len(generated_token_ids)
    return GreedyResult(
        example_id=int(example.example_id),
        prompt_length=int(prompt_ids.shape[1]),
        response=response_text,
        response_token_ids=tuple(int(token_id) for token_id in generated_token_ids),
        response_length=response_length,
        ended_with_eos=ended_with_eos,
        hit_max_length=bool(max_new_tokens > 0 and not ended_with_eos and response_length >= max_new_tokens),
        score=score,
        elapsed_sec=float(time.perf_counter() - start_time),
    )


def _result_row(
    *,
    result: GreedyResult,
    actor_name: str,
    actor_checkpoint_dir: str,
    actor_hf_dir: Path,
    example: ExampleRecord,
    prompt_index: int,
    assignment: WorkerAssignment | None,
    omit_prompt_text: bool,
    omit_response_token_ids: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "actor_name": actor_name,
        "actor_checkpoint_dir": actor_checkpoint_dir,
        "actor_hf_dir": str(actor_hf_dir),
        "prompt_index": int(prompt_index),
        "example_id": int(example.example_id),
        "response": result.response,
        "response_length": int(result.response_length),
        "ended_with_eos": bool(result.ended_with_eos),
        "hit_max_length": bool(result.hit_max_length),
        "task_score": float(result.score),
        "accuracy": float(result.score),
        "prompt_length": int(result.prompt_length),
        "elapsed_sec": float(result.elapsed_sec),
    }
    if assignment is not None:
        row.update(
            {
                "worker_id": int(assignment.worker_id),
                "device": assignment.device_name,
                "node_index": assignment.node_index,
                "node_ip": assignment.node_ip,
            }
        )
    if not omit_prompt_text:
        row["prompt"] = example.prompt_text
    if not omit_response_token_ids:
        row["response_token_ids"] = [int(token_id) for token_id in result.response_token_ids]
    return row


def _evaluate_assignment(
    *,
    args: argparse.Namespace,
    actor_checkpoint_dir: Path,
    actor_hf_dir: Path,
    examples: Sequence[ExampleRecord],
    assignment: WorkerAssignment,
    eos_token_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    set_decode_seed(args.seed + assignment.worker_id)
    device = resolve_device(assignment.device_name or args.device)
    dtype = resolve_dtype(args.dtype)
    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    actor = load_actor_model(
        actor_hf_dir,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    rows: list[dict[str, Any]] = []
    local_examples = examples[assignment.prompt_start : assignment.prompt_end]
    for local_offset, example in enumerate(
        tqdm(
            local_examples,
            desc=f"greedy[{args.actor_name}]/worker_{assignment.worker_id:03d}",
            position=assignment.worker_id,
            leave=False,
        )
    ):
        prompt_index = assignment.prompt_start + local_offset
        prompt_ids = _prompt_ids_tensor(
            example=example,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            device=device,
        )
        result = greedy_decode_response(
            actor=actor,
            tokenizer=tokenizer,
            example=example,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            use_actor_cache=not args.disable_actor_cache,
        )
        rows.append(
            _result_row(
                result=result,
                actor_name=args.actor_name,
                actor_checkpoint_dir=str(actor_checkpoint_dir),
                actor_hf_dir=actor_hf_dir,
                example=example,
                prompt_index=prompt_index,
                assignment=assignment,
                omit_prompt_text=args.omit_prompt_text,
                omit_response_token_ids=args.omit_response_token_ids,
            )
        )
    return rows


def _ray_worker_entry(**kwargs) -> list[dict[str, Any]]:
    return _evaluate_assignment(
        args=kwargs["args"],
        actor_checkpoint_dir=Path(kwargs["actor_checkpoint_dir"]),
        actor_hf_dir=Path(kwargs["actor_hf_dir"]),
        examples=kwargs["examples"],
        assignment=kwargs["assignment"],
        eos_token_ids=kwargs["eos_token_ids"],
    )


def _run_local(
    *,
    args: argparse.Namespace,
    actor_checkpoint_dir: Path,
    actor_hf_dir: Path,
    examples: Sequence[ExampleRecord],
    eos_token_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    worker_devices = args.worker_devices or [args.device]
    rows: list[dict[str, Any]] = []
    for assignment in _build_local_assignments(num_examples=len(examples), worker_devices=worker_devices):
        rows.extend(
            _evaluate_assignment(
                args=args,
                actor_checkpoint_dir=actor_checkpoint_dir,
                actor_hf_dir=actor_hf_dir,
                examples=examples,
                assignment=assignment,
                eos_token_ids=eos_token_ids,
            )
        )
    return rows


def _run_ray(
    *,
    args: argparse.Namespace,
    ray_module,
    actor_checkpoint_dir: Path,
    actor_hf_dir: Path,
    examples: Sequence[ExampleRecord],
    eos_token_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    worker_devices = args.worker_devices or ["cuda:0"]
    assignments = _build_ray_assignments(
        ray_module=ray_module,
        num_examples=len(examples),
        worker_devices=worker_devices,
    )
    refs = []
    for assignment in assignments:
        remote_options: dict[str, Any] = {"num_cpus": args.ray_num_cpus_per_worker}
        if _is_cuda_device_name(assignment.device_name):
            remote_options["num_gpus"] = 1
        if assignment.node_resource_key:
            remote_options["resources"] = {assignment.node_resource_key: RAY_NODE_RESOURCE_FRACTION}
        refs.append(
            ray_module.remote(**remote_options)(_ray_worker_entry).remote(
                args=args,
                actor_checkpoint_dir=str(actor_checkpoint_dir),
                actor_hf_dir=str(actor_hf_dir),
                examples=list(examples),
                assignment=assignment,
                eos_token_ids=eos_token_ids,
            )
        )

    rows: list[dict[str, Any]] = []
    pending = list(refs)
    with tqdm(total=len(examples), desc=f"Greedy eval {args.actor_name}", unit="prompt", dynamic_ncols=True) as progress:
        while pending:
            done, pending = ray_module.wait(pending, num_returns=1, timeout=RAY_PROGRESS_POLL_INTERVAL_SEC)
            for ref in done:
                worker_rows = ray_module.get(ref)
                rows.extend(worker_rows)
                progress.update(len(worker_rows))
    return sorted(rows, key=lambda row: int(row["prompt_index"]))


def _summary_metrics(
    *,
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    actor_hf_dir: Path,
    execution_backend: str,
) -> dict[str, Any]:
    scores = np.asarray([float(row["task_score"]) for row in rows], dtype=np.float64)
    lengths = np.asarray([float(row["response_length"]) for row in rows], dtype=np.float64)
    elapsed = np.asarray([float(row["elapsed_sec"]) for row in rows], dtype=np.float64)
    if scores.size == 0:
        raise ValueError("No examples were evaluated.")

    return {
        "actor_name": args.actor_name,
        "actor_checkpoint_dir": str(Path(args.actor_checkpoint_dir).resolve()),
        "actor_hf_dir": str(actor_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "execution_backend": execution_backend,
        "num_examples": int(scores.size),
        "greedy_accuracy": float(scores.mean()),
        "mean_score": float(scores.mean()),
        "score_sum": float(scores.sum()),
        "score_std": float(scores.std()),
        "mean_response_length": float(lengths.mean()),
        "std_response_length": float(lengths.std()),
        "fraction_ended_with_eos": float(np.mean([float(row["ended_with_eos"]) for row in rows])),
        "fraction_hit_max_length": float(np.mean([float(row["hit_max_length"]) for row in rows])),
        "total_model_elapsed_sec": float(elapsed.sum()),
        "mean_model_elapsed_sec_per_example": float(elapsed.mean()),
        "args": vars(args),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_decode_seed(args.seed)
    actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
    actor_hf_dir = ensure_merged_component_checkpoint(
        actor_checkpoint_dir,
        component="actor",
        merged_root=Path(args.actor_merged_root).resolve() if args.actor_merged_root else None,
        hf_source_dir=Path(args.actor_hf_source_dir).resolve() if args.actor_hf_source_dir else None,
        skip_merge=args.skip_merge,
    )
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

    ray_address = _resolve_ray_address(args.ray_address)
    execution_backend = "ray" if ray_address is not None else "local"
    if ray_address is not None:
        if ray is None:
            raise ImportError("Ray is required when --ray_address is set.")
        ray.init(address=ray_address)
        try:
            rows = _run_ray(
                args=args,
                ray_module=ray,
                actor_checkpoint_dir=actor_checkpoint_dir,
                actor_hf_dir=actor_hf_dir,
                examples=examples,
                eos_token_ids=eos_token_ids,
            )
        finally:
            ray.shutdown()
    else:
        rows = _run_local(
            args=args,
            actor_checkpoint_dir=actor_checkpoint_dir,
            actor_hf_dir=actor_hf_dir,
            examples=examples,
            eos_token_ids=eos_token_ids,
        )

    predictions_path = output_dir / "greedy_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output_file:
        for row in sorted(rows, key=lambda item: int(item["prompt_index"])):
            output_file.write(_json_line(row))

    metrics = _summary_metrics(rows=rows, args=args, actor_hf_dir=actor_hf_dir, execution_backend=execution_backend)
    summary_path = output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps({key: metrics[key] for key in ("num_examples", "greedy_accuracy", "mean_response_length")}, indent=2))
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
