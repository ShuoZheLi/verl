from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from queue import Empty
import shutil
import subprocess
import sys
import time
import traceback
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
from value_decoding.decoding import ActorSamplingMode, ActorStepper, sample_token_from_actor, set_decode_seed


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled at runtime
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

try:
    import ray
except ImportError:
    ray = None


TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)
RAY_NODE_RESOURCE_FRACTION = 1e-3
RAY_PROGRESS_POLL_INTERVAL_SEC = 0.2


def _make_main_module_importable() -> None:
    """Allow spawn workers to re-import this module when launched via `python -m`."""
    if __name__ != "__main__":
        return

    module_spec = globals().get("__spec__")
    canonical_name = getattr(module_spec, "name", None)
    if not canonical_name:
        return

    module = sys.modules.get(__name__)
    if module is None:
        return

    sys.modules[canonical_name] = module
    for obj in vars(module).values():
        if getattr(obj, "__module__", None) != __name__:
            continue
        try:
            obj.__module__ = canonical_name
        except Exception:
            pass


@dataclass(frozen=True)
class ActorSpec:
    actor_index: int
    actor_name: str
    checkpoint_dir: str
    merged_root: str | None
    hf_source_dir: str | None


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    device_name: str | None
    prompt_start: int
    prompt_end: int
    node_index: int | None = None
    node_ip: str | None = None
    node_resource_key: str | None = None
    local_worker_index: int | None = None

    @property
    def num_prompts(self) -> int:
        return max(self.prompt_end - self.prompt_start, 0)


@dataclass(frozen=True)
class RayNodeInfo:
    node_index: int
    node_ip: str
    node_resource_key: str
    node_name: str | None = None


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float) -> None:
        numeric = float(value)
        self.count += 1
        self.total += numeric
        self.total_sq += numeric * numeric

    def merge(self, other: "RunningMoments") -> None:
        self.count += int(other.count)
        self.total += float(other.total)
        self.total_sq += float(other.total_sq)

    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return float(self.total / self.count)

    def std(self) -> float | None:
        if self.count <= 0:
            return None
        mean_value = self.total / self.count
        variance = max((self.total_sq / self.count) - (mean_value * mean_value), 0.0)
        return float(math.sqrt(variance))


@dataclass(frozen=True)
class SampledResponse:
    sample_index: int
    seed: int
    prompt_length: int
    response_text: str
    response_token_ids: tuple[int, ...]
    token_entropies: tuple[float, ...]
    response_length: int
    ended_with_eos: bool
    hit_max_length: bool
    task_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Actor proposal quality diagnostic. For each frozen actor checkpoint, sample a response bank per prompt, "
            "score all responses with the task reward, and measure search headroom, diversity, and entropy."
        )
    )
    parser.add_argument("--actor_checkpoint_dirs", nargs="+", type=str, required=True)
    parser.add_argument("--actor_names", nargs="+", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, required=True)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--num_samples_per_prompt", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--worker_devices", nargs="+", default=None)
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help=(
            "Optional Ray cluster address for cross-node execution. When set, --worker_devices is treated as the "
            "node-local worker layout and is replicated across all alive Ray nodes. Use 'auto' to read $RAY_ADDRESS."
        ),
    )
    parser.add_argument(
        "--ray_num_cpus_per_worker",
        type=float,
        default=1.0,
        help="CPU resources reserved per Ray worker task when --ray_address is used.",
    )
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--actor_merged_roots", nargs="+", default=None)
    parser.add_argument("--actor_hf_source_dirs", nargs="+", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument(
        "--omit_prompt_text",
        action="store_true",
        help="Exclude the full prompt text from response_bank.jsonl rows to reduce disk usage.",
    )
    parser.add_argument(
        "--omit_response_token_ids",
        action="store_true",
        help="Exclude generated response token ids from response_bank.jsonl rows to reduce disk usage.",
    )
    parser.add_argument(
        "--store_token_entropies",
        action="store_true",
        help="Store the full token-level entropy trace for each response in response_bank.jsonl.",
    )
    parser.add_argument(
        "--disable_actor_cache",
        action="store_true",
        help="Disable KV-cache decoding. This is slower but can help with unusual model implementations.",
    )
    return parser.parse_args()


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


def _resolve_optional_per_actor_list(
    values: list[str] | None,
    *,
    num_actors: int,
    argument_name: str,
) -> list[str | None]:
    if values is None:
        return [None] * num_actors
    if len(values) != num_actors:
        raise ValueError(
            f"{argument_name} must contain exactly one entry per actor "
            f"({num_actors} expected, received {len(values)})."
        )
    return [value if value else None for value in values]


def _assignment_ranges(*, num_items: int, num_workers: int) -> list[tuple[int, int]]:
    if num_items <= 0:
        return []
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0 when items are present.")

    active_workers = min(num_workers, num_items)
    ranges: list[tuple[int, int]] = []
    start = 0
    base = num_items // active_workers
    remainder = num_items % active_workers
    for worker_id in range(active_workers):
        shard_size = base + (1 if worker_id < remainder else 0)
        end = start + shard_size
        ranges.append((start, end))
        start = end
    return ranges


def _build_worker_assignments(
    *,
    num_prompts: int,
    worker_devices: list[str | None],
) -> list[WorkerAssignment]:
    if not worker_devices:
        worker_devices = [None]
    ranges = _assignment_ranges(num_items=num_prompts, num_workers=len(worker_devices))
    assignments: list[WorkerAssignment] = []
    for worker_id, (start, end) in enumerate(ranges):
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                device_name=worker_devices[worker_id],
                prompt_start=start,
                prompt_end=end,
            )
        )
    return assignments


def _build_distributed_worker_assignments(
    *,
    num_prompts: int,
    worker_devices: list[str | None],
    ray_nodes: list[RayNodeInfo],
) -> list[WorkerAssignment]:
    if not worker_devices:
        worker_devices = [None]
    if not ray_nodes:
        raise ValueError("At least one Ray node is required.")
    if num_prompts <= 0:
        return []

    worker_descriptors: list[tuple[RayNodeInfo, int, str | None]] = []
    for local_worker_index, device_name in enumerate(worker_devices):
        for node in ray_nodes:
            worker_descriptors.append((node, local_worker_index, device_name))

    ranges = _assignment_ranges(num_items=num_prompts, num_workers=len(worker_descriptors))
    assignments: list[WorkerAssignment] = []
    for worker_id, (start, end) in enumerate(ranges):
        node, local_worker_index, device_name = worker_descriptors[worker_id]
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                device_name=device_name,
                prompt_start=start,
                prompt_end=end,
                node_index=node.node_index,
                node_ip=node.node_ip,
                node_resource_key=node.node_resource_key,
                local_worker_index=local_worker_index,
            )
        )
    return assignments


def _validate_visible_cuda_device(device: torch.device | None, *, label: str) -> None:
    if device is None or device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"{label} requested CUDA device {device}, but CUDA is not available in this worker.")
    if device.index is None:
        return
    visible_device_count = torch.cuda.device_count()
    if device.index >= visible_device_count:
        raise RuntimeError(
            f"{label} requested CUDA device {device}, but this worker only sees {visible_device_count} CUDA device(s)."
        )


def _resolve_ray_address(ray_address: str | None) -> str | None:
    if ray_address is None:
        return None
    normalized = str(ray_address).strip()
    if not normalized:
        return None
    if normalized.lower() == "auto":
        env_address = os.environ.get("RAY_ADDRESS")
        if not env_address:
            raise ValueError("--ray_address=auto was requested, but $RAY_ADDRESS is not set.")
        return env_address
    return normalized


def _require_ray():
    if ray is None:
        raise ImportError("Ray is required for cross-node actor proposal evaluation, but it is not installed.")
    return ray


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


def _discover_ray_nodes(ray_module) -> list[RayNodeInfo]:
    nodes: list[RayNodeInfo] = []
    for raw_node in ray_module.nodes():
        if not bool(raw_node.get("Alive")):
            continue
        node_ip = str(raw_node.get("NodeManagerAddress") or "").strip()
        if not node_ip:
            raise ValueError(f"Ray reported an alive node without NodeManagerAddress: {raw_node}")
        nodes.append(
            RayNodeInfo(
                node_index=-1,
                node_ip=node_ip,
                node_resource_key=_resolve_ray_node_resource_key(raw_node),
                node_name=(
                    str(raw_node.get("NodeName"))
                    if raw_node.get("NodeName") is not None
                    else (
                        str(raw_node.get("NodeManagerHostname"))
                        if raw_node.get("NodeManagerHostname") is not None
                        else None
                    )
                ),
            )
        )
    nodes.sort(key=lambda item: (item.node_ip, item.node_name or ""))
    return [
        RayNodeInfo(
            node_index=index,
            node_ip=node.node_ip,
            node_resource_key=node.node_resource_key,
            node_name=node.node_name,
        )
        for index, node in enumerate(nodes)
    ]


def _normalize_cuda_device_name(device_name: str | None, *, assume_default_cuda: bool) -> str | None:
    if device_name is None:
        return "cuda:0" if assume_default_cuda else None
    resolved = torch.device(device_name)
    if resolved.type != "cuda":
        return str(resolved)
    resolved_index = 0 if resolved.index is None else int(resolved.index)
    return f"cuda:{resolved_index}"


def _remap_cuda_device_name(device_name: str | None, *, cuda_slot_mapping: dict[int, int]) -> str | None:
    if device_name is None:
        return None
    resolved = torch.device(device_name)
    if resolved.type != "cuda":
        return str(resolved)
    resolved_index = 0 if resolved.index is None else int(resolved.index)
    if resolved_index not in cuda_slot_mapping:
        raise ValueError(
            f"CUDA device index {resolved_index} is missing from the Ray-visible slot mapping {cuda_slot_mapping}."
        )
    return f"cuda:{cuda_slot_mapping[resolved_index]}"


def _build_ray_node_execution_specs(
    *,
    worker_assignments: Sequence[WorkerAssignment],
    ray_num_cpus_per_worker: float,
) -> list[dict[str, Any]]:
    node_groups: dict[tuple[int | None, str | None, str | None], list[WorkerAssignment]] = {}
    for assignment in worker_assignments:
        group_key = (assignment.node_index, assignment.node_ip, assignment.node_resource_key)
        node_groups.setdefault(group_key, []).append(assignment)

    node_specs: list[dict[str, Any]] = []
    for group_key in sorted(node_groups, key=lambda item: (-1 if item[0] is None else int(item[0]), str(item[1]))):
        node_assignments = sorted(node_groups[group_key], key=lambda item: int(item.worker_id))
        normalized_assignments: list[tuple[WorkerAssignment, str | None]] = []
        referenced_cuda_slots: set[int] = set()

        for assignment in node_assignments:
            device_name = _normalize_cuda_device_name(assignment.device_name, assume_default_cuda=True)
            normalized_assignments.append((assignment, device_name))
            if device_name is None:
                continue
            resolved = torch.device(device_name)
            if resolved.type == "cuda":
                referenced_cuda_slots.add(0 if resolved.index is None else int(resolved.index))

        cuda_slot_mapping = {
            original_slot: remapped_slot
            for remapped_slot, original_slot in enumerate(sorted(referenced_cuda_slots))
        }
        remapped_assignments: list[WorkerAssignment] = []
        for assignment, device_name in normalized_assignments:
            remapped_assignments.append(
                WorkerAssignment(
                    worker_id=assignment.worker_id,
                    device_name=_remap_cuda_device_name(device_name, cuda_slot_mapping=cuda_slot_mapping),
                    prompt_start=assignment.prompt_start,
                    prompt_end=assignment.prompt_end,
                    node_index=assignment.node_index,
                    node_ip=assignment.node_ip,
                    node_resource_key=assignment.node_resource_key,
                    local_worker_index=assignment.local_worker_index,
                )
            )

        node_specs.append(
            {
                "node_index": group_key[0],
                "node_ip": group_key[1],
                "node_resource_key": group_key[2],
                "assignments": remapped_assignments,
                "num_gpus": float(len(referenced_cuda_slots)),
                "num_cpus": float(ray_num_cpus_per_worker) * float(len(remapped_assignments)),
                "cuda_slot_mapping": cuda_slot_mapping,
            }
        )
    return node_specs


def _progress_postfix(worker_progress: dict[int, dict[str, Any]]) -> str:
    parts: list[str] = []
    for worker_id in sorted(worker_progress):
        state = worker_progress[worker_id]
        done = int(state.get("done", 0))
        total = int(state.get("total", 0))
        parts.append(f"w{worker_id}:{done}/{total}")
    return " | ".join(parts)


def _tokenizer_fingerprint(model_dir: Path) -> dict[str, Any]:
    # `hashlib` is intentionally imported lazily to keep the module import list focused above.
    import hashlib

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
    return {"sha256": digest.hexdigest(), "files": included_files}


def _assert_shared_tokenizer(tokenizer_fingerprints: dict[str, dict[str, Any]]) -> None:
    fingerprints = {name: payload["sha256"] for name, payload in tokenizer_fingerprints.items()}
    unique_fingerprints = set(fingerprints.values())
    if len(unique_fingerprints) == 1:
        return
    details = ", ".join(f"{name}={fingerprint}" for name, fingerprint in fingerprints.items())
    raise ValueError(
        "Actor proposal diagnostic requires all compared actors to share the same tokenizer so prompt "
        f"truncation, entropy, and token-length measurements remain comparable. Fingerprints: {details}"
    )


def _sample_seed(base_seed: int, *, actor_index: int, prompt_index: int, sample_index: int) -> int:
    return int(base_seed + actor_index * 1_000_000 + prompt_index * 10_000 + sample_index)


def _entropy_quarter_means(token_entropies: Sequence[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not token_entropies:
        return None, None, None, None

    entropy_array = np.asarray(token_entropies, dtype=np.float64)
    quarters = np.array_split(entropy_array, 4)
    means: list[float | None] = []
    for quarter in quarters:
        means.append(None if quarter.size == 0 else float(quarter.mean()))
    return means[0], means[1], means[2], means[3]


def _entropy_summary(token_entropies: Sequence[float]) -> dict[str, Any]:
    if not token_entropies:
        return {
            "mean_response_entropy": 0.0,
            "sum_response_entropy": 0.0,
            "std_response_entropy": 0.0,
            "max_response_entropy": 0.0,
            "min_response_entropy": 0.0,
            "first_quarter_mean_entropy": None,
            "second_quarter_mean_entropy": None,
            "third_quarter_mean_entropy": None,
            "fourth_quarter_mean_entropy": None,
        }

    entropy_array = np.asarray(token_entropies, dtype=np.float64)
    first_q, second_q, third_q, fourth_q = _entropy_quarter_means(token_entropies)
    return {
        "mean_response_entropy": float(entropy_array.mean()),
        "sum_response_entropy": float(entropy_array.sum()),
        "std_response_entropy": float(entropy_array.std()),
        "max_response_entropy": float(entropy_array.max()),
        "min_response_entropy": float(entropy_array.min()),
        "first_quarter_mean_entropy": first_q,
        "second_quarter_mean_entropy": second_q,
        "third_quarter_mean_entropy": third_q,
        "fourth_quarter_mean_entropy": fourth_q,
    }


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


def sample_actor_response_with_entropy(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    prompt_ids: torch.Tensor,
    sample_index: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
) -> SampledResponse:
    set_decode_seed(seed)
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    generated_token_ids: list[int] = []
    token_entropies: list[float] = []
    ended_with_eos = False

    for _step_index in range(max_new_tokens):
        logits = actor_state.current_logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = float((-(probs * log_probs).sum(dim=-1)).item())
        token_entropies.append(entropy)

        selected_token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=ActorSamplingMode.SAMPLE.value,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated_token_ids.append(selected_token_id)
        actor_state.append(selected_token_id)
        if selected_token_id in eos_token_ids:
            ended_with_eos = True
            break

    response_length = len(generated_token_ids)
    hit_max_length = bool(max_new_tokens > 0 and not ended_with_eos and response_length >= max_new_tokens)
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = float(score_response(example, response_text))
    return SampledResponse(
        sample_index=sample_index,
        seed=seed,
        prompt_length=int(prompt_ids.shape[1]),
        response_text=response_text,
        response_token_ids=tuple(int(token_id) for token_id in generated_token_ids),
        token_entropies=tuple(float(value) for value in token_entropies),
        response_length=response_length,
        ended_with_eos=ended_with_eos,
        hit_max_length=hit_max_length,
        task_score=task_score,
    )


def _build_response_row(
    *,
    actor_name: str,
    actor_checkpoint_dir: str,
    prompt_index: int,
    example: ExampleRecord,
    sampled_response: SampledResponse,
    omit_prompt_text: bool,
    omit_response_token_ids: bool,
    store_token_entropies: bool,
) -> dict[str, Any]:
    row = {
        "actor_name": actor_name,
        "actor_checkpoint_dir": actor_checkpoint_dir,
        "prompt_index": int(prompt_index),
        "example_id": int(example.example_id),
        "sample_index": int(sampled_response.sample_index),
        "seed": int(sampled_response.seed),
        "response": sampled_response.response_text,
        "response_length": int(sampled_response.response_length),
        "ended_with_eos": bool(sampled_response.ended_with_eos),
        "hit_max_length": bool(sampled_response.hit_max_length),
        "task_score": float(sampled_response.task_score),
        "accuracy": float(sampled_response.task_score),
        "prompt_length": int(sampled_response.prompt_length),
    }
    row.update(_entropy_summary(sampled_response.token_entropies))
    if not omit_prompt_text:
        row["prompt"] = example.prompt_text
    if not omit_response_token_ids:
        row["response_token_ids"] = [int(token_id) for token_id in sampled_response.response_token_ids]
    if store_token_entropies:
        row["token_entropies"] = [float(value) for value in sampled_response.token_entropies]
    return row


def _oracle_metric_key(num_samples_per_prompt: int) -> str:
    return f"oracle_best_of_{int(num_samples_per_prompt)}_accuracy"


def _score_sum_value(scores: np.ndarray) -> int | float:
    total = float(scores.sum())
    rounded = round(total)
    if np.allclose(scores, np.round(scores)) and abs(total - rounded) < 1e-9:
        return int(rounded)
    return total


def build_prompt_summary(
    *,
    actor_name: str,
    actor_checkpoint_dir: str,
    prompt_index: int,
    example: ExampleRecord,
    response_rows: Sequence[dict[str, Any]],
    num_samples_per_prompt: int,
) -> dict[str, Any]:
    if not response_rows:
        raise ValueError("Each prompt must contain at least one sampled response.")

    sorted_rows = sorted(response_rows, key=lambda row: int(row["sample_index"]))
    scores = np.asarray([float(row["task_score"]) for row in sorted_rows], dtype=np.float64)
    lengths = np.asarray([float(row["response_length"]) for row in sorted_rows], dtype=np.float64)
    entropies = np.asarray([float(row["mean_response_entropy"]) for row in sorted_rows], dtype=np.float64)
    normalized_responses = [str(row["response"]).strip() for row in sorted_rows]
    num_distinct_responses = len(set(normalized_responses))
    distinct_response_fraction = num_distinct_responses / max(num_samples_per_prompt, 1)
    oracle_key = _oracle_metric_key(num_samples_per_prompt)

    prompt_summary = {
        "actor_name": actor_name,
        "actor_checkpoint_dir": actor_checkpoint_dir,
        "prompt_index": int(prompt_index),
        "example_id": int(example.example_id),
        "num_samples": int(num_samples_per_prompt),
        "sampled_single_accuracy": float(scores[0]),
        "mean_sample_accuracy": float(scores.mean()),
        "oracle_best_of_n_accuracy": float(scores.max()),
        "has_success": int(float(scores.max()) > 0.0),
        "num_successes_in_bank": _score_sum_value(scores),
        "success_rate_in_bank": float(scores.mean()),
        "outcome_variance": float(scores.var()),
        "num_distinct_responses": int(num_distinct_responses),
        "distinct_response_fraction": float(distinct_response_fraction),
        "duplicate_response_fraction": float(1.0 - distinct_response_fraction),
        "mean_bank_response_entropy": float(entropies.mean()),
        "std_bank_response_entropy": float(entropies.std()),
        "mean_bank_response_length": float(lengths.mean()),
        "std_bank_response_length": float(lengths.std()),
    }
    prompt_summary[oracle_key] = prompt_summary["oracle_best_of_n_accuracy"]
    return prompt_summary


def _paired_bootstrap_from_differences(
    differences: Sequence[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    diff_array = np.asarray(differences, dtype=np.float64)
    if diff_array.size == 0:
        raise ValueError("Cannot bootstrap an empty set of prompt-level differences.")

    rng = np.random.default_rng(seed)
    sample_means = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        indices = rng.integers(0, diff_array.size, size=diff_array.size)
        sample_means[sample_index] = float(diff_array[indices].mean())

    ci_lower, ci_upper = np.quantile(sample_means, [0.025, 0.975]).tolist()
    return {
        "observed_difference": float(diff_array.mean()),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "num_prompts": int(diff_array.size),
        "num_bootstrap_samples": int(bootstrap_samples),
    }


class _RayProgressActor:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def put(self, event: dict[str, Any]) -> None:
        self._events.append(dict(event))

    def drain(self) -> list[dict[str, Any]]:
        events = self._events
        self._events = []
        return events


def _worker_entry(
    *,
    actor_spec: ActorSpec,
    actor_hf_dir: str,
    assignment: WorkerAssignment,
    examples: list[ExampleRecord],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    num_samples_per_prompt: int,
    temperature: float,
    top_p: float,
    top_k: int,
    base_seed: int,
    omit_prompt_text: bool,
    omit_response_token_ids: bool,
    store_token_entropies: bool,
    use_actor_cache: bool,
    worker_root: str,
    progress_queue=None,
) -> None:
    worker_dir = Path(worker_root) / f"worker_{assignment.worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    response_bank_path = worker_dir / "response_bank.jsonl"
    prompt_summary_path = worker_dir / "prompt_summary.jsonl"
    worker_summary_path = worker_dir / "worker_summary.json"
    error_path = worker_dir / "worker_error.txt"

    try:
        start_time = time.perf_counter()
        device = resolve_device(assignment.device_name)
        _validate_visible_cuda_device(device, label="worker_device")
        dtype = resolve_dtype(dtype_name)

        actor_hf_path = Path(actor_hf_dir)
        tokenizer = load_tokenizer(actor_hf_path, trust_remote_code=trust_remote_code)
        actor = load_actor_model(
            actor_hf_path,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        entropy_moments = RunningMoments()
        length_moments = RunningMoments()
        local_examples = examples[assignment.prompt_start : assignment.prompt_end]
        num_response_rows = 0
        num_prompt_rows = 0
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_started",
                    "worker_id": assignment.worker_id,
                    "worker_total_prompts": len(local_examples),
                }
            )

        with response_bank_path.open("w", encoding="utf-8") as response_file, prompt_summary_path.open(
            "w",
            encoding="utf-8",
        ) as prompt_file:
            for local_offset, example in enumerate(local_examples):
                prompt_index = assignment.prompt_start + local_offset
                prompt_ids = _prompt_ids_tensor(
                    example=example,
                    tokenizer=tokenizer,
                    max_prompt_length=max_prompt_length,
                    device=device,
                )
                response_rows: list[dict[str, Any]] = []
                for sample_index in range(num_samples_per_prompt):
                    seed = _sample_seed(
                        base_seed,
                        actor_index=actor_spec.actor_index,
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                    )
                    sampled_response = sample_actor_response_with_entropy(
                        actor=actor,
                        tokenizer=tokenizer,
                        example=example,
                        prompt_ids=prompt_ids,
                        sample_index=sample_index,
                        seed=seed,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_new_tokens=max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        use_actor_cache=use_actor_cache,
                    )
                    response_row = _build_response_row(
                        actor_name=actor_spec.actor_name,
                        actor_checkpoint_dir=actor_spec.checkpoint_dir,
                        prompt_index=prompt_index,
                        example=example,
                        sampled_response=sampled_response,
                        omit_prompt_text=omit_prompt_text,
                        omit_response_token_ids=omit_response_token_ids,
                        store_token_entropies=store_token_entropies,
                    )
                    response_file.write(_json_line(response_row))
                    response_rows.append(response_row)
                    entropy_moments.add(float(response_row["mean_response_entropy"]))
                    length_moments.add(float(response_row["response_length"]))
                    num_response_rows += 1

                prompt_summary = build_prompt_summary(
                    actor_name=actor_spec.actor_name,
                    actor_checkpoint_dir=actor_spec.checkpoint_dir,
                    prompt_index=prompt_index,
                    example=example,
                    response_rows=response_rows,
                    num_samples_per_prompt=num_samples_per_prompt,
                )
                prompt_file.write(_json_line(prompt_summary))
                num_prompt_rows += 1
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "type": "prompt_done",
                            "worker_id": assignment.worker_id,
                            "worker_completed_prompts": num_prompt_rows,
                            "worker_total_prompts": len(local_examples),
                        }
                    )

        worker_summary = {
            "worker_id": int(assignment.worker_id),
            "device": str(device),
            "prompt_start": int(assignment.prompt_start),
            "prompt_end": int(assignment.prompt_end),
            "num_prompts": int(num_prompt_rows),
            "num_responses": int(num_response_rows),
            "node_index": assignment.node_index,
            "node_ip": assignment.node_ip,
            "node_resource_key": assignment.node_resource_key,
            "local_worker_index": assignment.local_worker_index,
            "response_entropy_moments": asdict(entropy_moments),
            "response_length_moments": asdict(length_moments),
            "elapsed_sec": float(time.perf_counter() - start_time),
        }
        worker_summary_path.write_text(json.dumps(worker_summary, ensure_ascii=True, indent=2), encoding="utf-8")
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_done",
                    "worker_id": assignment.worker_id,
                    "worker_completed_prompts": num_prompt_rows,
                    "worker_total_prompts": len(local_examples),
                }
            )
    except Exception:
        trace = traceback.format_exc()
        error_path.write_text(trace, encoding="utf-8")
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_error",
                    "worker_id": assignment.worker_id,
                    "traceback": trace,
                }
            )
        raise


def _start_worker_processes(
    *,
    actor_spec: ActorSpec,
    actor_hf_dir: Path,
    assignments: Sequence[WorkerAssignment],
    examples: list[ExampleRecord],
    args: argparse.Namespace,
    eos_token_ids: tuple[int, ...],
    worker_root: Path,
) -> tuple[Any, list[tuple[mp.Process, WorkerAssignment]]]:
    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[tuple[mp.Process, WorkerAssignment]] = []
    for assignment in assignments:
        process = context.Process(
            target=_worker_entry,
            kwargs={
                "actor_spec": actor_spec,
                "actor_hf_dir": str(actor_hf_dir),
                "assignment": assignment,
                "examples": examples,
                "dtype_name": args.dtype,
                "trust_remote_code": args.trust_remote_code,
                "max_prompt_length": args.max_prompt_length,
                "max_new_tokens": args.max_new_tokens,
                "eos_token_ids": eos_token_ids,
                "num_samples_per_prompt": args.num_samples_per_prompt,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "base_seed": args.seed,
                "omit_prompt_text": args.omit_prompt_text,
                "omit_response_token_ids": args.omit_response_token_ids,
                "store_token_entropies": args.store_token_entropies,
                "use_actor_cache": not args.disable_actor_cache,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
            },
            name=f"actor_proposal_worker_{actor_spec.actor_name}_{assignment.worker_id}",
        )
        process.start()
        processes.append((process, assignment))
    return progress_queue, processes


def _assert_worker_processes_healthy(
    *,
    processes: Sequence[tuple[mp.Process, WorkerAssignment]],
    worker_root: Path,
) -> None:
    for process, assignment in processes:
        if process.exitcode not in (None, 0):
            error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
            if error_path.exists():
                raise RuntimeError(
                    f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n"
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")


def _join_worker_processes(
    *,
    processes: Sequence[tuple[mp.Process, WorkerAssignment]],
    worker_root: Path,
) -> None:
    for process, assignment in processes:
        process.join()
        if process.exitcode != 0:
            error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
            if error_path.exists():
                raise RuntimeError(
                    f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n"
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")


def _ray_node_entry_remote(**kwargs) -> dict[str, Any]:
    assignments: list[WorkerAssignment] = kwargs["assignments"]
    progress_actor = kwargs["progress_actor"]
    progress_queue, processes = _start_worker_processes(
        actor_spec=kwargs["actor_spec"],
        actor_hf_dir=Path(kwargs["actor_hf_dir"]),
        assignments=assignments,
        examples=kwargs["examples"],
        args=kwargs["args"],
        eos_token_ids=kwargs["eos_token_ids"],
        worker_root=Path(kwargs["worker_root"]),
    )
    worker_root = Path(kwargs["worker_root"])
    completed_workers = 0
    while completed_workers < len(assignments):
        try:
            event = progress_queue.get(timeout=RAY_PROGRESS_POLL_INTERVAL_SEC)
        except Empty:
            _assert_worker_processes_healthy(processes=processes, worker_root=worker_root)
            continue

        progress_actor.put.remote(event)
        if event.get("type") == "worker_done":
            completed_workers += 1
        elif event.get("type") == "worker_error":
            raise RuntimeError(
                f"Worker {event.get('worker_id')} reported an error.\n"
                f"{event.get('traceback', 'No traceback provided.')}"
            )

    _join_worker_processes(processes=processes, worker_root=worker_root)
    return {
        "node_index": kwargs["node_index"],
        "node_ip": kwargs["node_ip"],
        "worker_ids": [int(assignment.worker_id) for assignment in assignments],
    }


def run_ray_multi_worker(
    *,
    output_dir: Path,
    actor_spec: ActorSpec,
    actor_hf_dir: Path,
    examples: list[ExampleRecord],
    worker_assignments: list[WorkerAssignment],
    args: argparse.Namespace,
    eos_token_ids: tuple[int, ...],
    response_bank_file,
    prompt_summary_file,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not worker_assignments:
        raise ValueError("No worker assignments were created.")
    if args.ray_num_cpus_per_worker <= 0:
        raise ValueError(f"--ray_num_cpus_per_worker must be > 0, got {args.ray_num_cpus_per_worker}.")

    ray_module = _require_ray()
    if not ray_module.is_initialized():
        raise RuntimeError("Ray must be initialized before running cross-node actor proposal workers.")

    worker_root = output_dir / "_worker_tmp"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    progress_actor = ray_module.remote(num_cpus=0)(_RayProgressActor).remote()
    node_execution_specs = _build_ray_node_execution_specs(
        worker_assignments=worker_assignments,
        ray_num_cpus_per_worker=args.ray_num_cpus_per_worker,
    )
    node_remote = ray_module.remote(max_retries=0)(_ray_node_entry_remote)

    node_refs = []
    ref_to_node_spec: dict[Any, dict[str, Any]] = {}
    for node_spec in node_execution_specs:
        node_resource_key = node_spec["node_resource_key"]
        if node_resource_key is None:
            raise ValueError(
                f"Ray node execution spec for node {node_spec['node_ip']} is missing node_resource_key."
            )
        node_ref = node_remote.options(
            num_cpus=float(node_spec["num_cpus"]),
            num_gpus=float(node_spec["num_gpus"]),
            resources={node_resource_key: RAY_NODE_RESOURCE_FRACTION},
        ).remote(
            node_index=node_spec["node_index"],
            node_ip=node_spec["node_ip"],
            assignments=node_spec["assignments"],
            actor_spec=actor_spec,
            actor_hf_dir=str(actor_hf_dir),
            examples=examples,
            args=args,
            eos_token_ids=eos_token_ids,
            worker_root=str(worker_root),
            progress_actor=progress_actor,
        )
        node_refs.append(node_ref)
        ref_to_node_spec[node_ref] = node_spec

    total_prompts = len(examples)
    completed_prompts = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {"done": 0, "total": assignment.num_prompts} for assignment in worker_assignments
    }

    pending_refs = list(node_refs)
    with tqdm(
        total=total_prompts,
        desc=f"actor_proposal_diagnostic[{actor_spec.actor_name}]",
        unit="prompt",
        dynamic_ncols=True,
    ) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_prompts < total_prompts or completed_workers < len(worker_assignments):
            events = ray_module.get(progress_actor.drain.remote())
            for event in events:
                event_type = event.get("type")
                worker_id = int(event.get("worker_id", -1))
                if event_type == "worker_started":
                    worker_progress.setdefault(worker_id, {})
                    worker_progress[worker_id]["total"] = int(event.get("worker_total_prompts", 0))
                elif event_type == "prompt_done":
                    completed_prompts += 1
                    worker_progress.setdefault(worker_id, {})
                    worker_progress[worker_id]["done"] = int(event.get("worker_completed_prompts", 0))
                    worker_progress[worker_id]["total"] = int(event.get("worker_total_prompts", 0))
                    progress_bar.update(1)
                elif event_type == "worker_done":
                    completed_workers += 1
                    worker_progress.setdefault(worker_id, {})
                    worker_progress[worker_id]["done"] = int(event.get("worker_completed_prompts", 0))
                    worker_progress[worker_id]["total"] = int(event.get("worker_total_prompts", 0))
                elif event_type == "worker_error":
                    raise RuntimeError(
                        f"Worker {worker_id} reported an error.\n{event.get('traceback', 'No traceback provided.')}"
                    )
                progress_bar.set_postfix_str(_progress_postfix(worker_progress))

            if pending_refs:
                done_refs, pending_refs = ray_module.wait(
                    pending_refs,
                    num_returns=1,
                    timeout=RAY_PROGRESS_POLL_INTERVAL_SEC,
                )
                for done_ref in done_refs:
                    node_spec = ref_to_node_spec[done_ref]
                    try:
                        ray_module.get(done_ref)
                    except Exception as exc:
                        for assignment in node_spec["assignments"]:
                            error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
                            if error_path.exists():
                                raise RuntimeError(
                                    f"Worker {assignment.worker_id} failed on Ray node {assignment.node_ip}.\n"
                                    f"{error_path.read_text(encoding='utf-8')}"
                                ) from exc
                        raise RuntimeError(
                            f"Ray node task failed on node {node_spec['node_ip']} "
                            f"for workers {[assignment.worker_id for assignment in node_spec['assignments']]}."
                        ) from exc
            else:
                time.sleep(RAY_PROGRESS_POLL_INTERVAL_SEC)

    if node_refs:
        try:
            ray_module.get(node_refs)
        except Exception:
            for assignment in worker_assignments:
                error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
                if error_path.exists():
                    raise RuntimeError(
                        f"Worker {assignment.worker_id} failed on Ray node {assignment.node_ip}.\n"
                        f"{error_path.read_text(encoding='utf-8')}"
                    )
            raise

    return _collect_worker_outputs(
        worker_root=worker_root,
        assignments=worker_assignments,
        response_bank_file=response_bank_file,
        prompt_summary_file=prompt_summary_file,
    )


def _collect_worker_outputs(
    *,
    worker_root: Path,
    assignments: Sequence[WorkerAssignment],
    response_bank_file,
    prompt_summary_file,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_summaries: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for assignment in sorted(assignments, key=lambda item: int(item.worker_id)):
        worker_dir = worker_root / f"worker_{assignment.worker_id:03d}"
        response_path = worker_dir / "response_bank.jsonl"
        prompt_path = worker_dir / "prompt_summary.jsonl"
        summary_path = worker_dir / "worker_summary.json"

        with response_path.open("r", encoding="utf-8") as response_file:
            for line in response_file:
                if line.strip():
                    response_bank_file.write(line)

        with prompt_path.open("r", encoding="utf-8") as prompt_file:
            for line in prompt_file:
                if not line.strip():
                    continue
                prompt_summary_file.write(line)
                prompt_summaries.append(json.loads(line))

        worker_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return prompt_summaries, worker_summaries


def _aggregate_actor_metrics(
    *,
    actor_name: str,
    actor_checkpoint_dir: str,
    prompt_summaries: Sequence[dict[str, Any]],
    worker_summaries: Sequence[dict[str, Any]],
    num_samples_per_prompt: int,
) -> dict[str, Any]:
    if not prompt_summaries:
        raise ValueError(f"Actor {actor_name} produced no prompt summaries.")

    oracle_key = _oracle_metric_key(num_samples_per_prompt)
    entropy_moments = RunningMoments()
    length_moments = RunningMoments()
    for worker_summary in worker_summaries:
        entropy_moments.merge(RunningMoments(**worker_summary["response_entropy_moments"]))
        length_moments.merge(RunningMoments(**worker_summary["response_length_moments"]))

    actor_metrics = {
        "actor_checkpoint_dir": actor_checkpoint_dir,
        "num_prompts": int(len(prompt_summaries)),
        "num_responses": int(sum(int(worker_summary["num_responses"]) for worker_summary in worker_summaries)),
        "sampled_single_accuracy": float(
            np.mean([float(prompt_summary["sampled_single_accuracy"]) for prompt_summary in prompt_summaries])
        ),
        "mean_sample_accuracy": float(
            np.mean([float(prompt_summary["mean_sample_accuracy"]) for prompt_summary in prompt_summaries])
        ),
        "oracle_best_of_n_accuracy": float(
            np.mean([float(prompt_summary["oracle_best_of_n_accuracy"]) for prompt_summary in prompt_summaries])
        ),
        "fraction_prompts_with_at_least_one_success": float(
            np.mean([float(prompt_summary["has_success"]) for prompt_summary in prompt_summaries])
        ),
        "mean_num_successes_per_prompt": float(
            np.mean([float(prompt_summary["num_successes_in_bank"]) for prompt_summary in prompt_summaries])
        ),
        "mean_success_rate_in_bank": float(
            np.mean([float(prompt_summary["success_rate_in_bank"]) for prompt_summary in prompt_summaries])
        ),
        "mean_outcome_variance_within_prompt": float(
            np.mean([float(prompt_summary["outcome_variance"]) for prompt_summary in prompt_summaries])
        ),
        "mean_response_entropy": entropy_moments.mean(),
        "std_response_entropy_across_responses": entropy_moments.std(),
        "mean_response_length": length_moments.mean(),
        "mean_distinct_responses_per_prompt": float(
            np.mean([float(prompt_summary["num_distinct_responses"]) for prompt_summary in prompt_summaries])
        ),
        "mean_distinct_response_fraction": float(
            np.mean([float(prompt_summary["distinct_response_fraction"]) for prompt_summary in prompt_summaries])
        ),
        "mean_duplicate_response_fraction": float(
            np.mean([float(prompt_summary["duplicate_response_fraction"]) for prompt_summary in prompt_summaries])
        ),
    }
    actor_metrics[oracle_key] = actor_metrics["oracle_best_of_n_accuracy"]
    return actor_metrics


def _sorted_prompt_summaries(prompt_summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(prompt_summaries, key=lambda row: int(row["prompt_index"]))


def _pairwise_prompt_metric_differences(
    *,
    actor_a_prompts: Sequence[dict[str, Any]],
    actor_b_prompts: Sequence[dict[str, Any]],
) -> dict[str, list[float]]:
    prompts_a = _sorted_prompt_summaries(actor_a_prompts)
    prompts_b = _sorted_prompt_summaries(actor_b_prompts)
    if len(prompts_a) != len(prompts_b):
        raise ValueError("Actors must be evaluated on the same number of prompts for paired comparison.")

    differences: dict[str, list[float]] = {
        "sampled_single_accuracy": [],
        "mean_sample_accuracy": [],
        "oracle_best_of_n_accuracy": [],
        "fraction_prompts_with_at_least_one_success": [],
        "mean_outcome_variance_within_prompt": [],
        "mean_response_entropy": [],
        "mean_distinct_response_fraction": [],
        "mean_response_length": [],
    }
    for prompt_a, prompt_b in zip(prompts_a, prompts_b, strict=True):
        if int(prompt_a["prompt_index"]) != int(prompt_b["prompt_index"]):
            raise ValueError(
                "Actors must share the same prompt ordering for paired comparison, but prompt indices "
                f"{prompt_a['prompt_index']} and {prompt_b['prompt_index']} were encountered."
            )
        differences["sampled_single_accuracy"].append(
            float(prompt_a["sampled_single_accuracy"]) - float(prompt_b["sampled_single_accuracy"])
        )
        differences["mean_sample_accuracy"].append(
            float(prompt_a["mean_sample_accuracy"]) - float(prompt_b["mean_sample_accuracy"])
        )
        differences["oracle_best_of_n_accuracy"].append(
            float(prompt_a["oracle_best_of_n_accuracy"]) - float(prompt_b["oracle_best_of_n_accuracy"])
        )
        differences["fraction_prompts_with_at_least_one_success"].append(
            float(prompt_a["has_success"]) - float(prompt_b["has_success"])
        )
        differences["mean_outcome_variance_within_prompt"].append(
            float(prompt_a["outcome_variance"]) - float(prompt_b["outcome_variance"])
        )
        differences["mean_response_entropy"].append(
            float(prompt_a["mean_bank_response_entropy"]) - float(prompt_b["mean_bank_response_entropy"])
        )
        differences["mean_distinct_response_fraction"].append(
            float(prompt_a["distinct_response_fraction"]) - float(prompt_b["distinct_response_fraction"])
        )
        differences["mean_response_length"].append(
            float(prompt_a["mean_bank_response_length"]) - float(prompt_b["mean_bank_response_length"])
        )
    return differences


def build_pairwise_comparisons(
    *,
    actor_prompt_summaries: dict[str, list[dict[str, Any]]],
    actor_names: Sequence[str],
    bootstrap_samples: int,
    base_seed: int,
    num_samples_per_prompt: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    comparisons: dict[str, dict[str, Any]] = {}
    paired_bootstrap: dict[str, dict[str, Any]] = {}
    oracle_key = _oracle_metric_key(num_samples_per_prompt)

    for pair_index, (actor_a, actor_b) in enumerate(combinations(actor_names, 2)):
        comparison_name = f"{actor_a}_minus_{actor_b}"
        differences = _pairwise_prompt_metric_differences(
            actor_a_prompts=actor_prompt_summaries[actor_a],
            actor_b_prompts=actor_prompt_summaries[actor_b],
        )

        comparisons[comparison_name] = {
            "comparison_name": comparison_name,
            "sampled_single_accuracy_diff": float(np.mean(differences["sampled_single_accuracy"])),
            "mean_sample_accuracy_diff": float(np.mean(differences["mean_sample_accuracy"])),
            "oracle_best_of_n_accuracy_diff": float(np.mean(differences["oracle_best_of_n_accuracy"])),
            "fraction_prompts_with_at_least_one_success_diff": float(
                np.mean(differences["fraction_prompts_with_at_least_one_success"])
            ),
            "mean_outcome_variance_within_prompt_diff": float(
                np.mean(differences["mean_outcome_variance_within_prompt"])
            ),
            "mean_response_entropy_diff": float(np.mean(differences["mean_response_entropy"])),
            "mean_distinct_response_fraction_diff": float(
                np.mean(differences["mean_distinct_response_fraction"])
            ),
            "mean_response_length_diff": float(np.mean(differences["mean_response_length"])),
        }
        comparisons[comparison_name][f"{oracle_key}_diff"] = comparisons[comparison_name]["oracle_best_of_n_accuracy_diff"]

        paired_bootstrap[comparison_name] = {
            "sampled_single_accuracy": _paired_bootstrap_from_differences(
                differences["sampled_single_accuracy"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 1,
            ),
            "mean_sample_accuracy": _paired_bootstrap_from_differences(
                differences["mean_sample_accuracy"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 2,
            ),
            "oracle_best_of_n_accuracy": _paired_bootstrap_from_differences(
                differences["oracle_best_of_n_accuracy"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 3,
            ),
            "fraction_prompts_with_at_least_one_success": _paired_bootstrap_from_differences(
                differences["fraction_prompts_with_at_least_one_success"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 4,
            ),
            "mean_outcome_variance_within_prompt": _paired_bootstrap_from_differences(
                differences["mean_outcome_variance_within_prompt"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 5,
            ),
            "mean_response_entropy": _paired_bootstrap_from_differences(
                differences["mean_response_entropy"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 6,
            ),
            "distinct_response_fraction": _paired_bootstrap_from_differences(
                differences["mean_distinct_response_fraction"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 7,
            ),
            "response_length": _paired_bootstrap_from_differences(
                differences["mean_response_length"],
                bootstrap_samples=bootstrap_samples,
                seed=base_seed + pair_index * 10_000 + 8,
            ),
        }
        paired_bootstrap[comparison_name][oracle_key] = paired_bootstrap[comparison_name]["oracle_best_of_n_accuracy"]

    return comparisons, paired_bootstrap


def _plot_actor_headroom(
    *,
    actor_metrics: dict[str, dict[str, Any]],
    actor_names: Sequence[str],
    num_samples_per_prompt: int,
    output_path: Path,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plot generation but could not be imported."
        ) from MATPLOTLIB_IMPORT_ERROR

    oracle_key = _oracle_metric_key(num_samples_per_prompt)
    x_positions = np.arange(len(actor_names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.0, 4.8))

    axis.bar(
        x_positions - width,
        [float(actor_metrics[name]["sampled_single_accuracy"]) for name in actor_names],
        width=width,
        label="Sampled single",
    )
    axis.bar(
        x_positions,
        [float(actor_metrics[name]["mean_sample_accuracy"]) for name in actor_names],
        width=width,
        label="Mean sample",
    )
    axis.bar(
        x_positions + width,
        [float(actor_metrics[name][oracle_key]) for name in actor_names],
        width=width,
        label=f"Oracle best-of-{num_samples_per_prompt}",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(list(actor_names), rotation=15, ha="right")
    axis.set_ylabel("Task Score / Accuracy")
    axis.set_title("Actor Proposal Headroom")
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_prompt_scatter(
    *,
    actor_prompt_summaries: dict[str, list[dict[str, Any]]],
    actor_names: Sequence[str],
    x_field: str,
    y_field: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plot generation but could not be imported."
        ) from MATPLOTLIB_IMPORT_ERROR

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for actor_name in actor_names:
        rows = _sorted_prompt_summaries(actor_prompt_summaries[actor_name])
        xs = [float(row[x_field]) for row in rows]
        ys = [float(row[y_field]) for row in rows]
        axis.scatter(xs, ys, alpha=0.7, s=28, label=actor_name)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_outcome_variance(
    *,
    actor_metrics: dict[str, dict[str, Any]],
    actor_names: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plot generation but could not be imported."
        ) from MATPLOTLIB_IMPORT_ERROR

    x_positions = np.arange(len(actor_names))
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.bar(
        x_positions,
        [float(actor_metrics[name]["mean_outcome_variance_within_prompt"]) for name in actor_names],
        width=0.55,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(list(actor_names), rotation=15, ha="right")
    axis.set_ylabel("Mean Within-Prompt Outcome Variance")
    axis.set_title("Outcome Variance by Actor")
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _write_output_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    actor_names: Sequence[str],
    actor_metrics: dict[str, dict[str, Any]],
) -> None:
    oracle_phrase = f"oracle_best_of_{args.num_samples_per_prompt}_accuracy"
    lines: list[str] = [
        "# Actor Proposal Quality Diagnostic",
        "",
        "This diagnostic measures the proposal quality and search headroom of each frozen actor.",
        "",
        "Key quantities:",
        "- sampled_single_accuracy: pass@1-like performance from one sample.",
        f"- {oracle_phrase}: upper bound if a perfect selector chooses the best response in the bank.",
        "- fraction_prompts_with_at_least_one_success: how often search has any correct option available.",
        "- outcome_variance_within_prompt: whether sampled responses from the same prompt have mixed outcomes; high variance means there is meaningful selection opportunity.",
        "- response_entropy and distinct_response_fraction: proposal diversity diagnostics.",
        "",
        "Interpretation:",
        f"- High {oracle_phrase} with low sampled_single_accuracy means large search headroom.",
        f"- Low {oracle_phrase} means chunk/value guidance cannot help much because the actor rarely proposes correct responses.",
        "- High entropy/diversity may create more search headroom, but may also make critic ranking harder in later experiments.",
        "",
        "Implementation notes:",
        "- Token entropy is computed from the raw actor next-token distribution before temperature/top-p/top-k sampling is applied, matching the requested experiment definition.",
        "- Distinct responses are computed after `response.strip()` normalization.",
        "",
        "Run config:",
        f"- Dataset: `{args.dataset_path}`",
        f"- Actor names: `{list(actor_names)}`",
        f"- Max examples: `{args.max_examples}`",
        f"- Num samples per prompt: `{args.num_samples_per_prompt}`",
        f"- Temperature / top-p / top-k: `{args.temperature}` / `{args.top_p}` / `{args.top_k}`",
        f"- Seed: `{args.seed}`",
        f"- Worker devices: `{args.worker_devices}`",
        "",
        "Files:",
        "- `response_bank.jsonl`: one row per sampled response with task score, entropy diagnostics, and optional token ids / entropy traces.",
        "- `prompt_summary.jsonl`: one row per actor/prompt with headroom, diversity, entropy, and length aggregates.",
        "- `summary_metrics.json`: actor-level aggregates, pairwise comparisons, and paired bootstrap confidence intervals.",
        "- `actor_headroom_accuracy.png`: grouped bar chart for sampled single, mean sample, and oracle best-of-N accuracy.",
        "- `entropy_vs_oracle_success.png`: prompt-level entropy vs oracle best-of-N success scatter plot.",
        "- `diversity_vs_oracle_success.png`: prompt-level diversity vs oracle best-of-N success scatter plot.",
        "- `outcome_variance_by_actor.png`: actor-level within-prompt outcome variance bar chart.",
        "",
        "Quick read:",
    ]
    for actor_name in actor_names:
        metrics = actor_metrics[actor_name]
        lines.append(
            f"- {actor_name}: sampled_single={metrics['sampled_single_accuracy']:.6f}, "
            f"mean_sample={metrics['mean_sample_accuracy']:.6f}, "
            f"oracle_best_of_n={metrics['oracle_best_of_n_accuracy']:.6f}, "
            f"has_success={metrics['fraction_prompts_with_at_least_one_success']:.6f}, "
            f"mean_entropy={metrics['mean_response_entropy']:.6f}, "
            f"distinct_fraction={metrics['mean_distinct_response_fraction']:.6f}"
        )

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if len(args.actor_checkpoint_dirs) != len(args.actor_names):
        raise ValueError(
            "--actor_checkpoint_dirs and --actor_names must have the same number of entries "
            f"(received {len(args.actor_checkpoint_dirs)} and {len(args.actor_names)})."
        )
    if len(set(args.actor_names)) != len(args.actor_names):
        raise ValueError("--actor_names must be unique so outputs can be keyed by actor name.")
    if args.max_examples <= 0:
        raise ValueError(f"--max_examples must be > 0, got {args.max_examples}")
    if args.num_samples_per_prompt <= 0:
        raise ValueError(f"--num_samples_per_prompt must be > 0, got {args.num_samples_per_prompt}")
    if args.max_prompt_length <= 0:
        raise ValueError(f"--max_prompt_length must be > 0, got {args.max_prompt_length}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max_new_tokens must be > 0, got {args.max_new_tokens}")
    if args.bootstrap_samples <= 0:
        raise ValueError(f"--bootstrap_samples must be > 0, got {args.bootstrap_samples}")
    if args.ray_num_cpus_per_worker <= 0:
        raise ValueError(f"--ray_num_cpus_per_worker must be > 0, got {args.ray_num_cpus_per_worker}")
    if args.top_k < 0:
        raise ValueError(f"--top_k must be >= 0, got {args.top_k}")
    if not args.skip_plots and plt is None:
        raise RuntimeError(
            "matplotlib is required for plot generation but could not be imported."
        ) from MATPLOTLIB_IMPORT_ERROR

    num_actors = len(args.actor_names)
    actor_merged_roots = _resolve_optional_per_actor_list(
        args.actor_merged_roots,
        num_actors=num_actors,
        argument_name="--actor_merged_roots",
    )
    actor_hf_source_dirs = _resolve_optional_per_actor_list(
        args.actor_hf_source_dirs,
        num_actors=num_actors,
        argument_name="--actor_hf_source_dirs",
    )

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_root = output_dir / "_worker_tmp"
    worker_root.mkdir(parents=True, exist_ok=True)

    actor_specs = [
        ActorSpec(
            actor_index=actor_index,
            actor_name=actor_name,
            checkpoint_dir=str(Path(checkpoint_dir).resolve()),
            merged_root=None if actor_merged_roots[actor_index] is None else str(Path(actor_merged_roots[actor_index]).resolve()),
            hf_source_dir=(
                None
                if actor_hf_source_dirs[actor_index] is None
                else str(Path(actor_hf_source_dirs[actor_index]).resolve())
            ),
        )
        for actor_index, (actor_name, checkpoint_dir) in enumerate(
            zip(args.actor_names, args.actor_checkpoint_dirs, strict=True)
        )
    ]

    actor_hf_dirs: dict[str, Path] = {}
    tokenizer_fingerprints: dict[str, dict[str, Any]] = {}
    eos_token_ids_by_actor: dict[str, tuple[int, ...]] = {}
    for actor_spec in actor_specs:
        actor_hf_dir = ensure_merged_component_checkpoint(
            Path(actor_spec.checkpoint_dir),
            component="actor",
            merged_root=Path(actor_spec.merged_root) if actor_spec.merged_root else None,
            hf_source_dir=Path(actor_spec.hf_source_dir) if actor_spec.hf_source_dir else None,
            skip_merge=args.skip_merge,
        )
        actor_hf_dirs[actor_spec.actor_name] = actor_hf_dir
        tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
        tokenizer_fingerprints[actor_spec.actor_name] = _tokenizer_fingerprint(actor_hf_dir)
        eos_token_ids_by_actor[actor_spec.actor_name] = resolve_eos_token_ids(actor_hf_dir, tokenizer)
    _assert_shared_tokenizer(tokenizer_fingerprints)

    reference_tokenizer = load_tokenizer(
        actor_hf_dirs[actor_specs[0].actor_name],
        trust_remote_code=args.trust_remote_code,
    )
    examples = load_examples(
        args.dataset_path,
        tokenizer=reference_tokenizer,
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

    worker_devices = [device if device else None for device in (args.worker_devices or [None])]
    ray_address = _resolve_ray_address(args.ray_address)
    execution_backend = "ray" if ray_address is not None else "local"
    ray_nodes: list[RayNodeInfo] = []
    if ray_address is not None:
        ray_module = _require_ray()
        if not ray_module.is_initialized():
            ray_module.init(address=ray_address)
        ray_nodes = _discover_ray_nodes(ray_module)
        if not ray_nodes:
            raise ValueError("Ray is connected, but no alive Ray nodes were discovered.")
        worker_assignments = _build_distributed_worker_assignments(
            num_prompts=len(examples),
            worker_devices=worker_devices,
            ray_nodes=ray_nodes,
        )
    else:
        worker_assignments = _build_worker_assignments(
            num_prompts=len(examples),
            worker_devices=worker_devices,
        )
    multi_worker_enabled = len(worker_assignments) > 1

    actor_prompt_summaries: dict[str, list[dict[str, Any]]] = {}
    actor_metrics: dict[str, dict[str, Any]] = {}
    actor_worker_summaries: dict[str, list[dict[str, Any]]] = {}

    response_bank_path = output_dir / "response_bank.jsonl"
    prompt_summary_path = output_dir / "prompt_summary.jsonl"
    summary_metrics_path = output_dir / "summary_metrics.json"
    actor_headroom_path = output_dir / "actor_headroom_accuracy.png"
    entropy_scatter_path = output_dir / "entropy_vs_oracle_success.png"
    diversity_scatter_path = output_dir / "diversity_vs_oracle_success.png"
    outcome_variance_path = output_dir / "outcome_variance_by_actor.png"

    with response_bank_path.open("w", encoding="utf-8") as response_bank_file, prompt_summary_path.open(
        "w",
        encoding="utf-8",
    ) as prompt_summary_file:
        for actor_spec in actor_specs:
            actor_worker_dir = worker_root / actor_spec.actor_name
            if actor_worker_dir.exists():
                shutil.rmtree(actor_worker_dir)
            actor_worker_dir.mkdir(parents=True, exist_ok=True)

            eos_token_ids = eos_token_ids_by_actor[actor_spec.actor_name]
            actor_hf_dir = actor_hf_dirs[actor_spec.actor_name]

            if ray_address is not None:
                prompt_summaries, worker_summaries = run_ray_multi_worker(
                    output_dir=actor_worker_dir,
                    actor_spec=actor_spec,
                    actor_hf_dir=actor_hf_dir,
                    examples=examples,
                    worker_assignments=worker_assignments,
                    args=args,
                    eos_token_ids=eos_token_ids,
                    response_bank_file=response_bank_file,
                    prompt_summary_file=prompt_summary_file,
                )
            elif len(worker_assignments) == 1:
                progress_bar = tqdm(
                    total=len(examples),
                    desc=f"actor_proposal_diagnostic[{actor_spec.actor_name}]",
                    unit="prompt",
                    dynamic_ncols=True,
                )
                try:
                    _worker_entry(
                        actor_spec=actor_spec,
                        actor_hf_dir=str(actor_hf_dir),
                        assignment=worker_assignments[0],
                        examples=examples,
                        dtype_name=args.dtype,
                        trust_remote_code=args.trust_remote_code,
                        max_prompt_length=args.max_prompt_length,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        num_samples_per_prompt=args.num_samples_per_prompt,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        base_seed=args.seed,
                        omit_prompt_text=args.omit_prompt_text,
                        omit_response_token_ids=args.omit_response_token_ids,
                        store_token_entropies=args.store_token_entropies,
                        use_actor_cache=not args.disable_actor_cache,
                        worker_root=str(actor_worker_dir),
                        progress_queue=None,
                    )
                finally:
                    progress_bar.update(len(examples))
                    progress_bar.close()
                prompt_summaries, worker_summaries = _collect_worker_outputs(
                    worker_root=actor_worker_dir,
                    assignments=worker_assignments,
                    response_bank_file=response_bank_file,
                    prompt_summary_file=prompt_summary_file,
                )
            else:
                progress_queue, processes = _start_worker_processes(
                    actor_spec=actor_spec,
                    actor_hf_dir=actor_hf_dir,
                    assignments=worker_assignments,
                    examples=examples,
                    args=args,
                    eos_token_ids=eos_token_ids,
                    worker_root=actor_worker_dir,
                )
                worker_progress: dict[int, dict[str, Any]] = {
                    assignment.worker_id: {"done": 0, "total": assignment.num_prompts}
                    for assignment in worker_assignments
                }
                completed_prompts = 0
                completed_workers = 0
                with tqdm(
                    total=len(examples),
                    desc=f"actor_proposal_diagnostic[{actor_spec.actor_name}]",
                    unit="prompt",
                    dynamic_ncols=True,
                ) as progress_bar:
                    progress_bar.set_postfix_str(_progress_postfix(worker_progress))
                    while True:
                        alive = any(process.is_alive() for process, _assignment in processes)
                        try:
                            event = progress_queue.get(timeout=0.2)
                        except Empty:
                            _assert_worker_processes_healthy(processes=processes, worker_root=actor_worker_dir)
                            if not alive:
                                break
                            continue
                        event_type = event.get("type")
                        worker_id = int(event.get("worker_id", -1))
                        if event_type == "worker_started":
                            worker_progress.setdefault(worker_id, {})
                            worker_progress[worker_id]["total"] = int(event.get("worker_total_prompts", 0))
                        elif event_type == "prompt_done":
                            completed_prompts += 1
                            worker_progress.setdefault(worker_id, {})
                            worker_progress[worker_id]["done"] = int(event.get("worker_completed_prompts", 0))
                            worker_progress[worker_id]["total"] = int(event.get("worker_total_prompts", 0))
                            progress_bar.update(1)
                        elif event_type == "worker_done":
                            completed_workers += 1
                            worker_progress.setdefault(worker_id, {})
                            worker_progress[worker_id]["done"] = int(event.get("worker_completed_prompts", 0))
                            worker_progress[worker_id]["total"] = int(event.get("worker_total_prompts", 0))
                        elif event_type == "worker_error":
                            raise RuntimeError(
                                f"Worker {worker_id} reported an error.\n"
                                f"{event.get('traceback', 'No traceback provided.')}"
                            )
                        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
                        _assert_worker_processes_healthy(processes=processes, worker_root=actor_worker_dir)
                        if (
                            not alive
                            and completed_prompts >= len(examples)
                            and completed_workers >= len(worker_assignments)
                        ):
                            break

                _join_worker_processes(processes=processes, worker_root=actor_worker_dir)
                prompt_summaries, worker_summaries = _collect_worker_outputs(
                    worker_root=actor_worker_dir,
                    assignments=worker_assignments,
                    response_bank_file=response_bank_file,
                    prompt_summary_file=prompt_summary_file,
                )
            actor_prompt_summaries[actor_spec.actor_name] = prompt_summaries
            actor_worker_summaries[actor_spec.actor_name] = worker_summaries
            actor_metrics[actor_spec.actor_name] = _aggregate_actor_metrics(
                actor_name=actor_spec.actor_name,
                actor_checkpoint_dir=actor_spec.checkpoint_dir,
                prompt_summaries=prompt_summaries,
                worker_summaries=worker_summaries,
                num_samples_per_prompt=args.num_samples_per_prompt,
            )

    pairwise_comparisons: dict[str, dict[str, Any]] = {}
    paired_bootstrap: dict[str, dict[str, Any]] = {}
    if len(actor_specs) >= 2:
        pairwise_comparisons, paired_bootstrap = build_pairwise_comparisons(
            actor_prompt_summaries=actor_prompt_summaries,
            actor_names=[actor_spec.actor_name for actor_spec in actor_specs],
            bootstrap_samples=args.bootstrap_samples,
            base_seed=args.seed + 50_000,
            num_samples_per_prompt=args.num_samples_per_prompt,
        )

    if not args.skip_plots:
        _plot_actor_headroom(
            actor_metrics=actor_metrics,
            actor_names=[actor_spec.actor_name for actor_spec in actor_specs],
            num_samples_per_prompt=args.num_samples_per_prompt,
            output_path=actor_headroom_path,
            dpi=args.plot_dpi,
        )
        _plot_prompt_scatter(
            actor_prompt_summaries=actor_prompt_summaries,
            actor_names=[actor_spec.actor_name for actor_spec in actor_specs],
            x_field="mean_bank_response_entropy",
            y_field="oracle_best_of_n_accuracy",
            xlabel="Mean Bank Response Entropy",
            ylabel=f"Oracle Best-of-{args.num_samples_per_prompt} Task Score",
            title="Entropy vs Oracle Success",
            output_path=entropy_scatter_path,
            dpi=args.plot_dpi,
        )
        _plot_prompt_scatter(
            actor_prompt_summaries=actor_prompt_summaries,
            actor_names=[actor_spec.actor_name for actor_spec in actor_specs],
            x_field="distinct_response_fraction",
            y_field="oracle_best_of_n_accuracy",
            xlabel="Distinct Response Fraction",
            ylabel=f"Oracle Best-of-{args.num_samples_per_prompt} Task Score",
            title="Diversity vs Oracle Success",
            output_path=diversity_scatter_path,
            dpi=args.plot_dpi,
        )
        _plot_outcome_variance(
            actor_metrics=actor_metrics,
            actor_names=[actor_spec.actor_name for actor_spec in actor_specs],
            output_path=outcome_variance_path,
            dpi=args.plot_dpi,
        )

    _write_output_readme(
        output_dir=output_dir,
        args=args,
        actor_names=[actor_spec.actor_name for actor_spec in actor_specs],
        actor_metrics=actor_metrics,
    )

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "execution_backend": execution_backend,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "output_dir": str(output_dir),
        "response_bank_path": str(response_bank_path),
        "prompt_summary_path": str(prompt_summary_path),
        "summary_metrics_path": str(summary_metrics_path),
        "actor_headroom_accuracy_path": None if args.skip_plots else str(actor_headroom_path),
        "entropy_vs_oracle_success_path": None if args.skip_plots else str(entropy_scatter_path),
        "diversity_vs_oracle_success_path": None if args.skip_plots else str(diversity_scatter_path),
        "outcome_variance_by_actor_path": None if args.skip_plots else str(outcome_variance_path),
        "num_prompts": int(len(examples)),
        "num_samples_per_prompt": int(args.num_samples_per_prompt),
        "sampling_config": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_new_tokens": int(args.max_new_tokens),
            "max_prompt_length": int(args.max_prompt_length),
        },
        "actors": actor_metrics,
        "pairwise_comparisons": pairwise_comparisons,
        "paired_bootstrap": paired_bootstrap,
        "ray_address": ray_address,
        "ray_nodes": [asdict(node) for node in ray_nodes],
        "multi_worker_enabled": multi_worker_enabled,
        "worker_devices": worker_devices,
        "worker_assignments": [asdict(assignment) for assignment in worker_assignments],
        "worker_summaries": actor_worker_summaries,
        "tokenizer_fingerprints": tokenizer_fingerprints,
        "eos_token_ids_by_actor": {
            actor_name: list(token_ids) for actor_name, token_ids in eos_token_ids_by_actor.items()
        },
        "run_args": vars(args),
    }
    with summary_metrics_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    return 0


_make_main_module_importable()

if __name__ == "__main__":
    raise SystemExit(main())
