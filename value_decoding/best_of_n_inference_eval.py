from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
from queue import Empty
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
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
    critic_sequence_last_values,
    sample_token_from_actor,
    set_decode_seed,
)


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled explicitly at runtime
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

METHOD_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "random_single_sample",
            {
                "kind": "random",
                "selection_score_field": None,
                "trajectory_value_field": None,
                "label": "Random single sample",
            },
        ),
        (
            "best_of_n_old_critic",
            {
                "kind": "argmax",
                "selection_score_field": "old_critic_final_trajectory_value",
                "trajectory_value_field": "old_critic_final_trajectory_value",
                "label": "Old critic",
            },
        ),
        (
            "best_of_n_new_critic",
            {
                "kind": "argmax",
                "selection_score_field": "new_critic_final_trajectory_value",
                "trajectory_value_field": "new_critic_final_trajectory_value",
                "label": "New critic",
            },
        ),
        (
            "best_of_n_actor_logprob",
            {
                "kind": "argmax",
                "selection_score_field": "actor_response_logprob",
                "trajectory_value_field": None,
                "label": "Actor log-prob",
            },
        ),
        (
            "best_of_n_actor_avg_logprob",
            {
                "kind": "argmax",
                "selection_score_field": "actor_response_avg_logprob",
                "trajectory_value_field": None,
                "label": "Actor avg log-prob",
            },
        ),
        (
            "oracle_best_in_bank",
            {
                "kind": "argmax",
                "selection_score_field": "task_score",
                "trajectory_value_field": None,
                "label": "Oracle best in bank",
            },
        ),
    ]
)

DEFAULT_N_VALUES = (1, 2, 4, 8, 16)
PRIMARY_PLOT_METHODS = (
    "random_single_sample",
    "best_of_n_old_critic",
    "best_of_n_new_critic",
    "best_of_n_actor_logprob",
    "oracle_best_in_bank",
)
REFERENCE_CRITIC_VALUE_ABS_TOL = 1e-2
RAY_NODE_RESOURCE_FRACTION = 1e-3
RAY_PROGRESS_POLL_INTERVAL_SEC = 0.2


def _make_main_module_importable() -> None:
    """Allow Ray/multiprocessing to re-import this module when launched via `python -m`."""
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
class SampledTrajectory:
    sample_idx: int
    sample_seed: int
    prompt_length: int
    full_sequence_token_ids: tuple[int, ...]
    response_text: str
    response_length: int
    eos_emitted: bool
    max_length_hit: bool
    task_score: float
    actor_response_logprob: float
    actor_response_avg_logprob: float


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    actor_device: str | None
    old_critic_device: str | None
    new_critic_device: str | None
    example_start: int
    example_end: int
    node_index: int | None = None
    node_ip: str | None = None
    node_resource_key: str | None = None
    local_worker_index: int | None = None

    @property
    def num_examples(self) -> int:
        return max(self.example_end - self.example_start, 0)


@dataclass(frozen=True)
class RayNodeInfo:
    node_index: int
    node_ip: str
    node_resource_key: str
    node_name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 end-to-end best-of-N inference evaluation. Generate a shared maximum response bank once "
            "per prompt, then evaluate multiple response-level selectors across N."
        )
    )
    parser.add_argument("--actor_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the frozen actor.")
    parser.add_argument(
        "--old_critic_checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint dir for the baseline critic.",
    )
    parser.add_argument(
        "--new_critic_checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint dir for the larger/new critic.",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for experiment artifacts.")
    parser.add_argument("--actor_merged_root", type=str, default=None, help="Optional merged HF root for actor.")
    parser.add_argument(
        "--actor_hf_source_dir",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face config/tokenizer directory for the actor merge. "
            "Use this when the raw actor checkpoint was copied without actor/huggingface."
        ),
    )
    parser.add_argument(
        "--old_critic_merged_root",
        type=str,
        default=None,
        help="Optional merged HF root for the old critic.",
    )
    parser.add_argument(
        "--old_critic_hf_source_dir",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face config/tokenizer directory for the old critic merge. "
            "Use this when the raw old critic checkpoint was copied without critic/huggingface."
        ),
    )
    parser.add_argument(
        "--new_critic_merged_root",
        type=str,
        default=None,
        help="Optional merged HF root for the new critic.",
    )
    parser.add_argument(
        "--new_critic_hf_source_dir",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face config/tokenizer directory for the new critic merge. "
            "Use this when the raw new critic checkpoint was copied without critic/huggingface."
        ),
    )
    parser.add_argument(
        "--reference_stage1_trajectory_bank",
        type=str,
        default=None,
        help=(
            "Optional Stage 1 trajectory_bank.jsonl for overlap validation. "
            "When provided, matching prompt/sample pairs must reproduce the same prompt slices, seeds, and "
            "generated responses exactly; critic values are checked within a small absolute tolerance to allow "
            "for legacy Stage 1 scoring drift."
        ),
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None, help="Optional response/ground-truth column key.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--n_values",
        nargs="+",
        type=int,
        default=list(DEFAULT_N_VALUES),
        help="Best-of-N sweep values. The full bank is generated once using max(n_values).",
    )
    parser.add_argument(
        "--max_bank_size",
        type=int,
        default=None,
        help="Optional explicit maximum bank size. Defaults to max(n_values).",
    )
    parser.add_argument("--critic_score_batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Fallback device for any unset model device.")
    parser.add_argument("--actor_device", type=str, default=None, help="Optional actor device override.")
    parser.add_argument("--old_critic_device", type=str, default=None, help="Optional old critic device override.")
    parser.add_argument("--new_critic_device", type=str, default=None, help="Optional new critic device override.")
    parser.add_argument(
        "--worker_layouts",
        nargs="+",
        default=None,
        help=(
            "Optional multi-worker device layouts. Each entry should be "
            "'actor_device,old_critic_device,new_critic_device' or a single device to reuse for all three."
        ),
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help=(
            "Optional Ray cluster address for cross-node execution. When set, --worker_layouts is treated as the "
            "node-local worker layout and is replicated across all alive Ray nodes. Use 'auto' to read $RAY_ADDRESS."
        ),
    )
    parser.add_argument(
        "--ray_num_cpus_per_worker",
        type=float,
        default=1.0,
        help="CPU resources reserved per Ray worker task when --ray_address is used.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--disable_actor_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--actor_sampling_mode",
        type=str,
        default=ActorSamplingMode.SAMPLE.value,
        choices=[mode.value for mode in ActorSamplingMode],
    )
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--actor_top_p", type=float, default=1.0)
    parser.add_argument("--actor_top_k", type=int, default=0)
    parser.add_argument("--skip_plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--plot_dpi", type=int, default=160)
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


def _resolve_model_device(explicit_device: str | None, fallback_device: str | None) -> torch.device:
    return resolve_device(explicit_device or fallback_device)


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _assignment_ranges(*, num_examples: int, num_workers: int) -> list[tuple[int, int]]:
    if num_examples <= 0:
        return []
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0 when examples are present.")

    active_workers = min(num_workers, num_examples)
    ranges: list[tuple[int, int]] = []
    start = 0
    base = num_examples // active_workers
    remainder = num_examples % active_workers
    for worker_id in range(active_workers):
        shard_size = base + (1 if worker_id < remainder else 0)
        end = start + shard_size
        ranges.append((start, end))
        start = end
    return ranges


def parse_worker_layouts(
    worker_layouts: list[str] | None,
    *,
    actor_device: str | None,
    old_critic_device: str | None,
    new_critic_device: str | None,
    default_device: str | None,
) -> list[tuple[str | None, str | None, str | None]]:
    if worker_layouts:
        parsed: list[tuple[str | None, str | None, str | None]] = []
        for raw_layout in worker_layouts:
            value = raw_layout.strip()
            if not value:
                continue
            parts = [part.strip() or None for part in value.split(",")]
            if len(parts) == 1:
                actor = parts[0]
                old_critic = parts[0]
                new_critic = parts[0]
            elif len(parts) == 3:
                actor, old_critic, new_critic = parts
            else:
                raise ValueError(
                    "--worker_layouts entries must be a single device or "
                    "'actor_device,old_critic_device,new_critic_device'"
                )

            actor = actor or actor_device or default_device
            old_critic = old_critic or old_critic_device or default_device or actor
            new_critic = new_critic or new_critic_device or default_device or old_critic or actor
            parsed.append((actor, old_critic, new_critic))

        if not parsed:
            raise ValueError("--worker_layouts was provided, but no valid layouts were parsed.")
        return parsed

    resolved_actor = actor_device or default_device
    resolved_old = old_critic_device or default_device or resolved_actor
    resolved_new = new_critic_device or default_device or resolved_old or resolved_actor
    return [(resolved_actor, resolved_old, resolved_new)]


def build_worker_assignments(
    *,
    num_examples: int,
    worker_layouts: list[tuple[str | None, str | None, str | None]],
) -> list[WorkerAssignment]:
    if not worker_layouts:
        raise ValueError("At least one worker layout is required.")
    if num_examples <= 0:
        return []

    assignments: list[WorkerAssignment] = []
    ranges = _assignment_ranges(num_examples=num_examples, num_workers=len(worker_layouts))
    for worker_id, (start, end) in enumerate(ranges):
        actor_dev, old_dev, new_dev = worker_layouts[worker_id]
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                actor_device=actor_dev,
                old_critic_device=old_dev,
                new_critic_device=new_dev,
                example_start=start,
                example_end=end,
            )
        )
    return assignments


def build_distributed_worker_assignments(
    *,
    num_examples: int,
    worker_layouts: list[tuple[str | None, str | None, str | None]],
    ray_nodes: list[RayNodeInfo],
) -> list[WorkerAssignment]:
    if not worker_layouts:
        raise ValueError("At least one worker layout is required.")
    if not ray_nodes:
        raise ValueError("At least one Ray node is required.")
    if num_examples <= 0:
        return []

    worker_descriptors: list[tuple[RayNodeInfo, int, str | None, str | None, str | None]] = []
    for local_worker_index, (actor_dev, old_dev, new_dev) in enumerate(worker_layouts):
        for node in ray_nodes:
            worker_descriptors.append((node, local_worker_index, actor_dev, old_dev, new_dev))

    assignments: list[WorkerAssignment] = []
    ranges = _assignment_ranges(num_examples=num_examples, num_workers=len(worker_descriptors))
    for worker_id, (start, end) in enumerate(ranges):
        node, local_worker_index, actor_dev, old_dev, new_dev = worker_descriptors[worker_id]
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                actor_device=actor_dev,
                old_critic_device=old_dev,
                new_critic_device=new_dev,
                example_start=start,
                example_end=end,
                node_index=node.node_index,
                node_ip=node.node_ip,
                node_resource_key=node.node_resource_key,
                local_worker_index=local_worker_index,
            )
        )
    return assignments


def worker_assignments_to_jsonable(assignments: Sequence[WorkerAssignment]) -> list[dict[str, Any]]:
    return [asdict(assignment) for assignment in assignments]


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
            f"{label} requested CUDA device {device}, but this worker only sees "
            f"{visible_device_count} CUDA device(s)."
        )


def _emit_progress(*, progress_queue, progress_actor, event: dict[str, Any]) -> None:
    if progress_queue is not None:
        progress_queue.put(event)
        return
    if progress_actor is not None:
        progress_actor.put.remote(event)


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
        raise ImportError("Ray is required for cross-node best-of-n evaluation, but it is not installed.")
    return ray


def _unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_ray_runtime_env(repo_root: Path) -> dict[str, Any]:
    existing_pythonpath = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    env_vars = {
        "PYTHONPATH": os.pathsep.join(_unique_preserving_order([str(repo_root), *existing_pythonpath])),
    }
    for env_name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "PYTHONUNBUFFERED",
        "TIKTOKEN_ENCODINGS_BASE",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_CACHE",
        "UV_CACHE_DIR",
    ):
        value = os.environ.get(env_name)
        if value:
            env_vars[env_name] = value
    return {"env_vars": env_vars}


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
        normalized_assignments: list[tuple[WorkerAssignment, str | None, str | None, str | None]] = []
        referenced_cuda_slots: set[int] = set()

        for assignment in node_assignments:
            actor_device_name = _normalize_cuda_device_name(assignment.actor_device, assume_default_cuda=True)
            old_critic_device_name = _normalize_cuda_device_name(
                assignment.old_critic_device if assignment.old_critic_device is not None else actor_device_name,
                assume_default_cuda=True,
            )
            new_critic_device_name = _normalize_cuda_device_name(
                assignment.new_critic_device if assignment.new_critic_device is not None else old_critic_device_name,
                assume_default_cuda=True,
            )
            normalized_assignments.append(
                (assignment, actor_device_name, old_critic_device_name, new_critic_device_name)
            )
            for device_name in (actor_device_name, old_critic_device_name, new_critic_device_name):
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
        for assignment, actor_device_name, old_critic_device_name, new_critic_device_name in normalized_assignments:
            remapped_assignments.append(
                WorkerAssignment(
                    worker_id=assignment.worker_id,
                    actor_device=_remap_cuda_device_name(actor_device_name, cuda_slot_mapping=cuda_slot_mapping),
                    old_critic_device=_remap_cuda_device_name(
                        old_critic_device_name,
                        cuda_slot_mapping=cuda_slot_mapping,
                    ),
                    new_critic_device=_remap_cuda_device_name(
                        new_critic_device_name,
                        cuda_slot_mapping=cuda_slot_mapping,
                    ),
                    example_start=assignment.example_start,
                    example_end=assignment.example_end,
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


def _tokenizer_fingerprint(model_dir: Path) -> dict[str, Any]:
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
        "Actor/critic tokenizer files do not match, so Stage 2 would not evaluate the same tokenized "
        f"trajectories across all models. Fingerprints: {details}"
    )


def _sample_seed(base_seed: int, *, example_id: int, sample_idx: int) -> int:
    return int(base_seed + (example_id + 1) * 1_000_003 + sample_idx * 9_973)


def _random_selector_seed(base_seed: int, *, example_id: int, bank_size: int) -> int:
    return int(base_seed + (example_id + 1) * 2_000_029 + bank_size * 97 + 17)


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


def sample_actor_trajectory(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    prompt_ids: torch.Tensor,
    sample_idx: int,
    seed: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
) -> SampledTrajectory:
    set_decode_seed(seed)
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    generated_token_ids: list[int] = []
    eos_emitted = False
    actor_response_logprob = 0.0

    for _step_index in range(max_new_tokens):
        logits = actor_state.current_logits
        actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
        selected_token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=actor_sampling_mode,
            temperature=actor_temperature,
            top_p=actor_top_p,
            top_k=actor_top_k,
        )
        actor_response_logprob += float(actor_log_probs[0, selected_token_id].item())
        generated_token_ids.append(selected_token_id)
        actor_state.append(selected_token_id)
        if selected_token_id in eos_token_ids:
            eos_emitted = True
            break

    response_length = len(generated_token_ids)
    max_length_hit = bool(max_new_tokens > 0 and not eos_emitted and response_length >= max_new_tokens)
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = float(score_response(example, response_text))
    actor_response_avg_logprob = actor_response_logprob / max(response_length, 1)
    full_sequence_token_ids = tuple(int(token_id) for token_id in actor_state.sequence_ids[0].detach().cpu().tolist())

    return SampledTrajectory(
        sample_idx=sample_idx,
        sample_seed=seed,
        prompt_length=int(prompt_ids.shape[1]),
        full_sequence_token_ids=full_sequence_token_ids,
        response_text=response_text,
        response_length=response_length,
        eos_emitted=eos_emitted,
        max_length_hit=max_length_hit,
        task_score=task_score,
        actor_response_logprob=float(actor_response_logprob),
        actor_response_avg_logprob=float(actor_response_avg_logprob),
    )


def _pad_sequence_batch(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot pad an empty batch of sequences.")

    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_token_id, device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), device=device, dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError("Encountered an empty sequence while preparing critic inputs.")
        sequence_tensor = torch.tensor(sequence, device=device, dtype=torch.long)
        input_ids[row_index, : len(sequence)] = sequence_tensor
        attention_mask[row_index, : len(sequence)] = 1

    return input_ids, attention_mask


def score_sequences_with_critic(
    critic,
    *,
    sequences: Sequence[Sequence[int]],
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
) -> list[float]:
    if batch_size <= 0:
        raise ValueError(f"critic_score_batch_size must be > 0, got {batch_size}")

    values: list[float] = []
    # Score one completed trajectory at a time to avoid padded-batch numerical drift.
    for sequence in sequences:
        input_ids = torch.tensor([list(sequence)], device=device, dtype=torch.long)
        sequence_value = critic_sequence_last_values(critic, input_ids)[0]
        values.append(float(sequence_value.item()))
    return values


def _load_reference_stage1_bank(reference_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    with reference_path.open("r", encoding="utf-8") as reference_file:
        for line in reference_file:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["example_id"]), int(row["sample_idx"]))
            records[key] = row
    return records


def _validate_reference_row(
    *,
    generated_row: dict[str, Any],
    reference_row: dict[str, Any],
) -> dict[str, float]:
    exact_fields = (
        "example_id",
        "prompt_id",
        "sample_idx",
        "sample_seed",
        "data_source",
        "ground_truth",
        "prompt_text",
        "generated_response",
        "prompt_length",
        "response_length",
        "full_sequence_length",
        "eos_emitted",
        "max_length_hit",
        "task_score",
    )
    for field in exact_fields:
        if generated_row[field] != reference_row[field]:
            raise ValueError(
                f"Stage 2 overlap validation failed for field '{field}' on "
                f"example_id={generated_row['example_id']} sample_idx={generated_row['sample_idx']}: "
                f"generated={generated_row[field]!r} reference={reference_row[field]!r}"
            )

    critic_value_diffs: dict[str, float] = {}
    float_fields = (
        "old_critic_final_trajectory_value",
        "new_critic_final_trajectory_value",
    )
    for field in float_fields:
        diff = abs(float(generated_row[field]) - float(reference_row[field]))
        critic_value_diffs[field] = float(diff)
        if diff > REFERENCE_CRITIC_VALUE_ABS_TOL:
            raise ValueError(
                f"Stage 2 overlap validation failed for field '{field}' on "
                f"example_id={generated_row['example_id']} sample_idx={generated_row['sample_idx']}: "
                f"generated={generated_row[field]!r} reference={reference_row[field]!r}"
            )
    return critic_value_diffs


def _new_stage1_validation_summary(reference_path: Path) -> dict[str, Any]:
    return {
        "reference_stage1_trajectory_bank": str(reference_path),
        "critic_value_abs_tolerance": REFERENCE_CRITIC_VALUE_ABS_TOL,
        "num_rows_compared": 0,
        "max_overlapping_sample_idx": None,
        "max_abs_old_critic_value_diff": 0.0,
        "max_abs_new_critic_value_diff": 0.0,
    }


def _accumulate_stage1_validation_summary(
    target_summary: dict[str, Any] | None,
    update_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if target_summary is None:
        if update_summary is None:
            return None
        return dict(update_summary)
    if update_summary is None:
        return target_summary

    target_summary["num_rows_compared"] = int(target_summary["num_rows_compared"]) + int(update_summary["num_rows_compared"])
    if update_summary["max_overlapping_sample_idx"] is not None:
        target_summary["max_overlapping_sample_idx"] = (
            int(update_summary["max_overlapping_sample_idx"])
            if target_summary["max_overlapping_sample_idx"] is None
            else max(
                int(target_summary["max_overlapping_sample_idx"]),
                int(update_summary["max_overlapping_sample_idx"]),
            )
        )
    target_summary["max_abs_old_critic_value_diff"] = max(
        float(target_summary["max_abs_old_critic_value_diff"]),
        float(update_summary["max_abs_old_critic_value_diff"]),
    )
    target_summary["max_abs_new_critic_value_diff"] = max(
        float(target_summary["max_abs_new_critic_value_diff"]),
        float(update_summary["max_abs_new_critic_value_diff"]),
    )
    return target_summary


def process_example(
    *,
    actor,
    old_critic,
    new_critic,
    actor_tokenizer,
    old_critic_pad_token_id: int,
    new_critic_pad_token_id: int,
    example: ExampleRecord,
    n_values: Sequence[int],
    max_bank_size: int,
    actor_device: torch.device,
    old_critic_device: torch.device,
    new_critic_device: torch.device,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
    critic_score_batch_size: int,
    base_seed: int,
    reference_stage1_bank: dict[tuple[int, int], dict[str, Any]] | None,
    reference_stage1_summary: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    prompt_ids = _prompt_ids_tensor(
        example=example,
        tokenizer=actor_tokenizer,
        max_prompt_length=max_prompt_length,
        device=actor_device,
    )

    sampled_trajectories: list[SampledTrajectory] = []
    for sample_idx in range(max_bank_size):
        sampled_trajectory = sample_actor_trajectory(
            actor=actor,
            tokenizer=actor_tokenizer,
            example=example,
            prompt_ids=prompt_ids,
            sample_idx=sample_idx,
            seed=_sample_seed(base_seed, example_id=example.example_id, sample_idx=sample_idx),
            actor_sampling_mode=actor_sampling_mode,
            actor_temperature=actor_temperature,
            actor_top_p=actor_top_p,
            actor_top_k=actor_top_k,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            use_actor_cache=use_actor_cache,
        )
        sampled_trajectories.append(sampled_trajectory)

    full_sequences = [trajectory.full_sequence_token_ids for trajectory in sampled_trajectories]
    old_values = score_sequences_with_critic(
        old_critic,
        sequences=full_sequences,
        device=old_critic_device,
        pad_token_id=old_critic_pad_token_id,
        batch_size=critic_score_batch_size,
    )
    new_values = score_sequences_with_critic(
        new_critic,
        sequences=full_sequences,
        device=new_critic_device,
        pad_token_id=new_critic_pad_token_id,
        batch_size=critic_score_batch_size,
    )

    local_stage1_summary = None if reference_stage1_summary is None else dict(reference_stage1_summary)
    prompt_trajectory_rows: list[dict[str, Any]] = []
    for sampled_trajectory, old_value, new_value in zip(sampled_trajectories, old_values, new_values, strict=True):
        trajectory_row = {
            "example_id": int(example.example_id),
            "prompt_id": int(example.example_id),
            "sample_idx": int(sampled_trajectory.sample_idx),
            "sample_seed": int(sampled_trajectory.sample_seed),
            "data_source": example.data_source,
            "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
            "prompt_text": example.prompt_text,
            "generated_response": sampled_trajectory.response_text,
            "prompt_length": int(sampled_trajectory.prompt_length),
            "response_length": int(sampled_trajectory.response_length),
            "full_sequence_length": int(len(sampled_trajectory.full_sequence_token_ids)),
            "eos_emitted": bool(sampled_trajectory.eos_emitted),
            "max_length_hit": bool(sampled_trajectory.max_length_hit),
            "task_score": float(sampled_trajectory.task_score),
            "actor_response_logprob": float(sampled_trajectory.actor_response_logprob),
            "actor_response_avg_logprob": float(sampled_trajectory.actor_response_avg_logprob),
            "old_critic_final_trajectory_value": float(old_value),
            "new_critic_final_trajectory_value": float(new_value),
        }

        if reference_stage1_bank is not None and local_stage1_summary is not None:
            reference_key = (int(trajectory_row["example_id"]), int(trajectory_row["sample_idx"]))
            reference_row = reference_stage1_bank.get(reference_key)
            if reference_row is not None:
                critic_value_diffs = _validate_reference_row(
                    generated_row=trajectory_row,
                    reference_row=reference_row,
                )
                local_stage1_summary["num_rows_compared"] += 1
                current_max = local_stage1_summary["max_overlapping_sample_idx"]
                local_stage1_summary["max_overlapping_sample_idx"] = (
                    int(trajectory_row["sample_idx"])
                    if current_max is None
                    else max(int(current_max), int(trajectory_row["sample_idx"]))
                )
                local_stage1_summary["max_abs_old_critic_value_diff"] = max(
                    float(local_stage1_summary["max_abs_old_critic_value_diff"]),
                    float(critic_value_diffs["old_critic_final_trajectory_value"]),
                )
                local_stage1_summary["max_abs_new_critic_value_diff"] = max(
                    float(local_stage1_summary["max_abs_new_critic_value_diff"]),
                    float(critic_value_diffs["new_critic_final_trajectory_value"]),
                )

        prompt_trajectory_rows.append(trajectory_row)

    prompt_summary = build_prompt_summary(
        example=example,
        trajectory_rows=prompt_trajectory_rows,
        n_values=n_values,
        base_seed=base_seed,
    )
    return prompt_trajectory_rows, prompt_summary, local_stage1_summary


def _argmax_indices(values: Sequence[float]) -> list[int]:
    if not values:
        raise ValueError("Cannot take argmax of an empty list.")
    best_value = max(values)
    return [index for index, value in enumerate(values) if value == best_value]


def _select_method_indices(
    *,
    bank_rows: Sequence[dict[str, Any]],
    method_name: str,
    example_id: int,
    base_seed: int,
    bank_size: int,
) -> tuple[int, list[int], float | None]:
    if not bank_rows:
        raise ValueError("Each bank must contain at least one trajectory.")

    method_spec = METHOD_SPECS[method_name]
    if method_spec["kind"] == "random":
        rng = random.Random(_random_selector_seed(base_seed, example_id=example_id, bank_size=bank_size))
        position = int(rng.randrange(len(bank_rows)))
        return position, [position], None

    selection_score_field = method_spec["selection_score_field"]
    values = [float(row[selection_score_field]) for row in bank_rows]
    tied_positions = _argmax_indices(values)
    selected_position = tied_positions[0]
    selected_score = float(values[selected_position])
    return selected_position, tied_positions, selected_score


def build_prompt_summary(
    *,
    example: ExampleRecord,
    trajectory_rows: Sequence[dict[str, Any]],
    n_values: Sequence[int],
    base_seed: int,
) -> dict[str, Any]:
    if not trajectory_rows:
        raise ValueError("Each prompt must have at least one trajectory row.")

    sorted_rows = sorted(trajectory_rows, key=lambda row: int(row["sample_idx"]))

    prompt_summary: dict[str, Any] = {
        "example_id": int(example.example_id),
        "prompt_id": int(example.example_id),
        "prompt_text": example.prompt_text,
        "data_source": example.data_source,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "max_bank_size": len(sorted_rows),
        "sample_indices": [int(row["sample_idx"]) for row in sorted_rows],
        "sample_seeds": [int(row["sample_seed"]) for row in sorted_rows],
        "task_scores": [float(row["task_score"]) for row in sorted_rows],
        "response_lengths": [int(row["response_length"]) for row in sorted_rows],
        "eos_emitted": [bool(row["eos_emitted"]) for row in sorted_rows],
        "max_length_hit": [bool(row["max_length_hit"]) for row in sorted_rows],
        "actor_response_logprobs": [float(row["actor_response_logprob"]) for row in sorted_rows],
        "actor_response_avg_logprobs": [float(row["actor_response_avg_logprob"]) for row in sorted_rows],
        "old_critic_values": [float(row["old_critic_final_trajectory_value"]) for row in sorted_rows],
        "new_critic_values": [float(row["new_critic_final_trajectory_value"]) for row in sorted_rows],
        "by_n": {},
    }

    for bank_size in n_values:
        bank_rows = list(sorted_rows[:bank_size])
        task_scores = [float(row["task_score"]) for row in bank_rows]
        oracle_best_positions = _argmax_indices(task_scores)
        oracle_best_sample_indices = [int(bank_rows[position]["sample_idx"]) for position in oracle_best_positions]
        oracle_best_task_score = float(bank_rows[oracle_best_positions[0]]["task_score"])

        methods_payload: dict[str, Any] = {}
        for method_name, method_spec in METHOD_SPECS.items():
            selected_position, tied_positions, selected_selection_score = _select_method_indices(
                bank_rows=bank_rows,
                method_name=method_name,
                example_id=example.example_id,
                base_seed=base_seed,
                bank_size=bank_size,
            )
            selected_row = bank_rows[selected_position]
            selected_sample_idx = int(selected_row["sample_idx"])
            methods_payload[method_name] = {
                "selected_index": selected_sample_idx,
                "selected_tied_indices": [int(bank_rows[position]["sample_idx"]) for position in tied_positions],
                "selected_task_score": float(selected_row["task_score"]),
                "selected_response_length": int(selected_row["response_length"]),
                "selected_is_correct": bool(float(selected_row["task_score"]) == 1.0),
                "selected_is_oracle_best": selected_sample_idx in oracle_best_sample_indices,
                "selected_actor_response_logprob": float(selected_row["actor_response_logprob"]),
                "selected_actor_response_avg_logprob": float(selected_row["actor_response_avg_logprob"]),
                "selected_old_critic_final_trajectory_value": float(
                    selected_row["old_critic_final_trajectory_value"]
                ),
                "selected_new_critic_final_trajectory_value": float(
                    selected_row["new_critic_final_trajectory_value"]
                ),
                "selection_score_field": method_spec["selection_score_field"],
                "selected_selection_score": selected_selection_score,
            }

        prompt_summary["by_n"][str(bank_size)] = {
            "bank_size": int(bank_size),
            "oracle_best_index": int(bank_rows[oracle_best_positions[0]]["sample_idx"]),
            "oracle_best_indices": oracle_best_sample_indices,
            "oracle_best_task_score": oracle_best_task_score,
            "methods": methods_payload,
        }

    return prompt_summary


def _metric_from_prompt_rows(
    prompt_rows_n: Sequence[dict[str, Any]],
    *,
    method_name: str,
    metric_name: str,
    binary_task_scores: bool,
) -> float | None:
    if metric_name == "selected_mean_task_score":
        return _mean(
            [float(prompt_row["methods"][method_name]["selected_task_score"]) for prompt_row in prompt_rows_n]
        )

    if metric_name == "top1_hit_rate_against_oracle_best":
        return _mean(
            [
                1.0 if bool(prompt_row["methods"][method_name]["selected_is_oracle_best"]) else 0.0
                for prompt_row in prompt_rows_n
            ]
        )

    if metric_name == "conditional_success_recovery_rate":
        if not binary_task_scores:
            return None
        successful_prompt_rows = [
            prompt_row for prompt_row in prompt_rows_n if float(prompt_row["oracle_best_task_score"]) == 1.0
        ]
        if not successful_prompt_rows:
            return None
        return _mean(
            [
                1.0 if bool(prompt_row["methods"][method_name]["selected_is_correct"]) else 0.0
                for prompt_row in successful_prompt_rows
            ]
        )

    if metric_name == "mean_selected_response_length":
        return _mean(
            [int(prompt_row["methods"][method_name]["selected_response_length"]) for prompt_row in prompt_rows_n]
        )

    raise ValueError(f"Unsupported metric name: {metric_name}")


def _bootstrap_metric_difference(
    prompt_rows_n: Sequence[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
    metric_name: str,
    bootstrap_samples: int,
    seed: int,
    binary_task_scores: bool,
) -> dict[str, Any] | None:
    observed_a = _metric_from_prompt_rows(
        prompt_rows_n,
        method_name=method_a,
        metric_name=metric_name,
        binary_task_scores=binary_task_scores,
    )
    observed_b = _metric_from_prompt_rows(
        prompt_rows_n,
        method_name=method_b,
        metric_name=metric_name,
        binary_task_scores=binary_task_scores,
    )
    if observed_a is None or observed_b is None:
        return None

    rng = np.random.default_rng(seed)
    sample_differences: list[float] = []
    for _sample_index in range(bootstrap_samples):
        indices = rng.integers(0, len(prompt_rows_n), size=len(prompt_rows_n))
        sampled_prompt_rows = [prompt_rows_n[int(index)] for index in indices]
        sample_a = _metric_from_prompt_rows(
            sampled_prompt_rows,
            method_name=method_a,
            metric_name=metric_name,
            binary_task_scores=binary_task_scores,
        )
        sample_b = _metric_from_prompt_rows(
            sampled_prompt_rows,
            method_name=method_b,
            metric_name=metric_name,
            binary_task_scores=binary_task_scores,
        )
        if sample_a is None or sample_b is None:
            continue
        sample_differences.append(float(sample_a - sample_b))

    if not sample_differences:
        return None

    ci_lower, ci_upper = np.quantile(np.asarray(sample_differences, dtype=np.float64), [0.025, 0.975]).tolist()
    return {
        "observed_difference": float(observed_a - observed_b),
        "bootstrap_mean_difference": float(np.mean(np.asarray(sample_differences, dtype=np.float64))),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "num_prompts": int(len(prompt_rows_n)),
        "num_bootstrap_samples": int(bootstrap_samples),
        "metric_name": metric_name,
    }


def aggregate_metrics(
    *,
    prompt_rows: Sequence[dict[str, Any]],
    n_values: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    binary_task_scores: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    metrics_by_n: dict[str, dict[str, Any]] = {}
    comparisons_by_n: dict[str, dict[str, Any]] = {}

    comparison_specs = OrderedDict(
        [
            (
                "best_of_n_new_critic_minus_best_of_n_old_critic",
                ("best_of_n_new_critic", "best_of_n_old_critic"),
            ),
            (
                "best_of_n_new_critic_minus_random_single_sample",
                ("best_of_n_new_critic", "random_single_sample"),
            ),
            (
                "best_of_n_new_critic_minus_best_of_n_actor_logprob",
                ("best_of_n_new_critic", "best_of_n_actor_logprob"),
            ),
            (
                "best_of_n_new_critic_minus_best_of_n_actor_avg_logprob",
                ("best_of_n_new_critic", "best_of_n_actor_avg_logprob"),
            ),
            (
                "oracle_best_in_bank_minus_best_of_n_new_critic",
                ("oracle_best_in_bank", "best_of_n_new_critic"),
            ),
        ]
    )

    for bank_size in n_values:
        n_key = str(bank_size)
        prompt_rows_n = [prompt_row["by_n"][n_key] for prompt_row in prompt_rows]
        num_success_prompts = sum(
            1 for prompt_row in prompt_rows_n if float(prompt_row["oracle_best_task_score"]) == 1.0
        )

        method_metrics: dict[str, Any] = {}
        for method_name, method_spec in METHOD_SPECS.items():
            selected_scores = [
                float(prompt_row["methods"][method_name]["selected_task_score"]) for prompt_row in prompt_rows_n
            ]
            oracle_hits = [
                1.0 if bool(prompt_row["methods"][method_name]["selected_is_oracle_best"]) else 0.0
                for prompt_row in prompt_rows_n
            ]
            selected_lengths = [
                int(prompt_row["methods"][method_name]["selected_response_length"]) for prompt_row in prompt_rows_n
            ]
            selected_selection_scores = [
                float(prompt_row["methods"][method_name]["selected_selection_score"])
                for prompt_row in prompt_rows_n
                if prompt_row["methods"][method_name]["selected_selection_score"] is not None
            ]

            trajectory_value_field = method_spec["trajectory_value_field"]
            selected_trajectory_values: list[float] = []
            if trajectory_value_field is not None:
                selected_trajectory_values = [
                    float(prompt_row["methods"][method_name][f"selected_{trajectory_value_field}"])
                    for prompt_row in prompt_rows_n
                ]

            method_metrics[method_name] = {
                "N": int(bank_size),
                "method": method_name,
                "method_label": method_spec["label"],
                "num_prompts": int(len(prompt_rows_n)),
                "num_bank_success_prompts": int(num_success_prompts),
                "num_bank_trajectories": int(len(prompt_rows_n) * bank_size),
                "selected_mean_task_score": _mean(selected_scores),
                "top1_hit_rate_against_oracle_best": _mean(oracle_hits),
                "conditional_success_recovery_rate": _metric_from_prompt_rows(
                    prompt_rows_n,
                    method_name=method_name,
                    metric_name="conditional_success_recovery_rate",
                    binary_task_scores=binary_task_scores,
                ),
                "mean_selected_response_length": _mean(selected_lengths),
                "mean_selected_selection_score": _mean(selected_selection_scores),
                "mean_selected_trajectory_value": _mean(selected_trajectory_values),
                "selection_score_field": method_spec["selection_score_field"],
                "trajectory_value_field": trajectory_value_field,
            }

        metrics_by_n[n_key] = method_metrics

        comparison_payload: dict[str, Any] = {}
        for comparison_name, (method_a, method_b) in comparison_specs.items():
            score_difference = (
                float(method_metrics[method_a]["selected_mean_task_score"])
                - float(method_metrics[method_b]["selected_mean_task_score"])
            )
            oracle_hit_difference = (
                float(method_metrics[method_a]["top1_hit_rate_against_oracle_best"])
                - float(method_metrics[method_b]["top1_hit_rate_against_oracle_best"])
            )

            conditional_a = method_metrics[method_a]["conditional_success_recovery_rate"]
            conditional_b = method_metrics[method_b]["conditional_success_recovery_rate"]
            conditional_difference = None
            if conditional_a is not None and conditional_b is not None:
                conditional_difference = float(conditional_a - conditional_b)

            mean_length_difference = (
                float(method_metrics[method_a]["mean_selected_response_length"])
                - float(method_metrics[method_b]["mean_selected_response_length"])
            )

            comparison_payload[comparison_name] = {
                "method_a": method_a,
                "method_b": method_b,
                "selected_mean_task_score_difference": score_difference,
                "conditional_success_recovery_rate_difference": conditional_difference,
                "top1_hit_rate_against_oracle_best_difference": oracle_hit_difference,
                "mean_selected_response_length_difference": mean_length_difference,
                "paired_bootstrap": {
                    "selected_mean_task_score_difference": _bootstrap_metric_difference(
                        prompt_rows_n,
                        method_a=method_a,
                        method_b=method_b,
                        metric_name="selected_mean_task_score",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + bank_size * 100 + 1,
                        binary_task_scores=binary_task_scores,
                    ),
                    "conditional_success_recovery_rate_difference": _bootstrap_metric_difference(
                        prompt_rows_n,
                        method_a=method_a,
                        method_b=method_b,
                        metric_name="conditional_success_recovery_rate",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + bank_size * 100 + 2,
                        binary_task_scores=binary_task_scores,
                    ),
                    "top1_hit_rate_against_oracle_best_difference": _bootstrap_metric_difference(
                        prompt_rows_n,
                        method_a=method_a,
                        method_b=method_b,
                        metric_name="top1_hit_rate_against_oracle_best",
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + bank_size * 100 + 3,
                        binary_task_scores=binary_task_scores,
                    ),
                },
            }

        comparisons_by_n[n_key] = comparison_payload

    return metrics_by_n, comparisons_by_n


def _main_results_rows(metrics_by_n: dict[str, dict[str, Any]], n_values: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bank_size in n_values:
        n_key = str(bank_size)
        for method_name in METHOD_SPECS:
            rows.append(dict(metrics_by_n[n_key][method_name]))
    return rows


def _progress_postfix(worker_progress: dict[int, dict[str, Any]]) -> str:
    parts: list[str] = []
    for worker_id in sorted(worker_progress):
        state = worker_progress[worker_id]
        done = int(state.get("done", 0))
        total = int(state.get("total", 0))
        parts.append(f"w{worker_id}:{done}/{total}")
    return " | ".join(parts)


class _RayProgressActor:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def put(self, event: dict[str, Any]) -> None:
        self._events.append(dict(event))

    def drain(self) -> list[dict[str, Any]]:
        events = self._events
        self._events = []
        return events


def _start_local_worker_processes(
    *,
    assignments: Sequence[WorkerAssignment],
    actor_hf_dir: Path,
    old_critic_hf_dir: Path,
    new_critic_hf_dir: Path,
    examples: list[ExampleRecord],
    n_values: list[int],
    max_bank_size: int,
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
    critic_score_batch_size: int,
    seed: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    reference_stage1_trajectory_bank_path: str | None,
    worker_root: Path,
) -> tuple[Any, list[tuple[mp.Process, WorkerAssignment]]]:
    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[tuple[mp.Process, WorkerAssignment]] = []
    for assignment in assignments:
        process = context.Process(
            target=_worker_entry,
            kwargs={
                "assignment": assignment,
                "actor_hf_dir": str(actor_hf_dir),
                "old_critic_hf_dir": str(old_critic_hf_dir),
                "new_critic_hf_dir": str(new_critic_hf_dir),
                "examples": examples,
                "n_values": list(n_values),
                "max_bank_size": max_bank_size,
                "dtype_name": dtype_name,
                "trust_remote_code": trust_remote_code,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "eos_token_ids": eos_token_ids,
                "use_actor_cache": use_actor_cache,
                "critic_score_batch_size": critic_score_batch_size,
                "seed": seed,
                "actor_sampling_mode": actor_sampling_mode,
                "actor_temperature": actor_temperature,
                "actor_top_p": actor_top_p,
                "actor_top_k": actor_top_k,
                "reference_stage1_trajectory_bank_path": reference_stage1_trajectory_bank_path,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
                "progress_actor": None,
            },
            name=f"best_of_n_worker_{assignment.worker_id}",
        )
        process.start()
        processes.append((process, assignment))
    return progress_queue, processes


def _assert_local_processes_healthy(
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


def _join_local_processes(
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


def _worker_entry(
    *,
    assignment: WorkerAssignment,
    actor_hf_dir: str,
    old_critic_hf_dir: str,
    new_critic_hf_dir: str,
    examples: list[ExampleRecord],
    n_values: list[int],
    max_bank_size: int,
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
    critic_score_batch_size: int,
    seed: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    reference_stage1_trajectory_bank_path: str | None,
    worker_root: str,
    progress_queue=None,
    progress_actor=None,
) -> None:
    worker_dir = Path(worker_root) / f"worker_{assignment.worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    summary_path = worker_dir / "worker_summary.json"
    trajectory_path = worker_dir / "trajectory_bank.jsonl"
    prompt_summary_path = worker_dir / "prompt_level_summary.jsonl"
    error_path = worker_dir / "worker_error.txt"

    try:
        start_time = time.perf_counter()
        actor_device = resolve_device(assignment.actor_device)
        _validate_visible_cuda_device(actor_device, label="actor_device")
        old_critic_device = resolve_device(assignment.old_critic_device) if assignment.old_critic_device else actor_device
        _validate_visible_cuda_device(old_critic_device, label="old_critic_device")
        new_critic_device = resolve_device(assignment.new_critic_device) if assignment.new_critic_device else old_critic_device
        _validate_visible_cuda_device(new_critic_device, label="new_critic_device")
        dtype = resolve_dtype(dtype_name)

        actor_hf_path = Path(actor_hf_dir)
        old_critic_hf_path = Path(old_critic_hf_dir)
        new_critic_hf_path = Path(new_critic_hf_dir)
        shared_critic_checkpoint = _same_resolved_path(old_critic_hf_path, new_critic_hf_path)
        shared_critic_runtime = shared_critic_checkpoint and old_critic_device == new_critic_device

        actor_tokenizer = load_tokenizer(actor_hf_path, trust_remote_code=trust_remote_code)
        if shared_critic_checkpoint:
            shared_critic_tokenizer = load_tokenizer(old_critic_hf_path, trust_remote_code=trust_remote_code)
            old_critic_tokenizer = shared_critic_tokenizer
            new_critic_tokenizer = shared_critic_tokenizer
        else:
            old_critic_tokenizer = load_tokenizer(old_critic_hf_path, trust_remote_code=trust_remote_code)
            new_critic_tokenizer = load_tokenizer(new_critic_hf_path, trust_remote_code=trust_remote_code)
        actor = load_actor_model(
            actor_hf_path,
            dtype=dtype,
            device=actor_device,
            trust_remote_code=trust_remote_code,
        )
        if shared_critic_runtime:
            shared_critic = load_critic_model(
                old_critic_hf_path,
                dtype=dtype,
                device=old_critic_device,
                trust_remote_code=trust_remote_code,
            )
            old_critic = shared_critic
            new_critic = shared_critic
        else:
            old_critic = load_critic_model(
                old_critic_hf_path,
                dtype=dtype,
                device=old_critic_device,
                trust_remote_code=trust_remote_code,
            )
            new_critic = load_critic_model(
                new_critic_hf_path,
                dtype=dtype,
                device=new_critic_device,
                trust_remote_code=trust_remote_code,
            )

        reference_stage1_bank = None
        reference_stage1_summary = None
        if reference_stage1_trajectory_bank_path:
            reference_path = Path(reference_stage1_trajectory_bank_path)
            reference_stage1_bank = _load_reference_stage1_bank(reference_path)
            reference_stage1_summary = _new_stage1_validation_summary(reference_path)

        old_pad_token_id = int(old_critic_tokenizer.pad_token_id)
        new_pad_token_id = int(new_critic_tokenizer.pad_token_id)

        local_examples = examples[assignment.example_start : assignment.example_end]
        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_started",
                "worker_id": assignment.worker_id,
                "worker_total_prompts": len(local_examples),
            },
        )

        local_stage1_summary = None if reference_stage1_summary is None else dict(reference_stage1_summary)
        completed_prompts = 0
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file, prompt_summary_path.open(
            "w",
            encoding="utf-8",
        ) as prompt_file:
            for example in local_examples:
                prompt_trajectory_rows, prompt_summary, prompt_stage1_summary = process_example(
                    actor=actor,
                    old_critic=old_critic,
                    new_critic=new_critic,
                    actor_tokenizer=actor_tokenizer,
                    old_critic_pad_token_id=old_pad_token_id,
                    new_critic_pad_token_id=new_pad_token_id,
                    example=example,
                    n_values=n_values,
                    max_bank_size=max_bank_size,
                    actor_device=actor_device,
                    old_critic_device=old_critic_device,
                    new_critic_device=new_critic_device,
                    actor_sampling_mode=actor_sampling_mode,
                    actor_temperature=actor_temperature,
                    actor_top_p=actor_top_p,
                    actor_top_k=actor_top_k,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    use_actor_cache=use_actor_cache,
                    critic_score_batch_size=critic_score_batch_size,
                    base_seed=seed,
                    reference_stage1_bank=reference_stage1_bank,
                    reference_stage1_summary=reference_stage1_summary,
                )
                for trajectory_row in prompt_trajectory_rows:
                    trajectory_file.write(_json_line(trajectory_row))
                prompt_file.write(_json_line(prompt_summary))
                local_stage1_summary = _accumulate_stage1_validation_summary(
                    local_stage1_summary,
                    prompt_stage1_summary,
                )
                completed_prompts += 1
                _emit_progress(
                    progress_queue=progress_queue,
                    progress_actor=progress_actor,
                    event={
                        "type": "prompt_done",
                        "worker_id": assignment.worker_id,
                        "worker_completed_prompts": completed_prompts,
                        "worker_total_prompts": len(local_examples),
                    },
                )

        summary_payload = {
            "worker_id": assignment.worker_id,
            "actor_device": str(actor_device),
            "old_critic_device": str(old_critic_device),
            "new_critic_device": str(new_critic_device),
            "node_index": assignment.node_index,
            "node_ip": assignment.node_ip,
            "node_resource_key": assignment.node_resource_key,
            "local_worker_index": assignment.local_worker_index,
            "example_start": assignment.example_start,
            "example_end": assignment.example_end,
            "num_examples": assignment.num_examples,
            "shared_critic_checkpoint": bool(shared_critic_checkpoint),
            "shared_critic_runtime": bool(shared_critic_runtime),
            "runtime_sec": time.perf_counter() - start_time,
            "reference_stage1_validation": local_stage1_summary,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_done",
                "worker_id": assignment.worker_id,
                "worker_completed_prompts": completed_prompts,
                "worker_total_prompts": len(local_examples),
            },
        )
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_error",
                "worker_id": assignment.worker_id,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _collect_worker_outputs(
    *,
    worker_root: Path,
    assignments: Sequence[WorkerAssignment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    trajectory_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    merged_stage1_summary = None

    for assignment in assignments:
        worker_dir = worker_root / f"worker_{assignment.worker_id:03d}"
        with (worker_dir / "trajectory_bank.jsonl").open("r", encoding="utf-8") as trajectory_file:
            for line in trajectory_file:
                if line.strip():
                    trajectory_rows.append(json.loads(line))
        with (worker_dir / "prompt_level_summary.jsonl").open("r", encoding="utf-8") as prompt_file:
            for line in prompt_file:
                if line.strip():
                    prompt_rows.append(json.loads(line))
        with (worker_dir / "worker_summary.json").open("r", encoding="utf-8") as summary_file:
            worker_summary = json.load(summary_file)
            worker_summaries.append(worker_summary)
            merged_stage1_summary = _accumulate_stage1_validation_summary(
                merged_stage1_summary,
                worker_summary.get("reference_stage1_validation"),
            )

    trajectory_rows.sort(key=lambda row: (int(row["example_id"]), int(row["sample_idx"])))
    prompt_rows.sort(key=lambda row: int(row["example_id"]))
    worker_summaries.sort(key=lambda row: int(row["worker_id"]))
    return trajectory_rows, prompt_rows, worker_summaries, merged_stage1_summary


def _ray_node_entry_remote(**kwargs) -> dict[str, Any]:
    assignments: list[WorkerAssignment] = kwargs["assignments"]
    progress_actor = kwargs["progress_actor"]
    progress_queue, processes = _start_local_worker_processes(
        assignments=assignments,
        actor_hf_dir=Path(kwargs["actor_hf_dir"]),
        old_critic_hf_dir=Path(kwargs["old_critic_hf_dir"]),
        new_critic_hf_dir=Path(kwargs["new_critic_hf_dir"]),
        examples=kwargs["examples"],
        n_values=kwargs["n_values"],
        max_bank_size=kwargs["max_bank_size"],
        dtype_name=kwargs["dtype_name"],
        trust_remote_code=kwargs["trust_remote_code"],
        max_prompt_length=kwargs["max_prompt_length"],
        max_new_tokens=kwargs["max_new_tokens"],
        eos_token_ids=kwargs["eos_token_ids"],
        use_actor_cache=kwargs["use_actor_cache"],
        critic_score_batch_size=kwargs["critic_score_batch_size"],
        seed=kwargs["seed"],
        actor_sampling_mode=kwargs["actor_sampling_mode"],
        actor_temperature=kwargs["actor_temperature"],
        actor_top_p=kwargs["actor_top_p"],
        actor_top_k=kwargs["actor_top_k"],
        reference_stage1_trajectory_bank_path=kwargs["reference_stage1_trajectory_bank_path"],
        worker_root=Path(kwargs["worker_root"]),
    )
    worker_root = Path(kwargs["worker_root"])
    completed_workers = 0
    while completed_workers < len(assignments):
        try:
            event = progress_queue.get(timeout=RAY_PROGRESS_POLL_INTERVAL_SEC)
        except Empty:
            _assert_local_processes_healthy(processes=processes, worker_root=worker_root)
            continue

        progress_actor.put.remote(event)
        if event.get("type") == "worker_done":
            completed_workers += 1
        elif event.get("type") == "worker_error":
            raise RuntimeError(
                f"Worker {event.get('worker_id')} reported an error.\n"
                f"{event.get('traceback', 'No traceback provided.')}"
            )

    _join_local_processes(processes=processes, worker_root=worker_root)
    return {
        "node_index": kwargs["node_index"],
        "node_ip": kwargs["node_ip"],
        "worker_ids": [int(assignment.worker_id) for assignment in assignments],
    }


def run_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    old_critic_hf_dir: Path,
    new_critic_hf_dir: Path,
    examples: list[ExampleRecord],
    worker_layouts: list[tuple[str | None, str | None, str | None]],
    n_values: list[int],
    max_bank_size: int,
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
    critic_score_batch_size: int,
    seed: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    reference_stage1_trajectory_bank_path: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    assignments = build_worker_assignments(num_examples=len(examples), worker_layouts=worker_layouts)
    if not assignments:
        raise ValueError("No worker assignments were created.")

    worker_root = output_dir / "_worker_tmp"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    progress_queue, processes = _start_local_worker_processes(
        assignments=assignments,
        actor_hf_dir=actor_hf_dir,
        old_critic_hf_dir=old_critic_hf_dir,
        new_critic_hf_dir=new_critic_hf_dir,
        examples=examples,
        n_values=n_values,
        max_bank_size=max_bank_size,
        dtype_name=dtype_name,
        trust_remote_code=trust_remote_code,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        use_actor_cache=use_actor_cache,
        critic_score_batch_size=critic_score_batch_size,
        seed=seed,
        actor_sampling_mode=actor_sampling_mode,
        actor_temperature=actor_temperature,
        actor_top_p=actor_top_p,
        actor_top_k=actor_top_k,
        reference_stage1_trajectory_bank_path=reference_stage1_trajectory_bank_path,
        worker_root=worker_root,
    )

    total_prompts = len(examples)
    completed_prompts = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {"done": 0, "total": assignment.num_examples} for assignment in assignments
    }

    with tqdm(total=total_prompts, desc="best_of_n_inference_eval", unit="prompt", dynamic_ncols=True) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_prompts < total_prompts or completed_workers < len(assignments):
            try:
                event = progress_queue.get(timeout=0.2)
            except Empty:
                _assert_local_processes_healthy(processes=processes, worker_root=worker_root)
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
                    f"Worker {worker_id} reported an error.\n{event.get('traceback', 'No traceback provided.')}"
                )
            progress_bar.set_postfix_str(_progress_postfix(worker_progress))

    _join_local_processes(processes=processes, worker_root=worker_root)
    return _collect_worker_outputs(worker_root=worker_root, assignments=assignments)


def run_ray_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    old_critic_hf_dir: Path,
    new_critic_hf_dir: Path,
    examples: list[ExampleRecord],
    worker_assignments: list[WorkerAssignment],
    n_values: list[int],
    max_bank_size: int,
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
    critic_score_batch_size: int,
    seed: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    reference_stage1_trajectory_bank_path: str | None,
    ray_num_cpus_per_worker: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if not worker_assignments:
        raise ValueError("No worker assignments were created.")
    if ray_num_cpus_per_worker <= 0:
        raise ValueError(f"ray_num_cpus_per_worker must be > 0, got {ray_num_cpus_per_worker}.")

    ray_module = _require_ray()
    if not ray_module.is_initialized():
        raise RuntimeError("Ray must be initialized before running cross-node best-of-n workers.")

    worker_root = output_dir / "_worker_tmp"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    progress_actor = ray_module.remote(num_cpus=0)(_RayProgressActor).remote()
    node_execution_specs = _build_ray_node_execution_specs(
        worker_assignments=worker_assignments,
        ray_num_cpus_per_worker=ray_num_cpus_per_worker,
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
            actor_hf_dir=str(actor_hf_dir),
            old_critic_hf_dir=str(old_critic_hf_dir),
            new_critic_hf_dir=str(new_critic_hf_dir),
            examples=examples,
            n_values=list(n_values),
            max_bank_size=max_bank_size,
            dtype_name=dtype_name,
            trust_remote_code=trust_remote_code,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            use_actor_cache=use_actor_cache,
            critic_score_batch_size=critic_score_batch_size,
            seed=seed,
            actor_sampling_mode=actor_sampling_mode,
            actor_temperature=actor_temperature,
            actor_top_p=actor_top_p,
            actor_top_k=actor_top_k,
            reference_stage1_trajectory_bank_path=reference_stage1_trajectory_bank_path,
            worker_root=str(worker_root),
            progress_actor=progress_actor,
        )
        node_refs.append(node_ref)
        ref_to_node_spec[node_ref] = node_spec

    total_prompts = len(examples)
    completed_prompts = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {"done": 0, "total": assignment.num_examples} for assignment in worker_assignments
    }

    pending_refs = list(node_refs)
    with tqdm(total=total_prompts, desc="best_of_n_inference_eval", unit="prompt", dynamic_ncols=True) as progress_bar:
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

    return _collect_worker_outputs(worker_root=worker_root, assignments=worker_assignments)


def _plot_metric_vs_n(
    *,
    metrics_by_n: dict[str, dict[str, Any]],
    n_values: Sequence[int],
    metric_name: str,
    ylabel: str,
    output_path: Path,
    method_names: Sequence[str],
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plot generation but could not be imported."
        ) from MATPLOTLIB_IMPORT_ERROR

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for method_name in method_names:
        xs = list(n_values)
        ys = [metrics_by_n[str(bank_size)][method_name][metric_name] for bank_size in xs]
        axis.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            label=METHOD_SPECS[method_name]["label"],
        )

    axis.set_xlabel("N")
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(n_values))
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _write_output_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    n_values: Sequence[int],
    max_bank_size: int,
    metrics_by_n: dict[str, dict[str, Any]],
    comparisons_by_n: dict[str, dict[str, Any]],
    stage1_validation_summary: dict[str, Any] | None,
    shared_critic_checkpoint: bool,
) -> None:
    lines: list[str] = [
        "# Stage 2 Best-of-N Inference Evaluation",
        "",
        "This run implements a shared-bank response-level reranking evaluation:",
        f"- The frozen actor from `{args.actor_checkpoint_dir}` samples a maximum bank of `{max_bank_size}` responses per prompt exactly once.",
        "- For smaller N, the evaluator reuses the first N samples from that same bank.",
        "- The old critic, new critic, actor log-prob selectors, and random/oracle baselines all select from the exact same bank for each prompt.",
        "- Critic trajectory value is always read at the last token position of the full prompt+response sequence.",
        "",
        "## Methods",
        "- `random_single_sample`",
        "- `best_of_n_old_critic`",
        "- `best_of_n_new_critic`",
        "- `best_of_n_actor_logprob`",
        "- `best_of_n_actor_avg_logprob`",
        "- `oracle_best_in_bank`",
        "",
        "## Main Metrics",
        "- `selected_mean_task_score`: primary end-to-end task metric; on binary tasks this is accuracy.",
        "- `conditional_success_recovery_rate`: among prompts where the bank contains at least one correct response, how often the selector recovers one.",
        "- `top1_hit_rate_against_oracle_best`: fraction of prompts where the selector chose any highest-scoring response in the bank.",
        "- `mean_selected_response_length`: mean generated-token length of the selected response.",
        "",
        "## Interpretation",
        "- The main question is whether `best_of_n_new_critic` beats `random_single_sample`, `best_of_n_old_critic`, and `best_of_n_actor_logprob` at practical N values.",
        "- Do not overinterpret raw critic value magnitudes across critics; scales can differ.",
        "",
        "## Run Config",
        f"- Dataset: `{args.dataset_path}`",
        f"- Execution backend: `{'ray' if _resolve_ray_address(args.ray_address) is not None else 'local'}`",
        f"- Ray address: `{_resolve_ray_address(args.ray_address)}`",
        f"- N values: `{list(n_values)}`",
        f"- Max bank size: `{max_bank_size}`",
        f"- Sampling mode: `{args.actor_sampling_mode}`",
        f"- Temperature / top-p / top-k: `{args.actor_temperature}` / `{args.actor_top_p}` / `{args.actor_top_k}`",
        f"- Seed: `{args.seed}`",
    ]

    if shared_critic_checkpoint:
        lines.extend(
            [
                "- Shared critic checkpoint mode: `best_of_n_old_critic` and `best_of_n_new_critic` reuse the same critic checkpoint in this run.",
                "- Old/new critic metrics are therefore expected to match unless you override the CLI manually.",
            ]
        )

    if stage1_validation_summary is not None:
        lines.extend(
            [
                "",
                "## Stage 1 Overlap Validation",
                f"- Reference bank: `{stage1_validation_summary['reference_stage1_trajectory_bank']}`",
                f"- Critic value abs tolerance: `{stage1_validation_summary['critic_value_abs_tolerance']}`",
                f"- Rows compared: `{stage1_validation_summary['num_rows_compared']}`",
                f"- Max overlapping sample index: `{stage1_validation_summary['max_overlapping_sample_idx']}`",
                f"- Max |old critic value diff|: `{stage1_validation_summary['max_abs_old_critic_value_diff']}`",
                f"- Max |new critic value diff|: `{stage1_validation_summary['max_abs_new_critic_value_diff']}`",
            ]
        )

    lines.extend(["", "## Quick Read"])
    for bank_size in n_values:
        n_key = str(bank_size)
        old_metric = metrics_by_n[n_key]["best_of_n_old_critic"]["selected_mean_task_score"]
        new_metric = metrics_by_n[n_key]["best_of_n_new_critic"]["selected_mean_task_score"]
        random_metric = metrics_by_n[n_key]["random_single_sample"]["selected_mean_task_score"]
        logp_metric = metrics_by_n[n_key]["best_of_n_actor_logprob"]["selected_mean_task_score"]
        oracle_metric = metrics_by_n[n_key]["oracle_best_in_bank"]["selected_mean_task_score"]
        delta_new_old = comparisons_by_n[n_key]["best_of_n_new_critic_minus_best_of_n_old_critic"][
            "selected_mean_task_score_difference"
        ]
        lines.append(
            f"- N={bank_size}: random={random_metric:.6f}, old={old_metric:.6f}, "
            f"new={new_metric:.6f}, actor_logprob={logp_metric:.6f}, oracle={oracle_metric:.6f}, "
            f"new-old={delta_new_old:.6f}"
        )

    lines.extend(
        [
            "",
            "## Files",
            "- `trajectory_bank.jsonl`: one row per sampled trajectory with task score, actor log-prob, and both critic values.",
            "- `prompt_level_summary.jsonl`: one row per prompt with per-N selector outcomes.",
            "- `summary_metrics.json`: aggregate metrics and paired bootstrap comparisons by N.",
            "- `main_results.csv`: compact flat table with one row per method and N.",
            "- `accuracy_vs_n.png`: selected mean task score vs N.",
            "- `conditional_success_recovery_vs_n.png`: recovery rate vs N.",
            "- `mean_selected_response_length_vs_n.png`: selected response length vs N.",
        ]
    )

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    n_values = sorted(set(int(value) for value in args.n_values))
    if not n_values:
        raise ValueError("--n_values must contain at least one value.")
    if any(value <= 0 for value in n_values):
        raise ValueError(f"--n_values must all be > 0, got {n_values}")
    max_bank_size = int(args.max_bank_size) if args.max_bank_size is not None else int(max(n_values))
    if max_bank_size <= 0:
        raise ValueError(f"--max_bank_size must be > 0, got {max_bank_size}")
    if max_bank_size < max(n_values):
        raise ValueError(
            f"--max_bank_size ({max_bank_size}) must be >= max(n_values) ({max(n_values)})"
        )
    if args.bootstrap_samples <= 0:
        raise ValueError(f"--bootstrap_samples must be > 0, got {args.bootstrap_samples}")
    if args.critic_score_batch_size <= 0:
        raise ValueError(f"--critic_score_batch_size must be > 0, got {args.critic_score_batch_size}")
    if args.ray_num_cpus_per_worker <= 0:
        raise ValueError(f"--ray_num_cpus_per_worker must be > 0, got {args.ray_num_cpus_per_worker}")
    if not args.skip_plots and plt is None:
        raise RuntimeError(
            "matplotlib is required for plot generation but could not be imported."
        ) from MATPLOTLIB_IMPORT_ERROR

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
    old_critic_checkpoint_dir = Path(args.old_critic_checkpoint_dir).resolve()
    new_critic_checkpoint_dir = Path(args.new_critic_checkpoint_dir).resolve()

    actor_hf_dir = ensure_merged_component_checkpoint(
        actor_checkpoint_dir,
        component="actor",
        merged_root=Path(args.actor_merged_root).resolve() if args.actor_merged_root else None,
        hf_source_dir=Path(args.actor_hf_source_dir).resolve() if args.actor_hf_source_dir else None,
        skip_merge=args.skip_merge,
    )
    old_critic_hf_dir = ensure_merged_component_checkpoint(
        old_critic_checkpoint_dir,
        component="critic",
        merged_root=Path(args.old_critic_merged_root).resolve() if args.old_critic_merged_root else None,
        hf_source_dir=Path(args.old_critic_hf_source_dir).resolve() if args.old_critic_hf_source_dir else None,
        skip_merge=args.skip_merge,
    )
    new_critic_hf_dir = ensure_merged_component_checkpoint(
        new_critic_checkpoint_dir,
        component="critic",
        merged_root=Path(args.new_critic_merged_root).resolve() if args.new_critic_merged_root else None,
        hf_source_dir=Path(args.new_critic_hf_source_dir).resolve() if args.new_critic_hf_source_dir else None,
        skip_merge=args.skip_merge,
    )
    shared_critic_checkpoint = _same_resolved_path(old_critic_hf_dir, new_critic_hf_dir)

    dtype = resolve_dtype(args.dtype)
    actor_tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    if shared_critic_checkpoint:
        shared_critic_tokenizer = load_tokenizer(old_critic_hf_dir, trust_remote_code=args.trust_remote_code)
        old_critic_tokenizer = shared_critic_tokenizer
        new_critic_tokenizer = shared_critic_tokenizer
    else:
        old_critic_tokenizer = load_tokenizer(old_critic_hf_dir, trust_remote_code=args.trust_remote_code)
        new_critic_tokenizer = load_tokenizer(new_critic_hf_dir, trust_remote_code=args.trust_remote_code)
    tokenizer_fingerprints = {
        "actor": _tokenizer_fingerprint(actor_hf_dir),
        "old_critic": _tokenizer_fingerprint(old_critic_hf_dir),
        "new_critic": _tokenizer_fingerprint(new_critic_hf_dir),
    }
    _assert_shared_tokenizer(tokenizer_fingerprints)

    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, actor_tokenizer)
    examples = load_examples(
        args.dataset_path,
        tokenizer=actor_tokenizer,
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

    worker_layouts = parse_worker_layouts(
        args.worker_layouts,
        actor_device=args.actor_device,
        old_critic_device=args.old_critic_device,
        new_critic_device=args.new_critic_device,
        default_device=args.device,
    )
    ray_address = _resolve_ray_address(args.ray_address)
    execution_backend = "ray" if ray_address is not None else "local"
    ray_nodes: list[RayNodeInfo] = []
    if ray_address is not None:
        ray_module = _require_ray()
        if not ray_module.is_initialized():
            # Rely on the launcher to propagate PYTHONPATH / caches / environment.
            # Passing runtime_env during ray.init(address=...) has been unreliable on this cluster.
            ray_module.init(address=ray_address)
        ray_nodes = _discover_ray_nodes(ray_module)
        if not ray_nodes:
            raise ValueError("Ray is connected, but no alive Ray nodes were discovered.")
        worker_assignments = build_distributed_worker_assignments(
            num_examples=len(examples),
            worker_layouts=worker_layouts,
            ray_nodes=ray_nodes,
        )
    else:
        worker_assignments = build_worker_assignments(num_examples=len(examples), worker_layouts=worker_layouts)
    multi_worker_enabled = len(worker_assignments) > 1

    if ray_address is not None:
        trajectory_rows, prompt_rows, worker_summaries, stage1_validation_summary = run_ray_multi_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            old_critic_hf_dir=old_critic_hf_dir,
            new_critic_hf_dir=new_critic_hf_dir,
            examples=examples,
            worker_assignments=worker_assignments,
            n_values=list(n_values),
            max_bank_size=max_bank_size,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            use_actor_cache=not args.disable_actor_cache,
            critic_score_batch_size=args.critic_score_batch_size,
            seed=args.seed,
            actor_sampling_mode=args.actor_sampling_mode,
            actor_temperature=args.actor_temperature,
            actor_top_p=args.actor_top_p,
            actor_top_k=args.actor_top_k,
            reference_stage1_trajectory_bank_path=(
                str(Path(args.reference_stage1_trajectory_bank).resolve())
                if args.reference_stage1_trajectory_bank
                else None
            ),
            ray_num_cpus_per_worker=args.ray_num_cpus_per_worker,
        )
        actor_device = None
        old_critic_device = None
        new_critic_device = None
    elif multi_worker_enabled:
        trajectory_rows, prompt_rows, worker_summaries, stage1_validation_summary = run_multi_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            old_critic_hf_dir=old_critic_hf_dir,
            new_critic_hf_dir=new_critic_hf_dir,
            examples=examples,
            worker_layouts=worker_layouts,
            n_values=list(n_values),
            max_bank_size=max_bank_size,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            use_actor_cache=not args.disable_actor_cache,
            critic_score_batch_size=args.critic_score_batch_size,
            seed=args.seed,
            actor_sampling_mode=args.actor_sampling_mode,
            actor_temperature=args.actor_temperature,
            actor_top_p=args.actor_top_p,
            actor_top_k=args.actor_top_k,
            reference_stage1_trajectory_bank_path=(
                str(Path(args.reference_stage1_trajectory_bank).resolve())
                if args.reference_stage1_trajectory_bank
                else None
            ),
        )
        actor_device = None
        old_critic_device = None
        new_critic_device = None
    else:
        assignment = worker_assignments[0]
        actor_device = resolve_device(assignment.actor_device)
        old_critic_device = resolve_device(assignment.old_critic_device) if assignment.old_critic_device else actor_device
        new_critic_device = resolve_device(assignment.new_critic_device) if assignment.new_critic_device else old_critic_device
        shared_critic_runtime = shared_critic_checkpoint and old_critic_device == new_critic_device

        actor = load_actor_model(
            actor_hf_dir,
            dtype=dtype,
            device=actor_device,
            trust_remote_code=args.trust_remote_code,
        )
        if shared_critic_runtime:
            shared_critic = load_critic_model(
                old_critic_hf_dir,
                dtype=dtype,
                device=old_critic_device,
                trust_remote_code=args.trust_remote_code,
            )
            old_critic = shared_critic
            new_critic = shared_critic
        else:
            old_critic = load_critic_model(
                old_critic_hf_dir,
                dtype=dtype,
                device=old_critic_device,
                trust_remote_code=args.trust_remote_code,
            )
            new_critic = load_critic_model(
                new_critic_hf_dir,
                dtype=dtype,
                device=new_critic_device,
                trust_remote_code=args.trust_remote_code,
            )

        old_pad_token_id = int(old_critic_tokenizer.pad_token_id)
        new_pad_token_id = int(new_critic_tokenizer.pad_token_id)

        reference_stage1_bank: dict[tuple[int, int], dict[str, Any]] | None = None
        reference_stage1_template = None
        if args.reference_stage1_trajectory_bank:
            reference_path = Path(args.reference_stage1_trajectory_bank).resolve()
            reference_stage1_bank = _load_reference_stage1_bank(reference_path)
            reference_stage1_template = _new_stage1_validation_summary(reference_path)

        trajectory_rows = []
        prompt_rows = []
        stage1_validation_summary = None
        with tqdm(examples, desc="best_of_n_inference_eval", unit="prompt", dynamic_ncols=True) as progress_bar:
            for example in progress_bar:
                progress_bar.set_postfix_str(f"example_id={example.example_id}")
                prompt_trajectory_rows, prompt_summary, prompt_stage1_summary = process_example(
                    actor=actor,
                    old_critic=old_critic,
                    new_critic=new_critic,
                    actor_tokenizer=actor_tokenizer,
                    old_critic_pad_token_id=old_pad_token_id,
                    new_critic_pad_token_id=new_pad_token_id,
                    example=example,
                    n_values=n_values,
                    max_bank_size=max_bank_size,
                    actor_device=actor_device,
                    old_critic_device=old_critic_device,
                    new_critic_device=new_critic_device,
                    actor_sampling_mode=args.actor_sampling_mode,
                    actor_temperature=args.actor_temperature,
                    actor_top_p=args.actor_top_p,
                    actor_top_k=args.actor_top_k,
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    use_actor_cache=not args.disable_actor_cache,
                    critic_score_batch_size=args.critic_score_batch_size,
                    base_seed=args.seed,
                    reference_stage1_bank=reference_stage1_bank,
                    reference_stage1_summary=reference_stage1_template,
                )
                trajectory_rows.extend(prompt_trajectory_rows)
                prompt_rows.append(prompt_summary)
                stage1_validation_summary = _accumulate_stage1_validation_summary(
                    stage1_validation_summary,
                    prompt_stage1_summary,
                )

        worker_summaries = [
            {
                "worker_id": 0,
                "actor_device": str(actor_device),
                "old_critic_device": str(old_critic_device),
                "new_critic_device": str(new_critic_device),
                "node_index": None,
                "node_ip": None,
                "node_resource_key": None,
                "local_worker_index": None,
                "example_start": 0,
                "example_end": len(examples),
                "num_examples": len(examples),
                "shared_critic_checkpoint": bool(shared_critic_checkpoint),
                "shared_critic_runtime": bool(shared_critic_runtime),
                "reference_stage1_validation": stage1_validation_summary,
            }
        ]

    binary_task_scores = set(float(row["task_score"]) for row in trajectory_rows).issubset({0.0, 1.0})
    metrics_by_n, comparisons_by_n = aggregate_metrics(
        prompt_rows=prompt_rows,
        n_values=n_values,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed + 70_001,
        binary_task_scores=binary_task_scores,
    )

    trajectory_bank_path = output_dir / "trajectory_bank.jsonl"
    prompt_summary_path = output_dir / "prompt_level_summary.jsonl"
    summary_metrics_path = output_dir / "summary_metrics.json"
    main_results_path = output_dir / "main_results.csv"
    accuracy_plot_path = output_dir / "accuracy_vs_n.png"
    recovery_plot_path = output_dir / "conditional_success_recovery_vs_n.png"
    length_plot_path = output_dir / "mean_selected_response_length_vs_n.png"

    with trajectory_bank_path.open("w", encoding="utf-8") as trajectory_bank_file:
        for trajectory_row in trajectory_rows:
            trajectory_bank_file.write(_json_line(trajectory_row))

    with prompt_summary_path.open("w", encoding="utf-8") as prompt_summary_file:
        for prompt_row in prompt_rows:
            prompt_summary_file.write(_json_line(prompt_row))

    csv_rows = _main_results_rows(metrics_by_n, n_values)
    csv_fieldnames = [
        "N",
        "method",
        "method_label",
        "num_prompts",
        "num_bank_success_prompts",
        "num_bank_trajectories",
        "selected_mean_task_score",
        "conditional_success_recovery_rate",
        "top1_hit_rate_against_oracle_best",
        "mean_selected_response_length",
        "mean_selected_selection_score",
        "mean_selected_trajectory_value",
        "selection_score_field",
        "trajectory_value_field",
    ]
    with main_results_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    if not args.skip_plots:
        _plot_metric_vs_n(
            metrics_by_n=metrics_by_n,
            n_values=n_values,
            metric_name="selected_mean_task_score",
            ylabel="Selected Mean Task Score",
            output_path=accuracy_plot_path,
            method_names=PRIMARY_PLOT_METHODS,
            dpi=args.plot_dpi,
        )
        _plot_metric_vs_n(
            metrics_by_n=metrics_by_n,
            n_values=n_values,
            metric_name="conditional_success_recovery_rate",
            ylabel="Conditional Success Recovery Rate",
            output_path=recovery_plot_path,
            method_names=PRIMARY_PLOT_METHODS,
            dpi=args.plot_dpi,
        )
        _plot_metric_vs_n(
            metrics_by_n=metrics_by_n,
            n_values=n_values,
            metric_name="mean_selected_response_length",
            ylabel="Mean Selected Response Length",
            output_path=length_plot_path,
            method_names=PRIMARY_PLOT_METHODS,
            dpi=args.plot_dpi,
        )

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "execution_backend": execution_backend,
        "actor_checkpoint_dir": str(actor_checkpoint_dir),
        "old_critic_checkpoint_dir": str(old_critic_checkpoint_dir),
        "new_critic_checkpoint_dir": str(new_critic_checkpoint_dir),
        "merged_actor_dir": str(actor_hf_dir),
        "merged_old_critic_dir": str(old_critic_hf_dir),
        "merged_new_critic_dir": str(new_critic_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "output_dir": str(output_dir),
        "trajectory_bank_path": str(trajectory_bank_path),
        "prompt_summary_path": str(prompt_summary_path),
        "main_results_path": str(main_results_path),
        "accuracy_plot_path": None if args.skip_plots else str(accuracy_plot_path),
        "conditional_success_recovery_plot_path": None if args.skip_plots else str(recovery_plot_path),
        "mean_selected_response_length_plot_path": None if args.skip_plots else str(length_plot_path),
        "num_prompts": len(prompt_rows),
        "num_trajectories": len(trajectory_rows),
        "n_values": list(n_values),
        "max_bank_size": max_bank_size,
        "methods": list(METHOD_SPECS.keys()),
        "multi_worker_enabled": multi_worker_enabled,
        "shared_critic_checkpoint": bool(shared_critic_checkpoint),
        "dtype": args.dtype,
        "ray_address": ray_address,
        "ray_nodes": [asdict(node) for node in ray_nodes],
        "devices": None
        if actor_device is None or old_critic_device is None or new_critic_device is None
        else {
            "actor": str(actor_device),
            "old_critic": str(old_critic_device),
            "new_critic": str(new_critic_device),
        },
        "worker_layouts": [list(layout) for layout in worker_layouts],
        "worker_assignments": worker_assignments_to_jsonable(worker_assignments),
        "worker_summaries": worker_summaries,
        "eos_token_ids": list(eos_token_ids),
        "tokenizer_fingerprints": tokenizer_fingerprints,
        "shared_response_bank": True,
        "binary_task_scores": binary_task_scores,
        "reference_stage1_validation": stage1_validation_summary,
        "run_args": vars(args),
        "metrics_by_n": metrics_by_n,
        "comparisons_by_n": comparisons_by_n,
    }
    with summary_metrics_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    _write_output_readme(
        output_dir=output_dir,
        args=args,
        n_values=n_values,
        max_bank_size=max_bank_size,
        metrics_by_n=metrics_by_n,
        comparisons_by_n=comparisons_by_n,
        stage1_validation_summary=stage1_validation_summary,
        shared_critic_checkpoint=shared_critic_checkpoint,
    )
    return 0


_make_main_module_importable()

if __name__ == "__main__":
    raise SystemExit(main())
