from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
from queue import Empty
import random
import re
import shutil
import subprocess
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
    critic_sequence_values,
    sample_token_from_actor,
    set_decode_seed,
)


TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)
PREFIX_BUCKETS = ("early", "middle", "late")
RESERVED_METHOD_NAMES = {"random_chunk", "actor_logprob", "oracle_best_chunk"}
CRITIC_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
TRACE_MANIFEST_FILENAME = "chunk_benchmark_trace_manifest.json"
TRACE_BANK_REQUIRED_FIELDS = (
    "prompt_text",
    "prompt_token_ids",
    "prefix_token_ids",
    "chunk_token_ids",
    "chunk_end_sequence_token_ids",
)


@dataclass(frozen=True)
class CriticSpec:
    name: str
    checkpoint_dir: Path
    merged_root: Path | None = None
    device: str | None = None


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    actor_device: str | None
    critic_devices: tuple[str | None, ...]
    example_start: int
    example_end: int

    @property
    def num_examples(self) -> int:
        return max(self.example_end - self.example_start, 0)


@dataclass(frozen=True)
class SampledActorContinuation:
    prefix_length: int
    continuation_token_ids: tuple[int, ...]
    full_sequence_token_ids: tuple[int, ...]
    continuation_length: int
    continuation_text: str
    eos_emitted: bool
    max_length_hit: bool
    sum_actor_logprob: float


@dataclass(frozen=True)
class PrefixState:
    prefix_id: int
    bucket: str
    step_index: int
    prompt_length: int
    prefix_token_ids: tuple[int, ...]
    reference_response_length: int
    reference_response_eos_emitted: bool
    reference_response_max_length_hit: bool


@dataclass(frozen=True)
class MethodSpec:
    name: str
    method_type: str
    selector_score_field: str | None
    selected_final_score_key: str
    selected_chunk_id_key: str
    selected_chunk_length_key: str
    selected_completed_response_length_key: str
    selected_contains_eos_key: str
    selected_is_oracle_best_key: str
    selected_selector_value_key: str | None = None
    pairwise_correct_key: str | None = None
    pairwise_rankable_key: str | None = None
    pairwise_accuracy_key: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline chunk-ranking benchmark: collect shared prefix states from a frozen actor, sample a shared chunk "
            "bank per prefix, score each chunk boundary with one or more critics, roll each chunk to completion once, "
            "and compare ranking quality across critics and baselines."
        )
    )
    parser.add_argument(
        "--actor_checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint dir for the frozen actor. Required when generating a fresh trace bank.",
    )
    parser.add_argument(
        "--critic",
        action="append",
        nargs=2,
        metavar=("NAME", "CHECKPOINT_DIR"),
        default=[],
        help="Repeatable critic specification. Example: --critic old /path/to/ckpt --critic new /path/to/ckpt2",
    )
    parser.add_argument(
        "--critic_merged_root",
        action="append",
        nargs=2,
        metavar=("NAME", "MERGED_ROOT"),
        default=[],
        help="Optional merged HF root override for a named critic.",
    )
    parser.add_argument(
        "--critic_device",
        action="append",
        nargs=2,
        metavar=("NAME", "DEVICE"),
        default=[],
        help="Optional default device override for a named critic.",
    )
    parser.add_argument(
        "--old_critic_checkpoint_dir",
        type=str,
        default=None,
        help="Optional convenience alias for --critic old <checkpoint_dir>.",
    )
    parser.add_argument(
        "--new_critic_checkpoint_dir",
        type=str,
        default=None,
        help="Optional convenience alias for --critic new <checkpoint_dir>.",
    )
    parser.add_argument(
        "--old_critic_merged_root",
        type=str,
        default=None,
        help="Optional convenience alias for --critic_merged_root old <dir>.",
    )
    parser.add_argument(
        "--new_critic_merged_root",
        type=str,
        default=None,
        help="Optional convenience alias for --critic_merged_root new <dir>.",
    )
    parser.add_argument(
        "--old_critic_device",
        type=str,
        default=None,
        help="Optional convenience alias for --critic_device old <device>.",
    )
    parser.add_argument(
        "--new_critic_device",
        type=str,
        default=None,
        help="Optional convenience alias for --critic_device new <device>.",
    )
    parser.add_argument(
        "--existing_candidate_bank",
        type=str,
        default=None,
        help=(
            "Optional existing chunk_benchmark_candidates.jsonl trace bank to rescore without regenerating actor "
            "traces."
        ),
    )
    parser.add_argument(
        "--existing_trace_manifest",
        type=str,
        default=None,
        help=(
            "Optional existing chunk_benchmark_trace_manifest.json. Defaults to the manifest next to "
            "--existing_candidate_bank."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Evaluation parquet dataset. Required when generating a fresh trace bank.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for experiment artifacts.")
    parser.add_argument("--actor_merged_root", type=str, default=None, help="Optional merged HF root for actor.")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None, help="Optional response/ground-truth column key.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--chunk_size", type=int, default=32, help="Maximum chunk length m.")
    parser.add_argument("--num_chunk_candidates", type=int, default=8, help="Chunk bank size K per prefix.")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Fallback device for any unset model device.")
    parser.add_argument("--actor_device", type=str, default=None, help="Optional actor device override.")
    parser.add_argument(
        "--worker_layouts",
        nargs="+",
        default=None,
        help=(
            "Optional prompt-sharded multi-worker layouts. Each entry should be a single device to reuse for the "
            "actor and all critics, or 'actor_device,critic_device_for_name1,...' in the same order as the critics."
        ),
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
    return parser.parse_args()


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.isclose(x_arr.std(), 0.0) or np.isclose(y_arr.std(), 0.0):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_arr, kind="mergesort")
    ranks = np.empty(len(values_arr), dtype=np.float64)

    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values_arr[order[end + 1]] == values_arr[order[start]]:
            end += 1
        average_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = average_rank
        start = end + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_average_ranks(xs).tolist(), _average_ranks(ys).tolist())


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


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _critic_end_value_key(name: str) -> str:
    return f"{name}_critic_end_value"


def _critic_mean_value_key(name: str) -> str:
    return f"{name}_critic_mean_value"


def _critic_pairwise_correct_key(name: str) -> str:
    return f"{name}_critic_pairwise_correct_pairs"


def _critic_pairwise_rankable_key(name: str) -> str:
    return f"{name}_critic_pairwise_rankable_pairs"


def _critic_pairwise_accuracy_key(name: str) -> str:
    return f"{name}_critic_pairwise_accuracy"


def _critic_selected_chunk_id_key(name: str) -> str:
    return f"{name}_critic_selected_chunk_id"


def _critic_selected_chunk_end_value_key(name: str) -> str:
    return f"{name}_critic_selected_chunk_end_value"


def _critic_selected_chunk_mean_value_key(name: str) -> str:
    return f"{name}_critic_selected_chunk_mean_value"


def _critic_selected_final_score_key(name: str) -> str:
    return f"{name}_critic_selected_final_task_score"


def _critic_selected_chunk_length_key(name: str) -> str:
    return f"{name}_critic_selected_chunk_length"


def _critic_selected_completed_response_length_key(name: str) -> str:
    return f"{name}_critic_selected_completed_response_length"


def _critic_selected_contains_eos_key(name: str) -> str:
    return f"{name}_critic_selected_chunk_contains_eos"


def _critic_selected_is_oracle_key(name: str) -> str:
    return f"{name}_critic_selected_is_oracle_best"


def _named_mapping(entries: Sequence[Sequence[str]], *, entry_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_entry in entries:
        if len(raw_entry) != 2:
            raise ValueError(f"Each {entry_name} entry must contain exactly two values, got: {raw_entry}")
        name = str(raw_entry[0]).strip()
        value = str(raw_entry[1]).strip()
        if not name:
            raise ValueError(f"{entry_name} critic name cannot be empty.")
        if name in mapping:
            raise ValueError(f"Duplicate {entry_name} entry for critic '{name}'.")
        mapping[name] = value
    return mapping


def _collect_critic_specs(args: argparse.Namespace, *, allow_empty: bool = False) -> list[CriticSpec]:
    critic_checkpoint_map = _named_mapping(args.critic, entry_name="--critic")
    ordered_names = list(critic_checkpoint_map.keys())

    for convenience_name, checkpoint_dir in (
        ("old", args.old_critic_checkpoint_dir),
        ("new", args.new_critic_checkpoint_dir),
    ):
        if checkpoint_dir is None:
            continue
        if convenience_name in critic_checkpoint_map:
            raise ValueError(
                f"Critic '{convenience_name}' was provided both via --critic and via the convenience "
                f"--{convenience_name}_critic_checkpoint_dir flag."
            )
        critic_checkpoint_map[convenience_name] = checkpoint_dir
        ordered_names.append(convenience_name)

    if not ordered_names:
        if allow_empty:
            return []
        raise ValueError(
            "No critics were configured. Provide one or more --critic NAME CHECKPOINT_DIR entries, or use the "
            "--old_critic_checkpoint_dir / --new_critic_checkpoint_dir convenience flags."
        )

    merged_root_map = _named_mapping(args.critic_merged_root, entry_name="--critic_merged_root")
    device_map = _named_mapping(args.critic_device, entry_name="--critic_device")

    for convenience_name, merged_root in (
        ("old", args.old_critic_merged_root),
        ("new", args.new_critic_merged_root),
    ):
        if merged_root is None:
            continue
        if convenience_name in merged_root_map:
            raise ValueError(
                f"Critic '{convenience_name}' was provided both via --critic_merged_root and via the convenience "
                f"--{convenience_name}_critic_merged_root flag."
            )
        merged_root_map[convenience_name] = merged_root

    for convenience_name, device_name in (
        ("old", args.old_critic_device),
        ("new", args.new_critic_device),
    ):
        if device_name is None:
            continue
        if convenience_name in device_map:
            raise ValueError(
                f"Critic '{convenience_name}' was provided both via --critic_device and via the convenience "
                f"--{convenience_name}_critic_device flag."
            )
        device_map[convenience_name] = device_name

    unknown_override_names = (set(merged_root_map) | set(device_map)) - set(ordered_names)
    if unknown_override_names:
        raise ValueError(
            "Named critic overrides were provided for critics that do not exist in this run: "
            f"{sorted(unknown_override_names)}"
        )

    critics: list[CriticSpec] = []
    for name in ordered_names:
        if name in RESERVED_METHOD_NAMES:
            raise ValueError(
                f"Critic name '{name}' is reserved by a built-in baseline method. Choose a different critic name."
            )
        if not CRITIC_NAME_RE.match(name):
            raise ValueError(
                f"Critic name '{name}' is invalid. Use letters, numbers, and underscores only, starting with a "
                "letter."
            )
        critics.append(
            CriticSpec(
                name=name,
                checkpoint_dir=Path(critic_checkpoint_map[name]).resolve(),
                merged_root=Path(merged_root_map[name]).resolve() if name in merged_root_map else None,
                device=device_map.get(name),
            )
        )
    return critics


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
    return {
        "sha256": digest.hexdigest(),
        "files": included_files,
    }


def _assert_shared_tokenizer(tokenizer_fingerprints: dict[str, dict[str, Any]]) -> None:
    fingerprints = {name: payload["sha256"] for name, payload in tokenizer_fingerprints.items()}
    unique_fingerprints = set(fingerprints.values())
    if len(unique_fingerprints) == 1:
        return
    details = ", ".join(f"{name}={fingerprint}" for name, fingerprint in fingerprints.items())
    raise ValueError(
        "Actor/critic tokenizer files do not match, so the benchmark would not compare values on the exact same "
        f"token sequences. Fingerprints: {details}"
    )


def parse_worker_layouts(
    worker_layouts: list[str] | None,
    *,
    critic_specs: Sequence[CriticSpec],
    actor_device: str | None,
    default_device: str | None,
) -> list[tuple[str | None, tuple[str | None, ...]]]:
    critic_default_devices = {critic.name: critic.device for critic in critic_specs}
    critic_names = [critic.name for critic in critic_specs]
    num_critics = len(critic_names)

    if worker_layouts:
        parsed: list[tuple[str | None, tuple[str | None, ...]]] = []
        for raw_layout in worker_layouts:
            value = raw_layout.strip()
            if not value:
                continue
            parts = [part.strip() or None for part in value.split(",")]
            if len(parts) == 1:
                actor = parts[0]
                critic_parts = [parts[0] for _ in critic_names]
            elif len(parts) == num_critics + 1:
                actor = parts[0]
                critic_parts = parts[1:]
            else:
                raise ValueError(
                    "--worker_layouts entries must be either a single device or "
                    f"'actor_device,{','.join(f'critic_device_for_{name}' for name in critic_names)}'"
                )

            resolved_actor = actor or actor_device or default_device
            resolved_critics: list[str | None] = []
            for critic_name, explicit_critic_device in zip(critic_names, critic_parts, strict=True):
                resolved_critics.append(
                    explicit_critic_device
                    or critic_default_devices.get(critic_name)
                    or default_device
                    or resolved_actor
                )
            parsed.append((resolved_actor, tuple(resolved_critics)))

        if not parsed:
            raise ValueError("--worker_layouts was provided, but no valid layouts were parsed.")
        return parsed

    resolved_actor = actor_device or default_device
    resolved_critics = tuple(
        critic_default_devices.get(critic_name) or default_device or resolved_actor for critic_name in critic_names
    )
    return [(resolved_actor, resolved_critics)]


def build_worker_assignments(
    *,
    num_examples: int,
    worker_layouts: list[tuple[str | None, tuple[str | None, ...]]],
) -> list[WorkerAssignment]:
    if not worker_layouts:
        raise ValueError("At least one worker layout is required.")
    if num_examples <= 0:
        return []

    active_workers = min(len(worker_layouts), num_examples)
    assignments: list[WorkerAssignment] = []
    start = 0
    base = num_examples // active_workers
    remainder = num_examples % active_workers
    for worker_id in range(active_workers):
        shard_size = base + (1 if worker_id < remainder else 0)
        end = start + shard_size
        actor_dev, critic_devs = worker_layouts[worker_id]
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                actor_device=actor_dev,
                critic_devices=tuple(critic_devs),
                example_start=start,
                example_end=end,
            )
        )
        start = end
    return assignments


def worker_assignments_to_jsonable(assignments: Sequence[WorkerAssignment]) -> list[dict[str, Any]]:
    return [asdict(assignment) for assignment in assignments]


def build_runtime_notes(
    *,
    worker_layouts: Sequence[tuple[str | None, tuple[str | None, ...]]],
    critic_names: Sequence[str],
) -> list[str]:
    notes: list[str] = []
    if len(worker_layouts) <= 1:
        return notes

    actor_device_counts: dict[str, int] = {}
    critic_device_counts: dict[str, dict[str, int]] = {critic_name: {} for critic_name in critic_names}
    for actor_device, critic_devices in worker_layouts:
        if actor_device is not None:
            actor_device_counts[actor_device] = actor_device_counts.get(actor_device, 0) + 1
        for critic_name, critic_device in zip(critic_names, critic_devices, strict=True):
            if critic_device is None:
                continue
            device_counts = critic_device_counts[critic_name]
            device_counts[critic_device] = device_counts.get(critic_device, 0) + 1

    duplicated_actor_devices = sorted(device for device, count in actor_device_counts.items() if count > 1)
    if duplicated_actor_devices:
        notes.append(
            "Some worker layouts reuse actor devices across multiple worker processes: "
            f"{duplicated_actor_devices}. Each worker loads its own actor copy."
        )

    for critic_name in critic_names:
        duplicated_critic_devices = sorted(
            device for device, count in critic_device_counts[critic_name].items() if count > 1
        )
        if duplicated_critic_devices:
            notes.append(
                f"Some worker layouts reuse the {critic_name} critic device across multiple worker processes: "
                f"{duplicated_critic_devices}. Each worker loads its own critic copy on that device, so prompt-"
                "sharded multi-worker runs require enough GPU memory for duplicated model replicas."
            )
    return notes


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row))


def _infer_trace_manifest_path(
    *,
    existing_candidate_bank: Path,
    explicit_manifest_path: str | None,
) -> Path:
    if explicit_manifest_path is not None:
        return Path(explicit_manifest_path).resolve()
    return existing_candidate_bank.resolve().parent / TRACE_MANIFEST_FILENAME


def _infer_existing_critic_names(candidate_rows: Sequence[dict[str, Any]]) -> list[str]:
    if not candidate_rows:
        return []
    sample_row = candidate_rows[0]
    suffix = "_critic_end_value"
    critic_names: list[str] = []
    for key in sample_row:
        if not key.endswith(suffix):
            continue
        critic_name = key[: -len(suffix)]
        if _critic_mean_value_key(critic_name) not in sample_row:
            raise ValueError(
                f"Candidate bank row contains '{key}' but is missing '{_critic_mean_value_key(critic_name)}'."
            )
        critic_names.append(critic_name)
    return sorted(critic_names)


def _group_candidate_rows_by_prefix(
    candidate_rows: Sequence[dict[str, Any]],
) -> "OrderedDict[tuple[int, int], list[dict[str, Any]]]":
    grouped: "OrderedDict[tuple[int, int], list[dict[str, Any]]]" = OrderedDict()
    for row in sorted(candidate_rows, key=lambda item: (int(item["example_id"]), int(item["prefix_id"]), int(item["candidate_chunk_id"]))):
        prefix_key = (int(row["example_id"]), int(row["prefix_id"]))
        grouped.setdefault(prefix_key, []).append(row)
    return grouped


def validate_candidate_bank(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    require_trace_fields: bool,
) -> None:
    if not candidate_rows:
        raise ValueError("Candidate bank is empty.")

    required_fields = {
        "example_id",
        "prompt_id",
        "data_source",
        "ground_truth",
        "prefix_id",
        "prefix_bucket",
        "prefix_step_index",
        "prompt_length",
        "prefix_sequence_length",
        "prefix_generated_length",
        "reference_rollout_seed",
        "reference_response_length",
        "reference_response_eos_emitted",
        "reference_response_max_length_hit",
        "candidate_chunk_id",
        "chunk_token_ids",
        "realized_chunk_length",
        "chunk_contains_eos",
        "chunk_logprob",
        "chunk_end_sequence_length",
        "completed_response_length",
        "completed_response_eos_emitted",
        "completed_response_max_length_hit",
        "final_task_score",
    }
    if require_trace_fields:
        required_fields.update(TRACE_BANK_REQUIRED_FIELDS)

    sample_row = candidate_rows[0]
    missing_fields = sorted(field for field in required_fields if field not in sample_row)
    if missing_fields:
        raise ValueError(
            "Candidate bank is missing required fields for this operation: "
            f"{missing_fields}. If this bank was generated before reusable trace-bank support was added, "
            "please regenerate it."
        )

    critic_names = _infer_existing_critic_names(candidate_rows)
    grouped_rows = _group_candidate_rows_by_prefix(candidate_rows)
    for prefix_key, rows in grouped_rows.items():
        candidate_ids = [int(row["candidate_chunk_id"]) for row in rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Prefix {prefix_key} contains duplicate candidate_chunk_id values: {candidate_ids}")

        reference_row = rows[0]
        shared_fields = (
            "prompt_id",
            "prompt_length",
            "prefix_bucket",
            "prefix_step_index",
            "prefix_sequence_length",
            "prefix_generated_length",
            "reference_rollout_seed",
            "reference_response_length",
            "reference_response_eos_emitted",
            "reference_response_max_length_hit",
            "data_source",
            "ground_truth",
        )
        if require_trace_fields:
            shared_fields = shared_fields + ("prompt_text", "prompt_token_ids", "prefix_token_ids")

        for row in rows[1:]:
            for field in shared_fields:
                if row[field] != reference_row[field]:
                    raise ValueError(
                        f"Candidate bank prefix {prefix_key} is inconsistent for field '{field}': "
                        f"{reference_row[field]!r} vs {row[field]!r}"
                    )

        for row in rows:
            realized_chunk_length = int(row["realized_chunk_length"])
            chunk_token_ids = [int(token_id) for token_id in row["chunk_token_ids"]]
            if len(chunk_token_ids) != realized_chunk_length:
                raise ValueError(
                    f"Prefix {prefix_key} candidate {row['candidate_chunk_id']} has realized_chunk_length="
                    f"{realized_chunk_length} but {len(chunk_token_ids)} chunk_token_ids."
                )

            prefix_sequence_length = int(row["prefix_sequence_length"])
            prompt_length = int(row["prompt_length"])
            prefix_step_index = int(row["prefix_step_index"])
            if prefix_sequence_length != prompt_length + prefix_step_index:
                raise ValueError(
                    f"Prefix {prefix_key} is inconsistent: prefix_sequence_length={prefix_sequence_length}, "
                    f"prompt_length={prompt_length}, prefix_step_index={prefix_step_index}."
                )

            if require_trace_fields:
                prompt_token_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
                prefix_token_ids = [int(token_id) for token_id in row["prefix_token_ids"]]
                chunk_end_sequence_token_ids = [int(token_id) for token_id in row["chunk_end_sequence_token_ids"]]

                if len(prompt_token_ids) != prompt_length:
                    raise ValueError(
                        f"Prefix {prefix_key} has prompt_length={prompt_length} but {len(prompt_token_ids)} prompt_token_ids."
                    )
                if len(prefix_token_ids) != prefix_sequence_length:
                    raise ValueError(
                        f"Prefix {prefix_key} has prefix_sequence_length={prefix_sequence_length} but "
                        f"{len(prefix_token_ids)} prefix_token_ids."
                    )
                if prompt_token_ids != prefix_token_ids[:prompt_length]:
                    raise ValueError(f"Prefix {prefix_key} prompt_token_ids do not match prefix_token_ids[:prompt_length].")
                if len(chunk_end_sequence_token_ids) != prefix_sequence_length + realized_chunk_length:
                    raise ValueError(
                        f"Prefix {prefix_key} candidate {row['candidate_chunk_id']} has chunk_end_sequence_length="
                        f"{len(chunk_end_sequence_token_ids)} but expected {prefix_sequence_length + realized_chunk_length}."
                    )
                if prefix_token_ids != chunk_end_sequence_token_ids[:prefix_sequence_length]:
                    raise ValueError(
                        f"Prefix {prefix_key} candidate {row['candidate_chunk_id']} chunk_end_sequence_token_ids do "
                        "not start with prefix_token_ids."
                    )
                if chunk_token_ids != chunk_end_sequence_token_ids[prefix_sequence_length:]:
                    raise ValueError(
                        f"Prefix {prefix_key} candidate {row['candidate_chunk_id']} chunk_token_ids do not match the "
                        "tail of chunk_end_sequence_token_ids."
                    )

            if "completed_response_token_ids" in row:
                completed_response_token_ids = [int(token_id) for token_id in row["completed_response_token_ids"]]
                if len(completed_response_token_ids) != int(row["completed_response_length"]):
                    raise ValueError(
                        f"Prefix {prefix_key} candidate {row['candidate_chunk_id']} has completed_response_length="
                        f"{row['completed_response_length']} but {len(completed_response_token_ids)} completed_response_token_ids."
                    )

            for critic_name in critic_names:
                if _critic_end_value_key(critic_name) not in row or _critic_mean_value_key(critic_name) not in row:
                    raise ValueError(
                        f"Candidate bank prefix {prefix_key} candidate {row['candidate_chunk_id']} is missing critic "
                        f"score fields for '{critic_name}'."
                    )


def _example_record_from_trace_row(row: dict[str, Any]) -> ExampleRecord:
    return ExampleRecord(
        example_id=int(row["example_id"]),
        prompt_text=str(row["prompt_text"]),
        data_source=str(row["data_source"]),
        ground_truth=row["ground_truth"],
        prompt_token_ids=tuple(int(token_id) for token_id in row["prompt_token_ids"]),
    )


def _prefix_state_from_trace_row(row: dict[str, Any]) -> PrefixState:
    return PrefixState(
        prefix_id=int(row["prefix_id"]),
        bucket=str(row["prefix_bucket"]),
        step_index=int(row["prefix_step_index"]),
        prompt_length=int(row["prompt_length"]),
        prefix_token_ids=tuple(int(token_id) for token_id in row["prefix_token_ids"]),
        reference_response_length=int(row["reference_response_length"]),
        reference_response_eos_emitted=bool(row["reference_response_eos_emitted"]),
        reference_response_max_length_hit=bool(row["reference_response_max_length_hit"]),
    )


def rebuild_prefix_rows_from_candidate_rows(
    *,
    candidate_rows: Sequence[dict[str, Any]],
    base_seed: int,
    critic_names: Sequence[str],
) -> list[dict[str, Any]]:
    validate_candidate_bank(candidate_rows, require_trace_fields=True)
    prefix_rows: list[dict[str, Any]] = []
    for _prefix_key, rows in _group_candidate_rows_by_prefix(candidate_rows).items():
        reference_row = rows[0]
        prefix_rows.append(
            build_prefix_summary(
                example=_example_record_from_trace_row(reference_row),
                prefix_state=_prefix_state_from_trace_row(reference_row),
                candidate_rows=rows,
                critic_names=critic_names,
                base_seed=base_seed,
                reference_rollout_seed=int(reference_row["reference_rollout_seed"]),
            )
        )
    return prefix_rows


def _reference_rollout_seed(base_seed: int, *, example_id: int) -> int:
    return int(base_seed + (example_id + 1) * 1_000_003 + 11)


def _chunk_candidate_seed(base_seed: int, *, example_id: int, prefix_id: int, candidate_id: int) -> int:
    return int(
        base_seed
        + (example_id + 1) * 1_000_003
        + (prefix_id + 1) * 1_000_000_007
        + candidate_id * 97_003
        + 101
    )


def _completion_seed(base_seed: int, *, example_id: int, prefix_id: int, candidate_id: int) -> int:
    return int(
        base_seed
        + (example_id + 1) * 2_000_029
        + (prefix_id + 1) * 1_000_000_007
        + candidate_id * 193_001
        + 503
    )


def _random_selector_seed(base_seed: int, *, example_id: int, prefix_id: int) -> int:
    return int(base_seed + (example_id + 1) * 3_000_043 + (prefix_id + 1) * 53_111 + 17)


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


def sample_actor_continuation(
    *,
    actor,
    tokenizer,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    seed: int,
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
) -> SampledActorContinuation:
    set_decode_seed(seed)
    actor_state = ActorStepper(actor, prefix_ids, use_cache=use_actor_cache)
    continuation_token_ids: list[int] = []
    eos_emitted = False
    sum_actor_logprob = 0.0

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
        sum_actor_logprob += float(actor_log_probs[0, selected_token_id].item())
        continuation_token_ids.append(selected_token_id)
        actor_state.append(selected_token_id)
        if selected_token_id in eos_token_ids:
            eos_emitted = True
            break

    continuation_length = len(continuation_token_ids)
    max_length_hit = bool(max_new_tokens > 0 and not eos_emitted and continuation_length >= max_new_tokens)
    full_sequence_token_ids = tuple(int(token_id) for token_id in actor_state.sequence_ids[0].detach().cpu().tolist())

    return SampledActorContinuation(
        prefix_length=int(prefix_ids.shape[1]),
        continuation_token_ids=tuple(int(token_id) for token_id in continuation_token_ids),
        full_sequence_token_ids=full_sequence_token_ids,
        continuation_length=continuation_length,
        continuation_text=tokenizer.decode(continuation_token_ids, skip_special_tokens=True),
        eos_emitted=eos_emitted,
        max_length_hit=max_length_hit,
        sum_actor_logprob=float(sum_actor_logprob),
    )


def _prefix_bucket_indices(num_generated_tokens: int) -> dict[str, list[int]]:
    if num_generated_tokens <= 0:
        return {bucket: [] for bucket in PREFIX_BUCKETS}

    indices = list(range(num_generated_tokens))
    early_end = int(math.ceil(num_generated_tokens * 0.25))
    middle_end = int(math.ceil(num_generated_tokens * 0.75))
    return {
        "early": indices[:early_end],
        "middle": indices[early_end:middle_end],
        "late": indices[middle_end:],
    }


def select_prefix_states(
    *,
    prompt_ids: torch.Tensor,
    reference_rollout: SampledActorContinuation,
) -> list[PrefixState]:
    generated_token_ids = list(reference_rollout.continuation_token_ids)
    bucket_indices = _prefix_bucket_indices(len(generated_token_ids))
    prompt_token_ids = [int(token_id) for token_id in prompt_ids[0].detach().cpu().tolist()]

    prefix_states: list[PrefixState] = []
    for bucket in PREFIX_BUCKETS:
        candidate_indices = bucket_indices[bucket]
        if not candidate_indices:
            continue
        step_index = candidate_indices[(len(candidate_indices) - 1) // 2]
        prefix_token_ids = tuple(prompt_token_ids + generated_token_ids[:step_index])
        prefix_states.append(
            PrefixState(
                prefix_id=len(prefix_states),
                bucket=bucket,
                step_index=int(step_index),
                prompt_length=len(prompt_token_ids),
                prefix_token_ids=prefix_token_ids,
                reference_response_length=int(reference_rollout.continuation_length),
                reference_response_eos_emitted=bool(reference_rollout.eos_emitted),
                reference_response_max_length_hit=bool(reference_rollout.max_length_hit),
            )
        )
    return prefix_states


def _score_chunk_with_critics(
    *,
    critic_models: dict[str, Any],
    critic_devices: dict[str, torch.device],
    full_sequence_token_ids: Sequence[int],
    prefix_length: int,
    chunk_length: int,
) -> dict[str, dict[str, float]]:
    critic_scores: dict[str, dict[str, float]] = {}
    for critic_name, critic in critic_models.items():
        critic_device = critic_devices[critic_name]
        input_ids = torch.tensor([list(full_sequence_token_ids)], device=critic_device, dtype=torch.long)
        values = critic_sequence_values(critic, input_ids)[0]
        chunk_values = values[prefix_length : prefix_length + chunk_length]
        if chunk_values.numel() != chunk_length:
            raise RuntimeError(
                f"Chunk value extraction length mismatch for critic '{critic_name}': "
                f"expected {chunk_length}, got {int(chunk_values.numel())}"
            )
        critic_scores[critic_name] = {
            "end_value": float(chunk_values[-1].item()),
            "mean_value": float(chunk_values.mean().item()),
        }
    return critic_scores


def _argmax_indices(values: Sequence[float]) -> list[int]:
    if not values:
        raise ValueError("Cannot take argmax of an empty list.")
    best_value = max(values)
    return [index for index, value in enumerate(values) if value == best_value]


def _pairwise_ranking_stats(task_scores: Sequence[float], ranking_scores: Sequence[float]) -> dict[str, Any]:
    if len(task_scores) != len(ranking_scores):
        raise ValueError("task_scores and ranking_scores must have the same length.")

    correct_pairs = 0.0
    rankable_pairs = 0
    for left_index in range(len(task_scores)):
        for right_index in range(left_index + 1, len(task_scores)):
            left_score = float(task_scores[left_index])
            right_score = float(task_scores[right_index])
            if left_score == right_score:
                continue
            rankable_pairs += 1
            left_value = float(ranking_scores[left_index])
            right_value = float(ranking_scores[right_index])
            if left_value == right_value:
                correct_pairs += 0.5
                continue
            ordered_correctly = (left_value > right_value) == (left_score > right_score)
            correct_pairs += 1.0 if ordered_correctly else 0.0

    return {
        "correct_pairs": float(correct_pairs),
        "rankable_pairs": int(rankable_pairs),
        "accuracy": (float(correct_pairs) / rankable_pairs) if rankable_pairs > 0 else None,
    }


def _candidate_identity(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in row["chunk_token_ids"])


def build_method_specs(critic_names: Sequence[str]) -> "OrderedDict[str, MethodSpec]":
    specs: "OrderedDict[str, MethodSpec]" = OrderedDict()
    specs["random_chunk"] = MethodSpec(
        name="random_chunk",
        method_type="baseline",
        selector_score_field=None,
        selected_final_score_key="random_selected_final_task_score",
        selected_chunk_id_key="random_selected_chunk_id",
        selected_chunk_length_key="random_selected_chunk_length",
        selected_completed_response_length_key="random_selected_completed_response_length",
        selected_contains_eos_key="random_selected_chunk_contains_eos",
        selected_is_oracle_best_key="random_selected_is_oracle_best",
    )
    specs["actor_logprob"] = MethodSpec(
        name="actor_logprob",
        method_type="baseline",
        selector_score_field="chunk_logprob",
        selected_final_score_key="actor_logprob_selected_final_task_score",
        selected_chunk_id_key="actor_logprob_selected_chunk_id",
        selected_chunk_length_key="actor_logprob_selected_chunk_length",
        selected_completed_response_length_key="actor_logprob_selected_completed_response_length",
        selected_contains_eos_key="actor_logprob_selected_chunk_contains_eos",
        selected_is_oracle_best_key="actor_logprob_selected_is_oracle_best",
        selected_selector_value_key="actor_logprob_selected_chunk_logprob",
        pairwise_correct_key="actor_logprob_pairwise_correct_pairs",
        pairwise_rankable_key="actor_logprob_pairwise_rankable_pairs",
        pairwise_accuracy_key="actor_logprob_pairwise_accuracy",
    )
    for critic_name in critic_names:
        specs[critic_name] = MethodSpec(
            name=critic_name,
            method_type="critic",
            selector_score_field=_critic_end_value_key(critic_name),
            selected_final_score_key=_critic_selected_final_score_key(critic_name),
            selected_chunk_id_key=_critic_selected_chunk_id_key(critic_name),
            selected_chunk_length_key=_critic_selected_chunk_length_key(critic_name),
            selected_completed_response_length_key=_critic_selected_completed_response_length_key(critic_name),
            selected_contains_eos_key=_critic_selected_contains_eos_key(critic_name),
            selected_is_oracle_best_key=_critic_selected_is_oracle_key(critic_name),
            selected_selector_value_key=_critic_selected_chunk_end_value_key(critic_name),
            pairwise_correct_key=_critic_pairwise_correct_key(critic_name),
            pairwise_rankable_key=_critic_pairwise_rankable_key(critic_name),
            pairwise_accuracy_key=_critic_pairwise_accuracy_key(critic_name),
        )
    specs["oracle_best_chunk"] = MethodSpec(
        name="oracle_best_chunk",
        method_type="baseline",
        selector_score_field=None,
        selected_final_score_key="oracle_best_chunk_score",
        selected_chunk_id_key="oracle_best_chunk_id",
        selected_chunk_length_key="oracle_best_chunk_length",
        selected_completed_response_length_key="oracle_best_completed_response_length",
        selected_contains_eos_key="oracle_best_chunk_contains_eos",
        selected_is_oracle_best_key="oracle_best_is_oracle_best",
        selected_selector_value_key="oracle_best_chunk_score",
    )
    return specs


def build_prefix_summary(
    *,
    example: ExampleRecord,
    prefix_state: PrefixState,
    candidate_rows: Sequence[dict[str, Any]],
    critic_names: Sequence[str],
    base_seed: int,
    reference_rollout_seed: int,
) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError("Each prefix must have at least one candidate row.")

    sorted_rows = sorted(candidate_rows, key=lambda row: int(row["candidate_chunk_id"]))
    task_scores = [float(row["final_task_score"]) for row in sorted_rows]
    chunk_logprobs = [float(row["chunk_logprob"]) for row in sorted_rows]
    candidate_chunk_ids = [int(row["candidate_chunk_id"]) for row in sorted_rows]
    candidate_chunk_lengths = [int(row["realized_chunk_length"]) for row in sorted_rows]
    candidate_completed_lengths = [int(row["completed_response_length"]) for row in sorted_rows]
    candidate_contains_eos = [bool(row["chunk_contains_eos"]) for row in sorted_rows]

    oracle_best_positions = _argmax_indices(task_scores)
    actor_logprob_positions = _argmax_indices(chunk_logprobs)
    random_rng = random.Random(
        _random_selector_seed(base_seed, example_id=example.example_id, prefix_id=prefix_state.prefix_id)
    )
    random_position = int(random_rng.randrange(len(sorted_rows)))

    actor_pairwise = _pairwise_ranking_stats(task_scores, chunk_logprobs)
    oracle_best_position = oracle_best_positions[0]
    actor_logprob_position = actor_logprob_positions[0]

    chunk_identities = [_candidate_identity(row) for row in sorted_rows]
    unique_chunk_count = len(set(chunk_identities))
    duplicate_candidate_count = len(sorted_rows) - unique_chunk_count
    duplicate_candidate_fraction = duplicate_candidate_count / len(sorted_rows)

    prefix_summary: dict[str, Any] = {
        "example_id": int(example.example_id),
        "prompt_id": int(example.example_id),
        "data_source": example.data_source,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "prefix_id": int(prefix_state.prefix_id),
        "prefix_bucket": prefix_state.bucket,
        "prefix_step_index": int(prefix_state.step_index),
        "prompt_length": int(prefix_state.prompt_length),
        "prefix_sequence_length": int(len(prefix_state.prefix_token_ids)),
        "prefix_generated_length": int(prefix_state.step_index),
        "reference_rollout_seed": int(reference_rollout_seed),
        "reference_response_length": int(prefix_state.reference_response_length),
        "reference_response_eos_emitted": bool(prefix_state.reference_response_eos_emitted),
        "reference_response_max_length_hit": bool(prefix_state.reference_response_max_length_hit),
        "num_candidates": len(sorted_rows),
        "candidate_chunk_ids": candidate_chunk_ids,
        "candidate_chunk_lengths": candidate_chunk_lengths,
        "candidate_completed_response_lengths": candidate_completed_lengths,
        "candidate_final_task_scores": task_scores,
        "num_rankable_pairs": int(actor_pairwise["rankable_pairs"]),
        "num_unique_chunks": int(unique_chunk_count),
        "num_duplicate_candidates": int(duplicate_candidate_count),
        "duplicate_candidate_fraction": float(duplicate_candidate_fraction),
        "has_duplicate_chunks": bool(duplicate_candidate_count > 0),
        "prefix_has_any_success": bool(any(score == 1.0 for score in task_scores)),
        "oracle_best_chunk_id": int(sorted_rows[oracle_best_position]["candidate_chunk_id"]),
        "oracle_best_chunk_ids": [int(sorted_rows[position]["candidate_chunk_id"]) for position in oracle_best_positions],
        "oracle_best_chunk_score": float(sorted_rows[oracle_best_position]["final_task_score"]),
        "oracle_best_chunk_length": int(sorted_rows[oracle_best_position]["realized_chunk_length"]),
        "oracle_best_completed_response_length": int(sorted_rows[oracle_best_position]["completed_response_length"]),
        "oracle_best_chunk_contains_eos": bool(sorted_rows[oracle_best_position]["chunk_contains_eos"]),
        "oracle_best_is_oracle_best": True,
        "random_selected_chunk_id": int(sorted_rows[random_position]["candidate_chunk_id"]),
        "random_selected_final_task_score": float(sorted_rows[random_position]["final_task_score"]),
        "random_selected_chunk_length": int(sorted_rows[random_position]["realized_chunk_length"]),
        "random_selected_completed_response_length": int(sorted_rows[random_position]["completed_response_length"]),
        "random_selected_chunk_contains_eos": bool(sorted_rows[random_position]["chunk_contains_eos"]),
        "random_selected_is_oracle_best": random_position in oracle_best_positions,
        "actor_logprob_selected_chunk_id": int(sorted_rows[actor_logprob_position]["candidate_chunk_id"]),
        "actor_logprob_selected_chunk_ids": [
            int(sorted_rows[position]["candidate_chunk_id"]) for position in actor_logprob_positions
        ],
        "actor_logprob_selected_chunk_logprob": float(sorted_rows[actor_logprob_position]["chunk_logprob"]),
        "actor_logprob_selected_final_task_score": float(sorted_rows[actor_logprob_position]["final_task_score"]),
        "actor_logprob_selected_chunk_length": int(sorted_rows[actor_logprob_position]["realized_chunk_length"]),
        "actor_logprob_selected_completed_response_length": int(
            sorted_rows[actor_logprob_position]["completed_response_length"]
        ),
        "actor_logprob_selected_chunk_contains_eos": bool(sorted_rows[actor_logprob_position]["chunk_contains_eos"]),
        "actor_logprob_selected_is_oracle_best": actor_logprob_position in oracle_best_positions,
        "actor_logprob_pairwise_correct_pairs": float(actor_pairwise["correct_pairs"]),
        "actor_logprob_pairwise_rankable_pairs": int(actor_pairwise["rankable_pairs"]),
        "actor_logprob_pairwise_accuracy": actor_pairwise["accuracy"],
    }

    for critic_name in critic_names:
        end_value_key = _critic_end_value_key(critic_name)
        mean_value_key = _critic_mean_value_key(critic_name)
        critic_end_values = [float(row[end_value_key]) for row in sorted_rows]
        critic_pairwise = _pairwise_ranking_stats(task_scores, critic_end_values)
        selected_positions = _argmax_indices(critic_end_values)
        selected_position = selected_positions[0]
        selected_row = sorted_rows[selected_position]

        prefix_summary.update(
            {
                _critic_selected_chunk_id_key(critic_name): int(selected_row["candidate_chunk_id"]),
                f"{critic_name}_critic_selected_chunk_ids": [
                    int(sorted_rows[position]["candidate_chunk_id"]) for position in selected_positions
                ],
                _critic_selected_chunk_end_value_key(critic_name): float(selected_row[end_value_key]),
                _critic_selected_chunk_mean_value_key(critic_name): float(selected_row[mean_value_key]),
                _critic_selected_final_score_key(critic_name): float(selected_row["final_task_score"]),
                _critic_selected_chunk_length_key(critic_name): int(selected_row["realized_chunk_length"]),
                _critic_selected_completed_response_length_key(critic_name): int(
                    selected_row["completed_response_length"]
                ),
                _critic_selected_contains_eos_key(critic_name): bool(selected_row["chunk_contains_eos"]),
                _critic_selected_is_oracle_key(critic_name): selected_position in oracle_best_positions,
                _critic_pairwise_correct_key(critic_name): float(critic_pairwise["correct_pairs"]),
                _critic_pairwise_rankable_key(critic_name): int(critic_pairwise["rankable_pairs"]),
                _critic_pairwise_accuracy_key(critic_name): critic_pairwise["accuracy"],
            }
        )

    return prefix_summary


def process_prefix(
    *,
    actor,
    critic_models: dict[str, Any],
    tokenizer,
    example: ExampleRecord,
    prefix_state: PrefixState,
    actor_device: torch.device,
    critic_devices: dict[str, torch.device],
    max_new_tokens: int,
    chunk_size: int,
    num_chunk_candidates: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    base_seed: int,
    critic_names: Sequence[str],
    reference_rollout_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefix_ids = torch.tensor([list(prefix_state.prefix_token_ids)], device=actor_device, dtype=torch.long)
    prefix_length = int(prefix_ids.shape[1])
    prompt_token_ids = list(prefix_state.prefix_token_ids[: prefix_state.prompt_length])
    remaining_total_budget = max_new_tokens - int(prefix_state.step_index)
    max_chunk_len = min(chunk_size, remaining_total_budget)
    if max_chunk_len <= 0:
        raise ValueError(
            f"Prefix example_id={example.example_id} prefix_id={prefix_state.prefix_id} has no remaining token budget."
        )

    candidate_rows: list[dict[str, Any]] = []
    for candidate_id in range(num_chunk_candidates):
        chunk_seed = _chunk_candidate_seed(
            base_seed,
            example_id=example.example_id,
            prefix_id=prefix_state.prefix_id,
            candidate_id=candidate_id,
        )
        completion_seed = _completion_seed(
            base_seed,
            example_id=example.example_id,
            prefix_id=prefix_state.prefix_id,
            candidate_id=candidate_id,
        )
        chunk_rollout = sample_actor_continuation(
            actor=actor,
            tokenizer=tokenizer,
            prefix_ids=prefix_ids,
            max_new_tokens=max_chunk_len,
            seed=chunk_seed,
            actor_sampling_mode=actor_sampling_mode,
            actor_temperature=actor_temperature,
            actor_top_p=actor_top_p,
            actor_top_k=actor_top_k,
            eos_token_ids=eos_token_ids,
            use_actor_cache=use_actor_cache,
        )
        if chunk_rollout.continuation_length <= 0:
            raise RuntimeError("Chunk candidate generation produced zero tokens, which should be impossible.")

        critic_scores = _score_chunk_with_critics(
            critic_models=critic_models,
            critic_devices=critic_devices,
            full_sequence_token_ids=chunk_rollout.full_sequence_token_ids,
            prefix_length=prefix_length,
            chunk_length=chunk_rollout.continuation_length,
        )

        final_full_sequence_token_ids = chunk_rollout.full_sequence_token_ids
        continuation_after_chunk_length = 0
        final_eos_emitted = bool(chunk_rollout.eos_emitted)
        if not chunk_rollout.eos_emitted:
            remaining_completion_budget = max_new_tokens - int(prefix_state.step_index) - chunk_rollout.continuation_length
            if remaining_completion_budget > 0:
                chunk_end_ids = torch.tensor(
                    [list(chunk_rollout.full_sequence_token_ids)],
                    device=actor_device,
                    dtype=prefix_ids.dtype,
                )
                completion_rollout = sample_actor_continuation(
                    actor=actor,
                    tokenizer=tokenizer,
                    prefix_ids=chunk_end_ids,
                    max_new_tokens=remaining_completion_budget,
                    seed=completion_seed,
                    actor_sampling_mode=actor_sampling_mode,
                    actor_temperature=actor_temperature,
                    actor_top_p=actor_top_p,
                    actor_top_k=actor_top_k,
                    eos_token_ids=eos_token_ids,
                    use_actor_cache=use_actor_cache,
                )
                final_full_sequence_token_ids = completion_rollout.full_sequence_token_ids
                continuation_after_chunk_length = int(completion_rollout.continuation_length)
                final_eos_emitted = bool(completion_rollout.eos_emitted)

        completed_response_token_ids = final_full_sequence_token_ids[prefix_state.prompt_length :]
        completed_response_text = tokenizer.decode(list(completed_response_token_ids), skip_special_tokens=True)
        completed_response_length = len(completed_response_token_ids)
        final_max_length_hit = bool(max_new_tokens > 0 and not final_eos_emitted and completed_response_length >= max_new_tokens)
        final_task_score = float(score_response(example, completed_response_text))

        candidate_row: dict[str, Any] = {
            "example_id": int(example.example_id),
            "prompt_id": int(example.example_id),
            "data_source": example.data_source,
            "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
            "prompt_text": example.prompt_text,
            "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
            "prefix_id": int(prefix_state.prefix_id),
            "prefix_bucket": prefix_state.bucket,
            "prefix_step_index": int(prefix_state.step_index),
            "prompt_length": int(prefix_state.prompt_length),
            "prefix_sequence_length": int(prefix_length),
            "prefix_generated_length": int(prefix_state.step_index),
            "prefix_token_ids": [int(token_id) for token_id in prefix_state.prefix_token_ids],
            "reference_rollout_seed": int(reference_rollout_seed),
            "reference_response_length": int(prefix_state.reference_response_length),
            "reference_response_eos_emitted": bool(prefix_state.reference_response_eos_emitted),
            "reference_response_max_length_hit": bool(prefix_state.reference_response_max_length_hit),
            "candidate_chunk_id": int(candidate_id),
            "chunk_seed": int(chunk_seed),
            "completion_seed": int(completion_seed),
            "chunk_token_ids": [int(token_id) for token_id in chunk_rollout.continuation_token_ids],
            "chunk_text": chunk_rollout.continuation_text,
            "realized_chunk_length": int(chunk_rollout.continuation_length),
            "chunk_contains_eos": bool(chunk_rollout.eos_emitted),
            "chunk_logprob": float(chunk_rollout.sum_actor_logprob),
            "chunk_end_sequence_length": int(len(chunk_rollout.full_sequence_token_ids)),
            "chunk_end_sequence_token_ids": [int(token_id) for token_id in chunk_rollout.full_sequence_token_ids],
            "continuation_after_chunk_length": int(continuation_after_chunk_length),
            "completed_response_length": int(completed_response_length),
            "completed_response_eos_emitted": bool(final_eos_emitted),
            "completed_response_max_length_hit": bool(final_max_length_hit),
            "completed_response_token_ids": [int(token_id) for token_id in completed_response_token_ids],
            "completed_response_text": completed_response_text,
            "final_task_score": float(final_task_score),
        }
        for critic_name in critic_names:
            candidate_row[_critic_end_value_key(critic_name)] = float(critic_scores[critic_name]["end_value"])
            candidate_row[_critic_mean_value_key(critic_name)] = float(critic_scores[critic_name]["mean_value"])
        candidate_rows.append(candidate_row)

    prefix_summary = build_prefix_summary(
        example=example,
        prefix_state=prefix_state,
        candidate_rows=candidate_rows,
        critic_names=critic_names,
        base_seed=base_seed,
        reference_rollout_seed=reference_rollout_seed,
    )
    return candidate_rows, prefix_summary


def process_example(
    *,
    actor,
    critic_models: dict[str, Any],
    tokenizer,
    example: ExampleRecord,
    actor_device: torch.device,
    critic_devices: dict[str, torch.device],
    max_prompt_length: int,
    max_new_tokens: int,
    chunk_size: int,
    num_chunk_candidates: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    base_seed: int,
    critic_names: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_ids = _prompt_ids_tensor(
        example=example,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        device=actor_device,
    )
    reference_seed = _reference_rollout_seed(base_seed, example_id=example.example_id)
    reference_rollout = sample_actor_continuation(
        actor=actor,
        tokenizer=tokenizer,
        prefix_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        seed=reference_seed,
        actor_sampling_mode=actor_sampling_mode,
        actor_temperature=actor_temperature,
        actor_top_p=actor_top_p,
        actor_top_k=actor_top_k,
        eos_token_ids=eos_token_ids,
        use_actor_cache=use_actor_cache,
    )
    prefix_states = select_prefix_states(prompt_ids=prompt_ids, reference_rollout=reference_rollout)

    candidate_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    for prefix_state in prefix_states:
        prefix_candidate_rows, prefix_summary = process_prefix(
            actor=actor,
            critic_models=critic_models,
            tokenizer=tokenizer,
            example=example,
            prefix_state=prefix_state,
            actor_device=actor_device,
            critic_devices=critic_devices,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            num_chunk_candidates=num_chunk_candidates,
            eos_token_ids=eos_token_ids,
            actor_sampling_mode=actor_sampling_mode,
            actor_temperature=actor_temperature,
            actor_top_p=actor_top_p,
            actor_top_k=actor_top_k,
            use_actor_cache=use_actor_cache,
            base_seed=base_seed,
            critic_names=critic_names,
            reference_rollout_seed=reference_seed,
        )
        candidate_rows.extend(prefix_candidate_rows)
        prefix_rows.append(prefix_summary)
    return candidate_rows, prefix_rows


def score_existing_candidate_rows_with_critics(
    *,
    candidate_rows: Sequence[dict[str, Any]],
    critic_specs: Sequence[CriticSpec],
    dtype_name: str,
    trust_remote_code: bool,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    validate_candidate_bank(candidate_rows, require_trace_fields=True)
    if not critic_specs:
        return [dict(row) for row in candidate_rows], {}

    existing_critic_names = set(_infer_existing_critic_names(candidate_rows))
    requested_critic_names = [critic.name for critic in critic_specs]
    overlapping_names = sorted(existing_critic_names & set(requested_critic_names))
    if overlapping_names:
        raise ValueError(
            "The existing trace bank already contains critic scores for: "
            f"{overlapping_names}. To avoid silently overwriting saved scores, please use distinct critic names."
        )

    dtype = resolve_dtype(dtype_name)
    critic_devices = {
        critic.name: resolve_device(critic.device)
        for critic in critic_specs
    }
    critic_models = {
        critic.name: load_critic_model(
            critic.checkpoint_dir,
            dtype=dtype,
            device=critic_devices[critic.name],
            trust_remote_code=trust_remote_code,
        )
        for critic in critic_specs
    }

    updated_rows: list[dict[str, Any]] = []
    with tqdm(candidate_rows, desc="rescore_trace_bank", unit="candidate", dynamic_ncols=True) as progress_bar:
        for row in progress_bar:
            prefix_token_ids = [int(token_id) for token_id in row["prefix_token_ids"]]
            chunk_token_ids = [int(token_id) for token_id in row["chunk_token_ids"]]
            chunk_end_sequence_token_ids = [int(token_id) for token_id in row["chunk_end_sequence_token_ids"]]
            prefix_length = len(prefix_token_ids)
            chunk_length = len(chunk_token_ids)
            critic_scores = _score_chunk_with_critics(
                critic_models=critic_models,
                critic_devices=critic_devices,
                full_sequence_token_ids=chunk_end_sequence_token_ids,
                prefix_length=prefix_length,
                chunk_length=chunk_length,
            )
            updated_row = dict(row)
            for critic_name in requested_critic_names:
                updated_row[_critic_end_value_key(critic_name)] = float(critic_scores[critic_name]["end_value"])
                updated_row[_critic_mean_value_key(critic_name)] = float(critic_scores[critic_name]["mean_value"])
            updated_rows.append(updated_row)
    return updated_rows, {name: str(device) for name, device in critic_devices.items()}


def build_trace_manifest_payload(
    *,
    mode: str,
    output_dir: Path,
    args: argparse.Namespace,
    actor_checkpoint_dir: str | None,
    actor_hf_dir: str | None,
    dataset_path: str | None,
    actor_tokenizer_fingerprint: dict[str, Any] | None,
    critic_names: Sequence[str],
    critic_checkpoint_dirs: dict[str, str | None],
    merged_critic_dirs: dict[str, str | None],
    critic_tokenizer_fingerprints: dict[str, dict[str, Any]],
    base_seed: int,
    source_candidate_bank: str | None = None,
    source_trace_manifest: str | None = None,
    source_scored_critic_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "actor_checkpoint_dir": actor_checkpoint_dir,
        "merged_actor_dir": actor_hf_dir,
        "dataset_path": dataset_path,
        "seed": int(base_seed),
        "prompt_key": args.prompt_key,
        "response_key": args.response_key,
        "start_index": args.start_index,
        "max_examples": args.max_examples,
        "shuffle_examples": bool(args.shuffle_examples),
        "max_prompt_length": int(args.max_prompt_length),
        "max_new_tokens": int(args.max_new_tokens),
        "chunk_size": int(args.chunk_size),
        "num_chunk_candidates": int(args.num_chunk_candidates),
        "actor_sampling_mode": args.actor_sampling_mode,
        "actor_temperature": float(args.actor_temperature),
        "actor_top_p": float(args.actor_top_p),
        "actor_top_k": int(args.actor_top_k),
        "actor_tokenizer_fingerprint": actor_tokenizer_fingerprint,
        "critic_checkpoint_dirs": critic_checkpoint_dirs,
        "merged_critic_dirs": merged_critic_dirs,
        "critic_tokenizer_fingerprints": critic_tokenizer_fingerprints,
        "scored_critic_names": list(critic_names),
        "source_scored_critic_names": None if source_scored_critic_names is None else list(source_scored_critic_names),
        "source_candidate_bank": source_candidate_bank,
        "source_trace_manifest": source_trace_manifest,
        "candidate_bank_path": str(output_dir / "chunk_benchmark_candidates.jsonl"),
        "prefix_summary_path": str(output_dir / "chunk_benchmark_prefix_summary.jsonl"),
        "summary_metrics_path": str(output_dir / "chunk_benchmark_summary_metrics.json"),
        "required_candidate_fields_for_rescoring": list(TRACE_BANK_REQUIRED_FIELDS),
        "trace_bank_notes": [
            "Rescoring new critics uses the saved chunk_end_sequence_token_ids and does not rerun the actor.",
            "Prefix summaries and metrics are rebuilt from the saved candidate bank plus the manifest seed.",
        ],
    }


def validate_trace_manifest_for_rescoring(
    *,
    trace_manifest: dict[str, Any],
    candidate_rows: Sequence[dict[str, Any]],
) -> int:
    if "seed" not in trace_manifest:
        raise ValueError(
            "Trace manifest is missing the generation seed needed to rebuild random-baseline prefix summaries."
        )
    if "actor_tokenizer_fingerprint" not in trace_manifest:
        raise ValueError(
            "Trace manifest is missing actor_tokenizer_fingerprint, so critic/tokenizer compatibility cannot be checked."
        )

    if "chunk_size" in trace_manifest:
        expected_chunk_size = int(trace_manifest["chunk_size"])
        observed_max_chunk_size = max(int(row["realized_chunk_length"]) for row in candidate_rows)
        if observed_max_chunk_size > expected_chunk_size:
            raise ValueError(
                f"Trace bank contains realized chunk length {observed_max_chunk_size}, which exceeds manifest "
                f"chunk_size={expected_chunk_size}."
            )

    if "num_chunk_candidates" in trace_manifest:
        expected_num_candidates = int(trace_manifest["num_chunk_candidates"])
        grouped_rows = _group_candidate_rows_by_prefix(candidate_rows)
        for prefix_key, rows in grouped_rows.items():
            if len(rows) != expected_num_candidates:
                raise ValueError(
                    f"Trace bank prefix {prefix_key} has {len(rows)} candidates, but the manifest says "
                    f"num_chunk_candidates={expected_num_candidates}."
                )

    return int(trace_manifest["seed"])


def assert_trace_bank_tokenizer_compatibility(
    *,
    trace_manifest: dict[str, Any],
    critic_tokenizer_fingerprints: dict[str, dict[str, Any]],
) -> None:
    actor_fingerprint = trace_manifest["actor_tokenizer_fingerprint"]["sha256"]
    mismatches = {
        critic_name: payload["sha256"]
        for critic_name, payload in critic_tokenizer_fingerprints.items()
        if payload["sha256"] != actor_fingerprint
    }
    if mismatches:
        mismatch_text = ", ".join(f"{name}={fingerprint}" for name, fingerprint in mismatches.items())
        raise ValueError(
            "One or more critics do not share the tokenizer fingerprint stored in the trace manifest. "
            f"Mismatches: {mismatch_text}"
        )


def _metric_from_prefix_rows(
    prefix_rows: Sequence[dict[str, Any]],
    *,
    method_spec: MethodSpec,
    metric_name: str,
    binary_task_scores: bool,
) -> float | None:
    if metric_name == "top1_selected_mean_task_score":
        return _mean([float(row[method_spec.selected_final_score_key]) for row in prefix_rows])

    if metric_name == "top1_chunk_selection_accuracy":
        if not binary_task_scores:
            return None
        return _mean([float(row[method_spec.selected_final_score_key]) for row in prefix_rows])

    if metric_name == "chunk_success_recovery_rate":
        if not binary_task_scores:
            return None
        successful_prefix_rows = [
            row for row in prefix_rows if float(row["oracle_best_chunk_score"]) == 1.0
        ]
        if not successful_prefix_rows:
            return None
        return _mean([float(row[method_spec.selected_final_score_key]) for row in successful_prefix_rows])

    if metric_name == "top1_hit_rate_against_oracle_best":
        return _mean(
            [1.0 if bool(row[method_spec.selected_is_oracle_best_key]) else 0.0 for row in prefix_rows]
        )

    raise ValueError(f"Unsupported metric name: {metric_name}")


def _bootstrap_mean_difference(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if len(baseline_values) != len(candidate_values):
        raise ValueError("Paired bootstrap inputs must have the same length.")
    if not baseline_values:
        return None

    baseline = np.asarray(baseline_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_differences = np.empty(bootstrap_samples, dtype=np.float64)

    for sample_index in range(bootstrap_samples):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        sample_differences[sample_index] = float(np.mean(candidate[indices] - baseline[indices]))

    observed_difference = float(np.mean(candidate - baseline))
    ci_lower, ci_upper = np.quantile(sample_differences, [0.025, 0.975]).tolist()
    return {
        "observed_difference": observed_difference,
        "bootstrap_mean_difference": float(np.mean(sample_differences)),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "num_prefixes": int(len(baseline)),
        "num_bootstrap_samples": int(bootstrap_samples),
    }


def _bootstrap_pairwise_accuracy_difference(
    prefix_rows: Sequence[dict[str, Any]],
    *,
    baseline_correct_key: str,
    candidate_correct_key: str,
    rankable_key: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not prefix_rows:
        return None

    baseline_correct = np.asarray([float(row[baseline_correct_key]) for row in prefix_rows], dtype=np.float64)
    candidate_correct = np.asarray([float(row[candidate_correct_key]) for row in prefix_rows], dtype=np.float64)
    rankable_pairs = np.asarray([int(row[rankable_key]) for row in prefix_rows], dtype=np.int64)
    total_rankable_pairs = int(rankable_pairs.sum())
    if total_rankable_pairs <= 0:
        return None

    rng = np.random.default_rng(seed)
    sample_differences = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        indices = rng.integers(0, len(prefix_rows), size=len(prefix_rows))
        sampled_rankable_pairs = int(rankable_pairs[indices].sum())
        if sampled_rankable_pairs <= 0:
            sample_differences[sample_index] = 0.0
            continue
        sampled_baseline_accuracy = float(baseline_correct[indices].sum() / sampled_rankable_pairs)
        sampled_candidate_accuracy = float(candidate_correct[indices].sum() / sampled_rankable_pairs)
        sample_differences[sample_index] = sampled_candidate_accuracy - sampled_baseline_accuracy

    observed_baseline_accuracy = float(baseline_correct.sum() / total_rankable_pairs)
    observed_candidate_accuracy = float(candidate_correct.sum() / total_rankable_pairs)
    ci_lower, ci_upper = np.quantile(sample_differences, [0.025, 0.975]).tolist()
    return {
        "observed_difference": observed_candidate_accuracy - observed_baseline_accuracy,
        "bootstrap_mean_difference": float(np.mean(sample_differences)),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "num_prefixes": int(len(prefix_rows)),
        "num_bootstrap_samples": int(bootstrap_samples),
        "total_rankable_pairs": total_rankable_pairs,
    }


def _aggregate_method_metrics(
    *,
    method_spec: MethodSpec,
    prefix_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    critic_names: Sequence[str],
    binary_task_scores: bool,
) -> dict[str, Any]:
    selected_final_scores = [float(row[method_spec.selected_final_score_key]) for row in prefix_rows]
    selected_chunk_lengths = [int(row[method_spec.selected_chunk_length_key]) for row in prefix_rows]
    selected_completed_lengths = [int(row[method_spec.selected_completed_response_length_key]) for row in prefix_rows]
    selected_contains_eos = [
        1.0 if bool(row[method_spec.selected_contains_eos_key]) else 0.0 for row in prefix_rows
    ]
    oracle_hits = [1.0 if bool(row[method_spec.selected_is_oracle_best_key]) else 0.0 for row in prefix_rows]

    weighted_pairwise_accuracy = None
    mean_prefix_pairwise_accuracy = None
    total_rankable_pairs = None
    prefixes_with_rankable_pairs = None
    if method_spec.pairwise_correct_key is not None and method_spec.pairwise_rankable_key is not None:
        pairwise_correct = [float(row[method_spec.pairwise_correct_key]) for row in prefix_rows]
        pairwise_rankable = [int(row[method_spec.pairwise_rankable_key]) for row in prefix_rows]
        total_rankable_pairs = int(sum(pairwise_rankable))
        prefixes_with_rankable_pairs = sum(1 for count in pairwise_rankable if count > 0)
        if total_rankable_pairs > 0:
            weighted_pairwise_accuracy = float(sum(pairwise_correct) / total_rankable_pairs)
        if method_spec.pairwise_accuracy_key is not None:
            mean_prefix_pairwise_accuracy = _mean(
                [
                    float(row[method_spec.pairwise_accuracy_key])
                    for row in prefix_rows
                    if row[method_spec.pairwise_accuracy_key] is not None
                ]
            )

    selector_scores: list[float] = []
    if method_spec.selector_score_field is not None:
        selector_scores = [float(row[method_spec.selector_score_field]) for row in candidate_rows]

    task_scores = [float(row["final_task_score"]) for row in candidate_rows]
    method_metrics = {
        "method": method_spec.name,
        "method_label": f"{method_spec.name}_critic" if method_spec.method_type == "critic" else method_spec.name,
        "method_type": method_spec.method_type,
        "num_prefixes": len(prefix_rows),
        "num_candidates": len(candidate_rows),
        "weighted_pairwise_ranking_accuracy": weighted_pairwise_accuracy,
        "pairwise_rankable_pairs": total_rankable_pairs,
        "prefixes_with_rankable_pairs": prefixes_with_rankable_pairs,
        "mean_prefix_pairwise_ranking_accuracy": mean_prefix_pairwise_accuracy,
        "top1_selected_mean_task_score": _mean(selected_final_scores),
        "top1_chunk_selection_accuracy": _mean(selected_final_scores) if binary_task_scores else None,
        "chunk_success_recovery_rate": _metric_from_prefix_rows(
            prefix_rows,
            method_spec=method_spec,
            metric_name="chunk_success_recovery_rate",
            binary_task_scores=binary_task_scores,
        ),
        "top1_hit_rate_against_oracle_best": _mean(oracle_hits),
        "mean_selected_chunk_length": _mean(selected_chunk_lengths),
        "mean_selected_completed_response_length": _mean(selected_completed_lengths),
        "fraction_selected_chunks_with_eos": _mean(selected_contains_eos),
        "global_pearson_score_vs_task_score": _pearson(selector_scores, task_scores) if selector_scores else None,
        "global_spearman_score_vs_task_score": _spearman(selector_scores, task_scores) if selector_scores else None,
    }

    if method_spec.selected_selector_value_key is not None:
        method_metrics["mean_selected_selector_score"] = _mean(
            [float(row[method_spec.selected_selector_value_key]) for row in prefix_rows]
        )
    else:
        method_metrics["mean_selected_selector_score"] = None

    if method_spec.method_type == "critic":
        critic_name = method_spec.name
        end_values = [float(row[_critic_end_value_key(critic_name)]) for row in candidate_rows]
        mean_values = [float(row[_critic_mean_value_key(critic_name)]) for row in candidate_rows]
        method_metrics["global_pearson_end_value_vs_task_score"] = _pearson(end_values, task_scores)
        method_metrics["global_spearman_end_value_vs_task_score"] = _spearman(end_values, task_scores)
        method_metrics["global_pearson_mean_value_vs_task_score"] = _pearson(mean_values, task_scores)
        method_metrics["global_spearman_mean_value_vs_task_score"] = _spearman(mean_values, task_scores)
        method_metrics["mean_selected_chunk_end_value"] = _mean(
            [float(row[_critic_selected_chunk_end_value_key(critic_name)]) for row in prefix_rows]
        )
        method_metrics["mean_selected_chunk_mean_value"] = _mean(
            [float(row[_critic_selected_chunk_mean_value_key(critic_name)]) for row in prefix_rows]
        )
    else:
        method_metrics["global_pearson_end_value_vs_task_score"] = None
        method_metrics["global_spearman_end_value_vs_task_score"] = None
        method_metrics["global_pearson_mean_value_vs_task_score"] = None
        method_metrics["global_spearman_mean_value_vs_task_score"] = None
        method_metrics["mean_selected_chunk_end_value"] = None
        method_metrics["mean_selected_chunk_mean_value"] = None

    return method_metrics


def _aggregate_length_diagnostics(
    *,
    candidate_rows: Sequence[dict[str, Any]],
    critic_names: Sequence[str],
) -> dict[str, Any]:
    candidate_rows_by_prefix: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        prefix_key = (int(row["example_id"]), int(row["prefix_id"]))
        candidate_rows_by_prefix.setdefault(prefix_key, []).append(row)

    by_length: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_length.setdefault(int(row["realized_chunk_length"]), []).append(row)

    diagnostics: dict[str, Any] = {}
    for chunk_length, rows in sorted(by_length.items()):
        task_scores = [float(row["final_task_score"]) for row in rows]
        payload: dict[str, Any] = {
            "num_candidates": len(rows),
            "mean_final_task_score": _mean(task_scores),
            "mean_chunk_logprob": _mean([float(row["chunk_logprob"]) for row in rows]),
            "global_pearson_chunk_logprob_vs_task_score": _pearson(
                [float(row["chunk_logprob"]) for row in rows],
                task_scores,
            ),
            "global_spearman_chunk_logprob_vs_task_score": _spearman(
                [float(row["chunk_logprob"]) for row in rows],
                task_scores,
            ),
            "ranking_by_method": {},
            "critics": {},
        }

        actor_pairwise_correct = 0.0
        actor_pairwise_rankable = 0
        actor_prefix_accuracies: list[float] = []
        for prefix_rows in candidate_rows_by_prefix.values():
            same_length_rows = [
                row for row in prefix_rows if int(row["realized_chunk_length"]) == chunk_length
            ]
            if len(same_length_rows) < 2:
                continue
            pairwise = _pairwise_ranking_stats(
                [float(row["final_task_score"]) for row in same_length_rows],
                [float(row["chunk_logprob"]) for row in same_length_rows],
            )
            actor_pairwise_correct += float(pairwise["correct_pairs"])
            actor_pairwise_rankable += int(pairwise["rankable_pairs"])
            if pairwise["accuracy"] is not None:
                actor_prefix_accuracies.append(float(pairwise["accuracy"]))
        payload["ranking_by_method"]["actor_logprob"] = {
            "weighted_pairwise_ranking_accuracy": (
                actor_pairwise_correct / actor_pairwise_rankable if actor_pairwise_rankable > 0 else None
            ),
            "pairwise_rankable_pairs": int(actor_pairwise_rankable),
            "mean_prefix_pairwise_ranking_accuracy": _mean(actor_prefix_accuracies),
            "prefixes_with_rankable_pairs": len(actor_prefix_accuracies),
        }

        for critic_name in critic_names:
            end_values = [float(row[_critic_end_value_key(critic_name)]) for row in rows]
            mean_values = [float(row[_critic_mean_value_key(critic_name)]) for row in rows]
            critic_pairwise_correct = 0.0
            critic_pairwise_rankable = 0
            critic_prefix_accuracies: list[float] = []
            for prefix_rows in candidate_rows_by_prefix.values():
                same_length_rows = [
                    row for row in prefix_rows if int(row["realized_chunk_length"]) == chunk_length
                ]
                if len(same_length_rows) < 2:
                    continue
                pairwise = _pairwise_ranking_stats(
                    [float(row["final_task_score"]) for row in same_length_rows],
                    [float(row[_critic_end_value_key(critic_name)]) for row in same_length_rows],
                )
                critic_pairwise_correct += float(pairwise["correct_pairs"])
                critic_pairwise_rankable += int(pairwise["rankable_pairs"])
                if pairwise["accuracy"] is not None:
                    critic_prefix_accuracies.append(float(pairwise["accuracy"]))

            payload["critics"][critic_name] = {
                "mean_end_value": _mean(end_values),
                "mean_mean_value": _mean(mean_values),
                "global_pearson_end_value_vs_task_score": _pearson(end_values, task_scores),
                "global_spearman_end_value_vs_task_score": _spearman(end_values, task_scores),
                "global_pearson_mean_value_vs_task_score": _pearson(mean_values, task_scores),
                "global_spearman_mean_value_vs_task_score": _spearman(mean_values, task_scores),
                "weighted_pairwise_ranking_accuracy_same_length_pairs": (
                    critic_pairwise_correct / critic_pairwise_rankable if critic_pairwise_rankable > 0 else None
                ),
                "pairwise_rankable_same_length_pairs": int(critic_pairwise_rankable),
                "mean_prefix_pairwise_ranking_accuracy_same_length_pairs": _mean(critic_prefix_accuracies),
                "prefixes_with_rankable_same_length_pairs": len(critic_prefix_accuracies),
            }
        diagnostics[str(chunk_length)] = payload
    return diagnostics


def aggregate_metrics(
    *,
    candidate_rows: Sequence[dict[str, Any]],
    prefix_rows: Sequence[dict[str, Any]],
    critic_names: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not prefix_rows:
        raise ValueError("No prefix rows were produced, so the benchmark cannot be aggregated.")

    method_specs = build_method_specs(critic_names)
    task_scores = [float(row["final_task_score"]) for row in candidate_rows]
    binary_task_scores = set(task_scores).issubset({0.0, 1.0})

    method_metrics = OrderedDict(
        (
            method_name,
            _aggregate_method_metrics(
                method_spec=method_spec,
                prefix_rows=prefix_rows,
                candidate_rows=candidate_rows,
                critic_names=critic_names,
                binary_task_scores=binary_task_scores,
            ),
        )
        for method_name, method_spec in method_specs.items()
    )

    by_prefix_bucket: dict[str, Any] = {}
    for bucket in PREFIX_BUCKETS:
        bucket_prefix_rows = [row for row in prefix_rows if row["prefix_bucket"] == bucket]
        bucket_candidate_rows = [row for row in candidate_rows if row["prefix_bucket"] == bucket]
        if not bucket_prefix_rows:
            continue
        by_prefix_bucket[bucket] = {
            method_name: _aggregate_method_metrics(
                method_spec=method_spec,
                prefix_rows=bucket_prefix_rows,
                candidate_rows=bucket_candidate_rows,
                critic_names=critic_names,
                binary_task_scores=binary_task_scores,
            )
            for method_name, method_spec in method_specs.items()
        }

    comparisons: dict[str, Any] = {}
    comparison_index = 0
    for baseline_index, baseline_name in enumerate(critic_names):
        for candidate_name in critic_names[baseline_index + 1 :]:
            comparison_key = f"{candidate_name}_minus_{baseline_name}"
            comparison_payload = {
                "weighted_pairwise_ranking_accuracy": None,
                "mean_prefix_pairwise_ranking_accuracy": None,
                "top1_selected_mean_task_score": (
                    method_metrics[candidate_name]["top1_selected_mean_task_score"]
                    - method_metrics[baseline_name]["top1_selected_mean_task_score"]
                    if method_metrics[candidate_name]["top1_selected_mean_task_score"] is not None
                    and method_metrics[baseline_name]["top1_selected_mean_task_score"] is not None
                    else None
                ),
                "top1_chunk_selection_accuracy": (
                    method_metrics[candidate_name]["top1_chunk_selection_accuracy"]
                    - method_metrics[baseline_name]["top1_chunk_selection_accuracy"]
                    if method_metrics[candidate_name]["top1_chunk_selection_accuracy"] is not None
                    and method_metrics[baseline_name]["top1_chunk_selection_accuracy"] is not None
                    else None
                ),
                "chunk_success_recovery_rate": (
                    method_metrics[candidate_name]["chunk_success_recovery_rate"]
                    - method_metrics[baseline_name]["chunk_success_recovery_rate"]
                    if method_metrics[candidate_name]["chunk_success_recovery_rate"] is not None
                    and method_metrics[baseline_name]["chunk_success_recovery_rate"] is not None
                    else None
                ),
                "paired_bootstrap": {
                    "weighted_pairwise_ranking_accuracy": _bootstrap_pairwise_accuracy_difference(
                        prefix_rows,
                        baseline_correct_key=_critic_pairwise_correct_key(baseline_name),
                        candidate_correct_key=_critic_pairwise_correct_key(candidate_name),
                        rankable_key=_critic_pairwise_rankable_key(baseline_name),
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + comparison_index * 13 + 1,
                    ),
                    "top1_selected_mean_task_score": _bootstrap_mean_difference(
                        [float(row[_critic_selected_final_score_key(baseline_name)]) for row in prefix_rows],
                        [float(row[_critic_selected_final_score_key(candidate_name)]) for row in prefix_rows],
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + comparison_index * 13 + 2,
                    ),
                    "chunk_success_recovery_rate": _bootstrap_mean_difference(
                        [
                            float(row[_critic_selected_final_score_key(baseline_name)])
                            for row in prefix_rows
                            if float(row["oracle_best_chunk_score"]) == 1.0
                        ],
                        [
                            float(row[_critic_selected_final_score_key(candidate_name)])
                            for row in prefix_rows
                            if float(row["oracle_best_chunk_score"]) == 1.0
                        ],
                        bootstrap_samples=bootstrap_samples,
                        seed=bootstrap_seed + comparison_index * 13 + 3,
                    )
                    if binary_task_scores
                    else None,
                },
            }
            comparison_payload["weighted_pairwise_ranking_accuracy"] = (
                method_metrics[candidate_name]["weighted_pairwise_ranking_accuracy"]
                - method_metrics[baseline_name]["weighted_pairwise_ranking_accuracy"]
                if method_metrics[candidate_name]["weighted_pairwise_ranking_accuracy"] is not None
                and method_metrics[baseline_name]["weighted_pairwise_ranking_accuracy"] is not None
                else None
            )
            comparison_payload["mean_prefix_pairwise_ranking_accuracy"] = (
                method_metrics[candidate_name]["mean_prefix_pairwise_ranking_accuracy"]
                - method_metrics[baseline_name]["mean_prefix_pairwise_ranking_accuracy"]
                if method_metrics[candidate_name]["mean_prefix_pairwise_ranking_accuracy"] is not None
                and method_metrics[baseline_name]["mean_prefix_pairwise_ranking_accuracy"] is not None
                else None
            )
            comparisons[comparison_key] = comparison_payload
            if binary_task_scores:
                comparison_payload["paired_bootstrap"]["top1_chunk_selection_accuracy"] = comparison_payload[
                    "paired_bootstrap"
                ]["top1_selected_mean_task_score"]
            else:
                comparison_payload["paired_bootstrap"]["top1_chunk_selection_accuracy"] = None
            comparison_index += 1

    for critic_name in critic_names:
        comparison_key = f"{critic_name}_minus_actor_logprob"
        comparisons[comparison_key] = {
            "weighted_pairwise_ranking_accuracy": (
                method_metrics[critic_name]["weighted_pairwise_ranking_accuracy"]
                - method_metrics["actor_logprob"]["weighted_pairwise_ranking_accuracy"]
                if method_metrics[critic_name]["weighted_pairwise_ranking_accuracy"] is not None
                and method_metrics["actor_logprob"]["weighted_pairwise_ranking_accuracy"] is not None
                else None
            ),
            "mean_prefix_pairwise_ranking_accuracy": (
                method_metrics[critic_name]["mean_prefix_pairwise_ranking_accuracy"]
                - method_metrics["actor_logprob"]["mean_prefix_pairwise_ranking_accuracy"]
                if method_metrics[critic_name]["mean_prefix_pairwise_ranking_accuracy"] is not None
                and method_metrics["actor_logprob"]["mean_prefix_pairwise_ranking_accuracy"] is not None
                else None
            ),
            "top1_selected_mean_task_score": (
                method_metrics[critic_name]["top1_selected_mean_task_score"]
                - method_metrics["actor_logprob"]["top1_selected_mean_task_score"]
                if method_metrics[critic_name]["top1_selected_mean_task_score"] is not None
                and method_metrics["actor_logprob"]["top1_selected_mean_task_score"] is not None
                else None
            ),
            "top1_chunk_selection_accuracy": (
                method_metrics[critic_name]["top1_chunk_selection_accuracy"]
                - method_metrics["actor_logprob"]["top1_chunk_selection_accuracy"]
                if method_metrics[critic_name]["top1_chunk_selection_accuracy"] is not None
                and method_metrics["actor_logprob"]["top1_chunk_selection_accuracy"] is not None
                else None
            ),
            "chunk_success_recovery_rate": (
                method_metrics[critic_name]["chunk_success_recovery_rate"]
                - method_metrics["actor_logprob"]["chunk_success_recovery_rate"]
                if method_metrics[critic_name]["chunk_success_recovery_rate"] is not None
                and method_metrics["actor_logprob"]["chunk_success_recovery_rate"] is not None
                else None
            ),
            "paired_bootstrap": {
                "weighted_pairwise_ranking_accuracy": _bootstrap_pairwise_accuracy_difference(
                    prefix_rows,
                    baseline_correct_key="actor_logprob_pairwise_correct_pairs",
                    candidate_correct_key=_critic_pairwise_correct_key(critic_name),
                    rankable_key="actor_logprob_pairwise_rankable_pairs",
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + comparison_index * 17 + 1,
                ),
                "top1_selected_mean_task_score": _bootstrap_mean_difference(
                    [float(row["actor_logprob_selected_final_task_score"]) for row in prefix_rows],
                    [float(row[_critic_selected_final_score_key(critic_name)]) for row in prefix_rows],
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + comparison_index * 17 + 2,
                ),
                "chunk_success_recovery_rate": _bootstrap_mean_difference(
                    [
                        float(row["actor_logprob_selected_final_task_score"])
                        for row in prefix_rows
                        if float(row["oracle_best_chunk_score"]) == 1.0
                    ],
                    [
                        float(row[_critic_selected_final_score_key(critic_name)])
                        for row in prefix_rows
                        if float(row["oracle_best_chunk_score"]) == 1.0
                    ],
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + comparison_index * 17 + 3,
                )
                if binary_task_scores
                else None,
            },
        }
        if binary_task_scores:
            comparisons[comparison_key]["paired_bootstrap"]["top1_chunk_selection_accuracy"] = comparisons[
                comparison_key
            ]["paired_bootstrap"]["top1_selected_mean_task_score"]
        else:
            comparisons[comparison_key]["paired_bootstrap"]["top1_chunk_selection_accuracy"] = None
        comparison_index += 1

    aggregate_payload = {
        "binary_task_scores": binary_task_scores,
        "num_prefixes": len(prefix_rows),
        "num_candidates": len(candidate_rows),
        "prefixes_with_rankable_pairs": sum(1 for row in prefix_rows if int(row["num_rankable_pairs"]) > 0),
        "prefixes_with_any_success": (
            sum(1 for row in prefix_rows if float(row["oracle_best_chunk_score"]) == 1.0)
            if binary_task_scores
            else None
        ),
        "prefix_bucket_counts": {
            bucket: sum(1 for row in prefix_rows if row["prefix_bucket"] == bucket) for bucket in PREFIX_BUCKETS
        },
        "duplicate_chunk_bank_rate": _mean([1.0 if bool(row["has_duplicate_chunks"]) else 0.0 for row in prefix_rows]),
        "mean_duplicate_candidate_fraction": _mean(
            [float(row["duplicate_candidate_fraction"]) for row in prefix_rows]
        ),
        "methods": method_metrics,
        "by_prefix_bucket": by_prefix_bucket,
        "by_realized_chunk_length": _aggregate_length_diagnostics(
            candidate_rows=candidate_rows,
            critic_names=critic_names,
        ),
        "comparisons": comparisons,
    }
    return aggregate_payload


def _main_results_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name, payload in metrics["methods"].items():
        row = dict(payload)
        row["comparison_key"] = None
        rows.append(row)
    return rows


def _progress_postfix(worker_progress: dict[int, dict[str, Any]]) -> str:
    parts: list[str] = []
    for worker_id in sorted(worker_progress):
        state = worker_progress[worker_id]
        done = int(state.get("done", 0))
        total = int(state.get("total", 0))
        parts.append(f"w{worker_id}:{done}/{total}")
    return " | ".join(parts)


def _worker_entry(
    *,
    assignment: WorkerAssignment,
    actor_hf_dir: str,
    critic_specs_json: list[dict[str, Any]],
    examples: list[ExampleRecord],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    chunk_size: int,
    num_chunk_candidates: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    seed: int,
    worker_root: str,
    progress_queue,
) -> None:
    worker_dir = Path(worker_root) / f"worker_{assignment.worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    summary_path = worker_dir / "worker_summary.json"
    error_path = worker_dir / "worker_error.txt"
    candidate_path = worker_dir / "chunk_benchmark_candidates.jsonl"
    prefix_path = worker_dir / "chunk_benchmark_prefix_summary.jsonl"

    try:
        start_time = time.perf_counter()
        dtype = resolve_dtype(dtype_name)
        actor_device = resolve_device(assignment.actor_device)
        critic_specs = [
            CriticSpec(
                name=str(item["name"]),
                checkpoint_dir=Path(str(item["checkpoint_dir"])),
                merged_root=Path(str(item["merged_root"])) if item["merged_root"] is not None else None,
                device=str(item["device"]) if item["device"] is not None else None,
            )
            for item in critic_specs_json
        ]
        if len(assignment.critic_devices) != len(critic_specs):
            raise ValueError(
                f"Worker {assignment.worker_id} received {len(assignment.critic_devices)} critic devices for "
                f"{len(critic_specs)} critics."
            )
        critic_devices = {
            critic.name: resolve_device(device_name)
            for critic, device_name in zip(critic_specs, assignment.critic_devices, strict=True)
        }

        tokenizer = load_tokenizer(Path(actor_hf_dir), trust_remote_code=trust_remote_code)
        actor = load_actor_model(
            Path(actor_hf_dir),
            dtype=dtype,
            device=actor_device,
            trust_remote_code=trust_remote_code,
        )
        critic_models = {
            critic.name: load_critic_model(
                critic.checkpoint_dir,
                dtype=dtype,
                device=critic_devices[critic.name],
                trust_remote_code=trust_remote_code,
            )
            for critic in critic_specs
        }

        local_examples = examples[assignment.example_start : assignment.example_end]
        total_tasks = len(local_examples)
        completed_tasks = 0
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_started",
                    "worker_id": assignment.worker_id,
                    "worker_total_tasks": total_tasks,
                }
            )

        num_candidate_rows = 0
        num_prefix_rows = 0
        with candidate_path.open("w", encoding="utf-8") as candidate_file, prefix_path.open(
            "w",
            encoding="utf-8",
        ) as prefix_file:
            for example in local_examples:
                example_candidate_rows, example_prefix_rows = process_example(
                    actor=actor,
                    critic_models=critic_models,
                    tokenizer=tokenizer,
                    example=example,
                    actor_device=actor_device,
                    critic_devices=critic_devices,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    chunk_size=chunk_size,
                    num_chunk_candidates=num_chunk_candidates,
                    eos_token_ids=eos_token_ids,
                    actor_sampling_mode=actor_sampling_mode,
                    actor_temperature=actor_temperature,
                    actor_top_p=actor_top_p,
                    actor_top_k=actor_top_k,
                    use_actor_cache=use_actor_cache,
                    base_seed=seed,
                    critic_names=[critic.name for critic in critic_specs],
                )
                for row in example_candidate_rows:
                    candidate_file.write(_json_line(row))
                for row in example_prefix_rows:
                    prefix_file.write(_json_line(row))
                num_candidate_rows += len(example_candidate_rows)
                num_prefix_rows += len(example_prefix_rows)
                completed_tasks += 1
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "type": "task_done",
                            "worker_id": assignment.worker_id,
                            "worker_completed_tasks": completed_tasks,
                            "worker_total_tasks": total_tasks,
                        }
                    )

        summary = {
            "worker_id": assignment.worker_id,
            "actor_device": str(actor_device),
            "critic_devices": {name: str(device) for name, device in critic_devices.items()},
            "example_start": assignment.example_start,
            "example_end": assignment.example_end,
            "num_examples": assignment.num_examples,
            "num_candidate_rows": int(num_candidate_rows),
            "num_prefix_rows": int(num_prefix_rows),
            "runtime_sec": time.perf_counter() - start_time,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=True, indent=2)

        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "worker_done",
                    "worker_id": assignment.worker_id,
                    "worker_completed_tasks": completed_tasks,
                    "worker_total_tasks": total_tasks,
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


def run_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_specs: Sequence[CriticSpec],
    examples: list[ExampleRecord],
    worker_layouts: list[tuple[str | None, tuple[str | None, ...]]],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    chunk_size: int,
    num_chunk_candidates: int,
    eos_token_ids: tuple[int, ...],
    actor_sampling_mode: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_top_k: int,
    use_actor_cache: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[WorkerAssignment]]:
    assignments = build_worker_assignments(num_examples=len(examples), worker_layouts=worker_layouts)
    if not assignments:
        raise ValueError("No worker assignments were created.")

    worker_root = output_dir / "_worker_tmp"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes: list[tuple[mp.Process, WorkerAssignment]] = []
    critic_specs_json = [
        {
            "name": critic.name,
            "checkpoint_dir": str(critic.checkpoint_dir),
            "merged_root": None if critic.merged_root is None else str(critic.merged_root),
            "device": critic.device,
        }
        for critic in critic_specs
    ]

    for assignment in assignments:
        process = context.Process(
            target=_worker_entry,
            kwargs={
                "assignment": assignment,
                "actor_hf_dir": str(actor_hf_dir),
                "critic_specs_json": critic_specs_json,
                "examples": examples,
                "dtype_name": dtype_name,
                "trust_remote_code": trust_remote_code,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "chunk_size": chunk_size,
                "num_chunk_candidates": num_chunk_candidates,
                "eos_token_ids": eos_token_ids,
                "actor_sampling_mode": actor_sampling_mode,
                "actor_temperature": actor_temperature,
                "actor_top_p": actor_top_p,
                "actor_top_k": actor_top_k,
                "use_actor_cache": use_actor_cache,
                "seed": seed,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
            },
            name=f"chunk_ranking_benchmark_worker_{assignment.worker_id}",
        )
        process.start()
        processes.append((process, assignment))

    total_tasks = len(examples)
    completed_tasks = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {
            "done": 0,
            "total": assignment.num_examples,
        }
        for assignment in assignments
    }

    with tqdm(total=total_tasks, desc="chunk_ranking_benchmark", unit="prompt", dynamic_ncols=True) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_tasks < total_tasks or completed_workers < len(assignments):
            try:
                event = progress_queue.get(timeout=0.2)
            except Empty:
                for process, assignment in processes:
                    if process.exitcode not in (None, 0):
                        error_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_error.txt"
                        if error_path.exists():
                            raise RuntimeError(
                                f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n"
                                f"{error_path.read_text(encoding='utf-8')}"
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
                progress_bar.update(1)
            elif event_type == "worker_done":
                completed_workers += 1
                worker_progress.setdefault(worker_id, {})
                worker_progress[worker_id]["done"] = int(event.get("worker_completed_tasks", 0))
                worker_progress[worker_id]["total"] = int(event.get("worker_total_tasks", 0))
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
                raise RuntimeError(
                    f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n"
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")

    candidate_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    output_candidate_path = output_dir / "chunk_benchmark_candidates.jsonl"
    output_prefix_path = output_dir / "chunk_benchmark_prefix_summary.jsonl"

    with output_candidate_path.open("w", encoding="utf-8") as candidate_file, output_prefix_path.open(
        "w",
        encoding="utf-8",
    ) as prefix_file:
        for assignment in assignments:
            worker_dir = worker_root / f"worker_{assignment.worker_id:03d}"
            worker_candidate_path = worker_dir / "chunk_benchmark_candidates.jsonl"
            worker_prefix_path = worker_dir / "chunk_benchmark_prefix_summary.jsonl"

            with worker_candidate_path.open("r", encoding="utf-8") as worker_candidate_file:
                for line in worker_candidate_file:
                    if not line.strip():
                        continue
                    candidate_file.write(line)
                    candidate_rows.append(json.loads(line))

            with worker_prefix_path.open("r", encoding="utf-8") as worker_prefix_file:
                for line in worker_prefix_file:
                    if not line.strip():
                        continue
                    prefix_file.write(line)
                    prefix_rows.append(json.loads(line))

    worker_summaries: list[dict[str, Any]] = []
    for assignment in assignments:
        summary_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_summary.json"
        with summary_path.open("r", encoding="utf-8") as summary_file:
            worker_summaries.append(json.load(summary_file))
    worker_summaries.sort(key=lambda item: int(item["worker_id"]))
    return candidate_rows, prefix_rows, worker_summaries, assignments


def _write_output_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    critic_names: Sequence[str],
    metrics: dict[str, Any],
    trace_bank_mode: str,
) -> None:
    lines = [
        "# Chunk-Ranking Benchmark",
        "",
        "This benchmark evaluates a critic as a local chunk evaluator, not as an online controller.",
        "",
        "## Core Objects",
        "- A `prefix state` is the prompt plus a generated prefix selected from one reference actor rollout.",
        "- A `chunk` is a contiguous token block sampled autoregressively from that prefix, up to `chunk_size` tokens or EOS.",
        "- The primary critic score is the raw end-of-chunk value, i.e. the critic value at the last token of the chunk boundary state.",
        "- Mean-of-chunk value is also logged as a secondary diagnostic, but it is not the primary selection score.",
        "",
        "## Protocol",
        "- For each prompt, the frozen actor first generates one ordinary reference rollout.",
        "- Up to three prefix states are extracted from that rollout: one each from the early, middle, and late buckets.",
        "- For each prefix, the benchmark samples one shared chunk bank from the actor and reuses that exact bank for every critic.",
        "- Each candidate chunk is completed to a final response once under the same frozen actor, and that completed rollout is reused for every critic.",
        "- Duplicate chunks are kept by default. They are logged rather than removed because they are part of the actor-induced candidate distribution.",
        "",
        "## Main Metrics",
        "- `weighted_pairwise_ranking_accuracy`: pool all rankable within-prefix chunk pairs and score the ranking correctness, with exact score ties worth 0.5.",
        "- `mean_prefix_pairwise_ranking_accuracy`: compute pairwise accuracy per prefix, then average over prefixes with at least one rankable pair.",
        "- `top1_selected_mean_task_score`: mean final task score of the chunk chosen by the selector at each prefix.",
        "- `chunk_success_recovery_rate`: on binary tasks, among prefixes whose bank contains at least one successful chunk, the fraction where the selector picks a successful chunk.",
        "- `oracle_best_chunk`: upper bound from the best candidate already present in the sampled bank.",
        "",
        "## Trace Bank",
        "- The candidate bank is saved with enough token-level state to rescore later critics without rerunning the actor.",
        "- Each candidate row now stores prompt_token_ids, prefix_token_ids, chunk_token_ids, and chunk_end_sequence_token_ids.",
        "- The trace-manifest file stores the generation seed and tokenizer fingerprint needed to rebuild prefix summaries and validate new critics.",
        "",
        "## Run Config",
        f"- Trace-bank mode: `{trace_bank_mode}`",
        f"- Dataset: `{args.dataset_path}`",
        f"- Chunk size: `{args.chunk_size}`",
        f"- Number of chunk candidates: `{args.num_chunk_candidates}`",
        f"- Sampling mode: `{args.actor_sampling_mode}`",
        f"- Temperature / top-p / top-k: `{args.actor_temperature}` / `{args.actor_top_p}` / `{args.actor_top_k}`",
        f"- Max prompt length: `{args.max_prompt_length}`",
        f"- Max new tokens: `{args.max_new_tokens}`",
        f"- Seed: `{args.seed}`",
        f"- Critics: `{list(critic_names)}`",
        "",
        "## Quick Read",
    ]

    actor_logprob_metrics = metrics["methods"]["actor_logprob"]
    lines.append(
        f"- Actor log-prob weighted pairwise accuracy: "
        f"`{_format_metric(actor_logprob_metrics['weighted_pairwise_ranking_accuracy'])}`"
    )
    for critic_name in critic_names:
        critic_metrics = metrics["methods"][critic_name]
        lines.append(
            f"- {critic_name} critic weighted pairwise accuracy: "
            f"`{_format_metric(critic_metrics['weighted_pairwise_ranking_accuracy'])}`"
        )
        lines.append(
            f"- {critic_name} critic top-1 selected mean task score: "
            f"`{_format_metric(critic_metrics['top1_selected_mean_task_score'])}`"
        )
    lines.append(
        f"- Oracle-best-in-bank mean task score: "
        f"`{_format_metric(metrics['methods']['oracle_best_chunk']['top1_selected_mean_task_score'])}`"
    )

    lines.extend(
        [
            "",
            "## Files",
            "- `chunk_benchmark_candidates.jsonl`: one row per candidate chunk with chunk tokens, actor log-prob, critic scores, and final rollout score.",
            "- `chunk_benchmark_prefix_summary.jsonl`: one row per prefix with pairwise stats and top-1 selections for every selector.",
            "- `chunk_benchmark_summary_metrics.json`: aggregate metrics, bucketed diagnostics, and paired bootstrap comparisons.",
            "- `chunk_benchmark_main_results.csv`: flat summary table for critics and baselines.",
            f"- `{TRACE_MANIFEST_FILENAME}`: reusable trace-bank manifest for later critic rescoring.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk_size must be > 0, got {args.chunk_size}")
    if args.num_chunk_candidates <= 0:
        raise ValueError(f"--num_chunk_candidates must be > 0, got {args.num_chunk_candidates}")
    if args.bootstrap_samples <= 0:
        raise ValueError(f"--bootstrap_samples must be > 0, got {args.bootstrap_samples}")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rescore_mode = args.existing_candidate_bank is not None
    if rescore_mode and args.worker_layouts:
        raise ValueError(
            "--worker_layouts is only supported when generating a fresh trace bank. Rescoring an existing bank "
            "does not rerun the actor, so use per-critic device flags instead."
        )

    critic_specs = _collect_critic_specs(args, allow_empty=rescore_mode)
    benchmark_start_time = time.perf_counter()

    actor_checkpoint_dir_str: str | None = None
    merged_actor_dir_str: str | None = None
    effective_dataset_path: str | None = None
    actor_tokenizer_fingerprint: dict[str, Any] | None = None
    eos_token_ids: list[int] | None = None
    candidate_rows: list[dict[str, Any]]
    prefix_rows: list[dict[str, Any]]
    critic_names: list[str]
    multi_worker_enabled = False
    actor_device_summary: str | None = None
    critic_devices_summary: dict[str, str] | None = None
    worker_layouts_jsonable: list[list[str | None]] = []
    runtime_notes: list[str] = []
    worker_assignments_jsonable: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    merged_critic_dirs_for_summary: dict[str, str | None] = {}
    critic_checkpoint_dirs_for_summary: dict[str, str | None] = {}
    critic_tokenizer_fingerprints_for_summary: dict[str, dict[str, Any]] = {}
    trace_source_candidate_bank: str | None = None
    trace_source_manifest_path: str | None = None
    source_scored_critic_names: list[str] | None = None
    base_seed: int
    trace_bank_mode: str
    effective_output_args_overrides: dict[str, Any] = {}

    if rescore_mode:
        existing_candidate_bank = Path(args.existing_candidate_bank).resolve()
        if not existing_candidate_bank.exists():
            raise FileNotFoundError(f"Existing candidate bank not found: {existing_candidate_bank}")
        existing_trace_manifest_path = _infer_trace_manifest_path(
            existing_candidate_bank=existing_candidate_bank,
            explicit_manifest_path=args.existing_trace_manifest,
        )
        if not existing_trace_manifest_path.exists():
            raise FileNotFoundError(
                f"Trace manifest not found: {existing_trace_manifest_path}. Rescoring requires a manifest generated "
                "alongside a reusable trace bank."
            )

        trace_manifest = _load_json(existing_trace_manifest_path)
        candidate_rows = _load_jsonl(existing_candidate_bank)
        validate_candidate_bank(candidate_rows, require_trace_fields=True)
        base_seed = validate_trace_manifest_for_rescoring(
            trace_manifest=trace_manifest,
            candidate_rows=candidate_rows,
        )

        existing_scored_critic_names = trace_manifest.get("scored_critic_names")
        if existing_scored_critic_names is None:
            existing_scored_critic_names = _infer_existing_critic_names(candidate_rows)
        existing_scored_critic_names = [str(name) for name in existing_scored_critic_names]
        merged_critic_dirs_for_summary = {
            str(name): (None if path is None else str(path))
            for name, path in dict(trace_manifest.get("merged_critic_dirs", {})).items()
        }
        critic_checkpoint_dirs_for_summary = {
            str(name): (None if path is None else str(path))
            for name, path in dict(trace_manifest.get("critic_checkpoint_dirs", {})).items()
        }
        critic_tokenizer_fingerprints_for_summary = {
            str(name): payload
            for name, payload in dict(trace_manifest.get("critic_tokenizer_fingerprints", {})).items()
        }

        merged_new_critic_dirs: dict[str, Path] = {}
        if critic_specs:
            for critic in critic_specs:
                merged_new_critic_dirs[critic.name] = ensure_merged_component_checkpoint(
                    critic.checkpoint_dir,
                    component="critic",
                    merged_root=critic.merged_root,
                    skip_merge=args.skip_merge,
                )
            critic_tokenizer_fingerprints_for_summary = {
                **critic_tokenizer_fingerprints_for_summary,
                **{name: _tokenizer_fingerprint(path) for name, path in merged_new_critic_dirs.items()},
            }
            assert_trace_bank_tokenizer_compatibility(
                trace_manifest=trace_manifest,
                critic_tokenizer_fingerprints={
                    name: critic_tokenizer_fingerprints_for_summary[name] for name in merged_new_critic_dirs
                },
            )
            runtime_critic_specs = [
                CriticSpec(
                    name=critic.name,
                    checkpoint_dir=merged_new_critic_dirs[critic.name],
                    merged_root=critic.merged_root,
                    device=critic.device or args.device,
                )
                for critic in critic_specs
            ]
            candidate_rows, critic_devices_summary = score_existing_candidate_rows_with_critics(
                candidate_rows=candidate_rows,
                critic_specs=runtime_critic_specs,
                dtype_name=args.dtype,
                trust_remote_code=args.trust_remote_code,
            )
            merged_critic_dirs_for_summary.update({name: str(path) for name, path in merged_new_critic_dirs.items()})
            critic_checkpoint_dirs_for_summary.update(
                {critic.name: str(critic.checkpoint_dir) for critic in critic_specs}
            )
        else:
            critic_devices_summary = None
            runtime_notes.append(
                "No new critics were provided in rescore mode, so the trace bank was only validated and re-aggregated."
            )

        critic_names = existing_scored_critic_names + [
            critic.name for critic in critic_specs if critic.name not in existing_scored_critic_names
        ]
        prefix_rows = rebuild_prefix_rows_from_candidate_rows(
            candidate_rows=candidate_rows,
            base_seed=base_seed,
            critic_names=critic_names,
        )
        actor_checkpoint_dir_str = trace_manifest.get("actor_checkpoint_dir")
        merged_actor_dir_str = trace_manifest.get("merged_actor_dir")
        effective_dataset_path = trace_manifest.get("dataset_path")
        actor_tokenizer_fingerprint = trace_manifest.get("actor_tokenizer_fingerprint")
        eos_token_ids = trace_manifest.get("eos_token_ids")
        trace_source_candidate_bank = str(existing_candidate_bank)
        trace_source_manifest_path = str(existing_trace_manifest_path)
        source_scored_critic_names = existing_scored_critic_names
        trace_bank_mode = "rescored_existing_trace_bank"
        runtime_notes.append("Rescored an existing trace bank without rerunning the actor.")
        effective_output_args_overrides = {
            "dataset_path": trace_manifest.get("dataset_path"),
            "seed": base_seed,
            "prompt_key": trace_manifest.get("prompt_key", args.prompt_key),
            "response_key": trace_manifest.get("response_key", args.response_key),
            "start_index": trace_manifest.get("start_index", args.start_index),
            "max_examples": trace_manifest.get("max_examples", args.max_examples),
            "shuffle_examples": trace_manifest.get("shuffle_examples", args.shuffle_examples),
            "max_prompt_length": trace_manifest.get("max_prompt_length", args.max_prompt_length),
            "max_new_tokens": trace_manifest.get("max_new_tokens", args.max_new_tokens),
            "chunk_size": trace_manifest.get("chunk_size", args.chunk_size),
            "num_chunk_candidates": trace_manifest.get("num_chunk_candidates", args.num_chunk_candidates),
            "actor_sampling_mode": trace_manifest.get("actor_sampling_mode", args.actor_sampling_mode),
            "actor_temperature": trace_manifest.get("actor_temperature", args.actor_temperature),
            "actor_top_p": trace_manifest.get("actor_top_p", args.actor_top_p),
            "actor_top_k": trace_manifest.get("actor_top_k", args.actor_top_k),
        }
    else:
        if args.actor_checkpoint_dir is None:
            raise ValueError("--actor_checkpoint_dir is required when generating a fresh trace bank.")
        if args.dataset_path is None:
            raise ValueError("--dataset_path is required when generating a fresh trace bank.")

        actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
        critic_specs = _collect_critic_specs(args, allow_empty=False)
        critic_names = [critic.name for critic in critic_specs]

        actor_hf_dir = ensure_merged_component_checkpoint(
            actor_checkpoint_dir,
            component="actor",
            merged_root=Path(args.actor_merged_root).resolve() if args.actor_merged_root else None,
            skip_merge=args.skip_merge,
        )
        merged_critic_dirs: dict[str, Path] = {}
        for critic in critic_specs:
            merged_critic_dirs[critic.name] = ensure_merged_component_checkpoint(
                critic.checkpoint_dir,
                component="critic",
                merged_root=critic.merged_root,
                skip_merge=args.skip_merge,
            )

        actor_tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
        actor_tokenizer_fingerprint = _tokenizer_fingerprint(actor_hf_dir)
        critic_tokenizer_fingerprints_for_summary = {
            name: _tokenizer_fingerprint(path) for name, path in merged_critic_dirs.items()
        }
        tokenizer_fingerprints = {
            "actor": actor_tokenizer_fingerprint,
            **{f"{name}_critic": payload for name, payload in critic_tokenizer_fingerprints_for_summary.items()},
        }
        _assert_shared_tokenizer(tokenizer_fingerprints)

        eos_token_ids = list(resolve_eos_token_ids(actor_hf_dir, actor_tokenizer))
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

        runtime_critic_specs = [
            CriticSpec(
                name=critic.name,
                checkpoint_dir=merged_critic_dirs[critic.name],
                merged_root=critic.merged_root,
                device=critic.device,
            )
            for critic in critic_specs
        ]
        worker_layouts = parse_worker_layouts(
            args.worker_layouts,
            critic_specs=runtime_critic_specs,
            actor_device=args.actor_device,
            default_device=args.device,
        )
        runtime_notes = build_runtime_notes(worker_layouts=worker_layouts, critic_names=critic_names)
        worker_assignments = build_worker_assignments(num_examples=len(examples), worker_layouts=worker_layouts)
        multi_worker_enabled = len(worker_assignments) > 1
        worker_layouts_jsonable = [[layout[0], *list(layout[1])] for layout in worker_layouts]
        worker_assignments_jsonable = worker_assignments_to_jsonable(worker_assignments)

        if multi_worker_enabled:
            candidate_rows, prefix_rows, worker_summaries, worker_assignments = run_multi_worker(
                output_dir=output_dir,
                actor_hf_dir=actor_hf_dir,
                critic_specs=runtime_critic_specs,
                examples=examples,
                worker_layouts=worker_layouts,
                dtype_name=args.dtype,
                trust_remote_code=args.trust_remote_code,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                chunk_size=args.chunk_size,
                num_chunk_candidates=args.num_chunk_candidates,
                eos_token_ids=tuple(eos_token_ids),
                actor_sampling_mode=args.actor_sampling_mode,
                actor_temperature=args.actor_temperature,
                actor_top_p=args.actor_top_p,
                actor_top_k=args.actor_top_k,
                use_actor_cache=not args.disable_actor_cache,
                seed=args.seed,
            )
            worker_assignments_jsonable = worker_assignments_to_jsonable(worker_assignments)
        else:
            dtype = resolve_dtype(args.dtype)
            actor_device = resolve_device(worker_layouts[0][0])
            critic_devices = {
                critic.name: resolve_device(device_name)
                for critic, device_name in zip(runtime_critic_specs, worker_layouts[0][1], strict=True)
            }
            actor = load_actor_model(
                actor_hf_dir,
                dtype=dtype,
                device=actor_device,
                trust_remote_code=args.trust_remote_code,
            )
            critic_models = {
                critic.name: load_critic_model(
                    merged_critic_dirs[critic.name],
                    dtype=dtype,
                    device=critic_devices[critic.name],
                    trust_remote_code=args.trust_remote_code,
                )
                for critic in runtime_critic_specs
            }

            candidate_rows = []
            prefix_rows = []
            with tqdm(examples, desc="chunk_ranking_benchmark", unit="prompt", dynamic_ncols=True) as progress_bar:
                for example in progress_bar:
                    progress_bar.set_postfix_str(f"example_id={example.example_id}")
                    example_candidate_rows, example_prefix_rows = process_example(
                        actor=actor,
                        critic_models=critic_models,
                        tokenizer=actor_tokenizer,
                        example=example,
                        actor_device=actor_device,
                        critic_devices=critic_devices,
                        max_prompt_length=args.max_prompt_length,
                        max_new_tokens=args.max_new_tokens,
                        chunk_size=args.chunk_size,
                        num_chunk_candidates=args.num_chunk_candidates,
                        eos_token_ids=tuple(eos_token_ids),
                        actor_sampling_mode=args.actor_sampling_mode,
                        actor_temperature=args.actor_temperature,
                        actor_top_p=args.actor_top_p,
                        actor_top_k=args.actor_top_k,
                        use_actor_cache=not args.disable_actor_cache,
                        base_seed=args.seed,
                        critic_names=critic_names,
                    )
                    candidate_rows.extend(example_candidate_rows)
                    prefix_rows.extend(example_prefix_rows)

            actor_device_summary = str(actor_device)
            critic_devices_summary = {name: str(device) for name, device in critic_devices.items()}
            worker_summaries = [
                {
                    "worker_id": 0,
                    "actor_device": str(actor_device),
                    "critic_devices": {name: str(device) for name, device in critic_devices.items()},
                    "example_start": 0,
                    "example_end": len(examples),
                    "num_examples": len(examples),
                    "num_candidate_rows": len(candidate_rows),
                    "num_prefix_rows": len(prefix_rows),
                    "runtime_sec": time.perf_counter() - benchmark_start_time,
                }
            ]

        validate_candidate_bank(candidate_rows, require_trace_fields=True)
        actor_checkpoint_dir_str = str(actor_checkpoint_dir)
        merged_actor_dir_str = str(actor_hf_dir)
        effective_dataset_path = str(Path(args.dataset_path).resolve())
        merged_critic_dirs_for_summary = {name: str(path) for name, path in merged_critic_dirs.items()}
        critic_checkpoint_dirs_for_summary = {
            critic.name: str(critic.checkpoint_dir) for critic in critic_specs
        }
        if multi_worker_enabled:
            actor_device_summary = None
            critic_devices_summary = None
        trace_bank_mode = "generated_fresh_trace_bank"
        base_seed = int(args.seed)
        effective_output_args_overrides = {
            "dataset_path": effective_dataset_path,
            "seed": base_seed,
        }

    benchmark_wall_time_sec = time.perf_counter() - benchmark_start_time
    if not prefix_rows:
        raise ValueError(
            "The benchmark produced zero prefix states. This usually means no response tokens were available in the "
            "reference rollouts."
        )

    candidate_path = output_dir / "chunk_benchmark_candidates.jsonl"
    prefix_path = output_dir / "chunk_benchmark_prefix_summary.jsonl"
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(prefix_path, prefix_rows)

    summary_metrics = aggregate_metrics(
        candidate_rows=candidate_rows,
        prefix_rows=prefix_rows,
        critic_names=critic_names,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=base_seed + 9_973,
    )
    main_results_rows = _main_results_rows(summary_metrics)

    csv_path = output_dir / "chunk_benchmark_main_results.csv"
    fieldnames = [
        "method",
        "method_label",
        "method_type",
        "num_prefixes",
        "num_candidates",
        "weighted_pairwise_ranking_accuracy",
        "pairwise_rankable_pairs",
        "prefixes_with_rankable_pairs",
        "mean_prefix_pairwise_ranking_accuracy",
        "top1_selected_mean_task_score",
        "top1_chunk_selection_accuracy",
        "chunk_success_recovery_rate",
        "top1_hit_rate_against_oracle_best",
        "mean_selected_chunk_length",
        "mean_selected_completed_response_length",
        "fraction_selected_chunks_with_eos",
        "mean_selected_selector_score",
        "global_pearson_score_vs_task_score",
        "global_spearman_score_vs_task_score",
        "global_pearson_end_value_vs_task_score",
        "global_spearman_end_value_vs_task_score",
        "global_pearson_mean_value_vs_task_score",
        "global_spearman_mean_value_vs_task_score",
        "mean_selected_chunk_end_value",
        "mean_selected_chunk_mean_value",
        "comparison_key",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in main_results_rows:
            writer.writerow(row)

    effective_args = argparse.Namespace(**{**vars(args), **effective_output_args_overrides})

    trace_manifest_payload = build_trace_manifest_payload(
        mode=trace_bank_mode,
        output_dir=output_dir,
        args=effective_args,
        actor_checkpoint_dir=actor_checkpoint_dir_str,
        actor_hf_dir=merged_actor_dir_str,
        dataset_path=effective_dataset_path,
        actor_tokenizer_fingerprint=actor_tokenizer_fingerprint,
        critic_names=critic_names,
        critic_checkpoint_dirs=critic_checkpoint_dirs_for_summary,
        merged_critic_dirs=merged_critic_dirs_for_summary,
        critic_tokenizer_fingerprints=critic_tokenizer_fingerprints_for_summary,
        base_seed=base_seed,
        source_candidate_bank=trace_source_candidate_bank,
        source_trace_manifest=trace_source_manifest_path,
        source_scored_critic_names=source_scored_critic_names,
    )
    trace_manifest_payload["eos_token_ids"] = eos_token_ids
    trace_manifest_path = output_dir / TRACE_MANIFEST_FILENAME
    with trace_manifest_path.open("w", encoding="utf-8") as trace_manifest_file:
        json.dump(trace_manifest_payload, trace_manifest_file, ensure_ascii=True, indent=2)

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "trace_bank_mode": trace_bank_mode,
        "actor_checkpoint_dir": actor_checkpoint_dir_str,
        "critic_checkpoint_dirs": critic_checkpoint_dirs_for_summary,
        "merged_actor_dir": merged_actor_dir_str,
        "merged_critic_dirs": merged_critic_dirs_for_summary,
        "dataset_path": effective_dataset_path,
        "output_dir": str(output_dir),
        "source_candidate_bank": trace_source_candidate_bank,
        "source_trace_manifest": trace_source_manifest_path,
        "multi_worker_enabled": multi_worker_enabled,
        "actor_device": actor_device_summary,
        "critic_devices": critic_devices_summary,
        "worker_layouts": worker_layouts_jsonable,
        "runtime_notes": runtime_notes,
        "worker_assignments": worker_assignments_jsonable,
        "worker_summaries": worker_summaries,
        "dtype": args.dtype,
        "eos_token_ids": eos_token_ids,
        "tokenizer_fingerprints": {
            "actor": actor_tokenizer_fingerprint,
            "critics": critic_tokenizer_fingerprints_for_summary,
        },
        "run_args": vars(args),
        "critic_names": critic_names,
        "benchmark_wall_time_sec": benchmark_wall_time_sec,
        "trace_manifest_path": str(trace_manifest_path),
        "metrics": summary_metrics,
    }
    summary_path = output_dir / "chunk_benchmark_summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    _write_output_readme(
        output_dir=output_dir,
        args=effective_args,
        critic_names=critic_names,
        metrics=summary_metrics,
        trace_bank_mode=trace_bank_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
