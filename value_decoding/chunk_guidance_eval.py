from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from queue import Empty
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import ray
except ImportError:
    ray = None

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
from value_decoding.multi_worker import (
    RayNodeInfo,
    WorkerAssignment,
    build_distributed_worker_assignments,
    build_worker_assignments,
    parse_worker_pairs,
    worker_assignments_to_jsonable,
)


DEFAULT_CHUNK_SIZES = (2, 4)
DEFAULT_NUM_CHUNK_CANDIDATES_VALUES = (2,)
DEFAULT_BETAS = (0.0, 0.05, 0.1, 0.25)
STANDARD_TAIL_SUMMARY_LENGTHS = (2, 4, 8, 16)
DEFAULT_COMPARISON_BOOTSTRAP_SAMPLES = 1_000
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
class ChunkRunSpec:
    config_id: str
    method_name: str
    score_mode: str
    chunk_size: int | None = None
    num_chunk_candidates: int | None = None
    beta: float | None = None
    value_reducer: str | None = None
    comparison_value_reducer: str | None = None
    actor_sampling_mode: str = ActorSamplingMode.SAMPLE.value
    actor_temperature: float = 1.0
    actor_top_p: float = 1.0
    actor_top_k: int = 0

    @property
    def is_chunk_method(self) -> bool:
        return self.chunk_size is not None and self.num_chunk_candidates is not None


@dataclass(frozen=True)
class ChunkCandidate:
    candidate_index: int
    chunk_token_ids: tuple[int, ...]
    chunk_text: str
    chunk_length: int
    chunk_logprob: float
    chunk_values: tuple[float, ...]
    end_value: float | None
    mean_value: float | None
    contains_eos: bool
    token_logprobs: tuple[float, ...] = ()
    token_entropies: tuple[float, ...] = ()
    chunk_uncertainty: float = 0.0


@dataclass
class ChunkDecodeArtifacts:
    example_result: dict[str, Any]
    chunk_decision_results: list[dict[str, Any]]


@dataclass(frozen=True)
class ParsedValueReducer:
    raw_name: str
    canonical_name: str
    kind: str
    method_suffix: str
    tail_length: int | None = None
    alpha: float | None = None

    @property
    def alpha_id(self) -> str | None:
        if self.alpha is None:
            return None
        return _format_float_for_id(self.alpha)

    @property
    def selected_metric_key(self) -> str:
        if self.kind == "end":
            return "selected_chunk_end_value"
        if self.kind == "chunk_mean":
            return "selected_chunk_mean_value"
        if self.kind == "tail_mean":
            if self.tail_length is None:
                raise ValueError("tail_mean reducer is missing tail_length.")
            return f"selected_chunk_tail_mean_h{self.tail_length}"
        if self.kind == "tail_exp":
            if self.tail_length is None or self.alpha_id is None:
                raise ValueError("tail_exp reducer is missing tail_length or alpha.")
            return f"selected_chunk_exp_tail_value__h{self.tail_length}__a{self.alpha_id}"
        raise ValueError(f"Unsupported reducer kind: {self.kind}")

    @property
    def aggregate_metric_key(self) -> str:
        return "mean_" + self.selected_metric_key

    @property
    def winner_metric_fragment(self) -> str:
        if self.kind == "end":
            return "endvalue"
        if self.kind == "chunk_mean":
            return "meanvalue"
        return self.canonical_name

    @property
    def winner_candidate_values_key(self) -> str:
        return f"candidate_chunk_{self.winner_metric_fragment}_values"

    @property
    def winner_index_key(self) -> str:
        return f"{self.winner_metric_fragment}_chunk_winner_index"

    @property
    def winner_tied_indices_key(self) -> str:
        return f"{self.winner_metric_fragment}_chunk_winner_tied_indices"

    @property
    def winner_value_key(self) -> str:
        return f"{self.winner_metric_fragment}_chunk_winner_value"

    @property
    def selected_differs_from_winner_key(self) -> str:
        return f"selected_differs_from_{self.winner_metric_fragment}_winner"

    @property
    def fraction_diff_from_winner_key(self) -> str:
        return f"fraction_chunk_decisions_different_from_{self.winner_metric_fragment}_winner"

    @property
    def mean_fraction_diff_from_winner_key(self) -> str:
        return "mean_" + self.fraction_diff_from_winner_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk-level guidance evaluation with a frozen actor and the new critic. "
            "Supports ordinary actor sampling, chunk actor-only reranking, critic end-value reranking, "
            "and actor-uncertainty reranking."
        )
    )
    parser.add_argument("--actor_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the frozen actor.")
    parser.add_argument("--critic_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the new critic.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for experiment artifacts.")
    parser.add_argument("--actor_merged_root", type=str, default=None, help="Optional merged HF root for actor.")
    parser.add_argument("--critic_merged_root", type=str, default=None, help="Optional merged HF root for critic.")
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
        "--critic_hf_source_dir",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face config/tokenizer directory for the critic merge. "
            "Use this when the raw critic checkpoint was copied without critic/huggingface."
        ),
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None, help="Optional response/ground-truth column key.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Fallback device for both actor and critic.")
    parser.add_argument("--actor_device", type=str, default=None, help="Optional actor device override.")
    parser.add_argument("--critic_device", type=str, default=None, help="Optional critic device override.")
    parser.add_argument(
        "--worker_pairs",
        nargs="+",
        default=None,
        help=(
            "Optional prompt-sharded worker layouts. Each entry should be 'actor_device,critic_device' "
            "or a single device to reuse for both."
        ),
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help=(
            "Optional Ray cluster address for cross-node execution. When set, --worker_pairs is treated as the "
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
    parser.add_argument(
        "--disable_critic_model",
        action="store_true",
        help=(
            "Skip merging, loading, and scoring the critic. Only valid when every requested config is actor-only "
            "or uncertainty-only and no critic-based comparison reducer is requested."
        ),
    )
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
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=list(DEFAULT_CHUNK_SIZES))
    parser.add_argument(
        "--num_chunk_candidates_values",
        nargs="+",
        type=int,
        default=list(DEFAULT_NUM_CHUNK_CANDIDATES_VALUES),
    )
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument(
        "--value_reducers",
        nargs="+",
        default=["end"],
        help=(
            "Value reducers for actor+critic chunk guidance. Supported forms: "
            "'end', 'mean' (legacy whole-chunk mean), 'tail_mean_h4', 'tail_exp_h8_a0p85'."
        ),
    )
    parser.add_argument(
        "--comparison_value_reducer",
        type=str,
        default=None,
        help=(
            "Optional explicit shared-bank comparison reducer override used for diagnostics. "
            "Supported forms match --value_reducers. If unset, the runner auto-resolves a same-h comparison "
            "for tail-based reducers."
        ),
    )
    parser.add_argument(
        "--comparison_tail_h",
        type=int,
        default=None,
        help=(
            "Optional tail length for auto-resolved comparison reducers. If unset, a tail-based method reuses "
            "its own h for same-h comparison; non-tail methods skip the auto comparison unless this is set."
        ),
    )
    parser.add_argument(
        "--comparison_tail_exp_alpha",
        type=float,
        default=None,
        help=(
            "Optional alpha for auto-resolved tail-exp comparison reducers. If unset, the auto comparison uses "
            "tail_mean_h. If set, it uses tail_exp_h_aalpha with the chosen h."
        ),
    )
    parser.add_argument("--include_critic_only", action="store_true", help="Add optional critic-only chunk rerank configs.")
    parser.add_argument(
        "--include_uncertainty_only",
        action="store_true",
        help=(
            "Add chunk rerank configs that use the shared actor chunk bank and select the candidate with the "
            "minimum mean actor entropy."
        ),
    )
    parser.add_argument(
        "--skip_actor_only_baselines",
        action="store_true",
        help=(
            "Skip the ordinary actor-only sampling baseline and the chunk actor-only rerank baseline, "
            "while still allowing actor+critic and critic-only configs."
        ),
    )
    parser.add_argument(
        "--only_critic_only",
        action="store_true",
        help=(
            "Run only the critic-only chunk rerank configs for the requested chunk sizes / candidate counts / reducers. "
            "This skips actor-only and actor+critic configs."
        ),
    )
    parser.add_argument("--normalization_eps", type=float, default=1e-6)
    parser.add_argument("--debug_full_chunk_candidates", action="store_true")
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


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _format_float_for_id(value: float) -> str:
    formatted = f"{value:g}"
    return formatted.replace("-", "m").replace(".", "p")


def _parse_alpha_token(token: str) -> float:
    normalized = token.strip().replace("p", ".")
    try:
        parsed = float(normalized)
    except ValueError as exc:
        raise ValueError(f"Unable to parse reducer alpha from token '{token}'.") from exc
    if not (0.0 < parsed < 1.0):
        raise ValueError(f"Reducer alpha must be strictly between 0 and 1, got {parsed}.")
    return parsed


def parse_optional_value_reducer(value_reducer: str | None) -> ParsedValueReducer | None:
    if value_reducer is None:
        return None
    normalized = str(value_reducer).strip()
    if normalized == "" or normalized.lower() == "none":
        return None
    return parse_value_reducer(normalized)


@lru_cache(maxsize=None)
def parse_value_reducer(value_reducer: str) -> ParsedValueReducer:
    reducer = str(value_reducer).strip()
    if reducer == "end":
        return ParsedValueReducer(
            raw_name=reducer,
            canonical_name="end",
            kind="end",
            method_suffix="endvalue",
        )
    if reducer in {"mean", "chunk_mean", "full_mean"}:
        return ParsedValueReducer(
            raw_name=reducer,
            canonical_name="mean",
            kind="chunk_mean",
            method_suffix="meanvalue",
        )

    tail_mean_match = re.fullmatch(r"(?:tail_mean|tailmean)_h(\d+)", reducer)
    if tail_mean_match is not None:
        tail_length = int(tail_mean_match.group(1))
        if tail_length <= 0:
            raise ValueError(f"Reducer tail length must be > 0, got {tail_length}.")
        return ParsedValueReducer(
            raw_name=reducer,
            canonical_name=f"tail_mean_h{tail_length}",
            kind="tail_mean",
            method_suffix=f"tailmean_h{tail_length}",
            tail_length=tail_length,
        )

    tail_exp_match = re.fullmatch(r"(?:tail_exp|tailexp)_h(\d+)_(?:a|alpha)([0-9p.]+)", reducer)
    if tail_exp_match is not None:
        tail_length = int(tail_exp_match.group(1))
        if tail_length <= 0:
            raise ValueError(f"Reducer tail length must be > 0, got {tail_length}.")
        alpha = _parse_alpha_token(tail_exp_match.group(2))
        alpha_id = _format_float_for_id(alpha)
        return ParsedValueReducer(
            raw_name=reducer,
            canonical_name=f"tail_exp_h{tail_length}_a{alpha_id}",
            kind="tail_exp",
            method_suffix=f"tailexp_h{tail_length}_a{alpha_id}",
            tail_length=tail_length,
            alpha=alpha,
        )

    raise ValueError(
        "Unsupported value reducer "
        f"'{value_reducer}'. Supported forms: 'end', 'mean', 'tail_mean_h4', 'tail_exp_h8_a0p85'."
    )


def reduce_end(chunk_values: Sequence[float]) -> float:
    if len(chunk_values) == 0:
        raise ValueError("chunk_values must contain at least one value.")
    return float(chunk_values[-1])


def reduce_mean(chunk_values: Sequence[float]) -> float:
    if len(chunk_values) == 0:
        raise ValueError("chunk_values must contain at least one value.")
    values = np.asarray(chunk_values, dtype=np.float64)
    return float(values.mean())


def reduce_tail_mean(chunk_values: Sequence[float], tail_length: int) -> float:
    if tail_length <= 0:
        raise ValueError(f"tail_length must be > 0, got {tail_length}.")
    if len(chunk_values) == 0:
        raise ValueError("chunk_values must contain at least one value.")
    effective_tail_length = min(int(tail_length), len(chunk_values))
    values = np.asarray(chunk_values[-effective_tail_length:], dtype=np.float64)
    return float(values.mean())


def reduce_tail_exp(chunk_values: Sequence[float], tail_length: int, alpha: float) -> float:
    if tail_length <= 0:
        raise ValueError(f"tail_length must be > 0, got {tail_length}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")
    if len(chunk_values) == 0:
        raise ValueError("chunk_values must contain at least one value.")

    effective_tail_length = min(int(tail_length), len(chunk_values))
    tail_values = np.asarray(chunk_values[-effective_tail_length:], dtype=np.float64)
    raw_weights = np.asarray(
        [alpha ** (effective_tail_length - 1 - index) for index in range(effective_tail_length)],
        dtype=np.float64,
    )
    normalized_weights = raw_weights / raw_weights.sum()
    return float(np.dot(tail_values, normalized_weights))


def reduce_chunk_values(chunk_values: Sequence[float], reducer: ParsedValueReducer | str) -> float:
    reducer_spec = parse_value_reducer(reducer) if isinstance(reducer, str) else reducer
    if reducer_spec.kind == "end":
        return reduce_end(chunk_values)
    if reducer_spec.kind == "chunk_mean":
        return reduce_mean(chunk_values)
    if reducer_spec.kind == "tail_mean":
        if reducer_spec.tail_length is None:
            raise ValueError("tail_mean reducer is missing tail_length.")
        return reduce_tail_mean(chunk_values, reducer_spec.tail_length)
    if reducer_spec.kind == "tail_exp":
        if reducer_spec.tail_length is None or reducer_spec.alpha is None:
            raise ValueError("tail_exp reducer is missing tail_length or alpha.")
        return reduce_tail_exp(chunk_values, reducer_spec.tail_length, reducer_spec.alpha)
    raise ValueError(f"Unsupported reducer kind: {reducer_spec.kind}")


def _standard_tail_mean_summary(chunk_values: Sequence[float]) -> dict[int, float]:
    return {
        tail_length: reduce_tail_mean(chunk_values, tail_length)
        for tail_length in STANDARD_TAIL_SUMMARY_LENGTHS
    }


def _empty_tail_mean_summary() -> dict[int, None]:
    return {tail_length: None for tail_length in STANDARD_TAIL_SUMMARY_LENGTHS}


def _spec_requires_critic(spec: ChunkRunSpec) -> bool:
    return bool(
        spec.score_mode in {"actor_plus_critic", "critic_only"}
        or spec.value_reducer is not None
        or spec.comparison_value_reducer is not None
    )


def resolve_comparison_value_reducer(
    *,
    value_reducer: str | None,
    explicit_comparison_value_reducer: str | None,
    comparison_tail_h: int | None,
    comparison_tail_exp_alpha: float | None,
) -> str | None:
    explicit_reducer_spec = parse_optional_value_reducer(explicit_comparison_value_reducer)
    if explicit_reducer_spec is not None:
        return explicit_reducer_spec.canonical_name

    current_reducer_spec = parse_optional_value_reducer(value_reducer)
    reference_tail_h = comparison_tail_h
    if reference_tail_h is None and current_reducer_spec is not None:
        reference_tail_h = current_reducer_spec.tail_length

    if reference_tail_h is None:
        return None
    if reference_tail_h <= 0:
        raise ValueError(f"comparison_tail_h must be > 0, got {reference_tail_h}.")

    if comparison_tail_exp_alpha is None:
        reference_reducer_spec = parse_value_reducer(f"tail_mean_h{reference_tail_h}")
    else:
        if not (0.0 < comparison_tail_exp_alpha < 1.0):
            raise ValueError(
                "comparison_tail_exp_alpha must be strictly between 0 and 1, "
                f"got {comparison_tail_exp_alpha}."
            )
        alpha_id = _format_float_for_id(comparison_tail_exp_alpha)
        reference_reducer_spec = parse_value_reducer(f"tail_exp_h{reference_tail_h}_a{alpha_id}")
    return reference_reducer_spec.canonical_name


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


def build_run_specs(args: argparse.Namespace) -> list[ChunkRunSpec]:
    specs: list[ChunkRunSpec] = []
    seen_config_ids: set[str] = set()

    include_only_critic_only = bool(args.only_critic_only)
    skip_actor_only_baselines = bool(args.skip_actor_only_baselines or args.only_critic_only)
    include_critic_only = bool(args.include_critic_only or args.only_critic_only)
    include_uncertainty_only = bool(args.include_uncertainty_only and not args.only_critic_only)

    if not skip_actor_only_baselines:
        actor_only_spec = ChunkRunSpec(
            config_id="",
            method_name="actor_only_sample",
            score_mode="actor_only_sample",
            comparison_value_reducer=resolve_comparison_value_reducer(
                value_reducer=None,
                explicit_comparison_value_reducer=args.comparison_value_reducer,
                comparison_tail_h=args.comparison_tail_h,
                comparison_tail_exp_alpha=args.comparison_tail_exp_alpha,
            ),
            actor_sampling_mode=args.actor_sampling_mode,
            actor_temperature=args.actor_temperature,
            actor_top_p=args.actor_top_p,
            actor_top_k=args.actor_top_k,
        )
        actor_only_parts = [actor_only_spec.method_name, actor_only_spec.actor_sampling_mode]
        if actor_only_spec.actor_sampling_mode == ActorSamplingMode.SAMPLE.value:
            actor_only_parts.extend(
                [
                    f"temp{_format_float_for_id(actor_only_spec.actor_temperature)}",
                    f"top_p{_format_float_for_id(actor_only_spec.actor_top_p)}",
                    f"top_k{actor_only_spec.actor_top_k}",
                ]
            )
        actor_only_spec = ChunkRunSpec(**{**asdict(actor_only_spec), "config_id": "__".join(actor_only_parts)})
        specs.append(actor_only_spec)
        seen_config_ids.add(actor_only_spec.config_id)

    for chunk_size in args.chunk_sizes:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        for num_chunk_candidates in args.num_chunk_candidates_values:
            if num_chunk_candidates <= 0:
                raise ValueError(f"num_chunk_candidates must be > 0, got {num_chunk_candidates}")

            if not skip_actor_only_baselines:
                actor_only_chunk = ChunkRunSpec(
                    config_id=f"chunk_rerank_actor_only__m{chunk_size}__k{num_chunk_candidates}",
                    method_name="chunk_rerank_actor_only",
                    score_mode="actor_logprob_only",
                    chunk_size=chunk_size,
                    num_chunk_candidates=num_chunk_candidates,
                    comparison_value_reducer=resolve_comparison_value_reducer(
                        value_reducer=None,
                        explicit_comparison_value_reducer=args.comparison_value_reducer,
                        comparison_tail_h=args.comparison_tail_h,
                        comparison_tail_exp_alpha=args.comparison_tail_exp_alpha,
                    ),
                    actor_sampling_mode=args.actor_sampling_mode,
                    actor_temperature=args.actor_temperature,
                    actor_top_p=args.actor_top_p,
                    actor_top_k=args.actor_top_k,
                )
                if actor_only_chunk.config_id not in seen_config_ids:
                    specs.append(actor_only_chunk)
                    seen_config_ids.add(actor_only_chunk.config_id)

            if include_uncertainty_only:
                uncertainty_only_chunk = ChunkRunSpec(
                    config_id=f"chunk_rerank_uncertainty_meanentropy__m{chunk_size}__k{num_chunk_candidates}",
                    method_name="chunk_rerank_uncertainty_meanentropy",
                    score_mode="uncertainty_meanentropy",
                    chunk_size=chunk_size,
                    num_chunk_candidates=num_chunk_candidates,
                    comparison_value_reducer=None,
                    actor_sampling_mode=args.actor_sampling_mode,
                    actor_temperature=args.actor_temperature,
                    actor_top_p=args.actor_top_p,
                    actor_top_k=args.actor_top_k,
                )
                if uncertainty_only_chunk.config_id not in seen_config_ids:
                    specs.append(uncertainty_only_chunk)
                    seen_config_ids.add(uncertainty_only_chunk.config_id)

            for value_reducer in args.value_reducers:
                reducer_spec = parse_value_reducer(value_reducer)
                if not include_only_critic_only:
                    for beta in args.betas:
                        if beta <= 0.0:
                            continue
                        method_name = f"chunk_rerank_newcritic_{reducer_spec.method_suffix}"
                        spec = ChunkRunSpec(
                            config_id=(
                                f"{method_name}__m{chunk_size}__k{num_chunk_candidates}"
                                f"__beta{_format_float_for_id(beta)}"
                            ),
                            method_name=method_name,
                            score_mode="actor_plus_critic",
                            chunk_size=chunk_size,
                            num_chunk_candidates=num_chunk_candidates,
                            beta=float(beta),
                            value_reducer=reducer_spec.canonical_name,
                            comparison_value_reducer=resolve_comparison_value_reducer(
                                value_reducer=reducer_spec.canonical_name,
                                explicit_comparison_value_reducer=args.comparison_value_reducer,
                                comparison_tail_h=args.comparison_tail_h,
                                comparison_tail_exp_alpha=args.comparison_tail_exp_alpha,
                            ),
                            actor_sampling_mode=args.actor_sampling_mode,
                            actor_temperature=args.actor_temperature,
                            actor_top_p=args.actor_top_p,
                            actor_top_k=args.actor_top_k,
                        )
                        if spec.config_id not in seen_config_ids:
                            specs.append(spec)
                            seen_config_ids.add(spec.config_id)

                if include_critic_only:
                    method_name = f"chunk_rerank_critic_only_{reducer_spec.method_suffix}"
                    spec = ChunkRunSpec(
                        config_id=f"{method_name}__m{chunk_size}__k{num_chunk_candidates}",
                        method_name=method_name,
                        score_mode="critic_only",
                        chunk_size=chunk_size,
                        num_chunk_candidates=num_chunk_candidates,
                        beta=None,
                        value_reducer=reducer_spec.canonical_name,
                        comparison_value_reducer=resolve_comparison_value_reducer(
                            value_reducer=reducer_spec.canonical_name,
                            explicit_comparison_value_reducer=args.comparison_value_reducer,
                            comparison_tail_h=args.comparison_tail_h,
                            comparison_tail_exp_alpha=args.comparison_tail_exp_alpha,
                        ),
                        actor_sampling_mode=args.actor_sampling_mode,
                        actor_temperature=args.actor_temperature,
                        actor_top_p=args.actor_top_p,
                        actor_top_k=args.actor_top_k,
                    )
                    if spec.config_id not in seen_config_ids:
                        specs.append(spec)
                        seen_config_ids.add(spec.config_id)

    if not specs:
        raise ValueError("No run specifications were generated. Check the method-selection flags and grid settings.")
    return specs


def _ordinary_actor_seed(base_seed: int, *, example_id: int) -> int:
    return int(base_seed + (example_id + 1) * 1_000_003 + 11)


def _chunk_candidate_seed(
    base_seed: int,
    *,
    example_id: int,
    chunk_size: int,
    num_chunk_candidates: int,
    chunk_decision_index: int,
    candidate_index: int,
) -> int:
    return int(
        base_seed
        + (example_id + 1) * 1_000_003
        + chunk_size * 10_007
        + num_chunk_candidates * 100_003
        + chunk_decision_index * 1_000_000_007
        + candidate_index * 97_003
    )


def _zscore(values: Sequence[float], *, eps: float) -> list[float]:
    values_tensor = torch.tensor(values, dtype=torch.float32)
    mean = values_tensor.mean()
    std = values_tensor.std(unbiased=False)
    normalized = (values_tensor - mean) / (std + eps)
    return [float(value) for value in normalized.tolist()]


def _entropy_from_logits(logits: torch.Tensor) -> float:
    logits_fp32 = logits.float()
    log_probs = torch.log_softmax(logits_fp32, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.item())


def _select_argmax(values: Sequence[float]) -> tuple[int, list[int], float]:
    best_value = max(values)
    tied_indices = [index for index, value in enumerate(values) if value == best_value]
    selected_index = tied_indices[0]
    return selected_index, tied_indices, float(best_value)


def sample_actor_chunk(
    *,
    actor,
    critic,
    tokenizer,
    prefix_ids: torch.Tensor,
    actor_device: torch.device,
    critic_device: torch.device | None,
    max_chunk_len: int,
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
    candidate_index: int,
) -> ChunkCandidate:
    if max_chunk_len <= 0:
        raise ValueError(f"max_chunk_len must be > 0, got {max_chunk_len}")

    set_decode_seed(seed)
    prefix_length = int(prefix_ids.shape[1])
    actor_state = ActorStepper(actor, prefix_ids, use_cache=use_actor_cache)
    chunk_token_ids: list[int] = []
    token_logprobs: list[float] = []
    token_entropies: list[float] = []
    chunk_logprob = 0.0
    contains_eos = False

    for _chunk_token_index in range(max_chunk_len):
        logits = actor_state.current_logits
        actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_entropy = _entropy_from_logits(logits)
        token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        sampled_token_logprob = float(actor_log_probs[0, token_id].item())
        token_logprobs.append(sampled_token_logprob)
        token_entropies.append(token_entropy)
        chunk_logprob += sampled_token_logprob
        chunk_token_ids.append(token_id)
        actor_state.append(token_id)
        if token_id in eos_token_ids:
            contains_eos = True
            break

    if not chunk_token_ids:
        raise RuntimeError("Chunk candidate generation produced zero tokens, which should be impossible.")

    if critic is not None:
        if critic_device is None:
            raise ValueError("critic_device must be provided when critic scoring is enabled.")
        critic_prefix_ids = prefix_ids.to(critic_device)
        continuation_values = getattr(critic, "continuation_values", None)
        if callable(continuation_values):
            chunk_token_tensor = torch.tensor(
                chunk_token_ids,
                device=critic_device,
                dtype=critic_prefix_ids.dtype,
            )
            chunk_values = continuation_values(
                prefix_ids=critic_prefix_ids,
                continuation_ids=chunk_token_tensor,
            )
        else:
            full_sequence_ids = actor_state.sequence_ids.to(critic_device)
            values = critic_sequence_values(critic, full_sequence_ids)[0]
            chunk_values = values[prefix_length : prefix_length + len(chunk_token_ids)]
        if chunk_values.numel() != len(chunk_token_ids):
            raise RuntimeError("Chunk value extraction length mismatch.")
        chunk_values_tuple = tuple(float(value) for value in chunk_values.tolist())
        end_value: float | None = float(chunk_values[-1].item())
        mean_value: float | None = float(chunk_values.mean().item())
    else:
        chunk_values_tuple = ()
        end_value = None
        mean_value = None

    return ChunkCandidate(
        candidate_index=candidate_index,
        chunk_token_ids=tuple(int(token_id) for token_id in chunk_token_ids),
        chunk_text=tokenizer.decode(chunk_token_ids, skip_special_tokens=True),
        chunk_length=len(chunk_token_ids),
        chunk_logprob=float(chunk_logprob),
        chunk_values=chunk_values_tuple,
        end_value=end_value,
        mean_value=mean_value,
        contains_eos=contains_eos,
        token_logprobs=tuple(token_logprobs),
        token_entropies=tuple(token_entropies),
        chunk_uncertainty=float(np.mean(np.asarray(token_entropies, dtype=np.float64))),
    )


def sample_actor_only_response(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    prompt_ids: torch.Tensor,
    spec: ChunkRunSpec,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    seed: int,
    use_actor_cache: bool,
) -> ChunkDecodeArtifacts:
    set_decode_seed(seed)
    comparison_reducer_spec = parse_optional_value_reducer(spec.comparison_value_reducer)
    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    generated_token_ids: list[int] = []
    sum_actor_logprob = 0.0
    eos_emitted = False

    start_time = time.perf_counter()
    for _step_index in range(max_new_tokens):
        logits = actor_state.current_logits
        actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=spec.actor_sampling_mode,
            temperature=spec.actor_temperature,
            top_p=spec.actor_top_p,
            top_k=spec.actor_top_k,
        )
        sum_actor_logprob += float(actor_log_probs[0, token_id].item())
        generated_token_ids.append(token_id)
        actor_state.append(token_id)
        if token_id in eos_token_ids:
            eos_emitted = True
            break

    latency_sec = time.perf_counter() - start_time
    response_length = len(generated_token_ids)
    max_length_hit = bool(max_new_tokens > 0 and not eos_emitted and response_length >= max_new_tokens)
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = float(score_response(example, response_text))

    example_result = {
        "config_id": spec.config_id,
        "method_name": spec.method_name,
        "score_mode": spec.score_mode,
        "chunk_size": None,
        "num_chunk_candidates": None,
        "beta": None,
        "value_reducer": None,
        "value_reducer_kind": None,
        "value_reducer_tail_length": None,
        "value_reducer_alpha": None,
        "comparison_value_reducer": None if comparison_reducer_spec is None else comparison_reducer_spec.canonical_name,
        "comparison_value_reducer_kind": None if comparison_reducer_spec is None else comparison_reducer_spec.kind,
        "comparison_value_reducer_tail_length": (
            None if comparison_reducer_spec is None else comparison_reducer_spec.tail_length
        ),
        "comparison_value_reducer_alpha": None if comparison_reducer_spec is None else comparison_reducer_spec.alpha,
        "example_id": int(example.example_id),
        "prompt_id": int(example.example_id),
        "data_source": example.data_source,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "prompt_length": int(prompt_ids.shape[1]),
        "generated_response": response_text,
        "response_length": response_length,
        "eos_emitted": eos_emitted,
        "max_length_hit": max_length_hit,
        "task_score": task_score,
        "sum_response_actor_logprob": float(sum_actor_logprob),
        "num_chunk_decisions": None,
        "mean_realized_chunk_length": None,
        "mean_selected_chunk_logprob": None,
        "mean_selected_chunk_value": None,
        "mean_selected_chunk_reducer_value": None,
        "mean_selected_chunk_end_value": None,
        "mean_selected_chunk_mean_value": None,
        "mean_selected_chunk_uncertainty": None,
        "mean_selected_chunk_entropy_horizon_mean": None,
        "mean_selected_chunk_tail_mean_h2": None,
        "mean_selected_chunk_tail_mean_h4": None,
        "mean_selected_chunk_tail_mean_h8": None,
        "mean_selected_chunk_tail_mean_h16": None,
        "fraction_chunk_decisions_different_from_actor_only_chunk_winner": None,
        "fraction_chunk_decisions_different_from_endvalue_winner": None,
        "fraction_chunk_decisions_different_from_uncertainty_winner": None,
        "fraction_chunk_decisions_different_from_comparison_winner": None,
        "mean_selected_chunk_score_margin": None,
        "fraction_selected_chunks_with_eos": None,
        "total_decoding_steps": response_length,
        "latency_sec": latency_sec,
        "tokens_per_second": (response_length / latency_sec) if latency_sec > 0 else None,
    }
    if comparison_reducer_spec is not None:
        example_result["fraction_chunk_decisions_different_from_comparison_winner"] = None
        example_result[comparison_reducer_spec.fraction_diff_from_winner_key] = None
    return ChunkDecodeArtifacts(example_result=example_result, chunk_decision_results=[])


def run_chunk_guided_response(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    prompt_ids: torch.Tensor,
    spec: ChunkRunSpec,
    actor_device: torch.device,
    critic_device: torch.device | None,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    seed: int,
    use_actor_cache: bool,
    debug_full_chunk_candidates: bool,
) -> ChunkDecodeArtifacts:
    if not spec.is_chunk_method or spec.chunk_size is None or spec.num_chunk_candidates is None:
        raise ValueError(f"Chunk run spec {spec.config_id} is missing chunk settings.")

    reducer_spec = parse_value_reducer(spec.value_reducer) if spec.value_reducer is not None else None
    comparison_reducer_spec = parse_optional_value_reducer(spec.comparison_value_reducer)
    critic_enabled = critic is not None
    if not critic_enabled:
        if spec.score_mode not in {"actor_logprob_only", "uncertainty_meanentropy"}:
            raise ValueError(f"Spec {spec.config_id} requires critic scoring, but critic loading is disabled.")
        if reducer_spec is not None or comparison_reducer_spec is not None:
            raise ValueError(
                f"Spec {spec.config_id} requests critic-based reducer diagnostics, but critic loading is disabled."
            )

    generated_token_ids: list[int] = []
    current_sequence_ids = prompt_ids
    chunk_decision_results: list[dict[str, Any]] = []

    selected_chunk_lengths: list[int] = []
    selected_chunk_logprobs: list[float] = []
    selected_chunk_end_values: list[float] = []
    selected_chunk_mean_values: list[float] = []
    selected_chunk_uncertainties: list[float] = []
    selected_chunk_values: list[float] = []
    selected_chunk_tail_means: dict[int, list[float]] = {
        tail_length: [] for tail_length in STANDARD_TAIL_SUMMARY_LENGTHS
    }
    selected_chunk_score_margins: list[float] = []
    selected_chunks_with_eos: list[float] = []
    selected_diff_from_actor_only_flags: list[float] = []
    selected_diff_from_endvalue_flags: list[float] = []
    selected_diff_from_uncertainty_flags: list[float] = []
    selected_diff_from_comparison_flags: list[float] = []

    start_time = time.perf_counter()
    chunk_decision_index = 0
    eos_emitted = False

    while len(generated_token_ids) < max_new_tokens:
        remaining_tokens = max_new_tokens - len(generated_token_ids)
        max_chunk_len = min(spec.chunk_size, remaining_tokens)
        generated_length_before_chunk = len(generated_token_ids)
        current_prefix_length = int(current_sequence_ids.shape[1])

        candidates: list[ChunkCandidate] = []
        for candidate_index in range(spec.num_chunk_candidates):
            candidate = sample_actor_chunk(
                actor=actor,
                critic=critic,
                tokenizer=tokenizer,
                prefix_ids=current_sequence_ids,
                actor_device=actor_device,
                critic_device=critic_device,
                max_chunk_len=max_chunk_len,
                sampling_mode=spec.actor_sampling_mode,
                temperature=spec.actor_temperature,
                top_p=spec.actor_top_p,
                top_k=spec.actor_top_k,
                seed=_chunk_candidate_seed(
                    seed,
                    example_id=example.example_id,
                    chunk_size=spec.chunk_size,
                    num_chunk_candidates=spec.num_chunk_candidates,
                    chunk_decision_index=chunk_decision_index,
                    candidate_index=candidate_index,
                ),
                eos_token_ids=eos_token_ids,
                use_actor_cache=use_actor_cache,
                candidate_index=candidate_index,
            )
            candidates.append(candidate)

        raw_logprobs = [candidate.chunk_logprob for candidate in candidates]
        if critic_enabled:
            if any(candidate.end_value is None or candidate.mean_value is None for candidate in candidates):
                raise RuntimeError("Critic-enabled chunk candidate is missing end/mean value diagnostics.")
            raw_end_values = [float(candidate.end_value) for candidate in candidates]
            raw_mean_values = [float(candidate.mean_value) for candidate in candidates]
        else:
            raw_end_values = None
            raw_mean_values = None
        raw_uncertainties = [candidate.chunk_uncertainty for candidate in candidates]
        normalized_logprobs = _zscore(raw_logprobs, eps=normalization_eps)
        normalized_end_values = _zscore(raw_end_values, eps=normalization_eps) if raw_end_values is not None else None
        normalized_mean_values = _zscore(raw_mean_values, eps=normalization_eps) if raw_mean_values is not None else None
        normalized_uncertainties = _zscore(raw_uncertainties, eps=normalization_eps)
        if comparison_reducer_spec is not None and raw_end_values is not None and raw_mean_values is not None:
            if comparison_reducer_spec.kind == "end":
                raw_comparison_values = raw_end_values
                normalized_comparison_values = normalized_end_values
            elif comparison_reducer_spec.kind == "chunk_mean":
                raw_comparison_values = raw_mean_values
                normalized_comparison_values = normalized_mean_values
            else:
                raw_comparison_values = [
                    reduce_chunk_values(candidate.chunk_values, comparison_reducer_spec)
                    for candidate in candidates
                ]
                normalized_comparison_values = _zscore(raw_comparison_values, eps=normalization_eps)
        else:
            raw_comparison_values = None
            normalized_comparison_values = None
        if reducer_spec is None:
            raw_reducer_values = None
            normalized_reducer_values = None
        elif raw_end_values is None or raw_mean_values is None:
            raise ValueError(f"Spec {spec.config_id} requires critic values, but critic loading is disabled.")
        elif reducer_spec.kind == "end":
            raw_reducer_values = raw_end_values
            normalized_reducer_values = normalized_end_values
        elif reducer_spec.kind == "chunk_mean":
            raw_reducer_values = raw_mean_values
            normalized_reducer_values = normalized_mean_values
        else:
            raw_reducer_values = [reduce_chunk_values(candidate.chunk_values, reducer_spec) for candidate in candidates]
            normalized_reducer_values = _zscore(raw_reducer_values, eps=normalization_eps)

        (
            actor_only_chunk_winner_index,
            actor_only_chunk_winner_ties,
            actor_only_chunk_winner_score,
        ) = _select_argmax(raw_logprobs)
        if raw_end_values is not None:
            endvalue_chunk_winner_index, endvalue_chunk_winner_ties, endvalue_chunk_winner_score = _select_argmax(
                raw_end_values
            )
        else:
            endvalue_chunk_winner_index = None
            endvalue_chunk_winner_ties = None
            endvalue_chunk_winner_score = None
        uncertainty_chunk_winner_index, uncertainty_chunk_winner_ties, _uncertainty_chunk_winner_score = (
            _select_argmax([-float(value) for value in raw_uncertainties])
        )
        uncertainty_chunk_winner_value = float(raw_uncertainties[uncertainty_chunk_winner_index])
        if comparison_reducer_spec is not None and raw_comparison_values is not None:
            comparison_chunk_winner_index, comparison_chunk_winner_ties, comparison_chunk_winner_score = _select_argmax(
                raw_comparison_values
            )
        else:
            comparison_chunk_winner_index = None
            comparison_chunk_winner_ties = None
            comparison_chunk_winner_score = None

        if spec.score_mode == "actor_logprob_only":
            selection_scores = normalized_logprobs
            normalized_value_for_scoring = None
        elif spec.score_mode == "uncertainty_meanentropy":
            selection_scores = [float(-value) for value in normalized_uncertainties]
            normalized_value_for_scoring = None
        elif spec.score_mode == "actor_plus_critic":
            if spec.beta is None or reducer_spec is None or normalized_reducer_values is None:
                raise ValueError(f"Spec {spec.config_id} requires beta and value_reducer.")
            normalized_value_for_scoring = normalized_reducer_values
            selection_scores = [
                float(normalized_logprobs[index] + float(spec.beta) * normalized_value_for_scoring[index])
                for index in range(len(candidates))
            ]
        elif spec.score_mode == "critic_only":
            if reducer_spec is None or normalized_reducer_values is None:
                raise ValueError(f"Spec {spec.config_id} requires value_reducer.")
            normalized_value_for_scoring = normalized_reducer_values
            selection_scores = [float(value) for value in normalized_value_for_scoring]
        else:
            raise ValueError(f"Unsupported score_mode: {spec.score_mode}")

        selected_candidate_index, selected_tied_indices, selected_score = _select_argmax(selection_scores)
        sorted_selection_scores = sorted(selection_scores, reverse=True)
        selected_score_margin = (
            float(sorted_selection_scores[0] - sorted_selection_scores[1]) if len(sorted_selection_scores) > 1 else None
        )
        selected_candidate = candidates[selected_candidate_index]
        selected_reducer_value = (
            float(raw_reducer_values[selected_candidate_index]) if raw_reducer_values is not None else None
        )
        selected_tail_mean_summary = (
            _standard_tail_mean_summary(selected_candidate.chunk_values)
            if selected_candidate.chunk_values
            else _empty_tail_mean_summary()
        )

        selected_chunk_lengths.append(selected_candidate.chunk_length)
        selected_chunk_logprobs.append(selected_candidate.chunk_logprob)
        if selected_candidate.end_value is not None:
            selected_chunk_end_values.append(selected_candidate.end_value)
        if selected_candidate.mean_value is not None:
            selected_chunk_mean_values.append(selected_candidate.mean_value)
        selected_chunk_uncertainties.append(selected_candidate.chunk_uncertainty)
        if selected_reducer_value is not None:
            selected_chunk_values.append(selected_reducer_value)
        for tail_length, tail_mean_value in selected_tail_mean_summary.items():
            if tail_mean_value is not None:
                selected_chunk_tail_means[tail_length].append(tail_mean_value)
        if selected_score_margin is not None:
            selected_chunk_score_margins.append(selected_score_margin)
        selected_chunks_with_eos.append(1.0 if selected_candidate.contains_eos else 0.0)
        selected_diff_from_actor_only_flags.append(
            1.0 if selected_candidate_index != actor_only_chunk_winner_index else 0.0
        )
        if endvalue_chunk_winner_index is not None:
            selected_diff_from_endvalue_flags.append(
                1.0 if selected_candidate_index != endvalue_chunk_winner_index else 0.0
            )
        selected_diff_from_uncertainty_flags.append(
            1.0 if selected_candidate_index != uncertainty_chunk_winner_index else 0.0
        )
        if comparison_chunk_winner_index is not None:
            selected_diff_from_comparison_flags.append(
                1.0 if selected_candidate_index != comparison_chunk_winner_index else 0.0
            )

        chunk_tensor = torch.tensor(
            [list(selected_candidate.chunk_token_ids)],
            device=actor_device,
            dtype=current_sequence_ids.dtype,
        )
        current_sequence_ids = torch.cat([current_sequence_ids, chunk_tensor], dim=1)
        generated_token_ids.extend(int(token_id) for token_id in selected_candidate.chunk_token_ids)

        chunk_decision_result: dict[str, Any] = {
            "config_id": spec.config_id,
            "method_name": spec.method_name,
            "score_mode": spec.score_mode,
            "chunk_size": spec.chunk_size,
            "num_chunk_candidates": spec.num_chunk_candidates,
            "beta": spec.beta,
            "value_reducer": spec.value_reducer,
            "value_reducer_kind": None if reducer_spec is None else reducer_spec.kind,
            "value_reducer_tail_length": None if reducer_spec is None else reducer_spec.tail_length,
            "value_reducer_alpha": None if reducer_spec is None else reducer_spec.alpha,
            "comparison_value_reducer": (
                None if comparison_reducer_spec is None else comparison_reducer_spec.canonical_name
            ),
            "comparison_value_reducer_kind": None if comparison_reducer_spec is None else comparison_reducer_spec.kind,
            "comparison_value_reducer_tail_length": (
                None if comparison_reducer_spec is None else comparison_reducer_spec.tail_length
            ),
            "comparison_value_reducer_alpha": (
                None if comparison_reducer_spec is None else comparison_reducer_spec.alpha
            ),
            "example_id": int(example.example_id),
            "prompt_id": int(example.example_id),
            "chunk_decision_index": chunk_decision_index,
            "current_prefix_length": current_prefix_length,
            "generated_length_before_chunk": generated_length_before_chunk,
            "candidate_chunk_ids": [candidate.candidate_index for candidate in candidates],
            "candidate_chunk_token_ids": [list(candidate.chunk_token_ids) for candidate in candidates],
            "candidate_chunk_texts": [candidate.chunk_text for candidate in candidates],
            "candidate_chunk_lengths": [candidate.chunk_length for candidate in candidates],
            "candidate_chunk_logprobs": raw_logprobs,
            "candidate_chunk_end_values": raw_end_values,
            "candidate_chunk_mean_values": raw_mean_values,
            "candidate_chunk_uncertainties": raw_uncertainties,
            "candidate_chunk_contains_eos": [candidate.contains_eos for candidate in candidates],
            "actor_only_selected_chunk_index": actor_only_chunk_winner_index,
            "actor_only_chunk_winner_index": actor_only_chunk_winner_index,
            "actor_only_chunk_winner_tied_indices": actor_only_chunk_winner_ties,
            "actor_only_chunk_winner_logprob": actor_only_chunk_winner_score,
            "endvalue_selected_chunk_index": endvalue_chunk_winner_index,
            "endvalue_chunk_winner_index": endvalue_chunk_winner_index,
            "endvalue_chunk_winner_tied_indices": endvalue_chunk_winner_ties,
            "endvalue_chunk_winner_value": endvalue_chunk_winner_score,
            "uncertainty_selected_chunk_index": uncertainty_chunk_winner_index,
            "uncertainty_chunk_winner_index": uncertainty_chunk_winner_index,
            "uncertainty_chunk_winner_tied_indices": uncertainty_chunk_winner_ties,
            "uncertainty_chunk_winner_value": uncertainty_chunk_winner_value,
            "selected_chunk_index": selected_candidate_index,
            "selected_chunk_tied_indices": selected_tied_indices,
            "selected_chunk_token_ids": list(selected_candidate.chunk_token_ids),
            "selected_chunk_text": selected_candidate.chunk_text,
            "selected_chunk_len": selected_candidate.chunk_length,
            "selected_chunk_logprob": selected_candidate.chunk_logprob,
            "selected_chunk_end_value": selected_candidate.end_value,
            "selected_chunk_mean_value": selected_candidate.mean_value,
            "selected_chunk_uncertainty": selected_candidate.chunk_uncertainty,
            "selected_chunk_entropy_horizon_mean": selected_candidate.chunk_uncertainty,
            "selected_chunk_tail_mean_h2": selected_tail_mean_summary[2],
            "selected_chunk_tail_mean_h4": selected_tail_mean_summary[4],
            "selected_chunk_tail_mean_h8": selected_tail_mean_summary[8],
            "selected_chunk_tail_mean_h16": selected_tail_mean_summary[16],
            "selected_chunk_value": selected_reducer_value,
            "selected_chunk_reducer_value": selected_reducer_value,
            "selected_chunk_contains_eos": selected_candidate.contains_eos,
            "selected_chunk_selection_score": selected_score,
            "selected_chunk_score_margin": selected_score_margin,
            "selected_differs_from_actor_only_chunk_winner": selected_candidate_index != actor_only_chunk_winner_index,
            "selected_differs_from_endvalue_winner": (
                None
                if endvalue_chunk_winner_index is None
                else selected_candidate_index != endvalue_chunk_winner_index
            ),
            "selected_differs_from_uncertainty_winner": selected_candidate_index != uncertainty_chunk_winner_index,
        }
        if reducer_spec is not None:
            chunk_decision_result[reducer_spec.selected_metric_key] = selected_reducer_value
            chunk_decision_result["candidate_chunk_reducer_values"] = raw_reducer_values
        if comparison_reducer_spec is not None and raw_comparison_values is not None:
            chunk_decision_result["comparison_chunk_winner_index"] = comparison_chunk_winner_index
            chunk_decision_result["comparison_chunk_winner_tied_indices"] = comparison_chunk_winner_ties
            chunk_decision_result["comparison_chunk_winner_value"] = comparison_chunk_winner_score
            chunk_decision_result["selected_differs_from_comparison_winner"] = (
                selected_candidate_index != comparison_chunk_winner_index
            )
            chunk_decision_result[comparison_reducer_spec.winner_candidate_values_key] = raw_comparison_values
            chunk_decision_result[comparison_reducer_spec.winner_index_key] = comparison_chunk_winner_index
            chunk_decision_result[comparison_reducer_spec.winner_tied_indices_key] = comparison_chunk_winner_ties
            chunk_decision_result[comparison_reducer_spec.winner_value_key] = comparison_chunk_winner_score
            chunk_decision_result[comparison_reducer_spec.selected_differs_from_winner_key] = (
                selected_candidate_index != comparison_chunk_winner_index
            )
        if debug_full_chunk_candidates:
            chunk_decision_result.update(
                {
                    "candidate_normalized_chunk_logprobs": normalized_logprobs,
                    "candidate_normalized_chunk_end_values": normalized_end_values,
                    "candidate_normalized_chunk_mean_values": normalized_mean_values,
                    "candidate_normalized_chunk_uncertainties": normalized_uncertainties,
                    "candidate_selection_scores": selection_scores,
                    "candidate_chunk_token_logprobs": [list(candidate.token_logprobs) for candidate in candidates],
                    "candidate_chunk_token_entropies": [list(candidate.token_entropies) for candidate in candidates],
                    "candidate_chunk_token_values": [list(candidate.chunk_values) for candidate in candidates],
                    "selected_chunk_token_logprobs": list(selected_candidate.token_logprobs),
                    "selected_chunk_token_entropies": list(selected_candidate.token_entropies),
                    "selected_chunk_token_values": list(selected_candidate.chunk_values),
                }
            )
            if normalized_reducer_values is not None:
                chunk_decision_result["candidate_normalized_chunk_reducer_values"] = normalized_reducer_values
            if comparison_reducer_spec is not None and normalized_comparison_values is not None:
                comparison_normalized_key = comparison_reducer_spec.winner_candidate_values_key.replace(
                    "candidate_chunk_",
                    "candidate_normalized_chunk_",
                    1,
                )
                chunk_decision_result[comparison_normalized_key] = normalized_comparison_values

        chunk_decision_results.append(chunk_decision_result)
        chunk_decision_index += 1

        if selected_candidate.contains_eos:
            eos_emitted = True
            break

    latency_sec = time.perf_counter() - start_time
    response_length = len(generated_token_ids)
    max_length_hit = bool(max_new_tokens > 0 and not eos_emitted and response_length >= max_new_tokens)
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = float(score_response(example, response_text))

    example_result = {
        "config_id": spec.config_id,
        "method_name": spec.method_name,
        "score_mode": spec.score_mode,
        "chunk_size": spec.chunk_size,
        "num_chunk_candidates": spec.num_chunk_candidates,
        "beta": spec.beta,
        "value_reducer": spec.value_reducer,
        "value_reducer_kind": None if reducer_spec is None else reducer_spec.kind,
        "value_reducer_tail_length": None if reducer_spec is None else reducer_spec.tail_length,
        "value_reducer_alpha": None if reducer_spec is None else reducer_spec.alpha,
        "comparison_value_reducer": None if comparison_reducer_spec is None else comparison_reducer_spec.canonical_name,
        "comparison_value_reducer_kind": None if comparison_reducer_spec is None else comparison_reducer_spec.kind,
        "comparison_value_reducer_tail_length": (
            None if comparison_reducer_spec is None else comparison_reducer_spec.tail_length
        ),
        "comparison_value_reducer_alpha": None if comparison_reducer_spec is None else comparison_reducer_spec.alpha,
        "example_id": int(example.example_id),
        "prompt_id": int(example.example_id),
        "data_source": example.data_source,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "prompt_length": int(prompt_ids.shape[1]),
        "generated_response": response_text,
        "response_length": response_length,
        "eos_emitted": eos_emitted,
        "max_length_hit": max_length_hit,
        "task_score": task_score,
        "sum_response_actor_logprob": None,
        "num_chunk_decisions": chunk_decision_index,
        "mean_realized_chunk_length": _mean(selected_chunk_lengths),
        "mean_selected_chunk_logprob": _mean(selected_chunk_logprobs),
        "mean_selected_chunk_value": _mean(selected_chunk_values),
        "mean_selected_chunk_reducer_value": _mean(selected_chunk_values),
        "mean_selected_chunk_end_value": _mean(selected_chunk_end_values),
        "mean_selected_chunk_mean_value": _mean(selected_chunk_mean_values),
        "mean_selected_chunk_uncertainty": _mean(selected_chunk_uncertainties),
        "mean_selected_chunk_entropy_horizon_mean": _mean(selected_chunk_uncertainties),
        "mean_selected_chunk_tail_mean_h2": _mean(selected_chunk_tail_means[2]),
        "mean_selected_chunk_tail_mean_h4": _mean(selected_chunk_tail_means[4]),
        "mean_selected_chunk_tail_mean_h8": _mean(selected_chunk_tail_means[8]),
        "mean_selected_chunk_tail_mean_h16": _mean(selected_chunk_tail_means[16]),
        "fraction_chunk_decisions_different_from_actor_only_chunk_winner": _mean(selected_diff_from_actor_only_flags),
        "fraction_chunk_decisions_different_from_endvalue_winner": _mean(selected_diff_from_endvalue_flags),
        "fraction_chunk_decisions_different_from_uncertainty_winner": _mean(selected_diff_from_uncertainty_flags),
        "fraction_chunk_decisions_different_from_comparison_winner": _mean(selected_diff_from_comparison_flags),
        "mean_selected_chunk_score_margin": _mean(selected_chunk_score_margins),
        "fraction_selected_chunks_with_eos": _mean(selected_chunks_with_eos),
        "total_decoding_steps": response_length,
        "latency_sec": latency_sec,
        "tokens_per_second": (response_length / latency_sec) if latency_sec > 0 else None,
    }
    if reducer_spec is not None:
        example_result[reducer_spec.aggregate_metric_key] = example_result["mean_selected_chunk_reducer_value"]
    if comparison_reducer_spec is not None:
        example_result["fraction_chunk_decisions_different_from_comparison_winner"] = _mean(
            selected_diff_from_comparison_flags
        )
        example_result[comparison_reducer_spec.fraction_diff_from_winner_key] = _mean(selected_diff_from_comparison_flags)
    return ChunkDecodeArtifacts(example_result=example_result, chunk_decision_results=chunk_decision_results)


def process_example_for_spec(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    spec: ChunkRunSpec,
    actor_device: torch.device,
    critic_device: torch.device | None,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    seed: int,
    use_actor_cache: bool,
    debug_full_chunk_candidates: bool,
) -> ChunkDecodeArtifacts:
    prompt_ids = _prompt_ids_tensor(
        example=example,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        device=actor_device,
    )
    if spec.score_mode == "actor_only_sample":
        return sample_actor_only_response(
            actor=actor,
            tokenizer=tokenizer,
            example=example,
            prompt_ids=prompt_ids,
            spec=spec,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            seed=_ordinary_actor_seed(seed, example_id=example.example_id),
            use_actor_cache=use_actor_cache,
        )
    return run_chunk_guided_response(
        actor=actor,
        critic=critic,
        tokenizer=tokenizer,
        example=example,
        prompt_ids=prompt_ids,
        spec=spec,
        actor_device=actor_device,
        critic_device=critic_device,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        normalization_eps=normalization_eps,
        seed=seed,
        use_actor_cache=use_actor_cache,
        debug_full_chunk_candidates=debug_full_chunk_candidates,
    )


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
        raise ImportError("Ray is required for cross-node chunk guidance evaluation, but it is not installed.")
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
    critic_hf_dir: Path | None,
    examples: list[ExampleRecord],
    run_specs: list[ChunkRunSpec],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    use_actor_cache: bool,
    debug_full_chunk_candidates: bool,
    seed: int,
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
                "critic_hf_dir": None if critic_hf_dir is None else str(critic_hf_dir),
                "examples": examples,
                "run_specs": run_specs,
                "dtype_name": dtype_name,
                "trust_remote_code": trust_remote_code,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "eos_token_ids": eos_token_ids,
                "normalization_eps": normalization_eps,
                "use_actor_cache": use_actor_cache,
                "debug_full_chunk_candidates": debug_full_chunk_candidates,
                "seed": seed,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
            },
            name=f"chunk_guidance_worker_{assignment.worker_id}",
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
    critic_enabled: bool,
    ray_num_cpus_per_worker: float,
) -> list[dict[str, Any]]:
    node_groups: dict[tuple[int | None, str | None, str | None], list[WorkerAssignment]] = {}
    for assignment in worker_assignments:
        group_key = (assignment.node_index, assignment.node_ip, assignment.node_resource_key)
        node_groups.setdefault(group_key, []).append(assignment)

    node_specs: list[dict[str, Any]] = []
    for group_key in sorted(node_groups, key=lambda item: (-1 if item[0] is None else int(item[0]), str(item[1]))):
        node_assignments = sorted(node_groups[group_key], key=lambda item: int(item.worker_id))
        normalized_assignments: list[tuple[WorkerAssignment, str | None, str | None]] = []
        referenced_cuda_slots: set[int] = set()

        for assignment in node_assignments:
            actor_device_name = _normalize_cuda_device_name(
                assignment.actor_device,
                assume_default_cuda=True,
            )
            critic_device_name = (
                _normalize_cuda_device_name(
                    assignment.critic_device if assignment.critic_device is not None else actor_device_name,
                    assume_default_cuda=True,
                )
                if critic_enabled
                else None
            )
            normalized_assignments.append((assignment, actor_device_name, critic_device_name))
            for device_name in (actor_device_name, critic_device_name):
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
        for assignment, actor_device_name, critic_device_name in normalized_assignments:
            remapped_assignments.append(
                WorkerAssignment(
                    worker_id=assignment.worker_id,
                    actor_device=_remap_cuda_device_name(actor_device_name, cuda_slot_mapping=cuda_slot_mapping),
                    critic_device=_remap_cuda_device_name(critic_device_name, cuda_slot_mapping=cuda_slot_mapping),
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


def _ray_node_entry_remote(**kwargs) -> dict[str, Any]:
    assignments: list[WorkerAssignment] = kwargs["assignments"]
    progress_actor = kwargs["progress_actor"]
    progress_queue, processes = _start_local_worker_processes(
        assignments=assignments,
        actor_hf_dir=Path(kwargs["actor_hf_dir"]),
        critic_hf_dir=None if kwargs["critic_hf_dir"] is None else Path(kwargs["critic_hf_dir"]),
        examples=kwargs["examples"],
        run_specs=kwargs["run_specs"],
        dtype_name=kwargs["dtype_name"],
        trust_remote_code=kwargs["trust_remote_code"],
        max_prompt_length=kwargs["max_prompt_length"],
        max_new_tokens=kwargs["max_new_tokens"],
        eos_token_ids=kwargs["eos_token_ids"],
        normalization_eps=kwargs["normalization_eps"],
        use_actor_cache=kwargs["use_actor_cache"],
        debug_full_chunk_candidates=kwargs["debug_full_chunk_candidates"],
        seed=kwargs["seed"],
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


def _worker_entry(
    *,
    assignment: WorkerAssignment,
    actor_hf_dir: str,
    critic_hf_dir: str | None,
    examples: list[ExampleRecord],
    run_specs: list[ChunkRunSpec],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    use_actor_cache: bool,
    debug_full_chunk_candidates: bool,
    seed: int,
    worker_root: str,
    progress_queue=None,
    progress_actor=None,
) -> None:
    worker_dir = Path(worker_root) / f"worker_{assignment.worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    summary_path = worker_dir / "worker_summary.json"
    error_path = worker_dir / "worker_error.txt"

    try:
        start_time = time.perf_counter()
        actor_device = resolve_device(assignment.actor_device)
        _validate_visible_cuda_device(actor_device, label="actor_device")
        critic_device = (
            resolve_device(assignment.critic_device)
            if critic_hf_dir is not None and assignment.critic_device
            else None
        )
        _validate_visible_cuda_device(critic_device, label="critic_device")
        dtype = resolve_dtype(dtype_name)

        tokenizer = load_tokenizer(Path(actor_hf_dir), trust_remote_code=trust_remote_code)
        actor = load_actor_model(
            Path(actor_hf_dir),
            dtype=dtype,
            device=actor_device,
            trust_remote_code=trust_remote_code,
        )
        critic = (
            load_critic_model(
                Path(critic_hf_dir),
                dtype=dtype,
                device=critic_device,
                trust_remote_code=trust_remote_code,
            )
            if critic_hf_dir is not None and critic_device is not None
            else None
        )

        local_examples = examples[assignment.example_start : assignment.example_end]
        worker_total_tasks = len(local_examples) * len(run_specs)
        worker_completed_tasks = 0
        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_started",
                "worker_id": assignment.worker_id,
                "worker_total_tasks": worker_total_tasks,
            },
        )

        per_config_counts: dict[str, int] = {}
        per_config_start_wall_time_sec: dict[str, float] = {}
        per_config_end_wall_time_sec: dict[str, float] = {}
        per_config_runtime_sec: dict[str, float] = {}

        for spec in run_specs:
            config_start_perf = time.perf_counter()
            config_start_wall = time.time()
            per_example_path = worker_dir / f"per_example__{spec.config_id}.jsonl"
            chunk_decision_path = worker_dir / f"chunk_decisions__{spec.config_id}.jsonl"
            count = 0
            with per_example_path.open("w", encoding="utf-8") as per_example_file, chunk_decision_path.open(
                "w",
                encoding="utf-8",
            ) as chunk_decision_file:
                for example in local_examples:
                    artifacts = process_example_for_spec(
                        actor=actor,
                        critic=critic,
                        tokenizer=tokenizer,
                        example=example,
                        spec=spec,
                        actor_device=actor_device,
                        critic_device=critic_device,
                        max_prompt_length=max_prompt_length,
                        max_new_tokens=max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        normalization_eps=normalization_eps,
                        seed=seed,
                        use_actor_cache=use_actor_cache,
                        debug_full_chunk_candidates=debug_full_chunk_candidates,
                    )
                    per_example_file.write(_json_line(artifacts.example_result))
                    for chunk_decision_result in artifacts.chunk_decision_results:
                        chunk_decision_file.write(_json_line(chunk_decision_result))
                    count += 1
                    worker_completed_tasks += 1
                    _emit_progress(
                        progress_queue=progress_queue,
                        progress_actor=progress_actor,
                        event={
                            "type": "task_done",
                            "worker_id": assignment.worker_id,
                            "config_id": spec.config_id,
                            "worker_completed_tasks": worker_completed_tasks,
                            "worker_total_tasks": worker_total_tasks,
                        },
                    )

            per_config_counts[spec.config_id] = count
            per_config_start_wall_time_sec[spec.config_id] = config_start_wall
            per_config_end_wall_time_sec[spec.config_id] = time.time()
            per_config_runtime_sec[spec.config_id] = time.perf_counter() - config_start_perf

        summary_payload = {
            "worker_id": assignment.worker_id,
            "actor_device": str(actor_device),
            "critic_device": None if critic_device is None else str(critic_device),
            "node_index": assignment.node_index,
            "node_ip": assignment.node_ip,
            "node_resource_key": assignment.node_resource_key,
            "local_worker_index": assignment.local_worker_index,
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
            json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)
        _emit_progress(
            progress_queue=progress_queue,
            progress_actor=progress_actor,
            event={
                "type": "worker_done",
                "worker_id": assignment.worker_id,
                "worker_completed_tasks": worker_completed_tasks,
                "worker_total_tasks": worker_total_tasks,
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


def run_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path | None,
    examples: list[ExampleRecord],
    run_specs: list[ChunkRunSpec],
    worker_pairs: list[tuple[str | None, str | None]],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    use_actor_cache: bool,
    debug_full_chunk_candidates: bool,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    assignments = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)
    if not assignments:
        raise ValueError("No worker assignments were created.")

    worker_root = output_dir / "_worker_tmp"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    progress_queue, processes = _start_local_worker_processes(
        assignments=assignments,
        actor_hf_dir=actor_hf_dir,
        critic_hf_dir=critic_hf_dir,
        examples=examples,
        run_specs=run_specs,
        dtype_name=dtype_name,
        trust_remote_code=trust_remote_code,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        normalization_eps=normalization_eps,
        use_actor_cache=use_actor_cache,
        debug_full_chunk_candidates=debug_full_chunk_candidates,
        seed=seed,
        worker_root=worker_root,
    )

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

    with tqdm(total=total_tasks, desc="chunk_guidance_eval", unit="task", dynamic_ncols=True) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_tasks < total_tasks or completed_workers < len(assignments):
            try:
                event = progress_queue.get(timeout=0.2)
            except Empty:
                _assert_local_processes_healthy(processes=processes, worker_root=worker_root)
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

    _join_local_processes(processes=processes, worker_root=worker_root)

    return _collect_worker_outputs(
        output_dir=output_dir,
        worker_root=worker_root,
        assignments=assignments,
        run_specs=run_specs,
    )


def _collect_worker_outputs(
    *,
    output_dir: Path,
    worker_root: Path,
    assignments: Sequence[WorkerAssignment],
    run_specs: Sequence[ChunkRunSpec],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    per_example_path = output_dir / "per_example_results.jsonl"
    chunk_decision_path = output_dir / "chunk_decision_results.jsonl"
    example_results_by_config: dict[str, list[dict[str, Any]]] = {spec.config_id: [] for spec in run_specs}

    with per_example_path.open("w", encoding="utf-8") as per_example_file, chunk_decision_path.open(
        "w",
        encoding="utf-8",
    ) as chunk_decision_file:
        for spec in run_specs:
            for assignment in assignments:
                worker_dir = worker_root / f"worker_{assignment.worker_id:03d}"
                worker_example_path = worker_dir / f"per_example__{spec.config_id}.jsonl"
                worker_chunk_path = worker_dir / f"chunk_decisions__{spec.config_id}.jsonl"

                with worker_example_path.open("r", encoding="utf-8") as worker_example_file:
                    for line in worker_example_file:
                        if not line.strip():
                            continue
                        per_example_file.write(line)
                        example_results_by_config[spec.config_id].append(json.loads(line))

                with worker_chunk_path.open("r", encoding="utf-8") as worker_chunk_file:
                    shutil.copyfileobj(worker_chunk_file, chunk_decision_file)

    worker_summaries: list[dict[str, Any]] = []
    for assignment in assignments:
        summary_path = worker_root / f"worker_{assignment.worker_id:03d}" / "worker_summary.json"
        with summary_path.open("r", encoding="utf-8") as summary_file:
            worker_summaries.append(json.load(summary_file))
    worker_summaries.sort(key=lambda item: int(item["worker_id"]))
    return example_results_by_config, worker_summaries


def run_ray_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path | None,
    examples: list[ExampleRecord],
    run_specs: list[ChunkRunSpec],
    worker_assignments: list[WorkerAssignment],
    dtype_name: str,
    trust_remote_code: bool,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    use_actor_cache: bool,
    debug_full_chunk_candidates: bool,
    seed: int,
    ray_num_cpus_per_worker: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not worker_assignments:
        raise ValueError("No worker assignments were created.")
    if ray_num_cpus_per_worker <= 0:
        raise ValueError(f"ray_num_cpus_per_worker must be > 0, got {ray_num_cpus_per_worker}.")

    ray_module = _require_ray()
    if not ray_module.is_initialized():
        raise RuntimeError("Ray must be initialized before running cross-node chunk guidance workers.")

    worker_root = output_dir / "_worker_tmp"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    progress_actor = ray_module.remote(num_cpus=0)(_RayProgressActor).remote()
    node_execution_specs = _build_ray_node_execution_specs(
        worker_assignments=worker_assignments,
        critic_enabled=critic_hf_dir is not None,
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
            critic_hf_dir=None if critic_hf_dir is None else str(critic_hf_dir),
            examples=examples,
            run_specs=run_specs,
            dtype_name=dtype_name,
            trust_remote_code=trust_remote_code,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            normalization_eps=normalization_eps,
            use_actor_cache=use_actor_cache,
            debug_full_chunk_candidates=debug_full_chunk_candidates,
            seed=seed,
            worker_root=str(worker_root),
            progress_actor=progress_actor,
        )
        node_refs.append(node_ref)
        ref_to_node_spec[node_ref] = node_spec

    total_tasks = len(examples) * len(run_specs)
    completed_tasks = 0
    completed_workers = 0
    worker_progress: dict[int, dict[str, Any]] = {
        assignment.worker_id: {
            "done": 0,
            "total": assignment.num_examples * len(run_specs),
            "config_id": None,
        }
        for assignment in worker_assignments
    }

    pending_refs = list(node_refs)
    with tqdm(total=total_tasks, desc="chunk_guidance_eval", unit="task", dynamic_ncols=True) as progress_bar:
        progress_bar.set_postfix_str(_progress_postfix(worker_progress))
        while completed_tasks < total_tasks or completed_workers < len(worker_assignments):
            events = ray_module.get(progress_actor.drain.remote())
            for event in events:
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
        output_dir=output_dir,
        worker_root=worker_root,
        assignments=worker_assignments,
        run_specs=run_specs,
    )


def aggregate_results(
    spec: ChunkRunSpec,
    example_results: list[dict[str, Any]],
    *,
    wall_time_sec: float | None = None,
) -> dict[str, Any]:
    reducer_spec = parse_value_reducer(spec.value_reducer) if spec.value_reducer is not None else None
    comparison_reducer_spec = parse_optional_value_reducer(spec.comparison_value_reducer)
    task_scores = [float(result["task_score"]) for result in example_results]
    response_lengths = [int(result["response_length"]) for result in example_results]
    eos_rate = _mean([1.0 if bool(result["eos_emitted"]) else 0.0 for result in example_results])
    max_length_hit_rate = _mean([1.0 if bool(result["max_length_hit"]) else 0.0 for result in example_results])
    latencies = [float(result["latency_sec"]) for result in example_results]
    tokens_per_second = [
        float(result["tokens_per_second"])
        for result in example_results
        if result["tokens_per_second"] is not None
    ]
    total_generated_tokens = sum(int(result["total_decoding_steps"]) for result in example_results)

    if spec.is_chunk_method:
        num_chunk_decisions = [int(result["num_chunk_decisions"]) for result in example_results]
        realized_chunk_lengths = [
            float(result["mean_realized_chunk_length"])
            for result in example_results
            if result["mean_realized_chunk_length"] is not None
        ]
        selected_chunk_logprobs = [
            float(result["mean_selected_chunk_logprob"])
            for result in example_results
            if result["mean_selected_chunk_logprob"] is not None
        ]
        selected_chunk_values = [
            float(result["mean_selected_chunk_value"])
            for result in example_results
            if result["mean_selected_chunk_value"] is not None
        ]
        selected_chunk_end_values = [
            float(result["mean_selected_chunk_end_value"])
            for result in example_results
            if result["mean_selected_chunk_end_value"] is not None
        ]
        selected_chunk_mean_values = [
            float(result["mean_selected_chunk_mean_value"])
            for result in example_results
            if result["mean_selected_chunk_mean_value"] is not None
        ]
        selected_chunk_uncertainties = [
            float(result["mean_selected_chunk_uncertainty"])
            for result in example_results
            if result.get("mean_selected_chunk_uncertainty") is not None
        ]
        selected_chunk_tail_mean_h2 = [
            float(result["mean_selected_chunk_tail_mean_h2"])
            for result in example_results
            if result.get("mean_selected_chunk_tail_mean_h2") is not None
        ]
        selected_chunk_tail_mean_h4 = [
            float(result["mean_selected_chunk_tail_mean_h4"])
            for result in example_results
            if result.get("mean_selected_chunk_tail_mean_h4") is not None
        ]
        selected_chunk_tail_mean_h8 = [
            float(result["mean_selected_chunk_tail_mean_h8"])
            for result in example_results
            if result.get("mean_selected_chunk_tail_mean_h8") is not None
        ]
        selected_chunk_tail_mean_h16 = [
            float(result["mean_selected_chunk_tail_mean_h16"])
            for result in example_results
            if result.get("mean_selected_chunk_tail_mean_h16") is not None
        ]
        diff_from_actor_only = [
            float(result["fraction_chunk_decisions_different_from_actor_only_chunk_winner"])
            for result in example_results
            if result["fraction_chunk_decisions_different_from_actor_only_chunk_winner"] is not None
        ]
        diff_from_endvalue = [
            float(result["fraction_chunk_decisions_different_from_endvalue_winner"])
            for result in example_results
            if result.get("fraction_chunk_decisions_different_from_endvalue_winner") is not None
        ]
        diff_from_uncertainty = [
            float(result["fraction_chunk_decisions_different_from_uncertainty_winner"])
            for result in example_results
            if result.get("fraction_chunk_decisions_different_from_uncertainty_winner") is not None
        ]
        if comparison_reducer_spec is not None:
            diff_from_comparison = [
                float(result["fraction_chunk_decisions_different_from_comparison_winner"])
                for result in example_results
                if result.get("fraction_chunk_decisions_different_from_comparison_winner") is not None
            ]
        else:
            diff_from_comparison = []
        selected_chunk_score_margins = [
            float(result["mean_selected_chunk_score_margin"])
            for result in example_results
            if result["mean_selected_chunk_score_margin"] is not None
        ]
        fraction_selected_chunks_with_eos = [
            float(result["fraction_selected_chunks_with_eos"])
            for result in example_results
            if result["fraction_selected_chunks_with_eos"] is not None
        ]
        mean_num_chunk_decisions = _mean(num_chunk_decisions)
    else:
        realized_chunk_lengths = []
        selected_chunk_logprobs = []
        selected_chunk_values = []
        selected_chunk_end_values = []
        selected_chunk_mean_values = []
        selected_chunk_uncertainties = []
        selected_chunk_tail_mean_h2 = []
        selected_chunk_tail_mean_h4 = []
        selected_chunk_tail_mean_h8 = []
        selected_chunk_tail_mean_h16 = []
        diff_from_actor_only = []
        diff_from_endvalue = []
        diff_from_uncertainty = []
        diff_from_comparison = []
        selected_chunk_score_margins = []
        fraction_selected_chunks_with_eos = []
        mean_num_chunk_decisions = None

    binary_scores = set(task_scores).issubset({0.0, 1.0})
    total_latency = sum(latencies)
    row = {
        "config_id": spec.config_id,
        "method_name": spec.method_name,
        "score_mode": spec.score_mode,
        "chunk_size": spec.chunk_size,
        "num_chunk_candidates": spec.num_chunk_candidates,
        "beta": spec.beta,
        "value_reducer": spec.value_reducer,
        "value_reducer_kind": None if reducer_spec is None else reducer_spec.kind,
        "value_reducer_tail_length": None if reducer_spec is None else reducer_spec.tail_length,
        "value_reducer_alpha": None if reducer_spec is None else reducer_spec.alpha,
        "comparison_value_reducer": None if comparison_reducer_spec is None else comparison_reducer_spec.canonical_name,
        "comparison_value_reducer_kind": (
            None if comparison_reducer_spec is None else comparison_reducer_spec.kind
        ),
        "comparison_value_reducer_tail_length": (
            None if comparison_reducer_spec is None else comparison_reducer_spec.tail_length
        ),
        "comparison_value_reducer_alpha": None if comparison_reducer_spec is None else comparison_reducer_spec.alpha,
        "actor_sampling_mode": spec.actor_sampling_mode,
        "actor_temperature": spec.actor_temperature,
        "actor_top_p": spec.actor_top_p,
        "actor_top_k": spec.actor_top_k,
        "num_examples": len(example_results),
        "mean_task_score": _mean(task_scores),
        "mean_accuracy": _mean(task_scores) if binary_scores else None,
        "mean_response_length": _mean(response_lengths),
        "eos_rate": eos_rate,
        "max_length_hit_rate": max_length_hit_rate,
        "mean_num_chunk_decisions": mean_num_chunk_decisions,
        "mean_realized_chunk_length": _mean(realized_chunk_lengths) if spec.is_chunk_method else None,
        "mean_selected_chunk_logprob": _mean(selected_chunk_logprobs) if spec.is_chunk_method else None,
        "mean_selected_chunk_value": _mean(selected_chunk_values) if spec.is_chunk_method else None,
        "mean_selected_chunk_reducer_value": _mean(selected_chunk_values) if spec.is_chunk_method else None,
        "mean_selected_chunk_end_value": _mean(selected_chunk_end_values) if spec.is_chunk_method else None,
        "mean_selected_chunk_mean_value": _mean(selected_chunk_mean_values) if spec.is_chunk_method else None,
        "mean_selected_chunk_uncertainty": _mean(selected_chunk_uncertainties) if spec.is_chunk_method else None,
        "mean_selected_chunk_entropy_horizon_mean": (
            _mean(selected_chunk_uncertainties) if spec.is_chunk_method else None
        ),
        "mean_selected_chunk_tail_mean_h2": _mean(selected_chunk_tail_mean_h2) if spec.is_chunk_method else None,
        "mean_selected_chunk_tail_mean_h4": _mean(selected_chunk_tail_mean_h4) if spec.is_chunk_method else None,
        "mean_selected_chunk_tail_mean_h8": _mean(selected_chunk_tail_mean_h8) if spec.is_chunk_method else None,
        "mean_selected_chunk_tail_mean_h16": _mean(selected_chunk_tail_mean_h16) if spec.is_chunk_method else None,
        "mean_fraction_chunk_decisions_different_from_actor_only_chunk_winner": (
            _mean(diff_from_actor_only) if spec.is_chunk_method else None
        ),
        "mean_fraction_chunk_decisions_different_from_endvalue_winner": (
            _mean(diff_from_endvalue) if spec.is_chunk_method else None
        ),
        "mean_fraction_chunk_decisions_different_from_uncertainty_winner": (
            _mean(diff_from_uncertainty) if spec.is_chunk_method else None
        ),
        "mean_fraction_chunk_decisions_different_from_comparison_winner": (
            _mean(diff_from_comparison) if spec.is_chunk_method else None
        ),
        "mean_selected_chunk_score_margin": _mean(selected_chunk_score_margins) if spec.is_chunk_method else None,
        "fraction_selected_chunks_with_eos": _mean(fraction_selected_chunks_with_eos) if spec.is_chunk_method else None,
        "total_generated_tokens": total_generated_tokens,
        "sum_example_latency_sec": total_latency,
        "wall_time_sec": wall_time_sec,
        "overall_tokens_per_second": (
            total_generated_tokens / wall_time_sec
            if wall_time_sec is not None and wall_time_sec > 0
            else (total_generated_tokens / total_latency if total_latency > 0 else None)
        ),
        "mean_tokens_per_second": _mean(tokens_per_second),
    }
    if reducer_spec is not None:
        row[reducer_spec.aggregate_metric_key] = row["mean_selected_chunk_reducer_value"]
    if comparison_reducer_spec is not None:
        row[comparison_reducer_spec.mean_fraction_diff_from_winner_key] = row[
            "mean_fraction_chunk_decisions_different_from_comparison_winner"
        ]
    return row


def _aligned_metric_arrays(
    target_results: Sequence[dict[str, Any]],
    baseline_results: Sequence[dict[str, Any]],
    *,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_by_example_id = {int(result["example_id"]): result for result in target_results}
    baseline_by_example_id = {int(result["example_id"]): result for result in baseline_results}
    common_example_ids = sorted(set(target_by_example_id).intersection(baseline_by_example_id))

    target_values: list[float] = []
    baseline_values: list[float] = []
    for example_id in common_example_ids:
        target_value = target_by_example_id[example_id].get(key)
        baseline_value = baseline_by_example_id[example_id].get(key)
        if target_value is None or baseline_value is None:
            continue
        target_values.append(float(target_value))
        baseline_values.append(float(baseline_value))
    return (
        np.asarray(target_values, dtype=np.float64),
        np.asarray(baseline_values, dtype=np.float64),
    )


def _paired_metric_delta(
    target_results: Sequence[dict[str, Any]],
    baseline_results: Sequence[dict[str, Any]],
    *,
    key: str,
) -> float | None:
    target_values, baseline_values = _aligned_metric_arrays(target_results, baseline_results, key=key)
    if target_values.size == 0:
        return None
    return float(np.mean(target_values - baseline_values))


def _paired_bootstrap_mean_delta(
    target_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if target_values.size == 0 or baseline_values.size == 0 or bootstrap_samples <= 0:
        return None

    paired_deltas = target_values - baseline_values
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, paired_deltas.size, size=(bootstrap_samples, paired_deltas.size))
    bootstrap_means = paired_deltas[sample_indices].mean(axis=1)
    return {
        "mean_delta": float(paired_deltas.mean()),
        "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
        "bootstrap_samples": int(bootstrap_samples),
    }


def _build_single_reducer_comparison(
    *,
    target_spec: ChunkRunSpec,
    baseline_spec: ChunkRunSpec,
    relation: str,
    example_results_by_config: dict[str, list[dict[str, Any]]],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    target_reducer = parse_value_reducer(target_spec.value_reducer) if target_spec.value_reducer is not None else None
    baseline_reducer = (
        parse_value_reducer(baseline_spec.value_reducer) if baseline_spec.value_reducer is not None else None
    )
    target_comparison_reducer = (
        parse_optional_value_reducer(target_spec.comparison_value_reducer)
        if target_spec.comparison_value_reducer is not None
        else None
    )
    baseline_comparison_reducer = (
        parse_optional_value_reducer(baseline_spec.comparison_value_reducer)
        if baseline_spec.comparison_value_reducer is not None
        else None
    )
    shared_comparison_reducer = None
    if (
        target_comparison_reducer is not None
        and baseline_comparison_reducer is not None
        and target_comparison_reducer.canonical_name == baseline_comparison_reducer.canonical_name
    ):
        shared_comparison_reducer = target_comparison_reducer
    target_results = example_results_by_config[target_spec.config_id]
    baseline_results = example_results_by_config[baseline_spec.config_id]

    task_target, task_baseline = _aligned_metric_arrays(target_results, baseline_results, key="task_score")
    binary_scores = (
        task_target.size > 0
        and set(task_target.tolist()).issubset({0.0, 1.0})
        and set(task_baseline.tolist()).issubset({0.0, 1.0})
    )
    bootstrap_summary = (
        _paired_bootstrap_mean_delta(
            task_target,
            task_baseline,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        if binary_scores
        else None
    )

    comparison_row: dict[str, Any] = {
        "comparison_id": f"{target_spec.config_id}__minus__{baseline_spec.config_id}",
        "comparison_type": relation,
        "chunk_size": target_spec.chunk_size,
        "num_chunk_candidates": target_spec.num_chunk_candidates,
        "score_mode": target_spec.score_mode,
        "beta": target_spec.beta,
        "target_config_id": target_spec.config_id,
        "target_method_name": target_spec.method_name,
        "target_value_reducer": target_spec.value_reducer,
        "target_value_reducer_kind": None if target_reducer is None else target_reducer.kind,
        "target_value_reducer_tail_length": None if target_reducer is None else target_reducer.tail_length,
        "target_value_reducer_alpha": None if target_reducer is None else target_reducer.alpha,
        "baseline_config_id": baseline_spec.config_id,
        "baseline_method_name": baseline_spec.method_name,
        "baseline_value_reducer": baseline_spec.value_reducer,
        "baseline_value_reducer_kind": None if baseline_reducer is None else baseline_reducer.kind,
        "baseline_value_reducer_tail_length": None if baseline_reducer is None else baseline_reducer.tail_length,
        "baseline_value_reducer_alpha": None if baseline_reducer is None else baseline_reducer.alpha,
        "target_comparison_value_reducer": (
            None if target_comparison_reducer is None else target_comparison_reducer.canonical_name
        ),
        "baseline_comparison_value_reducer": (
            None if baseline_comparison_reducer is None else baseline_comparison_reducer.canonical_name
        ),
        "shared_comparison_value_reducer": (
            None if shared_comparison_reducer is None else shared_comparison_reducer.canonical_name
        ),
        "num_aligned_examples": int(task_target.size),
        "delta_mean_task_score": _paired_metric_delta(target_results, baseline_results, key="task_score"),
        "delta_mean_accuracy": _paired_metric_delta(target_results, baseline_results, key="task_score")
        if binary_scores
        else None,
        "delta_mean_response_length": _paired_metric_delta(target_results, baseline_results, key="response_length"),
        "delta_mean_selected_chunk_end_value": _paired_metric_delta(
            target_results,
            baseline_results,
            key="mean_selected_chunk_end_value",
        ),
        "delta_mean_selected_chunk_reducer_value": _paired_metric_delta(
            target_results,
            baseline_results,
            key="mean_selected_chunk_reducer_value",
        ),
        "delta_mean_fraction_chunk_decisions_different_from_endvalue_winner": _paired_metric_delta(
            target_results,
            baseline_results,
            key="fraction_chunk_decisions_different_from_endvalue_winner",
        ),
        "delta_mean_fraction_chunk_decisions_different_from_comparison_winner": (
            _paired_metric_delta(
                target_results,
                baseline_results,
                key="fraction_chunk_decisions_different_from_comparison_winner",
            )
            if shared_comparison_reducer is not None
            else None
        ),
    }
    if target_reducer is not None and target_reducer.tail_length is not None:
        tail_metric_key = f"mean_selected_chunk_tail_mean_h{target_reducer.tail_length}"
        comparison_row[f"delta_{tail_metric_key}"] = _paired_metric_delta(
            target_results,
            baseline_results,
            key=tail_metric_key,
        )
    if bootstrap_summary is not None:
        comparison_row["paired_bootstrap_accuracy_delta_mean"] = bootstrap_summary["mean_delta"]
        comparison_row["paired_bootstrap_accuracy_delta_ci_lower"] = bootstrap_summary["ci_lower"]
        comparison_row["paired_bootstrap_accuracy_delta_ci_upper"] = bootstrap_summary["ci_upper"]
        comparison_row["paired_bootstrap_accuracy_delta_samples"] = bootstrap_summary["bootstrap_samples"]
    else:
        comparison_row["paired_bootstrap_accuracy_delta_mean"] = None
        comparison_row["paired_bootstrap_accuracy_delta_ci_lower"] = None
        comparison_row["paired_bootstrap_accuracy_delta_ci_upper"] = None
        comparison_row["paired_bootstrap_accuracy_delta_samples"] = None
    return comparison_row


def build_reducer_comparisons(
    *,
    run_specs: Sequence[ChunkRunSpec],
    example_results_by_config: dict[str, list[dict[str, Any]]],
    bootstrap_seed: int,
    bootstrap_samples: int = DEFAULT_COMPARISON_BOOTSTRAP_SAMPLES,
) -> list[dict[str, Any]]:
    grouped_specs: dict[tuple[Any, ...], list[ChunkRunSpec]] = {}
    for spec in run_specs:
        if not spec.is_chunk_method or spec.value_reducer is None:
            continue
        group_key = (
            spec.score_mode,
            spec.chunk_size,
            spec.num_chunk_candidates,
            spec.beta,
            spec.actor_sampling_mode,
            spec.actor_temperature,
            spec.actor_top_p,
            spec.actor_top_k,
        )
        grouped_specs.setdefault(group_key, []).append(spec)

    comparisons: list[dict[str, Any]] = []
    for group_index, group_specs in enumerate(sorted(grouped_specs.values(), key=lambda specs: specs[0].config_id)):
        end_baseline = None
        tail_mean_by_length: dict[int, ChunkRunSpec] = {}
        for spec in group_specs:
            reducer_spec = parse_value_reducer(spec.value_reducer) if spec.value_reducer is not None else None
            if reducer_spec is None:
                continue
            if reducer_spec.kind == "end":
                end_baseline = spec
            elif reducer_spec.kind == "tail_mean" and reducer_spec.tail_length is not None:
                tail_mean_by_length[reducer_spec.tail_length] = spec

        for spec in sorted(group_specs, key=lambda item: item.config_id):
            reducer_spec = parse_value_reducer(spec.value_reducer) if spec.value_reducer is not None else None
            if reducer_spec is None or reducer_spec.kind not in {"tail_mean", "tail_exp"}:
                continue

            if end_baseline is not None:
                comparisons.append(
                    _build_single_reducer_comparison(
                        target_spec=spec,
                        baseline_spec=end_baseline,
                        relation="vs_endvalue_baseline",
                        example_results_by_config=example_results_by_config,
                        bootstrap_seed=bootstrap_seed + group_index * 1_000 + len(comparisons) * 31 + 7,
                        bootstrap_samples=bootstrap_samples,
                    )
                )

            if (
                reducer_spec.kind == "tail_exp"
                and reducer_spec.tail_length is not None
                and reducer_spec.tail_length in tail_mean_by_length
            ):
                comparisons.append(
                    _build_single_reducer_comparison(
                        target_spec=spec,
                        baseline_spec=tail_mean_by_length[reducer_spec.tail_length],
                        relation="vs_tailmean_same_h_baseline",
                        example_results_by_config=example_results_by_config,
                        bootstrap_seed=bootstrap_seed + group_index * 1_000 + len(comparisons) * 31 + 17,
                        bootstrap_samples=bootstrap_samples,
                    )
                )
    return comparisons


def _write_output_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    run_specs: Sequence[ChunkRunSpec],
    aggregate_rows: Sequence[dict[str, Any]],
) -> None:
    actor_only_row = next((row for row in aggregate_rows if row["method_name"] == "actor_only_sample"), None)
    chunk_actor_only_rows = [row for row in aggregate_rows if row["method_name"] == "chunk_rerank_actor_only"]
    uncertainty_rows = [row for row in aggregate_rows if row["method_name"] == "chunk_rerank_uncertainty_meanentropy"]
    critic_only_end_rows = [
        row
        for row in aggregate_rows
        if row["score_mode"] == "critic_only" and row.get("value_reducer") == "end"
    ]
    actor_plus_critic_rows = [row for row in aggregate_rows if row["score_mode"] == "actor_plus_critic"]

    best_actor_plus_critic = None
    if actor_plus_critic_rows:
        best_actor_plus_critic = max(
            actor_plus_critic_rows,
            key=lambda row: float(row["mean_task_score"]) if row["mean_task_score"] is not None else float("-inf"),
        )

    lines = [
        "# Chunk-Level Guidance Experiment",
        "",
        "This experiment treats a chunk as a contiguous block of generated tokens that is proposed and committed as a unit.",
        "",
        "At each chunk decision:",
        f"- sample K candidate chunks from the frozen actor, with K in `{sorted(set(spec.num_chunk_candidates for spec in run_specs if spec.num_chunk_candidates is not None))}`",
        f"- each chunk rolls out up to m tokens, with m in `{sorted(set(spec.chunk_size for spec in run_specs if spec.chunk_size is not None))}`",
        "- reuse the same candidate bank for every selector at that decoding state",
        "- score the candidates by actor log-prob only, mean actor entropy, actor+critic, or critic only depending on the config",
        "- if actor+critic is used, the score is zscore(chunk_logprob) + beta * zscore(chunk_value) within the candidate set",
        "- chunk uncertainty is the mean token entropy over the realized chunk, computed from the raw actor logits before sampling",
        "",
        (
            "Chunk value reducers support the end-of-chunk value, the legacy whole-chunk mean, "
            "tail means (`tail_mean_h*`), and exponentially weighted tail reducers (`tail_exp_h*_a*`)."
        ),
        "",
        "No training is performed in this experiment.",
        "",
        "## Run Config",
        f"- Dataset: `{args.dataset_path}`",
        f"- Execution backend: `{'ray' if _resolve_ray_address(args.ray_address) is not None else 'local'}`",
        f"- Ray address: `{_resolve_ray_address(args.ray_address)}`",
        f"- Chunk sizes: `{args.chunk_sizes}`",
        f"- Candidate counts: `{args.num_chunk_candidates_values}`",
        f"- Betas: `{args.betas}`",
        f"- Value reducers: `{args.value_reducers}`",
        f"- Explicit comparison reducer override: `{args.comparison_value_reducer}`",
        f"- Auto comparison tail h override: `{args.comparison_tail_h}`",
        f"- Auto comparison tail-exp alpha: `{args.comparison_tail_exp_alpha}`",
        f"- Include critic-only: `{args.include_critic_only}`",
        f"- Include uncertainty-only: `{args.include_uncertainty_only}`",
        f"- Critic model disabled: `{args.disable_critic_model}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Quick Read",
    ]
    if actor_only_row is not None:
        lines.append(f"- Ordinary actor-only sampling mean task score: `{actor_only_row['mean_task_score']:.6f}`")
    for row in chunk_actor_only_rows:
        lines.append(
            f"- Chunk actor-only m={row['chunk_size']} K={row['num_chunk_candidates']}: "
            f"`{row['mean_task_score']:.6f}`"
        )
    for row in uncertainty_rows:
        lines.append(
            f"- Chunk uncertainty m={row['chunk_size']} K={row['num_chunk_candidates']}: "
            f"`{row['mean_task_score']:.6f}`"
        )
    for row in critic_only_end_rows:
        lines.append(
            f"- Chunk critic end-value m={row['chunk_size']} K={row['num_chunk_candidates']}: "
            f"`{row['mean_task_score']:.6f}`"
        )
    if best_actor_plus_critic is not None:
        lines.append(
            f"- Best chunk actor+critic config: `{best_actor_plus_critic['config_id']}` "
            f"with mean task score `{best_actor_plus_critic['mean_task_score']:.6f}`"
        )

    lines.extend(
        [
            "",
            "## Files",
            "- `summary_metrics.json`",
            "- `main_results.csv`",
            "- `per_example_results.jsonl`",
            "- `chunk_decision_results.jsonl`",
            "- `reducer_comparisons.json` (when comparison rows are available)",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.chunk_sizes and any(value <= 0 for value in args.chunk_sizes):
        raise ValueError(f"All chunk sizes must be > 0, got {args.chunk_sizes}")
    if args.num_chunk_candidates_values and any(value <= 0 for value in args.num_chunk_candidates_values):
        raise ValueError(f"All num_chunk_candidates values must be > 0, got {args.num_chunk_candidates_values}")
    if args.comparison_tail_h is not None and args.comparison_tail_h <= 0:
        raise ValueError(f"comparison_tail_h must be > 0, got {args.comparison_tail_h}")
    if args.comparison_tail_exp_alpha is not None and not (0.0 < args.comparison_tail_exp_alpha < 1.0):
        raise ValueError(
            "comparison_tail_exp_alpha must be strictly between 0 and 1, "
            f"got {args.comparison_tail_exp_alpha}"
        )
    if args.ray_num_cpus_per_worker <= 0:
        raise ValueError(f"ray_num_cpus_per_worker must be > 0, got {args.ray_num_cpus_per_worker}")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
    critic_checkpoint_dir = Path(args.critic_checkpoint_dir).resolve()
    actor_hf_dir = ensure_merged_component_checkpoint(
        actor_checkpoint_dir,
        component="actor",
        merged_root=Path(args.actor_merged_root).resolve() if args.actor_merged_root else None,
        hf_source_dir=Path(args.actor_hf_source_dir).resolve() if args.actor_hf_source_dir else None,
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
        raise ValueError("No evaluation examples were loaded. Check dataset path and slicing arguments.")

    run_specs = build_run_specs(args)
    if not run_specs:
        raise ValueError("No run specifications were built from the provided arguments.")
    if args.disable_critic_model:
        invalid_specs = [spec.config_id for spec in run_specs if _spec_requires_critic(spec)]
        if invalid_specs:
            raise ValueError(
                "Critic loading is disabled, but the following configs still require critic values or diagnostics: "
                + ", ".join(invalid_specs)
            )

    critic_hf_dir = (
        None
        if args.disable_critic_model
        else ensure_merged_component_checkpoint(
            critic_checkpoint_dir,
            component="critic",
            merged_root=Path(args.critic_merged_root).resolve() if args.critic_merged_root else None,
            hf_source_dir=Path(args.critic_hf_source_dir).resolve() if args.critic_hf_source_dir else None,
            skip_merge=args.skip_merge,
        )
    )

    ray_address = _resolve_ray_address(args.ray_address)
    execution_backend = "ray" if ray_address is not None else "local"
    worker_pairs = parse_worker_pairs(
        args.worker_pairs,
        actor_device=args.actor_device,
        critic_device=args.critic_device,
        default_device=args.device,
    )
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
            worker_pairs=worker_pairs,
            ray_nodes=ray_nodes,
        )
    else:
        worker_assignments = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)
    multi_worker_enabled = len(worker_assignments) > 1

    if ray_address is not None:
        example_results_by_config, worker_summaries = run_ray_multi_worker(
            output_dir=output_dir,
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            run_specs=run_specs,
            worker_assignments=worker_assignments,
            dtype_name=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            normalization_eps=args.normalization_eps,
            use_actor_cache=not args.disable_actor_cache,
            debug_full_chunk_candidates=args.debug_full_chunk_candidates,
            seed=args.seed,
            ray_num_cpus_per_worker=args.ray_num_cpus_per_worker,
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
                if start_time_sec is not None and end_time_sec is not None:
                    start_times.append(float(start_time_sec))
                    end_times.append(float(end_time_sec))
            per_config_wall_times[spec.config_id] = (
                (max(end_times) - min(start_times)) if start_times and end_times else 0.0
            )
    elif multi_worker_enabled:
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
            debug_full_chunk_candidates=args.debug_full_chunk_candidates,
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
                if start_time_sec is not None and end_time_sec is not None:
                    start_times.append(float(start_time_sec))
                    end_times.append(float(end_time_sec))
            per_config_wall_times[spec.config_id] = (
                (max(end_times) - min(start_times)) if start_times and end_times else 0.0
            )
    else:
        actor_device = resolve_device(worker_pairs[0][0])
        critic_device = (
            resolve_device(worker_pairs[0][1])
            if worker_pairs[0][1] and not args.disable_critic_model
            else None
        )
        actor = load_actor_model(
            actor_hf_dir,
            dtype=dtype,
            device=actor_device,
            trust_remote_code=args.trust_remote_code,
        )
        critic = (
            load_critic_model(
                critic_hf_dir,
                dtype=dtype,
                device=critic_device,
                trust_remote_code=args.trust_remote_code,
            )
            if critic_hf_dir is not None and critic_device is not None
            else None
        )

        per_example_path = output_dir / "per_example_results.jsonl"
        chunk_decision_path = output_dir / "chunk_decision_results.jsonl"
        example_results_by_config = {spec.config_id: [] for spec in run_specs}
        per_config_wall_times: dict[str, float] = {}
        with per_example_path.open("w", encoding="utf-8") as per_example_file, chunk_decision_path.open(
            "w",
            encoding="utf-8",
        ) as chunk_decision_file, tqdm(
            total=len(examples) * len(run_specs),
            desc="chunk_guidance_eval",
            unit="task",
            dynamic_ncols=True,
        ) as progress_bar:
            for spec in run_specs:
                config_start_time = time.perf_counter()
                for example in examples:
                    progress_bar.set_postfix_str(f"config={spec.config_id} example_id={example.example_id}")
                    artifacts = process_example_for_spec(
                        actor=actor,
                        critic=critic,
                        tokenizer=tokenizer,
                        example=example,
                        spec=spec,
                        actor_device=actor_device,
                        critic_device=critic_device,
                        max_prompt_length=args.max_prompt_length,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        normalization_eps=args.normalization_eps,
                        seed=args.seed,
                        use_actor_cache=not args.disable_actor_cache,
                        debug_full_chunk_candidates=args.debug_full_chunk_candidates,
                    )
                    per_example_file.write(_json_line(artifacts.example_result))
                    for chunk_decision_result in artifacts.chunk_decision_results:
                        chunk_decision_file.write(_json_line(chunk_decision_result))
                    example_results_by_config[spec.config_id].append(artifacts.example_result)
                    progress_bar.update(1)
                per_config_wall_times[spec.config_id] = time.perf_counter() - config_start_time
        worker_summaries = [
            {
                "worker_id": 0,
                "actor_device": str(actor_device),
                "critic_device": None if critic_device is None else str(critic_device),
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
    reducer_comparisons = build_reducer_comparisons(
        run_specs=run_specs,
        example_results_by_config=example_results_by_config,
        bootstrap_seed=args.seed,
    )

    csv_path = output_dir / "main_results.csv"
    base_fieldnames = [
        "config_id",
        "method_name",
        "score_mode",
        "chunk_size",
        "num_chunk_candidates",
        "beta",
        "value_reducer",
        "value_reducer_kind",
        "value_reducer_tail_length",
        "value_reducer_alpha",
        "comparison_value_reducer",
        "comparison_value_reducer_kind",
        "comparison_value_reducer_tail_length",
        "comparison_value_reducer_alpha",
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
        "mean_num_chunk_decisions",
        "mean_realized_chunk_length",
        "mean_selected_chunk_logprob",
        "mean_selected_chunk_value",
        "mean_selected_chunk_reducer_value",
        "mean_selected_chunk_end_value",
        "mean_selected_chunk_mean_value",
        "mean_selected_chunk_uncertainty",
        "mean_selected_chunk_entropy_horizon_mean",
        "mean_selected_chunk_tail_mean_h2",
        "mean_selected_chunk_tail_mean_h4",
        "mean_selected_chunk_tail_mean_h8",
        "mean_selected_chunk_tail_mean_h16",
        "mean_fraction_chunk_decisions_different_from_actor_only_chunk_winner",
        "mean_fraction_chunk_decisions_different_from_endvalue_winner",
        "mean_fraction_chunk_decisions_different_from_uncertainty_winner",
        "mean_fraction_chunk_decisions_different_from_comparison_winner",
        "mean_selected_chunk_score_margin",
        "fraction_selected_chunks_with_eos",
        "total_generated_tokens",
        "sum_example_latency_sec",
        "wall_time_sec",
        "overall_tokens_per_second",
        "mean_tokens_per_second",
    ]
    dynamic_fieldnames = sorted({key for row in aggregate_rows for key in row.keys()} - set(base_fieldnames))
    fieldnames = base_fieldnames + dynamic_fieldnames
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    reducer_comparisons_path = output_dir / "reducer_comparisons.json"
    with reducer_comparisons_path.open("w", encoding="utf-8") as comparison_file:
        json.dump(reducer_comparisons, comparison_file, ensure_ascii=True, indent=2)

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "execution_backend": execution_backend,
        "actor_checkpoint_dir": str(actor_checkpoint_dir),
        "critic_checkpoint_dir": str(critic_checkpoint_dir),
        "merged_actor_dir": str(actor_hf_dir),
        "merged_critic_dir": None if critic_hf_dir is None else str(critic_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "output_dir": str(output_dir),
        "critic_model_disabled": bool(args.disable_critic_model),
        "multi_worker_enabled": multi_worker_enabled,
        "actor_device": None if actor_device is None else str(actor_device),
        "critic_device": None if critic_device is None else str(critic_device),
        "ray_address": ray_address,
        "ray_nodes": [asdict(node) for node in ray_nodes],
        "worker_pairs": [[actor, critic] for actor, critic in worker_pairs],
        "worker_assignments": worker_assignments_to_jsonable(worker_assignments),
        "worker_summaries": worker_summaries,
        "dtype": args.dtype,
        "eos_token_ids": list(eos_token_ids),
        "run_args": vars(args),
        "run_specs": [asdict(spec) for spec in run_specs],
        "aggregate_metrics": aggregate_rows,
        "reducer_comparisons": reducer_comparisons,
    }
    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)

    _write_output_readme(
        output_dir=output_dir,
        args=args,
        run_specs=run_specs,
        aggregate_rows=aggregate_rows,
    )
    return 0


_make_main_module_importable()

if __name__ == "__main__":
    raise SystemExit(main())
