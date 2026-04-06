from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from queue import Empty
import shutil
import subprocess
import time
import traceback
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
from value_decoding.multi_worker import (
    WorkerAssignment,
    build_worker_assignments,
    parse_worker_pairs,
    worker_assignments_to_jsonable,
)


DEFAULT_CHUNK_SIZES = (2, 4)
DEFAULT_NUM_CHUNK_CANDIDATES_VALUES = (2,)
DEFAULT_BETAS = (0.0, 0.05, 0.1, 0.25)


@dataclass(frozen=True)
class ChunkRunSpec:
    config_id: str
    method_name: str
    score_mode: str
    chunk_size: int | None = None
    num_chunk_candidates: int | None = None
    beta: float | None = None
    value_reducer: str | None = None
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
    end_value: float
    mean_value: float
    contains_eos: bool


@dataclass
class ChunkDecodeArtifacts:
    example_result: dict[str, Any]
    chunk_decision_results: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk-level guidance evaluation with a frozen actor and the new critic. "
            "Supports ordinary actor sampling, chunk actor-only reranking, and chunk actor+critic reranking."
        )
    )
    parser.add_argument("--actor_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the frozen actor.")
    parser.add_argument("--critic_checkpoint_dir", type=str, required=True, help="Checkpoint dir for the new critic.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for experiment artifacts.")
    parser.add_argument("--actor_merged_root", type=str, default=None, help="Optional merged HF root for actor.")
    parser.add_argument("--critic_merged_root", type=str, default=None, help="Optional merged HF root for critic.")
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
        choices=["end", "mean"],
        help="Value reducers for actor+critic chunk guidance. Default is the primary end-of-chunk reducer only.",
    )
    parser.add_argument("--include_critic_only", action="store_true", help="Add optional critic-only chunk rerank configs.")
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

    if not skip_actor_only_baselines:
        actor_only_spec = ChunkRunSpec(
            config_id="",
            method_name="actor_only_sample",
            score_mode="actor_only_sample",
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
                    actor_sampling_mode=args.actor_sampling_mode,
                    actor_temperature=args.actor_temperature,
                    actor_top_p=args.actor_top_p,
                    actor_top_k=args.actor_top_k,
                )
                if actor_only_chunk.config_id not in seen_config_ids:
                    specs.append(actor_only_chunk)
                    seen_config_ids.add(actor_only_chunk.config_id)

            for value_reducer in args.value_reducers:
                if not include_only_critic_only:
                    for beta in args.betas:
                        if beta <= 0.0:
                            continue
                        method_name = (
                            "chunk_rerank_newcritic_endvalue"
                            if value_reducer == "end"
                            else "chunk_rerank_newcritic_meanvalue"
                        )
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
                            value_reducer=value_reducer,
                            actor_sampling_mode=args.actor_sampling_mode,
                            actor_temperature=args.actor_temperature,
                            actor_top_p=args.actor_top_p,
                            actor_top_k=args.actor_top_k,
                        )
                        if spec.config_id not in seen_config_ids:
                            specs.append(spec)
                            seen_config_ids.add(spec.config_id)

                if include_critic_only:
                    method_name = (
                        "chunk_rerank_critic_only_endvalue"
                        if value_reducer == "end"
                        else "chunk_rerank_critic_only_meanvalue"
                    )
                    spec = ChunkRunSpec(
                        config_id=f"{method_name}__m{chunk_size}__k{num_chunk_candidates}",
                        method_name=method_name,
                        score_mode="critic_only",
                        chunk_size=chunk_size,
                        num_chunk_candidates=num_chunk_candidates,
                        beta=None,
                        value_reducer=value_reducer,
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
    critic_device: torch.device,
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
    chunk_logprob = 0.0
    contains_eos = False

    for _chunk_token_index in range(max_chunk_len):
        logits = actor_state.current_logits
        actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        chunk_logprob += float(actor_log_probs[0, token_id].item())
        chunk_token_ids.append(token_id)
        actor_state.append(token_id)
        if token_id in eos_token_ids:
            contains_eos = True
            break

    if not chunk_token_ids:
        raise RuntimeError("Chunk candidate generation produced zero tokens, which should be impossible.")

    full_sequence_ids = actor_state.sequence_ids.to(critic_device)
    values = critic_sequence_values(critic, full_sequence_ids)[0]
    chunk_values = values[prefix_length : prefix_length + len(chunk_token_ids)]
    if chunk_values.numel() != len(chunk_token_ids):
        raise RuntimeError("Chunk value extraction length mismatch.")

    return ChunkCandidate(
        candidate_index=candidate_index,
        chunk_token_ids=tuple(int(token_id) for token_id in chunk_token_ids),
        chunk_text=tokenizer.decode(chunk_token_ids, skip_special_tokens=True),
        chunk_length=len(chunk_token_ids),
        chunk_logprob=float(chunk_logprob),
        end_value=float(chunk_values[-1].item()),
        mean_value=float(chunk_values.mean().item()),
        contains_eos=contains_eos,
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
        "example_id": int(example.example_id),
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
        "mean_selected_chunk_end_value": None,
        "mean_selected_chunk_mean_value": None,
        "fraction_chunk_decisions_different_from_actor_only_chunk_winner": None,
        "mean_selected_chunk_score_margin": None,
        "fraction_selected_chunks_with_eos": None,
        "total_decoding_steps": response_length,
        "latency_sec": latency_sec,
        "tokens_per_second": (response_length / latency_sec) if latency_sec > 0 else None,
    }
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
    critic_device: torch.device,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    normalization_eps: float,
    seed: int,
    use_actor_cache: bool,
    debug_full_chunk_candidates: bool,
) -> ChunkDecodeArtifacts:
    if not spec.is_chunk_method or spec.chunk_size is None or spec.num_chunk_candidates is None:
        raise ValueError(f"Chunk run spec {spec.config_id} is missing chunk settings.")

    generated_token_ids: list[int] = []
    current_sequence_ids = prompt_ids
    chunk_decision_results: list[dict[str, Any]] = []

    selected_chunk_lengths: list[int] = []
    selected_chunk_logprobs: list[float] = []
    selected_chunk_end_values: list[float] = []
    selected_chunk_mean_values: list[float] = []
    selected_chunk_values: list[float] = []
    selected_chunk_score_margins: list[float] = []
    selected_chunks_with_eos: list[float] = []
    selected_diff_from_actor_only_flags: list[float] = []

    start_time = time.perf_counter()
    chunk_decision_index = 0
    eos_emitted = False

    while len(generated_token_ids) < max_new_tokens:
        remaining_tokens = max_new_tokens - len(generated_token_ids)
        max_chunk_len = min(spec.chunk_size, remaining_tokens)
        generated_length_before_chunk = len(generated_token_ids)

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
        raw_end_values = [candidate.end_value for candidate in candidates]
        raw_mean_values = [candidate.mean_value for candidate in candidates]
        normalized_logprobs = _zscore(raw_logprobs, eps=normalization_eps)
        normalized_end_values = _zscore(raw_end_values, eps=normalization_eps)
        normalized_mean_values = _zscore(raw_mean_values, eps=normalization_eps)

        actor_only_chunk_winner_index, actor_only_chunk_winner_ties, actor_only_chunk_winner_score = _select_argmax(raw_logprobs)

        if spec.score_mode == "actor_logprob_only":
            selection_scores = normalized_logprobs
            normalized_value_for_scoring = None
        elif spec.score_mode == "actor_plus_critic":
            if spec.beta is None or spec.value_reducer is None:
                raise ValueError(f"Spec {spec.config_id} requires beta and value_reducer.")
            normalized_value_for_scoring = normalized_end_values if spec.value_reducer == "end" else normalized_mean_values
            selection_scores = [
                float(normalized_logprobs[index] + float(spec.beta) * normalized_value_for_scoring[index])
                for index in range(len(candidates))
            ]
        elif spec.score_mode == "critic_only":
            if spec.value_reducer is None:
                raise ValueError(f"Spec {spec.config_id} requires value_reducer.")
            normalized_value_for_scoring = normalized_end_values if spec.value_reducer == "end" else normalized_mean_values
            selection_scores = [float(value) for value in normalized_value_for_scoring]
        else:
            raise ValueError(f"Unsupported score_mode: {spec.score_mode}")

        selected_candidate_index, selected_tied_indices, selected_score = _select_argmax(selection_scores)
        sorted_selection_scores = sorted(selection_scores, reverse=True)
        selected_score_margin = (
            float(sorted_selection_scores[0] - sorted_selection_scores[1]) if len(sorted_selection_scores) > 1 else None
        )
        selected_candidate = candidates[selected_candidate_index]

        selected_chunk_lengths.append(selected_candidate.chunk_length)
        selected_chunk_logprobs.append(selected_candidate.chunk_logprob)
        selected_chunk_end_values.append(selected_candidate.end_value)
        selected_chunk_mean_values.append(selected_candidate.mean_value)
        if spec.score_mode in {"actor_plus_critic", "critic_only"} and spec.value_reducer is not None:
            selected_chunk_values.append(
                selected_candidate.end_value if spec.value_reducer == "end" else selected_candidate.mean_value
            )
        if selected_score_margin is not None:
            selected_chunk_score_margins.append(selected_score_margin)
        selected_chunks_with_eos.append(1.0 if selected_candidate.contains_eos else 0.0)
        selected_diff_from_actor_only_flags.append(
            1.0 if selected_candidate_index != actor_only_chunk_winner_index else 0.0
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
            "example_id": int(example.example_id),
            "chunk_decision_index": chunk_decision_index,
            "generated_length_before_chunk": generated_length_before_chunk,
            "candidate_chunk_token_ids": [list(candidate.chunk_token_ids) for candidate in candidates],
            "candidate_chunk_texts": [candidate.chunk_text for candidate in candidates],
            "candidate_chunk_lengths": [candidate.chunk_length for candidate in candidates],
            "candidate_chunk_logprobs": raw_logprobs,
            "candidate_chunk_end_values": raw_end_values,
            "candidate_chunk_mean_values": raw_mean_values,
            "candidate_chunk_contains_eos": [candidate.contains_eos for candidate in candidates],
            "actor_only_chunk_winner_index": actor_only_chunk_winner_index,
            "actor_only_chunk_winner_tied_indices": actor_only_chunk_winner_ties,
            "actor_only_chunk_winner_logprob": actor_only_chunk_winner_score,
            "selected_chunk_index": selected_candidate_index,
            "selected_chunk_tied_indices": selected_tied_indices,
            "selected_chunk_token_ids": list(selected_candidate.chunk_token_ids),
            "selected_chunk_text": selected_candidate.chunk_text,
            "selected_chunk_len": selected_candidate.chunk_length,
            "selected_chunk_logprob": selected_candidate.chunk_logprob,
            "selected_chunk_end_value": selected_candidate.end_value,
            "selected_chunk_mean_value": selected_candidate.mean_value,
            "selected_chunk_value": (
                selected_candidate.end_value if spec.value_reducer == "end" else (
                    selected_candidate.mean_value if spec.value_reducer == "mean" else None
                )
            ),
            "selected_chunk_contains_eos": selected_candidate.contains_eos,
            "selected_chunk_selection_score": selected_score,
            "selected_chunk_score_margin": selected_score_margin,
            "selected_differs_from_actor_only_chunk_winner": selected_candidate_index != actor_only_chunk_winner_index,
        }
        if debug_full_chunk_candidates:
            chunk_decision_result.update(
                {
                    "candidate_normalized_chunk_logprobs": normalized_logprobs,
                    "candidate_normalized_chunk_end_values": normalized_end_values,
                    "candidate_normalized_chunk_mean_values": normalized_mean_values,
                    "candidate_selection_scores": selection_scores,
                }
            )

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
        "example_id": int(example.example_id),
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
        "mean_selected_chunk_end_value": _mean(selected_chunk_end_values),
        "mean_selected_chunk_mean_value": _mean(selected_chunk_mean_values),
        "fraction_chunk_decisions_different_from_actor_only_chunk_winner": _mean(selected_diff_from_actor_only_flags),
        "mean_selected_chunk_score_margin": _mean(selected_chunk_score_margins),
        "fraction_selected_chunks_with_eos": _mean(selected_chunks_with_eos),
        "total_decoding_steps": response_length,
        "latency_sec": latency_sec,
        "tokens_per_second": (response_length / latency_sec) if latency_sec > 0 else None,
    }
    return ChunkDecodeArtifacts(example_result=example_result, chunk_decision_results=chunk_decision_results)


def process_example_for_spec(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    spec: ChunkRunSpec,
    actor_device: torch.device,
    critic_device: torch.device,
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
    critic_hf_dir: str,
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

        summary_payload = {
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
            json.dump(summary_payload, summary_file, ensure_ascii=True, indent=2)
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


def run_multi_worker(
    *,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path,
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
                "debug_full_chunk_candidates": debug_full_chunk_candidates,
                "seed": seed,
                "worker_root": str(worker_root),
                "progress_queue": progress_queue,
            },
            name=f"chunk_guidance_worker_{assignment.worker_id}",
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

    with tqdm(total=total_tasks, desc="chunk_guidance_eval", unit="task", dynamic_ncols=True) as progress_bar:
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
                raise RuntimeError(
                    f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.\n"
                    f"{error_path.read_text(encoding='utf-8')}"
                )
            raise RuntimeError(f"Worker {assignment.worker_id} failed with exit code {process.exitcode}.")

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


def aggregate_results(
    spec: ChunkRunSpec,
    example_results: list[dict[str, Any]],
    *,
    wall_time_sec: float | None = None,
) -> dict[str, Any]:
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
        diff_from_actor_only = [
            float(result["fraction_chunk_decisions_different_from_actor_only_chunk_winner"])
            for result in example_results
            if result["fraction_chunk_decisions_different_from_actor_only_chunk_winner"] is not None
        ]
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
        diff_from_actor_only = []
        selected_chunk_score_margins = []
        fraction_selected_chunks_with_eos = []
        mean_num_chunk_decisions = None

    binary_scores = set(task_scores).issubset({0.0, 1.0})
    total_latency = sum(latencies)
    return {
        "config_id": spec.config_id,
        "method_name": spec.method_name,
        "score_mode": spec.score_mode,
        "chunk_size": spec.chunk_size,
        "num_chunk_candidates": spec.num_chunk_candidates,
        "beta": spec.beta,
        "value_reducer": spec.value_reducer,
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
        "mean_selected_chunk_end_value": _mean(selected_chunk_end_values) if spec.is_chunk_method else None,
        "mean_selected_chunk_mean_value": _mean(selected_chunk_mean_values) if spec.is_chunk_method else None,
        "mean_fraction_chunk_decisions_different_from_actor_only_chunk_winner": (
            _mean(diff_from_actor_only) if spec.is_chunk_method else None
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


def _write_output_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    run_specs: Sequence[ChunkRunSpec],
    aggregate_rows: Sequence[dict[str, Any]],
) -> None:
    actor_only_row = next((row for row in aggregate_rows if row["method_name"] == "actor_only_sample"), None)
    chunk_actor_only_rows = [row for row in aggregate_rows if row["method_name"] == "chunk_rerank_actor_only"]
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
        "- score the candidates by actor log-prob only, actor+critic, or critic only depending on the config",
        "- if actor+critic is used, the score is zscore(chunk_logprob) + beta * zscore(chunk_value) within the candidate set",
        "",
        "The primary chunk value reducer is end-of-chunk value. Mean-of-chunk value is available as an ablation.",
        "",
        "No training is performed in this experiment.",
        "",
        "## Run Config",
        f"- Dataset: `{args.dataset_path}`",
        f"- Chunk sizes: `{args.chunk_sizes}`",
        f"- Candidate counts: `{args.num_chunk_candidates_values}`",
        f"- Betas: `{args.betas}`",
        f"- Value reducers: `{args.value_reducers}`",
        f"- Include critic-only: `{args.include_critic_only}`",
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
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.chunk_sizes and any(value <= 0 for value in args.chunk_sizes):
        raise ValueError(f"All chunk sizes must be > 0, got {args.chunk_sizes}")
    if args.num_chunk_candidates_values and any(value <= 0 for value in args.num_chunk_candidates_values):
        raise ValueError(f"All num_chunk_candidates values must be > 0, got {args.num_chunk_candidates_values}")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
    critic_checkpoint_dir = Path(args.critic_checkpoint_dir).resolve()
    actor_hf_dir = ensure_merged_component_checkpoint(
        actor_checkpoint_dir,
        component="actor",
        merged_root=Path(args.actor_merged_root).resolve() if args.actor_merged_root else None,
        skip_merge=args.skip_merge,
    )
    critic_hf_dir = ensure_merged_component_checkpoint(
        critic_checkpoint_dir,
        component="critic",
        merged_root=Path(args.critic_merged_root).resolve() if args.critic_merged_root else None,
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

    worker_pairs = parse_worker_pairs(
        args.worker_pairs,
        actor_device=args.actor_device,
        critic_device=args.critic_device,
        default_device=args.device,
    )
    worker_assignments = build_worker_assignments(num_examples=len(examples), worker_pairs=worker_pairs)
    multi_worker_enabled = len(worker_assignments) > 1

    if multi_worker_enabled:
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
        critic_device = resolve_device(worker_pairs[0][1]) if worker_pairs[0][1] else actor_device
        actor = load_actor_model(
            actor_hf_dir,
            dtype=dtype,
            device=actor_device,
            trust_remote_code=args.trust_remote_code,
        )
        critic = load_critic_model(
            critic_hf_dir,
            dtype=dtype,
            device=critic_device,
            trust_remote_code=args.trust_remote_code,
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
                "critic_device": str(critic_device),
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

    csv_path = output_dir / "main_results.csv"
    fieldnames = [
        "config_id",
        "method_name",
        "score_mode",
        "chunk_size",
        "num_chunk_candidates",
        "beta",
        "value_reducer",
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
        "mean_selected_chunk_end_value",
        "mean_selected_chunk_mean_value",
        "mean_fraction_chunk_decisions_different_from_actor_only_chunk_winner",
        "mean_selected_chunk_score_margin",
        "fraction_selected_chunks_with_eos",
        "total_generated_tokens",
        "sum_example_latency_sec",
        "wall_time_sec",
        "overall_tokens_per_second",
        "mean_tokens_per_second",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "actor_checkpoint_dir": str(actor_checkpoint_dir),
        "critic_checkpoint_dir": str(critic_checkpoint_dir),
        "merged_actor_dir": str(actor_hf_dir),
        "merged_critic_dir": str(critic_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "output_dir": str(output_dir),
        "multi_worker_enabled": multi_worker_enabled,
        "actor_device": None if actor_device is None else str(actor_device),
        "critic_device": None if critic_device is None else str(critic_device),
        "worker_pairs": [[actor, critic] for actor, critic in worker_pairs],
        "worker_assignments": worker_assignments_to_jsonable(worker_assignments),
        "worker_summaries": worker_summaries,
        "dtype": args.dtype,
        "eos_token_ids": list(eos_token_ids),
        "run_args": vars(args),
        "run_specs": [asdict(spec) for spec in run_specs],
        "aggregate_metrics": aggregate_rows,
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


if __name__ == "__main__":
    raise SystemExit(main())
