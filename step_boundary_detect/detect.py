from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Sequence

import numpy as np
import torch

from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
    has_complete_hf_checkpoint,
    load_actor_model,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.data import ExampleRecord, load_examples, score_response
from value_decoding.decoding import ActorSamplingMode, ActorStepper, critic_last_token_values
from value_decoding.decoding import sample_token_from_actor, set_decode_seed


@dataclass(frozen=True)
class BoundaryCandidate:
    candidate_index: int
    token_ids: tuple[int, ...]
    text: str
    entropies: tuple[float, ...]
    token_logprobs: tuple[float, ...]
    chunk_logprob: float
    value_score: float | None
    stop_reason: str
    boundary_index: int | None
    boundary_entropy: float | None
    boundary_threshold: float | None
    boundary_reference_history_len: int | None
    contains_eos: bool


@dataclass(frozen=True)
class BoundaryDecodeArtifacts:
    example_result: dict[str, Any]
    chunk_results: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick entropy-boundary value-guided chunk decoding inspection. "
            "Finds correctly answered examples and writes segmented trajectories."
        )
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800",
    )
    parser.add_argument(
        "--critic_checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Optional separate critic checkpoint. If unset, VERL checkpoints use checkpoint_dir/critic. "
            "Plain Hugging Face actor directories run without critic value guidance."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/shuozhe/verl/step_boundary_detect/output",
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_scan_examples", type=int, default=128)
    parser.add_argument("--num_correct", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--min_chunk_length", type=int, default=4)
    parser.add_argument("--max_chunk_length", type=int, default=32)
    parser.add_argument("--entropy_percentile", type=float, default=0.9)
    parser.add_argument("--warmup_size", type=int, default=32)
    parser.add_argument(
        "--bootstrap_current_chunk_entropy",
        action="store_true",
        help=(
            "Let boundary detection use previous non-boundary entropies from the current candidate while "
            "H_ctx is still warming up. Without this, the first chunk cannot be entropy-split before m_max."
        ),
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--actor_device", type=str, default=None)
    parser.add_argument("--critic_device", type=str, default=None)
    parser.add_argument(
        "--worker_pairs",
        nargs="+",
        default=None,
        help=(
            "Optional multi-process GPU layout. Each entry is either 'cuda:N' to put actor+critic "
            "on one GPU, or 'actor_device,critic_device' to split a worker across two GPUs. "
            "Examples: --worker_pairs cuda:0 cuda:1 cuda:2 cuda:3, or "
            "--worker_pairs cuda:0,cuda:1 cuda:2,cuda:3."
        ),
    )
    parser.add_argument(
        "--actor_sampling_mode",
        type=str,
        default=ActorSamplingMode.SAMPLE.value,
        choices=[mode.value for mode in ActorSamplingMode],
    )
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--actor_top_p", type=float, default=1.0)
    parser.add_argument("--actor_top_k", type=int, default=0)
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="auto",
        choices=["auto", "value", "actor_logprob"],
        help=(
            "How to choose among K chunks. 'auto' uses critic value when a critic is loaded and "
            "falls back to actor chunk logprob for plain actor-only Hugging Face models."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument(
        "--disable_critic_model",
        action="store_true",
        help="Do not load a critic; select chunks by actor logprob unless --selection_mode value is requested.",
    )
    parser.add_argument("--disable_actor_cache", action="store_true")
    return parser.parse_args()


def _entropy_from_logits(logits: torch.Tensor) -> float:
    logits_fp32 = logits.float()
    log_probs = torch.log_softmax(logits_fp32, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.item())


def _candidate_seed(
    base_seed: int,
    *,
    example_id: int,
    chunk_index: int,
    candidate_index: int,
) -> int:
    return int(
        base_seed
        + (example_id + 1) * 1_000_003
        + chunk_index * 1_000_000_007
        + candidate_index * 97_003
    )


def _score_candidate_value(
    *,
    critic,
    prefix_ids: torch.Tensor,
    token_ids: Sequence[int],
    critic_device: torch.device | None,
) -> float | None:
    if not token_ids:
        raise ValueError("Cannot score an empty candidate chunk.")
    if critic is None:
        return None
    if critic_device is None:
        raise ValueError("critic_device must be provided when critic scoring is enabled.")

    candidate_tensor = torch.tensor([list(token_ids)], device=critic_device, dtype=torch.long)
    full_ids = torch.cat([prefix_ids.to(critic_device), candidate_tensor], dim=1)
    return float(critic_last_token_values(critic, full_ids)[0].item())


def _rollout_candidate(
    *,
    actor,
    critic,
    tokenizer,
    prefix_ids: torch.Tensor,
    actor_device: torch.device,
    critic_device: torch.device | None,
    candidate_index: int,
    max_chunk_len: int,
    min_chunk_length: int,
    entropy_history: Sequence[float],
    entropy_percentile: float,
    warmup_size: int,
    bootstrap_current_chunk_entropy: bool,
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
) -> BoundaryCandidate:
    if max_chunk_len <= 0:
        raise ValueError(f"max_chunk_len must be > 0, got {max_chunk_len}")

    set_decode_seed(seed)
    actor_state = ActorStepper(actor, prefix_ids.to(actor_device), use_cache=use_actor_cache)
    token_ids: list[int] = []
    entropies: list[float] = []
    token_logprobs: list[float] = []
    chunk_logprob = 0.0
    stop_reason = "max_length"
    boundary_index: int | None = None
    boundary_entropy: float | None = None
    boundary_threshold: float | None = None
    boundary_reference_history_len: int | None = None
    contains_eos = False
    candidate_reference_entropies: list[float] = []

    for step_index in range(1, max_chunk_len + 1):
        logits = actor_state.current_logits
        actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
        entropy = _entropy_from_logits(logits)
        token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_logprob = float(actor_log_probs[0, token_id].item())

        token_ids.append(int(token_id))
        entropies.append(entropy)
        token_logprobs.append(token_logprob)
        chunk_logprob += token_logprob
        actor_state.append(token_id)

        if token_id in eos_token_ids:
            contains_eos = True
            stop_reason = "eos"
            break

        if bootstrap_current_chunk_entropy:
            reference_history = [*entropy_history, *candidate_reference_entropies]
        else:
            reference_history = entropy_history

        if step_index >= min_chunk_length and len(reference_history) >= warmup_size:
            threshold = float(np.quantile(np.asarray(reference_history, dtype=np.float64), entropy_percentile))
            if entropy > threshold:
                stop_reason = "entropy_boundary"
                boundary_index = step_index
                boundary_entropy = entropy
                boundary_threshold = threshold
                boundary_reference_history_len = len(reference_history)
                break
        candidate_reference_entropies.append(float(entropy))

    value_score = _score_candidate_value(
        critic=critic,
        prefix_ids=prefix_ids,
        token_ids=token_ids,
        critic_device=critic_device,
    )
    return BoundaryCandidate(
        candidate_index=int(candidate_index),
        token_ids=tuple(token_ids),
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        entropies=tuple(float(value) for value in entropies),
        token_logprobs=tuple(float(value) for value in token_logprobs),
        chunk_logprob=float(chunk_logprob),
        value_score=value_score,
        stop_reason=stop_reason,
        boundary_index=boundary_index,
        boundary_entropy=boundary_entropy,
        boundary_threshold=boundary_threshold,
        boundary_reference_history_len=boundary_reference_history_len,
        contains_eos=contains_eos,
    )


def _resolve_selection_mode(candidates: Sequence[BoundaryCandidate], selection_mode: str) -> str:
    if selection_mode == "auto":
        if all(candidate.value_score is not None for candidate in candidates):
            return "value"
        return "actor_logprob"
    if selection_mode == "value" and any(candidate.value_score is None for candidate in candidates):
        raise ValueError("--selection_mode value requires a critic, but no critic value scores are available.")
    return selection_mode


def _candidate_selection_score(candidate: BoundaryCandidate, selection_mode: str) -> float:
    if selection_mode == "value":
        if candidate.value_score is None:
            raise ValueError("Cannot score candidate by value without a critic value score.")
        return float(candidate.value_score)
    if selection_mode == "actor_logprob":
        return float(candidate.chunk_logprob)
    raise ValueError(f"Unsupported effective selection mode: {selection_mode}")


def _select_best_candidate(
    candidates: Sequence[BoundaryCandidate],
    *,
    selection_mode: str,
) -> tuple[int, BoundaryCandidate, str, list[float]]:
    if not candidates:
        raise ValueError("No candidates to select from.")

    effective_selection_mode = _resolve_selection_mode(candidates, selection_mode)
    selection_scores = [
        _candidate_selection_score(candidate, effective_selection_mode)
        for candidate in candidates
    ]
    best_index = max(range(len(candidates)), key=lambda index: selection_scores[index])
    return best_index, candidates[best_index], effective_selection_mode, selection_scores


def _compact_candidate(candidate: BoundaryCandidate) -> dict[str, Any]:
    return {
        "candidate_index": candidate.candidate_index,
        "chunk_length": len(candidate.token_ids),
        "text": candidate.text,
        "chunk_logprob": candidate.chunk_logprob,
        "value_score": candidate.value_score,
        "stop_reason": candidate.stop_reason,
        "boundary_index": candidate.boundary_index,
        "boundary_entropy": candidate.boundary_entropy,
        "boundary_threshold": candidate.boundary_threshold,
        "boundary_reference_history_len": candidate.boundary_reference_history_len,
        "contains_eos": candidate.contains_eos,
        "mean_entropy": float(np.mean(candidate.entropies)) if candidate.entropies else None,
    }


def _decode_one_example(
    *,
    actor,
    critic,
    tokenizer,
    example: ExampleRecord,
    actor_device: torch.device,
    critic_device: torch.device | None,
    max_prompt_length: int,
    max_new_tokens: int,
    num_candidates: int,
    min_chunk_length: int,
    max_chunk_length: int,
    entropy_percentile: float,
    warmup_size: int,
    bootstrap_current_chunk_entropy: bool,
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    selection_mode: str,
    seed: int,
    eos_token_ids: tuple[int, ...],
    use_actor_cache: bool,
) -> BoundaryDecodeArtifacts:
    if example.prompt_token_ids is not None:
        prompt_token_ids = list(example.prompt_token_ids)
    else:
        tokenized = tokenizer(
            example.prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        prompt_token_ids = tokenized["input_ids"][0].tolist()

    current_sequence_ids = torch.tensor([prompt_token_ids], device=actor_device, dtype=torch.long)
    generated_token_ids: list[int] = []
    entropy_history: list[float] = []
    chunk_results: list[dict[str, Any]] = []
    eos_emitted = False

    start_time = time.perf_counter()
    chunk_index = 0
    while len(generated_token_ids) < max_new_tokens:
        remaining_tokens = max_new_tokens - len(generated_token_ids)
        max_chunk_len = min(max_chunk_length, remaining_tokens)
        entropy_history_len_before = len(entropy_history)

        candidates = [
            _rollout_candidate(
                actor=actor,
                critic=critic,
                tokenizer=tokenizer,
                prefix_ids=current_sequence_ids,
                actor_device=actor_device,
                critic_device=critic_device,
                candidate_index=candidate_index,
                max_chunk_len=max_chunk_len,
                min_chunk_length=min_chunk_length,
                entropy_history=entropy_history,
                entropy_percentile=entropy_percentile,
                warmup_size=warmup_size,
                bootstrap_current_chunk_entropy=bootstrap_current_chunk_entropy,
                sampling_mode=sampling_mode,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=_candidate_seed(
                    seed,
                    example_id=example.example_id,
                    chunk_index=chunk_index,
                    candidate_index=candidate_index,
                ),
                eos_token_ids=eos_token_ids,
                use_actor_cache=use_actor_cache,
            )
            for candidate_index in range(num_candidates)
        ]
        (
            selected_list_index,
            selected_candidate,
            effective_selection_mode,
            candidate_selection_scores,
        ) = _select_best_candidate(candidates, selection_mode=selection_mode)

        selected_tensor = torch.tensor(
            [list(selected_candidate.token_ids)],
            device=actor_device,
            dtype=current_sequence_ids.dtype,
        )
        current_sequence_ids = torch.cat([current_sequence_ids, selected_tensor], dim=1)
        generated_token_ids.extend(selected_candidate.token_ids)

        for entropy_index, entropy in enumerate(selected_candidate.entropies, start=1):
            if (
                selected_candidate.boundary_index is not None
                and entropy_index == selected_candidate.boundary_index
            ):
                continue
            entropy_history.append(float(entropy))

        chunk_results.append(
            {
                "example_id": int(example.example_id),
                "chunk_index": int(chunk_index),
                "generated_length_before_chunk": int(len(generated_token_ids) - len(selected_candidate.token_ids)),
                "selected_candidate_list_index": int(selected_list_index),
                "selected_candidate_index": int(selected_candidate.candidate_index),
                "selected_chunk_length": int(len(selected_candidate.token_ids)),
                "selected_text": selected_candidate.text,
                "selected_token_ids": list(selected_candidate.token_ids),
                "selected_entropies": list(selected_candidate.entropies),
                "selected_token_logprobs": list(selected_candidate.token_logprobs),
                "selected_chunk_logprob": float(selected_candidate.chunk_logprob),
                "selected_value_score": (
                    None if selected_candidate.value_score is None else float(selected_candidate.value_score)
                ),
                "selection_mode": effective_selection_mode,
                "selected_selection_score": float(candidate_selection_scores[selected_list_index]),
                "candidate_selection_scores": candidate_selection_scores,
                "stop_reason": selected_candidate.stop_reason,
                "boundary_index": selected_candidate.boundary_index,
                "boundary_entropy": selected_candidate.boundary_entropy,
                "boundary_threshold": selected_candidate.boundary_threshold,
                "boundary_reference_history_len": selected_candidate.boundary_reference_history_len,
                "entropy_history_len_before": int(entropy_history_len_before),
                "entropy_history_len_after": int(len(entropy_history)),
                "all_candidates": [_compact_candidate(candidate) for candidate in candidates],
            }
        )

        chunk_index += 1
        if selected_candidate.contains_eos:
            eos_emitted = True
            break

    latency_sec = time.perf_counter() - start_time
    response_length = len(generated_token_ids)
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = float(score_response(example, response_text))
    max_length_hit = bool(max_new_tokens > 0 and not eos_emitted and response_length >= max_new_tokens)

    example_result = {
        "example_id": int(example.example_id),
        "data_source": example.data_source,
        "prompt": example.prompt_text,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "generated_response": response_text,
        "response_length": response_length,
        "task_score": task_score,
        "critic_enabled": critic is not None,
        "selection_mode_requested": selection_mode,
        "eos_emitted": eos_emitted,
        "max_length_hit": max_length_hit,
        "num_chunks": len(chunk_results),
        "num_entropy_boundaries": sum(
            1 for chunk in chunk_results if chunk["stop_reason"] == "entropy_boundary"
        ),
        "latency_sec": latency_sec,
        "tokens_per_second": (response_length / latency_sec) if latency_sec > 0 else None,
    }
    return BoundaryDecodeArtifacts(example_result=example_result, chunk_results=chunk_results)


def _json_ready_result(artifacts: BoundaryDecodeArtifacts) -> dict[str, Any]:
    return {
        "example_result": artifacts.example_result,
        "chunk_results": artifacts.chunk_results,
    }


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _markdown_escape(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _format_chunk_header(chunk: dict[str, Any]) -> str:
    reason = chunk["stop_reason"]
    value_score = chunk.get("selected_value_score")
    if value_score is None:
        score_label = chunk.get("selection_mode", "score")
        score_value = f"{float(chunk['selected_selection_score']):.4f}"
        score_fragment = f"{score_label}={score_value}"
    else:
        score_fragment = f"V={float(value_score):.4f}"
    length = chunk["selected_chunk_length"]
    if reason == "entropy_boundary":
        entropy = chunk["boundary_entropy"]
        threshold = chunk["boundary_threshold"]
        return (
            f"#### Chunk {chunk['chunk_index'] + 1} "
            f"(len={length}, reason=entropy, H={entropy:.4f}, q-threshold={threshold:.4f}, {score_fragment})"
        )
    return f"#### Chunk {chunk['chunk_index'] + 1} (len={length}, reason={reason}, {score_fragment})"


def _write_markdown(path: Path, records: Sequence[dict[str, Any]], config: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Entropy-Boundary Inspection Results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    for result_index, record in enumerate(records, start=1):
        example = record["example_result"]
        chunks = record["chunk_results"]
        lines.append(f"## Correct Trajectory {result_index}: example_id={example['example_id']}")
        lines.append("")
        lines.append(f"- Ground truth: `{_markdown_escape(example['ground_truth'])}`")
        lines.append(f"- Score: `{example['task_score']}`")
        lines.append(f"- Response tokens: `{example['response_length']}`")
        lines.append(f"- Chunks: `{example['num_chunks']}`")
        lines.append(f"- Entropy-triggered boundaries: `{example['num_entropy_boundaries']}`")
        lines.append("")
        lines.append("### Prompt")
        lines.append("")
        lines.append(_markdown_escape(example["prompt"]))
        lines.append("")
        lines.append("### Segmented Response")
        lines.append("")
        for chunk in chunks:
            lines.append(_format_chunk_header(chunk))
            lines.append("")
            lines.append("```text")
            lines.append(_markdown_escape(chunk["selected_text"]).strip())
            lines.append("```")
            lines.append("")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_worker_pairs(
    worker_pairs: Sequence[str] | None,
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
                parsed.append((actor.strip() or None, critic.strip() or None))
            else:
                parsed.append((value, value))
        if not parsed:
            raise ValueError("--worker_pairs was provided, but no valid entries were parsed.")
        return parsed

    resolved_actor = actor_device or default_device
    resolved_critic = critic_device or default_device or resolved_actor
    return [(resolved_actor, resolved_critic)]


def _resolve_actor_hf_dir(checkpoint_dir: Path, *, skip_merge: bool) -> Path:
    if has_complete_hf_checkpoint(checkpoint_dir):
        return checkpoint_dir
    return ensure_merged_component_checkpoint(
        checkpoint_dir,
        component="actor",
        skip_merge=skip_merge,
    )


def _resolve_critic_hf_dir(
    *,
    checkpoint_dir: Path,
    critic_checkpoint_dir: Path | None,
    skip_merge: bool,
    disable_critic_model: bool,
) -> Path | None:
    if disable_critic_model:
        return None

    if critic_checkpoint_dir is not None:
        if has_complete_hf_checkpoint(critic_checkpoint_dir):
            return critic_checkpoint_dir
        return ensure_merged_component_checkpoint(
            critic_checkpoint_dir,
            component="critic",
            skip_merge=skip_merge,
        )

    if has_complete_hf_checkpoint(checkpoint_dir):
        return None

    return ensure_merged_component_checkpoint(
        checkpoint_dir,
        component="critic",
        skip_merge=skip_merge,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_candidates <= 0:
        raise ValueError("--num_candidates must be positive.")
    if args.min_chunk_length <= 0:
        raise ValueError("--min_chunk_length must be positive.")
    if args.max_chunk_length < args.min_chunk_length:
        raise ValueError("--max_chunk_length must be >= --min_chunk_length.")
    if not (0.0 < args.entropy_percentile < 1.0):
        raise ValueError("--entropy_percentile must be strictly between 0 and 1.")
    if args.warmup_size <= 0:
        raise ValueError("--warmup_size must be positive.")
    if args.num_correct <= 0:
        raise ValueError("--num_correct must be positive.")
    if args.max_scan_examples is not None and args.max_scan_examples <= 0:
        raise ValueError("--max_scan_examples must be positive when provided.")
    if args.max_prompt_length <= 0:
        raise ValueError("--max_prompt_length must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if args.actor_temperature < 0.0:
        raise ValueError("--actor_temperature must be non-negative.")
    if not (0.0 < args.actor_top_p <= 1.0):
        raise ValueError("--actor_top_p must be in (0, 1].")
    if args.actor_top_k < 0:
        raise ValueError("--actor_top_k must be non-negative.")
    if args.disable_critic_model and args.selection_mode == "value":
        raise ValueError("--selection_mode value cannot be used with --disable_critic_model.")


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int]:
    example_result = record["example_result"]
    return (
        int(example_result.get("example_id", -1)),
        int(example_result.get("worker_id", 0)),
    )


def _put_status(result_queue, worker_id: int, message: str) -> None:
    if result_queue is None:
        print(message, flush=True)
        return
    result_queue.put({"type": "status", "worker_id": worker_id, "message": message})


def _scan_examples_for_correct(
    *,
    actor_hf_dir: Path,
    critic_hf_dir: Path | None,
    examples: Sequence[ExampleRecord],
    args: argparse.Namespace,
    actor_device_name: str | None,
    critic_device_name: str | None,
    worker_id: int,
    num_workers: int,
    stop_event=None,
    result_queue=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dtype = resolve_dtype(args.dtype)
    actor_device = resolve_device(actor_device_name)
    critic_device = resolve_device(critic_device_name) if critic_hf_dir is not None else None
    use_actor_cache = not args.disable_actor_cache
    critic_status = "disabled" if critic_hf_dir is None else str(critic_device)

    _put_status(
        result_queue,
        worker_id,
        (
            f"[worker {worker_id}/{num_workers}] loading actor={actor_device} "
            f"critic={critic_status}; shard_examples={len(examples)}"
        ),
    )

    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, tokenizer)
    actor = load_actor_model(
        actor_hf_dir,
        dtype=dtype,
        device=actor_device,
        trust_remote_code=args.trust_remote_code,
    )
    if critic_hf_dir is None:
        critic = None
    else:
        critic = load_critic_model(
            critic_hf_dir,
            dtype=dtype,
            device=critic_device,
            trust_remote_code=args.trust_remote_code,
        )

    _put_status(
        result_queue,
        worker_id,
        f"[worker {worker_id}/{num_workers}] ready; eos={eos_token_ids}",
    )

    found_records: list[dict[str, Any]] = []
    attempted_records: list[dict[str, Any]] = []
    for local_index, example in enumerate(examples, start=1):
        if stop_event is not None and stop_event.is_set():
            break

        artifacts = _decode_one_example(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            actor_device=actor_device,
            critic_device=critic_device,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            num_candidates=args.num_candidates,
            min_chunk_length=args.min_chunk_length,
            max_chunk_length=args.max_chunk_length,
            entropy_percentile=args.entropy_percentile,
            warmup_size=args.warmup_size,
            bootstrap_current_chunk_entropy=args.bootstrap_current_chunk_entropy,
            sampling_mode=args.actor_sampling_mode,
            temperature=args.actor_temperature,
            top_p=args.actor_top_p,
            top_k=args.actor_top_k,
            selection_mode=args.selection_mode,
            seed=args.seed,
            eos_token_ids=eos_token_ids,
            use_actor_cache=use_actor_cache,
        )
        record = _json_ready_result(artifacts)
        record["example_result"].update(
            {
                "worker_id": int(worker_id),
                "worker_num_workers": int(num_workers),
                "worker_shard_index": int(local_index - 1),
                "worker_actor_device": str(actor_device),
                "worker_critic_device": None if critic_device is None else str(critic_device),
            }
        )
        for chunk in record["chunk_results"]:
            chunk["worker_id"] = int(worker_id)

        if result_queue is None:
            attempted_records.append(record)
            example_result = record["example_result"]
            is_correct = float(example_result["task_score"]) >= 1.0
            print(
                "example_id={example_id} score={score:.1f} len={length} chunks={chunks} "
                "entropy_boundaries={boundaries} correct={correct}".format(
                    example_id=example_result["example_id"],
                    score=float(example_result["task_score"]),
                    length=example_result["response_length"],
                    chunks=example_result["num_chunks"],
                    boundaries=example_result["num_entropy_boundaries"],
                    correct=is_correct,
                ),
                flush=True,
            )
            if is_correct:
                found_records.append(record)
                print(
                    f"  found {len(found_records)}/{args.num_correct} correct "
                    f"(worker shard position {local_index}/{len(examples)})",
                    flush=True,
                )
                if len(found_records) >= args.num_correct:
                    break
        else:
            result_queue.put(
                {
                    "type": "record",
                    "worker_id": worker_id,
                    "local_index": local_index,
                    "num_shard_examples": len(examples),
                    "record": record,
                }
            )

    return found_records, attempted_records


def _worker_entry(
    *,
    worker_id: int,
    num_workers: int,
    worker_pair: tuple[str | None, str | None],
    actor_hf_dir: str,
    critic_hf_dir: str | None,
    examples: Sequence[ExampleRecord],
    args: argparse.Namespace,
    stop_event,
    result_queue,
) -> None:
    try:
        _scan_examples_for_correct(
            actor_hf_dir=Path(actor_hf_dir),
            critic_hf_dir=None if critic_hf_dir is None else Path(critic_hf_dir),
            examples=examples,
            args=args,
            actor_device_name=worker_pair[0],
            critic_device_name=worker_pair[1],
            worker_id=worker_id,
            num_workers=num_workers,
            stop_event=stop_event,
            result_queue=result_queue,
        )
        result_queue.put({"type": "done", "worker_id": worker_id})
    except BaseException:
        result_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "traceback": traceback.format_exc(),
            }
        )


def _run_single_worker(
    *,
    actor_hf_dir: Path,
    critic_hf_dir: Path | None,
    examples: Sequence[ExampleRecord],
    args: argparse.Namespace,
    worker_pair: tuple[str | None, str | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"Scanning {len(examples)} examples for {args.num_correct} correct trajectories.", flush=True)
    found_records, attempted_records = _scan_examples_for_correct(
        actor_hf_dir=actor_hf_dir,
        critic_hf_dir=critic_hf_dir,
        examples=examples,
        args=args,
        actor_device_name=worker_pair[0],
        critic_device_name=worker_pair[1],
        worker_id=0,
        num_workers=1,
    )
    return found_records[: args.num_correct], attempted_records


def _run_multi_worker(
    *,
    actor_hf_dir: Path,
    critic_hf_dir: Path | None,
    examples: Sequence[ExampleRecord],
    args: argparse.Namespace,
    worker_pairs: Sequence[tuple[str | None, str | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return [], []

    active_workers = min(len(worker_pairs), len(examples))
    active_pairs = list(worker_pairs[:active_workers])
    print(
        f"Scanning {len(examples)} examples with {active_workers} workers for "
        f"{args.num_correct} total correct trajectories.",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    result_queue = ctx.Queue()
    processes: list[mp.Process] = []

    for worker_id, worker_pair in enumerate(active_pairs):
        shard_examples = list(examples[worker_id::active_workers])
        process = ctx.Process(
            target=_worker_entry,
            kwargs={
                "worker_id": worker_id,
                "num_workers": active_workers,
                "worker_pair": worker_pair,
                "actor_hf_dir": str(actor_hf_dir),
                "critic_hf_dir": None if critic_hf_dir is None else str(critic_hf_dir),
                "examples": shard_examples,
                "args": args,
                "stop_event": stop_event,
                "result_queue": result_queue,
            },
        )
        process.start()
        processes.append(process)

    found_records: list[dict[str, Any]] = []
    attempted_records: list[dict[str, Any]] = []
    finished_workers: set[int] = set()
    errors: list[str] = []

    while len(finished_workers) < len(processes):
        try:
            message = result_queue.get(timeout=1.0)
        except Empty:
            for worker_id, process in enumerate(processes):
                if worker_id in finished_workers:
                    continue
                if process.exitcode is not None and process.exitcode != 0:
                    finished_workers.add(worker_id)
                    errors.append(f"worker {worker_id} exited with code {process.exitcode}")
                    stop_event.set()
            continue

        message_type = message.get("type")
        worker_id = int(message.get("worker_id", -1))
        if message_type == "status":
            print(message["message"], flush=True)
            continue
        if message_type == "done":
            finished_workers.add(worker_id)
            print(f"[worker {worker_id}] done", flush=True)
            continue
        if message_type == "error":
            errors.append(f"[worker {worker_id}] failed:\n{message['traceback']}")
            finished_workers.add(worker_id)
            stop_event.set()
            continue
        if message_type != "record":
            errors.append(f"[worker {worker_id}] sent unknown message type: {message_type!r}")
            stop_event.set()
            continue

        record = message["record"]
        attempted_records.append(record)
        example_result = record["example_result"]
        is_correct = float(example_result["task_score"]) >= 1.0
        print(
            "[worker {worker_id}] example_id={example_id} score={score:.1f} len={length} chunks={chunks} "
            "entropy_boundaries={boundaries} correct={correct}".format(
                worker_id=worker_id,
                example_id=example_result["example_id"],
                score=float(example_result["task_score"]),
                length=example_result["response_length"],
                chunks=example_result["num_chunks"],
                boundaries=example_result["num_entropy_boundaries"],
                correct=is_correct,
            ),
            flush=True,
        )

        if is_correct and len(found_records) < args.num_correct:
            found_records.append(record)
            print(f"  found {len(found_records)}/{args.num_correct} correct total", flush=True)
            if len(found_records) >= args.num_correct:
                stop_event.set()

    for process in processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    if errors:
        raise RuntimeError("\n".join(errors))

    return sorted(found_records, key=_record_sort_key)[: args.num_correct], sorted(attempted_records, key=_record_sort_key)


def _write_run_outputs(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    actor_hf_dir: Path,
    critic_hf_dir: Path | None,
    eos_token_ids: tuple[int, ...],
    worker_pairs: Sequence[tuple[str | None, str | None]],
    found_records: Sequence[dict[str, Any]],
    attempted_records: Sequence[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    active_worker_ids = sorted(
        {
            int(record["example_result"]["worker_id"])
            for record in attempted_records
            if "worker_id" in record.get("example_result", {})
        }
    )
    config = {
        "created_at_utc": timestamp,
        "checkpoint_dir": str(checkpoint_dir),
        "actor_hf_dir": str(actor_hf_dir),
        "critic_hf_dir": None if critic_hf_dir is None else str(critic_hf_dir),
        "critic_enabled": critic_hf_dir is not None,
        "dataset_path": str(dataset_path),
        "num_candidates": args.num_candidates,
        "min_chunk_length": args.min_chunk_length,
        "max_chunk_length": args.max_chunk_length,
        "entropy_percentile": args.entropy_percentile,
        "warmup_size": args.warmup_size,
        "bootstrap_current_chunk_entropy": args.bootstrap_current_chunk_entropy,
        "max_new_tokens": args.max_new_tokens,
        "actor_sampling_mode": args.actor_sampling_mode,
        "actor_temperature": args.actor_temperature,
        "actor_top_p": args.actor_top_p,
        "actor_top_k": args.actor_top_k,
        "selection_mode_requested": args.selection_mode,
        "seed": args.seed,
        "worker_pairs": [[actor, critic] for actor, critic in worker_pairs],
        "num_worker_pairs_requested": len(worker_pairs),
        "num_workers_with_attempts": len(active_worker_ids),
        "worker_ids_with_attempts": active_worker_ids,
        "eos_token_ids": eos_token_ids,
        "num_correct_requested": args.num_correct,
        "num_correct_found": len(found_records),
        "num_examples_attempted": len(attempted_records),
    }

    found_jsonl = output_dir / f"correct_entropy_boundary_{timestamp}.jsonl"
    attempted_jsonl = output_dir / f"attempted_entropy_boundary_{timestamp}.jsonl"
    markdown_path = output_dir / f"correct_entropy_boundary_{timestamp}.md"
    summary_path = output_dir / f"summary_entropy_boundary_{timestamp}.json"

    _write_jsonl(found_jsonl, found_records)
    _write_jsonl(attempted_jsonl, attempted_records)
    _write_markdown(markdown_path, found_records, config)
    summary_path.write_text(json.dumps(config, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    return found_jsonl, attempted_jsonl, markdown_path, summary_path


def main() -> None:
    args = parse_args()
    _validate_args(args)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    critic_checkpoint_dir = Path(args.critic_checkpoint_dir).resolve() if args.critic_checkpoint_dir else None
    dataset_path = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_pairs = _parse_worker_pairs(
        args.worker_pairs,
        actor_device=args.actor_device,
        critic_device=args.critic_device,
        default_device=args.device,
    )
    if not args.bootstrap_current_chunk_entropy and args.max_chunk_length > args.warmup_size:
        print(
            "Warning: --bootstrap_current_chunk_entropy is off and max_chunk_length > warmup_size. "
            "The first chunk cannot trigger an entropy boundary until it reaches max_chunk_length/EOS, "
            "because H_ctx is empty before the first commit.",
            flush=True,
        )

    print(f"Preparing actor/critic from {checkpoint_dir}", flush=True)
    actor_hf_dir = _resolve_actor_hf_dir(checkpoint_dir, skip_merge=args.skip_merge)
    critic_hf_dir = _resolve_critic_hf_dir(
        checkpoint_dir=checkpoint_dir,
        critic_checkpoint_dir=critic_checkpoint_dir,
        skip_merge=args.skip_merge,
        disable_critic_model=args.disable_critic_model,
    )
    if critic_hf_dir is None:
        if args.selection_mode == "value":
            raise ValueError(
                "--selection_mode value requires a critic. Provide --critic_checkpoint_dir or use "
                "--selection_mode auto/actor_logprob."
            )
        print(
            "No critic checkpoint detected; running actor-only chunk selection "
            "(effective selection mode will be actor_logprob).",
            flush=True,
        )
    else:
        print(f"Using critic checkpoint: {critic_hf_dir}", flush=True)

    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, tokenizer)
    examples = load_examples(
        dataset_path,
        tokenizer=tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=args.start_index,
        max_examples=args.max_scan_examples,
        shuffle_examples=False,
        seed=args.seed,
        pretokenize_max_length=args.max_prompt_length,
    )

    if len(worker_pairs) <= 1:
        found_records, attempted_records = _run_single_worker(
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            args=args,
            worker_pair=worker_pairs[0],
        )
    else:
        found_records, attempted_records = _run_multi_worker(
            actor_hf_dir=actor_hf_dir,
            critic_hf_dir=critic_hf_dir,
            examples=examples,
            args=args,
            worker_pairs=worker_pairs,
        )

    found_jsonl, attempted_jsonl, markdown_path, summary_path = _write_run_outputs(
        args=args,
        checkpoint_dir=checkpoint_dir,
        dataset_path=dataset_path,
        output_dir=output_dir,
        actor_hf_dir=actor_hf_dir,
        critic_hf_dir=critic_hf_dir,
        eos_token_ids=eos_token_ids,
        worker_pairs=worker_pairs,
        found_records=found_records,
        attempted_records=attempted_records,
    )

    print(f"Wrote correct JSONL: {found_jsonl}", flush=True)
    print(f"Wrote attempted JSONL: {attempted_jsonl}", flush=True)
    print(f"Wrote segmented markdown: {markdown_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)

    if len(found_records) < args.num_correct:
        raise SystemExit(
            f"Only found {len(found_records)} correct trajectories after scanning {len(attempted_records)} examples."
        )


if __name__ == "__main__":
    main()
