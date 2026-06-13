from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from value_decoding.checkpointing import (
    load_actor_model,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.data import ExampleRecord, load_examples
from verl.utils.reward_score import default_compute_score
from value_decoding.decoding import critic_sequence_values

LLM = None
SamplingParams = None
TokensPrompt = None


def ensure_vllm_imported() -> None:
    global LLM, SamplingParams, TokensPrompt
    if LLM is not None and SamplingParams is not None:
        return
    try:
        from vllm import LLM as imported_llm
        from vllm import SamplingParams as imported_sampling_params
    except ImportError as exc:
        raise ImportError(
            "--generation_backend=vllm requires vLLM in the active environment. "
            "Use --generation_backend=torch if vLLM is unavailable."
        ) from exc
    try:
        from vllm.inputs.data import TokensPrompt as imported_tokens_prompt
    except Exception:
        imported_tokens_prompt = None
    LLM = imported_llm
    SamplingParams = imported_sampling_params
    TokensPrompt = imported_tokens_prompt


def resolve_vllm_dtype(dtype_name: str) -> str:
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype_name]


def vllm_output_token_ids(output: Any) -> list[int]:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is None:
        token_ids = getattr(output, "output_token_ids", None)
    if token_ids is None:
        return []
    return [int(token_id) for token_id in token_ids]


def make_vllm_prompt(prompt_token_ids: list[int]) -> Any:
    if TokensPrompt is not None:
        return TokensPrompt(prompt_token_ids=prompt_token_ids)
    return {"prompt_token_ids": prompt_token_ids}

@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: int
    prompt_id: int
    prompt: str
    response: str
    response_token_ids: list[int]
    reward: float
    data_source: str | None = None
    ground_truth: Any = None


@dataclass(frozen=True)
class ValueReadout:
    value: float
    response_length: int
    ended_with_eos: bool
    selected_response_index: int | None
    selected_full_index: int | None



def shard_sequence(items: list[Any], *, num_shards: int, shard_id: int) -> list[Any]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    if num_shards == 1:
        return items
    return [item for index, item in enumerate(items) if index % num_shards == shard_id]


def load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_output_dirs(input_dirs: list[Path], output_dir: Path, *, value_position: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        rows_path = input_dir / "trajectory_values.jsonl"
        if not rows_path.is_file():
            raise FileNotFoundError(f"Missing shard trajectory file: {rows_path}")
        rows.extend(load_rows_from_jsonl(rows_path))
    rows.sort(key=lambda row: (int(row.get("trajectory_id", 0)), int(row.get("prompt_id", 0))))
    if not rows:
        raise ValueError("No trajectory rows found in aggregate input dirs.")
    r_bar = float(np.mean([float(row["reward"]) for row in rows]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectory_values.jsonl").open("w", encoding="utf-8") as handle:
        normalized_rows: list[dict[str, Any]] = []
        for new_id, row in enumerate(rows):
            normalized = dict(row)
            normalized["trajectory_id"] = new_id
            normalized["r_bar"] = r_bar
            normalized["constant_loss"] = float((r_bar - float(normalized["reward"])) ** 2)
            normalized["critic_loss"] = float((float(normalized["critic_value"]) - float(normalized["reward"])) ** 2)
            normalized_rows.append(normalized)
            handle.write(json.dumps(normalized, ensure_ascii=False, default=_json_default) + "\n")
    summary = summarize(normalized_rows, value_position=value_position)
    critic_labels = sorted({str(row.get("critic_label")) for row in normalized_rows if row.get("critic_label") is not None})
    critic_dirs = sorted({str(row.get("critic_checkpoint_dir")) for row in normalized_rows if row.get("critic_checkpoint_dir") is not None})
    if len(critic_labels) == 1:
        summary["critic_label"] = critic_labels[0]
    if len(critic_dirs) == 1:
        summary["critic_checkpoint_dir"] = critic_dirs[0]
    write_json(output_dir / "summary_metrics.json", summary)
    write_json(
        output_dir / "config.json",
        {
            "aggregate_input_dirs": [str(path) for path in input_dirs],
            "value_position": value_position,
            "critic_labels": critic_labels,
            "critic_checkpoint_dirs": critic_dirs,
        },
    )
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a trained trajectory critic against the constant policy-accuracy baseline."
    )
    parser.add_argument("--actor_checkpoint_dir", type=str, required=True, help="Actor HF checkpoint directory.")
    parser.add_argument("--critic_checkpoint_dir", type=str, nargs="+", required=True, help="One or more critic HF checkpoint directories.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Evaluation parquet dataset path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for config/rows/summary outputs.")
    parser.add_argument("--max_examples", type=int, default=500, help="Maximum prompts to evaluate; negative means all.")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="Generated samples per prompt.")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt tokens for generation and critic scoring.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum generated response tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; <=0 uses greedy decoding.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling; 0 disables top-k.")
    parser.add_argument(
        "--value_position",
        choices=["pre_eos", "last_response", "tail_mean_4", "tail_mean_8", "tail_mean_16", "prompt_only"],
        nargs="+",
        default=["pre_eos"],
        help="One or more scalar critic readout positions.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation and shuffling.")
    parser.add_argument("--response_bank_path", type=str, default=None, help="Optional existing response bank JSONL.")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Prompt column in parquet dataset.")
    parser.add_argument(
        "--response_key",
        type=str,
        default="ground_truth",
        help="Ground-truth column in parquet dataset; use an empty string to read reward_model.ground_truth only.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Start prompt index in dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle prompts before max_examples selection.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Model dtype.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--critic_device", type=str, default=None, help="Optional separate critic device.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    parser.add_argument(
        "--math_dapo_binary_reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 0/1 math-DAPO rewards, matching ppo_freeze_actor_train_critic.sh by default.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Generation batch size for actor/vLLM generation.")
    parser.add_argument(
        "--generation_backend",
        choices=["torch", "vllm"],
        default="torch",
        help="Actor generation backend. vLLM accelerates trajectory generation; critic scoring remains torch.",
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.8, help="vLLM GPU memory utilization.")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size per evaluator process.")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None, help="Optional vLLM max_num_seqs.")
    parser.add_argument(
        "--vllm_enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enforce_eager to vLLM. Defaults true for robustness; use --no-vllm_enforce_eager for throughput.",
    )
    parser.add_argument("--num_shards", type=int, default=1, help="Number of dataset/trajectory shards for distributed evaluation.")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id in [0, num_shards).")
    parser.add_argument(
        "--always_write_run_subdirs",
        action="store_true",
        help="Always write critic/value-position outputs under <critic_label>/<value_position>; useful for shard aggregation.",
    )
    parser.add_argument(
        "--aggregate_input_dirs",
        nargs="*",
        default=None,
        help="If provided, aggregate existing shard output dirs and skip model loading/evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _safe_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_response_text(row: dict[str, Any]) -> str:
    for key in ("response", "generated_response", "solution_str", "completion", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(f"Response bank row lacks a response text field: keys={sorted(row.keys())}")


def _extract_prompt_id(row: dict[str, Any], fallback: int) -> int:
    for key in ("prompt_id", "example_id", "uid", "id"):
        if key in row:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                break
    return fallback


def _build_example_lookup(examples: Iterable[ExampleRecord]) -> dict[int, ExampleRecord]:
    return {int(example.example_id): example for example in examples}



def score_trajectory_response(example: ExampleRecord, response_text: str, *, math_dapo_binary_reward: bool) -> float:
    score = default_compute_score(
        data_source=example.data_source,
        solution_str=response_text,
        ground_truth=example.ground_truth,
        math_dapo_binary_reward=math_dapo_binary_reward,
    )
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)

def _score_bank_row(
    row: dict[str, Any],
    example: ExampleRecord,
    response: str,
    *,
    math_dapo_binary_reward: bool,
) -> float:
    for key in ("reward", "task_score", "score", "acc", "accuracy", "is_correct"):
        if key not in row:
            continue
        value = _safe_float(row[key])
        if value is not None:
            return value
    return float(score_trajectory_response(example, response, math_dapo_binary_reward=math_dapo_binary_reward))


def load_response_bank(
    path: Path,
    *,
    examples: list[ExampleRecord],
    tokenizer,
    max_trajectories: int | None,
    math_dapo_binary_reward: bool,
) -> list[TrajectoryRecord]:
    example_lookup = _build_example_lookup(examples)
    trajectories: list[TrajectoryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if max_trajectories is not None and len(trajectories) >= max_trajectories:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = _extract_prompt_id(row, fallback=line_index)
            example = example_lookup.get(prompt_id)
            if example is None:
                continue
            response = _extract_response_text(row)
            token_ids = row.get("response_token_ids") or row.get("generated_token_ids") or row.get("token_ids")
            if token_ids is None:
                token_ids = tokenizer(response, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            response_token_ids = [int(token_id) for token_id in token_ids]
            reward = _score_bank_row(
                row,
                example,
                response,
                math_dapo_binary_reward=math_dapo_binary_reward,
            )
            trajectories.append(
                TrajectoryRecord(
                    trajectory_id=len(trajectories),
                    prompt_id=prompt_id,
                    prompt=example.prompt_text,
                    response=response,
                    response_token_ids=response_token_ids,
                    reward=float(reward),
                    data_source=example.data_source,
                    ground_truth=example.ground_truth,
                )
            )
    return trajectories


def _sampling_kwargs(args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    do_sample = bool(args.temperature > 0.0)
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(args.temperature)
        kwargs["top_p"] = float(args.top_p)
        if int(args.top_k) > 0:
            kwargs["top_k"] = int(args.top_k)
    return kwargs


def _make_trajectory_record(
    *,
    trajectory_id: int,
    example: ExampleRecord,
    response: str,
    response_token_ids: list[int],
    math_dapo_binary_reward: bool,
) -> TrajectoryRecord:
    reward = float(
        score_trajectory_response(
            example,
            response,
            math_dapo_binary_reward=math_dapo_binary_reward,
        )
    )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        prompt_id=int(example.example_id),
        prompt=example.prompt_text,
        response=response,
        response_token_ids=response_token_ids,
        reward=reward,
        data_source=example.data_source,
        ground_truth=example.ground_truth,
    )


def generate_trajectories_with_torch(
    *,
    examples: list[ExampleRecord],
    actor,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> list[TrajectoryRecord]:
    trajectories: list[TrajectoryRecord] = []
    batch_size = max(1, int(args.batch_size))
    generation_kwargs = _sampling_kwargs(args, tokenizer)
    samples_per_prompt = max(1, int(args.num_samples_per_prompt))

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        prompt_texts = [example.prompt_text for example in batch]
        inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(args.max_prompt_length),
            return_token_type_ids=False,
        ).to(device)
        input_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = actor.generate(
                **inputs,
                num_return_sequences=samples_per_prompt,
                **generation_kwargs,
            )
        for output_index, generated_ids in enumerate(generated):
            example = batch[output_index // samples_per_prompt]
            generated_ids_list = [int(token_id) for token_id in generated_ids.detach().cpu().tolist()]
            response_token_ids = generated_ids_list[input_width:]
            if tokenizer.pad_token_id is not None:
                response_token_ids = [token_id for token_id in response_token_ids if token_id != int(tokenizer.pad_token_id)]
            response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
            trajectories.append(
                _make_trajectory_record(
                    trajectory_id=len(trajectories),
                    example=example,
                    response=response,
                    response_token_ids=response_token_ids,
                    math_dapo_binary_reward=bool(args.math_dapo_binary_reward),
                )
            )
    return trajectories


def build_vllm_engine(args: argparse.Namespace, actor_dir: Path) -> Any:
    ensure_vllm_imported()
    tensor_parallel_size = int(args.vllm_tensor_parallel_size)
    if tensor_parallel_size <= 0:
        raise ValueError("--vllm_tensor_parallel_size must be positive.")
    if not (0.0 < float(args.vllm_gpu_memory_utilization) <= 1.0):
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1].")
    kwargs: dict[str, Any] = {
        "model": str(actor_dir),
        "tokenizer": str(actor_dir),
        "dtype": resolve_vllm_dtype(args.dtype),
        "trust_remote_code": bool(args.trust_remote_code),
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "enforce_eager": bool(args.vllm_enforce_eager),
        "max_model_len": int(args.max_prompt_length) + int(args.max_new_tokens),
    }
    if args.vllm_max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(args.vllm_max_num_seqs)
    return LLM(**kwargs)



def maybe_append_vllm_stop_eos(response_token_ids: list[int], output: Any, tokenizer) -> list[int]:
    """Append EOS when vLLM stopped on EOS but omitted the stop token from token_ids.

    HF ``generate`` usually returns EOS in the generated ids. vLLM may stop because of
    ``stop_token_ids`` without including that stop token in ``output.token_ids``. For
    the ``pre_eos`` readout, we need the critic input to include EOS when EOS was the
    actual termination action, so the response-aligned EOS position maps to the raw
    value at the last non-EOS token (the state immediately before EOS).
    """
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return response_token_ids
    eos_token_id = int(eos_token_id)
    if response_token_ids and response_token_ids[-1] == eos_token_id:
        return response_token_ids

    finish_reason = getattr(output, "finish_reason", None)
    stop_reason = getattr(output, "stop_reason", None)
    stopped_on_eos = stop_reason == eos_token_id
    if isinstance(stop_reason, str):
        try:
            stopped_on_eos = int(stop_reason) == eos_token_id
        except ValueError:
            stopped_on_eos = False
    if finish_reason in {"stop", "eos"}:
        stopped_on_eos = True

    if stopped_on_eos:
        return [*response_token_ids, eos_token_id]
    return response_token_ids


def cleanup_vllm_engine(llm: Any) -> None:
    """Best-effort vLLM cleanup before loading the torch critic in the same process."""
    del llm
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment

        destroy_distributed_environment()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_vllm_sampling_params(args: argparse.Namespace, tokenizer) -> Any:
    ensure_vllm_imported()
    stop_token_ids = [] if tokenizer.eos_token_id is None else [int(tokenizer.eos_token_id)]
    temperature = float(args.temperature)
    return SamplingParams(
        n=max(1, int(args.num_samples_per_prompt)),
        temperature=temperature,
        top_p=float(args.top_p),
        top_k=0 if int(args.top_k) <= 0 else int(args.top_k),
        max_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        stop_token_ids=stop_token_ids,
        skip_special_tokens=True,
    )


def generate_trajectories_with_vllm(
    *,
    examples: list[ExampleRecord],
    llm: Any,
    tokenizer,
    args: argparse.Namespace,
) -> list[TrajectoryRecord]:
    trajectories: list[TrajectoryRecord] = []
    batch_size = max(1, int(args.batch_size))
    sampling_params = make_vllm_sampling_params(args, tokenizer)

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        prompts: list[Any] = []
        for example in batch:
            prompt_token_ids = tokenizer(
                example.prompt_text,
                truncation=True,
                max_length=int(args.max_prompt_length),
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            prompts.append(make_vllm_prompt([int(token_id) for token_id in prompt_token_ids]))

        request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        for example, request_output in zip(batch, request_outputs, strict=True):
            for output in request_output.outputs:
                response_token_ids = maybe_append_vllm_stop_eos(vllm_output_token_ids(output), output, tokenizer)
                response = getattr(output, "text", None)
                if response is None:
                    response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
                trajectories.append(
                    _make_trajectory_record(
                        trajectory_id=len(trajectories),
                        example=example,
                        response=response,
                        response_token_ids=response_token_ids,
                        math_dapo_binary_reward=bool(args.math_dapo_binary_reward),
                    )
                )
    return trajectories

def _first_eos_index(token_ids: list[int], eos_token_ids: set[int]) -> int | None:
    for index, token_id in enumerate(token_ids):
        if int(token_id) in eos_token_ids:
            return index
    return None


def response_aligned_values_from_raw(raw_values: torch.Tensor, prompt_length: int, response_length: int) -> torch.Tensor:
    """Return VERL-style values: value before each response token.

    For raw value-head outputs over ``prompt + response``, response token ``t`` is generated from the
    previous full-sequence position ``prompt_length + t - 1``. This mirrors VERL's slice
    ``[-response_length - 1:-1]`` for non-empty responses.
    """
    if response_length <= 0:
        raise ValueError("Cannot align critic values for an empty response.")
    if raw_values.dim() != 1:
        raise ValueError(f"Expected 1D raw_values, got shape {tuple(raw_values.shape)}")
    start = int(prompt_length) - 1
    end = start + int(response_length)
    if start < 0:
        raise ValueError("Prompt must contain at least one token for VERL-style value alignment.")
    if end > raw_values.shape[0]:
        raise ValueError(
            f"Not enough raw critic values for alignment: need end={end}, have {raw_values.shape[0]}."
        )
    return raw_values[start:end]


def select_trajectory_value(
    *,
    raw_values: torch.Tensor,
    prompt_length: int,
    response_token_ids: list[int],
    eos_token_ids: set[int],
    value_position: str,
) -> ValueReadout:
    response_length = len(response_token_ids)
    if response_length <= 0:
        raise ValueError("Cannot select a trajectory value for an empty generated response.")
    if raw_values.dim() != 1:
        raise ValueError(f"Expected 1D raw_values, got shape {tuple(raw_values.shape)}")

    eos_index = _first_eos_index(response_token_ids, eos_token_ids)
    ended_with_eos = eos_index is not None
    response_values = response_aligned_values_from_raw(raw_values, prompt_length, response_length)

    selected_response_index: int | None
    if value_position == "prompt_only":
        selected_full_index = int(prompt_length) - 1
        selected_response_index = None
        value = raw_values[selected_full_index]
    elif value_position == "pre_eos":
        selected_response_index = eos_index if eos_index is not None else response_length - 1
        value = response_values[selected_response_index]
        selected_full_index = int(prompt_length) + selected_response_index - 1
    elif value_position == "last_response":
        selected_response_index = response_length - 1
        value = response_values[selected_response_index]
        selected_full_index = int(prompt_length) + selected_response_index - 1
    elif value_position.startswith("tail_mean_"):
        tail_size = int(value_position.rsplit("_", maxsplit=1)[-1])
        terminal_exclusive = eos_index if eos_index is not None else response_length
        if terminal_exclusive <= 0:
            terminal_exclusive = response_length
        tail_start = max(0, terminal_exclusive - tail_size)
        tail_values = response_values[tail_start:terminal_exclusive]
        if tail_values.numel() == 0:
            tail_values = response_values[-min(tail_size, response_length) :]
            selected_response_index = response_length - 1
        else:
            selected_response_index = terminal_exclusive - 1
        value = tail_values.float().mean()
        selected_full_index = int(prompt_length) + selected_response_index - 1
    else:
        raise ValueError(f"Unsupported value_position={value_position!r}")

    return ValueReadout(
        value=float(value.float().item()),
        response_length=response_length,
        ended_with_eos=ended_with_eos,
        selected_response_index=selected_response_index,
        selected_full_index=selected_full_index,
    )


def compute_critic_readout(
    *,
    critic,
    tokenizer,
    trajectory: TrajectoryRecord,
    eos_token_ids: set[int],
    value_position: str,
    device: torch.device,
    max_prompt_length: int | None,
) -> ValueReadout:
    prompt_ids = tokenizer(
        trajectory.prompt,
        truncation=max_prompt_length is not None,
        max_length=max_prompt_length,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    prompt_ids = [int(token_id) for token_id in prompt_ids]
    response_ids = [int(token_id) for token_id in trajectory.response_token_ids]
    if not response_ids:
        raise ValueError(f"Trajectory {trajectory.trajectory_id} has an empty response.")
    full_ids = prompt_ids + response_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        raw_values = critic_sequence_values(critic, input_ids=input_ids, attention_mask=attention_mask)[0].detach().float().cpu()
    return select_trajectory_value(
        raw_values=raw_values,
        prompt_length=len(prompt_ids),
        response_token_ids=response_ids,
        eos_token_ids=eos_token_ids,
        value_position=value_position,
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    return _pearson(_rankdata_average_ties(x), _rankdata_average_ties(y))


def _auroc_binary(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive = labels == 1.0
    negative = labels == 0.0
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata_average_ties(scores)
    rank_sum_pos = float(np.sum(ranks[positive]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summarize(rows: list[dict[str, Any]], *, value_position: str) -> dict[str, Any]:
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row["critic_value"]) for row in rows], dtype=np.float64)
    if rewards.size == 0:
        raise ValueError("No trajectories were evaluated.")
    r_bar = float(np.mean(rewards))
    constant_mse = float(np.mean((r_bar - rewards) ** 2))
    critic_mse = float(np.mean((values - rewards) ** 2))
    relative = None if constant_mse == 0.0 else float((constant_mse - critic_mse) / constant_mse)
    correct_mask = rewards == 1.0
    wrong_mask = rewards == 0.0
    mean_correct = float(np.mean(values[correct_mask])) if np.any(correct_mask) else None
    mean_wrong = float(np.mean(values[wrong_mask])) if np.any(wrong_mask) else None
    value_gap = None if mean_correct is None or mean_wrong is None else float(mean_correct - mean_wrong)
    binary_labels = np.asarray([1.0 if reward >= 0.5 else 0.0 for reward in rewards], dtype=np.float64)
    return {
        "num_trajectories": int(rewards.size),
        "avg_policy_accuracy": r_bar,
        "constant_mse": constant_mse,
        "constant_mse_binary_formula": float(r_bar * (1.0 - r_bar)),
        "critic_mse": critic_mse,
        "relative_mse_improvement": relative,
        "mean_value_correct": mean_correct,
        "mean_value_wrong": mean_wrong,
        "value_gap_correct_minus_wrong": value_gap,
        "pearson_value_reward": _pearson(values, rewards),
        "spearman_value_reward": _spearman(values, rewards),
        "auroc_value_predicts_reward": _auroc_binary(values, binary_labels),
        "mean_critic_value": float(np.mean(values)),
        "std_critic_value": float(np.std(values)),
        "min_critic_value": float(np.min(values)),
        "max_critic_value": float(np.max(values)),
        "value_position": value_position,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")



def safe_path_name(value: str) -> str:
    safe = value.rstrip("/").replace("/", "__")
    safe = safe.replace(" ", "_").replace(":", "_")
    return safe or "root"


def critic_label_from_path(path: Path, used_labels: set[str]) -> str:
    parts = [part for part in path.parts if part]
    if len(parts) >= 2 and parts[-1] in {"critic", "actor"}:
        base = "__".join(parts[-3:]) if len(parts) >= 3 else "__".join(parts[-2:])
    elif len(parts) >= 2:
        base = "__".join(parts[-2:])
    else:
        base = path.name or "critic"
    label = safe_path_name(base)
    if label not in used_labels:
        used_labels.add(label)
        return label
    index = 1
    while f"{label}_{index}" in used_labels:
        index += 1
    unique = f"{label}_{index}"
    used_labels.add(unique)
    return unique


def write_trajectory_bank(output_dir: Path, trajectories: list[TrajectoryRecord], *, r_bar: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "trajectory_bank.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            row = {
                "trajectory_id": int(trajectory.trajectory_id),
                "prompt_id": int(trajectory.prompt_id),
                "prompt": trajectory.prompt,
                "response": trajectory.response,
                "response_token_ids": trajectory.response_token_ids,
                "reward": float(trajectory.reward),
                "r_bar": r_bar,
                "constant_loss": float((r_bar - float(trajectory.reward)) ** 2),
                "response_length": len(trajectory.response_token_ids),
                "data_source": trajectory.data_source,
                "ground_truth": trajectory.ground_truth,
            }
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def evaluate_critic_position(
    *,
    critic,
    critic_dir: Path,
    critic_label: str,
    tokenizer,
    trajectories: list[TrajectoryRecord],
    r_bar: float,
    eos_token_ids: set[int],
    value_position: str,
    output_dir: Path,
    device: torch.device,
    max_prompt_length: int | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows_path = output_dir / "trajectory_values.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            readout = compute_critic_readout(
                critic=critic,
                tokenizer=tokenizer,
                trajectory=trajectory,
                eos_token_ids=eos_token_ids,
                value_position=value_position,
                device=device,
                max_prompt_length=max_prompt_length,
            )
            constant_loss = float((r_bar - float(trajectory.reward)) ** 2)
            critic_loss = float((readout.value - float(trajectory.reward)) ** 2)
            row = {
                "trajectory_id": int(trajectory.trajectory_id),
                "prompt_id": int(trajectory.prompt_id),
                "prompt": trajectory.prompt,
                "response": trajectory.response,
                "response_token_ids": trajectory.response_token_ids,
                "reward": float(trajectory.reward),
                "r_bar": r_bar,
                "constant_loss": constant_loss,
                "critic_value": float(readout.value),
                "critic_loss": critic_loss,
                "critic_checkpoint_dir": str(critic_dir),
                "critic_label": critic_label,
                "value_position": value_position,
                "response_length": int(readout.response_length),
                "ended_with_eos": bool(readout.ended_with_eos),
                "selected_response_index": readout.selected_response_index,
                "selected_full_index": readout.selected_full_index,
                "data_source": trajectory.data_source,
                "ground_truth": trajectory.ground_truth,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()

    summary = summarize(rows, value_position=value_position)
    summary.update({"critic_checkpoint_dir": str(critic_dir), "critic_label": critic_label})
    run_config = dict(config)
    run_config.update(
        {
            "critic_checkpoint_dir": str(critic_dir),
            "critic_label": critic_label,
            "value_position": value_position,
            "output_dir": str(output_dir),
        }
    )
    write_json(output_dir / "config.json", run_config)
    write_json(output_dir / "summary_metrics.json", summary)
    return summary


def write_comparison_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    payload = {
        "num_runs": len(summaries),
        "runs": summaries,
        "best_by_critic_mse": min(summaries, key=lambda item: float(item["critic_mse"])),
        "best_by_relative_mse_improvement": max(
            summaries,
            key=lambda item: float("-inf") if item.get("relative_mse_improvement") is None else float(item["relative_mse_improvement"]),
        ),
    }
    write_json(output_dir / "comparison_summary.json", payload)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    value_positions = list(args.value_position)
    if not value_positions:
        raise ValueError("At least one --value_position must be provided.")
    primary_value_position = value_positions[0]

    aggregate_input_dirs = args.aggregate_input_dirs
    if aggregate_input_dirs:
        input_dirs = [Path(path).expanduser().resolve() for path in aggregate_input_dirs]
        summary = aggregate_output_dirs(input_dirs, output_dir, value_position=primary_value_position)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    actor_dir = Path(args.actor_checkpoint_dir).expanduser().resolve()
    critic_dirs = [Path(path).expanduser().resolve() for path in args.critic_checkpoint_dir]
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    response_key = args.response_key if args.response_key else None
    dtype = resolve_dtype(args.dtype)
    actor_device = resolve_device(args.device)
    critic_device = resolve_device(args.critic_device or args.device)

    tokenizer = load_tokenizer(actor_dir, trust_remote_code=args.trust_remote_code)
    examples = load_examples(
        dataset_path,
        tokenizer=tokenizer,
        prompt_key=args.prompt_key,
        response_key=response_key,
        start_index=int(args.start_index),
        max_examples=None if int(args.max_examples) < 0 else int(args.max_examples),
        shuffle_examples=bool(args.shuffle),
        seed=int(args.seed),
    )

    examples = shard_sequence(examples, num_shards=int(args.num_shards), shard_id=int(args.shard_id))

    if args.response_bank_path:
        trajectories = load_response_bank(
            Path(args.response_bank_path).expanduser().resolve(),
            examples=examples,
            tokenizer=tokenizer,
            max_trajectories=None if int(args.max_examples) < 0 else int(args.max_examples) * max(1, int(args.num_samples_per_prompt)),
            math_dapo_binary_reward=bool(args.math_dapo_binary_reward),
        )
    else:
        if args.generation_backend == "vllm":
            llm = build_vllm_engine(args, actor_dir)
            trajectories = generate_trajectories_with_vllm(
                examples=examples,
                llm=llm,
                tokenizer=tokenizer,
                args=args,
            )
            cleanup_vllm_engine(llm)
        else:
            actor = load_actor_model(actor_dir, dtype=dtype, device=actor_device, trust_remote_code=args.trust_remote_code)
            trajectories = generate_trajectories_with_torch(
                examples=examples,
                actor=actor,
                tokenizer=tokenizer,
                args=args,
                device=actor_device,
            )
            del actor
            if actor_device.type == "cuda":
                torch.cuda.empty_cache()

    eos_token_ids = set(resolve_eos_token_ids(actor_dir, tokenizer))
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(int(tokenizer.eos_token_id))

    rewards = np.asarray([trajectory.reward for trajectory in trajectories], dtype=np.float64)
    if rewards.size == 0:
        raise ValueError("No trajectories available for evaluation.")
    r_bar = float(np.mean(rewards))
    write_trajectory_bank(output_dir, trajectories, r_bar=r_bar)

    base_config = vars(args).copy()
    base_config.update(
        {
            "actor_checkpoint_dir": str(actor_dir),
            "critic_checkpoint_dirs": [str(path) for path in critic_dirs],
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "num_loaded_examples": len(examples),
            "num_trajectories": len(trajectories),
            "avg_policy_accuracy": r_bar,
            "eos_token_ids": sorted(eos_token_ids),
        }
    )
    write_json(output_dir / "config.json", base_config)

    used_labels: set[str] = set()
    critic_labels = [critic_label_from_path(path, used_labels) for path in critic_dirs]
    multi_run = bool(args.always_write_run_subdirs) or len(critic_dirs) > 1 or len(value_positions) > 1
    summaries: list[dict[str, Any]] = []

    for critic_dir, critic_label in zip(critic_dirs, critic_labels, strict=True):
        critic = load_critic_model(critic_dir, dtype=dtype, device=critic_device, trust_remote_code=args.trust_remote_code)
        for value_position in value_positions:
            run_output_dir = output_dir if not multi_run else output_dir / critic_label / value_position
            summary = evaluate_critic_position(
                critic=critic,
                critic_dir=critic_dir,
                critic_label=critic_label,
                tokenizer=tokenizer,
                trajectories=trajectories,
                r_bar=r_bar,
                eos_token_ids=eos_token_ids,
                value_position=value_position,
                output_dir=run_output_dir,
                device=critic_device,
                max_prompt_length=int(args.max_prompt_length) if int(args.max_prompt_length) > 0 else None,
                config=base_config,
            )
            summaries.append(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        del critic
        gc.collect()
        if critic_device.type == "cuda":
            torch.cuda.empty_cache()

    if multi_run:
        write_comparison_summary(output_dir, summaries)
    else:
        write_json(output_dir / "summary_metrics.json", summaries[0])


if __name__ == "__main__":
    main()
