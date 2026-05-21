from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# vLLM must see this before any CUDA initialization; otherwise its engine can
# fork after torch.cuda has been touched and fail with "Cannot re-initialize
# CUDA in forked subprocess" on Slurm/TACC.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# This environment can lack a linkable libcudart for FlashInfer's JIT sampler
# in some runs. vLLM has a non-FlashInfer sampler fallback, so default to that
# unless the caller explicitly overrides the variable.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.data import ExampleRecord, load_examples, score_response
from value_decoding.decoding import ActorSamplingMode, critic_sequence_values, set_decode_seed

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
except ImportError:
    LLM = None
    SamplingParams = None
    TokensPrompt = None


@dataclass(frozen=True)
class EvalConfig:
    dataset_path: Path
    checkpoint_paths: list[Path]
    output_dir: Path
    prompt_key: str
    response_key: str | None
    start_index: int
    max_examples: int | None
    shuffle_examples: bool
    seed: int
    max_prompt_length: int
    max_new_tokens: int
    batch_size: int
    actor_micro_batch_size: int
    dtype: str
    device: str | None
    actor_sampling_mode: str
    actor_temperature: float
    actor_top_p: float
    actor_top_k: int
    generation_backend: str
    vllm_gpu_memory_utilization: float
    vllm_tensor_parallel_size: int
    vllm_max_model_len: int | None
    vllm_max_num_seqs: int | None
    vllm_enforce_eager: bool
    gamma: float
    lam: float
    cliprange_value: float
    loss_agg_mode: str
    trust_remote_code: bool
    skip_merge: bool
    hf_source_dir: Path | None
    merged_checkpoint_root: Path | None
    save_trajectories: bool
    require_critic: bool


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate critic training metrics on a held-out parquet set by generating actor rollouts, "
            "scoring rewards, and running the critic on the resulting trajectories."
        )
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--checkpoint_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("value_decoding/output/critic_test_metrics"))
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4, help="Critic evaluation batch size.")
    parser.add_argument("--actor_micro_batch_size", type=int, default=1, help="Generation batch size for HF actor.generate.")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--actor_sampling_mode", choices=[mode.value for mode in ActorSamplingMode], default="greedy")
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--actor_top_p", type=float, default=1.0)
    parser.add_argument("--actor_top_k", type=int, default=0)
    parser.add_argument("--generation_backend", choices=("torch", "vllm"), default="torch")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--cliprange_value", type=float, default=0.5)
    parser.add_argument("--loss_agg_mode", choices=("token-mean", "seq-mean-token-sum", "seq-mean-token-mean"), default="token-mean")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--hf_source_dir", type=Path, default=None)
    parser.add_argument(
        "--merged_checkpoint_root",
        type=Path,
        default=None,
        help=(
            "Optional shared directory for merged Hugging Face actor/critic checkpoints. "
            "Use this when running multiple seeds into separate output directories to avoid re-merging FSDP shards."
        ),
    )
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument(
        "--require_critic",
        action="store_true",
        help="Raise an error if a checkpoint does not look like a critic/value-head checkpoint.",
    )
    args = parser.parse_args()

    if args.start_index < 0:
        raise ValueError(f"--start_index must be >= 0, got {args.start_index}")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError(f"--max_examples must be > 0, got {args.max_examples}")
    if args.max_prompt_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--max_prompt_length and --max_new_tokens must be > 0")
    if args.batch_size <= 0 or args.actor_micro_batch_size <= 0:
        raise ValueError("--batch_size and --actor_micro_batch_size must be > 0")
    if args.actor_top_k < 0:
        raise ValueError(f"--actor_top_k must be >= 0, got {args.actor_top_k}")
    if not 0.0 < args.actor_top_p <= 1.0:
        raise ValueError(f"--actor_top_p must be in (0, 1], got {args.actor_top_p}")
    if not 0.0 < args.vllm_gpu_memory_utilization <= 1.0:
        raise ValueError(
            "--vllm_gpu_memory_utilization must be in (0, 1], "
            f"got {args.vllm_gpu_memory_utilization}"
        )
    if args.vllm_tensor_parallel_size <= 0:
        raise ValueError(f"--vllm_tensor_parallel_size must be > 0, got {args.vllm_tensor_parallel_size}")
    if args.vllm_max_model_len is not None and args.vllm_max_model_len <= 0:
        raise ValueError(f"--vllm_max_model_len must be > 0, got {args.vllm_max_model_len}")
    if args.vllm_max_num_seqs is not None and args.vllm_max_num_seqs <= 0:
        raise ValueError(f"--vllm_max_num_seqs must be > 0, got {args.vllm_max_num_seqs}")

    return EvalConfig(**vars(args))


def _checkpoint_name(path: Path) -> str:
    path = path.expanduser().resolve()
    if path.name in {"actor", "critic"} and path.parent.name:
        return f"{path.parent.name}_{path.name}"
    return path.name or path.parent.name


def _load_actor(model_dir: Path, *, dtype: torch.dtype, device: torch.device, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def _resolve_vllm_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.lower()
    if normalized == "bf16":
        return "bfloat16"
    if normalized == "fp16":
        return "float16"
    if normalized == "fp32":
        return "float32"
    return dtype_name


def _load_vllm_actor(model_dir: Path, *, config: EvalConfig):
    if LLM is None:
        raise ImportError("vLLM is required when --generation_backend=vllm, but it is not installed.")
    kwargs = {
        "model": str(model_dir),
        "tokenizer": str(model_dir),
        "dtype": _resolve_vllm_dtype_name(config.dtype),
        "trust_remote_code": config.trust_remote_code,
        "tensor_parallel_size": int(config.vllm_tensor_parallel_size),
        "gpu_memory_utilization": float(config.vllm_gpu_memory_utilization),
        "enforce_eager": bool(config.vllm_enforce_eager),
    }
    if config.vllm_max_model_len is not None:
        kwargs["max_model_len"] = int(config.vllm_max_model_len)
    if config.vllm_max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(config.vllm_max_num_seqs)
    return LLM(**kwargs)


def _resolve_component_dirs(config: EvalConfig, checkpoint_path: Path, output_dir: Path) -> tuple[Path, Path]:
    merged_root = config.merged_checkpoint_root or output_dir / "merged_hf"
    merged_root = merged_root / _checkpoint_name(checkpoint_path)
    actor_dir = ensure_merged_component_checkpoint(
        checkpoint_path,
        component="actor",
        merged_root=merged_root,
        hf_source_dir=config.hf_source_dir,
        skip_merge=config.skip_merge,
    )
    critic_dir = ensure_merged_component_checkpoint(
        checkpoint_path,
        component="critic",
        merged_root=merged_root,
        hf_source_dir=config.hf_source_dir,
        skip_merge=config.skip_merge,
    )
    return actor_dir, critic_dir


def _checkpoint_has_weight_key(model_dir: Path, fragments: tuple[str, ...]) -> bool:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            return False
        return any(any(fragment in key for fragment in fragments) for key in weight_map)
    return False


def _looks_like_critic_checkpoint(model_dir: Path, *, trust_remote_code: bool = False) -> bool:
    try:
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    except Exception:
        return False
    if getattr(config, "value_head_type", None) is not None:
        return True
    if getattr(config, "num_labels", None) == 1:
        return True
    if _checkpoint_has_weight_key(model_dir, ("v_head", "score.", "classifier.", "prompt_prior_head")):
        return True
    return False


def _accuracy_metrics(trajectories: list[dict[str, Any]]) -> dict[str, float]:
    rewards = [float(row["reward"]) for row in trajectories if row.get("response_mask")]
    accuracy = sum(rewards) / max(len(rewards), 1)
    return {
        "num_examples": float(len(trajectories)),
        "num_valid_examples": float(len(rewards)),
        "reward_mean": accuracy,
        "accuracy": accuracy,
        "acc/mean@1": accuracy,
    }


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_response_mask(response_ids: list[int], eos_token_ids: tuple[int, ...]) -> list[int]:
    eos_set = set(int(token_id) for token_id in eos_token_ids)
    response_mask: list[int] = []
    seen_eos = False
    for token_id in response_ids:
        if seen_eos:
            response_mask.append(0)
            continue
        response_mask.append(1)
        if int(token_id) in eos_set:
            seen_eos = True
    return response_mask


def _trim_response_ids(response_ids: Iterable[int], *, tokenizer, eos_token_ids: tuple[int, ...]) -> list[int]:
    eos_set = set(int(token_id) for token_id in eos_token_ids)
    trimmed: list[int] = []
    for token_id in response_ids:
        token_id = int(token_id)
        if tokenizer.pad_token_id is not None and token_id == int(tokenizer.pad_token_id):
            continue
        trimmed.append(token_id)
        if token_id in eos_set:
            break
    return trimmed


def _generate_batch(
    *,
    actor,
    tokenizer,
    examples: list[ExampleRecord],
    device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[dict[str, Any]]:
    prompts = [example.prompt_text for example in examples]
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_token_type_ids=False,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    prompt_lengths = attention_mask.long().sum(dim=-1)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": list(eos_token_ids) if len(eos_token_ids) > 1 else (eos_token_ids[0] if eos_token_ids else tokenizer.eos_token_id),
        "return_dict_in_generate": True,
        "use_cache": True,
    }
    if sampling_mode == ActorSamplingMode.GREEDY.value or temperature <= 0.0:
        generation_kwargs.update({"do_sample": False})
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if top_k > 0:
            generation_kwargs["top_k"] = top_k

    with torch.inference_mode():
        generated = actor.generate(**generation_kwargs)

    sequences = generated.sequences
    input_width = int(input_ids.shape[1])
    rows: list[dict[str, Any]] = []
    for row_idx, example in enumerate(examples):
        prompt_length = int(prompt_lengths[row_idx].item())
        prompt_ids = input_ids[row_idx][attention_mask[row_idx].bool()].detach().cpu().tolist()
        # HF generate appends new tokens after the padded input width, not after each
        # row's unpadded prompt length. This matches both left- and right-padded batches.
        response_ids = _trim_response_ids(
            sequences[row_idx, input_width:].detach().cpu().tolist(),
            tokenizer=tokenizer,
            eos_token_ids=eos_token_ids,
        )
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        score = score_response(example, response_text)
        response_mask = _build_response_mask(response_ids, eos_token_ids)
        rows.append(
            {
                "example": example,
                "prompt_ids": [int(x) for x in prompt_ids],
                "response_ids": [int(x) for x in response_ids],
                "response_mask": response_mask,
                "response_text": response_text,
                "reward": float(score),
            }
        )
    return rows


def generate_trajectories_torch(
    *,
    actor,
    tokenizer,
    examples: list[ExampleRecord],
    config: EvalConfig,
    device: torch.device,
    eos_token_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for batch in tqdm(list(_chunks(examples, config.actor_micro_batch_size)), desc="Generating", unit="batch"):
        trajectories.extend(
            _generate_batch(
                actor=actor,
                tokenizer=tokenizer,
                examples=batch,
                device=device,
                max_prompt_length=config.max_prompt_length,
                max_new_tokens=config.max_new_tokens,
                eos_token_ids=eos_token_ids,
                sampling_mode=config.actor_sampling_mode,
                temperature=config.actor_temperature,
                top_p=config.actor_top_p,
                top_k=config.actor_top_k,
            )
        )
    return trajectories


def _vllm_output_token_ids(output: Any) -> tuple[int, ...]:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is None:
        token_ids = getattr(output, "output_token_ids", None)
    if token_ids is None:
        return ()
    return tuple(int(token_id) for token_id in token_ids)


def _vllm_output_text(output: Any, tokenizer) -> str:
    text = getattr(output, "text", None)
    if text is not None:
        return str(text)
    return tokenizer.decode(_vllm_output_token_ids(output), skip_special_tokens=True)


def generate_trajectories_vllm(
    *,
    llm,
    tokenizer,
    examples: list[ExampleRecord],
    config: EvalConfig,
    eos_token_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    if SamplingParams is None:
        raise ImportError("vLLM is required when --generation_backend=vllm, but it is not installed.")

    trajectories: list[dict[str, Any]] = []
    vllm_top_k = 0 if int(config.actor_top_k) <= 0 else int(config.actor_top_k)
    do_greedy = config.actor_sampling_mode == ActorSamplingMode.GREEDY.value or config.actor_temperature <= 0.0
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0 if do_greedy else float(config.actor_temperature),
        top_p=1.0 if do_greedy else float(config.actor_top_p),
        top_k=0 if do_greedy else vllm_top_k,
        max_tokens=int(config.max_new_tokens),
        seed=int(config.seed),
        stop_token_ids=list(eos_token_ids),
        skip_special_tokens=True,
    )

    for batch in tqdm(list(_chunks(examples, config.actor_micro_batch_size)), desc="Generating", unit="batch"):
        prompt_token_ids_by_example: list[list[int]] = []
        prompts: list[Any] = []
        for example in batch:
            if example.prompt_token_ids is not None:
                prompt_token_ids = list(example.prompt_token_ids)
            else:
                tokenized = tokenizer(
                    example.prompt_text,
                    truncation=True,
                    max_length=config.max_prompt_length,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                prompt_token_ids = [int(token_id) for token_id in tokenized["input_ids"]]
            prompt_token_ids_by_example.append(prompt_token_ids)
            prompts.append(
                TokensPrompt(prompt_token_ids=prompt_token_ids)
                if TokensPrompt is not None
                else {"prompt_token_ids": prompt_token_ids}
            )

        request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        if len(request_outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(request_outputs)} outputs for {len(batch)} prompts.")
        for example, prompt_ids, request_output in zip(batch, prompt_token_ids_by_example, request_outputs, strict=True):
            if not request_output.outputs:
                response_ids: list[int] = []
                response_text = ""
            else:
                output = request_output.outputs[0]
                response_ids = _trim_response_ids(
                    _vllm_output_token_ids(output),
                    tokenizer=tokenizer,
                    eos_token_ids=eos_token_ids,
                )
                # Decode from the exact token IDs we store/score so text and IDs
                # cannot diverge after EOS/pad trimming.
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            score = score_response(example, response_text)
            trajectories.append(
                {
                    "example": example,
                    "prompt_ids": [int(token_id) for token_id in prompt_ids],
                    "response_ids": response_ids,
                    "response_mask": _build_response_mask(response_ids, eos_token_ids),
                    "response_text": response_text,
                    "reward": float(score),
                }
            )
    return trajectories


def _pad_2d(rows: list[list[int]], *, pad_value: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(row) for row in rows), default=0)
    if max_len <= 0:
        raise ValueError("Cannot pad an empty token batch.")
    tensor = torch.full((len(rows), max_len), int(pad_value), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for idx, row in enumerate(rows):
        if not row:
            continue
        values = torch.tensor(row, dtype=torch.long, device=device)
        tensor[idx, : values.numel()] = values
        mask[idx, : values.numel()] = 1
    return tensor, mask


def _token_level_rewards(rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    token_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    lengths = response_mask.long().sum(dim=-1)
    valid_rows = lengths > 0
    if valid_rows.any():
        last_indices = lengths[valid_rows] - 1
        token_rewards[valid_rows, last_indices] = rewards[valid_rows].float()
    return token_rewards


def _compute_gae_returns(
    *,
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    with torch.no_grad():
        next_values = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
        last_gae_lam = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
        advantages_reversed: list[torch.Tensor] = []
        for t in reversed(range(token_level_rewards.shape[-1])):
            delta = token_level_rewards[:, t] + gamma * next_values - values[:, t]
            candidate = delta + gamma * lam * last_gae_lam
            mask_t = response_mask[:, t].to(dtype=values.dtype)
            next_values = values[:, t] * mask_t + (1 - mask_t) * next_values
            last_gae_lam = candidate * mask_t + (1 - mask_t) * last_gae_lam
            advantages_reversed.append(last_gae_lam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        return advantages + values


def _aggregate_loss(loss_mat: torch.Tensor, response_mask: torch.Tensor, mode: str) -> torch.Tensor:
    mask = response_mask.to(dtype=loss_mat.dtype)
    if mode == "token-mean":
        return (loss_mat * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    if mode == "seq-mean-token-sum":
        seq_losses = (loss_mat * mask).sum(dim=-1)
        seq_mask = (mask.sum(dim=-1) > 0).to(dtype=loss_mat.dtype)
        return (seq_losses * seq_mask).sum() / torch.clamp(seq_mask.sum(), min=1.0)
    if mode == "seq-mean-token-mean":
        token_counts = mask.sum(dim=-1)
        seq_losses = (loss_mat * mask).sum(dim=-1) / torch.clamp(token_counts, min=1.0)
        seq_mask = (token_counts > 0).to(dtype=loss_mat.dtype)
        return (seq_losses * seq_mask).sum() / torch.clamp(seq_mask.sum(), min=1.0)
    raise ValueError(f"Unsupported loss_agg_mode: {mode}")


def _vf_loss(
    *,
    vpreds: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str,
) -> float:
    # At evaluation time there is no "old critic" snapshot distinct from the
    # checkpoint being evaluated, so callers pass old_values=vpreds. This makes
    # PPO clipping a no-op while preserving the training loss formula.
    vpredclipped = torch.clamp(vpreds, old_values - cliprange_value, old_values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped = torch.maximum(vf_losses1, vf_losses2)
    return float((0.5 * _aggregate_loss(clipped, response_mask, loss_agg_mode)).detach().cpu().item())


def _endpoint_values(
    full_values: torch.Tensor,
    *,
    prompt_lengths: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Extract prompt-end and full-trajectory-end critic values.

    These endpoint diagnostics only need two state values per example:
    ``prompt_len - 1`` and the last response-aligned value position. This matches
    PPO training, where critic values are sliced as ``[-response_len - 1 : -1]``.
    Treat rows whose
    endpoint indices are unavailable as invalid for endpoint metrics rather than
    falling back to an earlier response position.
    """
    if full_values.dim() != 2:
        raise ValueError(f"Expected full_values shape (batch, seq), got {tuple(full_values.shape)}")
    if prompt_lengths.dim() != 1 or prompt_lengths.shape[0] != full_values.shape[0]:
        raise ValueError(
            f"prompt_lengths must have shape ({full_values.shape[0]},), got {tuple(prompt_lengths.shape)}"
        )
    if response_mask.dim() != 2 or response_mask.shape[0] != full_values.shape[0]:
        raise ValueError(
            f"response_mask must have shape ({full_values.shape[0]}, response_width), "
            f"got {tuple(response_mask.shape)}"
        )
    response_lengths = response_mask.to(dtype=torch.bool).sum(dim=-1).to(dtype=torch.long)
    prompt_end_indices = prompt_lengths.to(device=full_values.device, dtype=torch.long) - 1
    response_lengths = response_lengths.to(device=full_values.device)
    trajectory_end_indices = prompt_end_indices + response_lengths - 1
    endpoint_available = (
        (prompt_end_indices >= 0)
        & (response_lengths > 0)
        & (trajectory_end_indices < full_values.shape[1])
    )
    missing_endpoint_examples = int((~endpoint_available & (response_lengths > 0)).sum().item())
    valid_rows = endpoint_available
    if not valid_rows.any():
        empty = full_values.new_empty((0,), dtype=full_values.dtype)
        return valid_rows, empty, empty, missing_endpoint_examples
    prompt_end_values = full_values[valid_rows].gather(
        1, prompt_end_indices[valid_rows].long().unsqueeze(-1)
    ).squeeze(-1)
    trajectory_end_values = full_values[valid_rows].gather(
        1, trajectory_end_indices[valid_rows].long().unsqueeze(-1)
    ).squeeze(-1)
    return valid_rows, prompt_end_values, trajectory_end_values, missing_endpoint_examples


def _response_aligned_values(
    full_values: torch.Tensor,
    *,
    prompt_lengths: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Extract response-aligned critic values using the same offset as PPO training.

    For a sequence ``prompt + response``, the critic value at ``prompt_len - 1`` is
    the prompt-end state before generating the first response token. The next
    positions align with subsequent response states. This per-row gather is needed
    because prompts have variable lengths inside a padded evaluation batch.

    Some critic implementations return values only up to the longest valid
    unpadded sequence in a batch instead of the padded input width. When a row
    needs value positions beyond that returned width, keep the available prefix
    and clear the corresponding tail of ``response_mask`` so downstream metrics
    are computed over positions that actually have critic predictions.
    """
    response_width = response_mask.shape[1]
    if response_width <= 0:
        raise ValueError(f"response_width must be > 0, got {response_width}")
    if full_values.dim() != 2:
        raise ValueError(f"Expected full_values shape (batch, seq), got {tuple(full_values.shape)}")
    if full_values.shape[1] <= 0:
        raise ValueError("Critic returned zero value positions.")
    if prompt_lengths.dim() != 1 or prompt_lengths.shape[0] != full_values.shape[0]:
        raise ValueError(
            f"prompt_lengths must have shape ({full_values.shape[0]},), got {tuple(prompt_lengths.shape)}"
        )
    if response_mask.dim() != 2 or response_mask.shape[0] != full_values.shape[0]:
        raise ValueError(
            f"response_mask must have shape ({full_values.shape[0]}, response_width), "
            f"got {tuple(response_mask.shape)}"
        )
    start = prompt_lengths.to(device=full_values.device, dtype=torch.long) - 1
    if torch.any(start < 0):
        raise ValueError("Each prompt must contain at least one token.")

    offsets = torch.arange(response_width, device=full_values.device, dtype=torch.long)
    gather_indices = start.unsqueeze(1) + offsets.unsqueeze(0)
    available_mask = gather_indices < full_values.shape[1]
    safe_indices = gather_indices.clamp(min=0, max=full_values.shape[1] - 1)
    values = full_values.gather(1, safe_indices) * available_mask.to(dtype=full_values.dtype)

    aligned_mask = response_mask * available_mask.to(dtype=response_mask.dtype)
    truncated_positions = response_mask.to(dtype=torch.bool) & ~available_mask
    truncated_tokens = int(truncated_positions.sum().detach().cpu().item())
    truncated_examples = int(truncated_positions.any(dim=-1).sum().detach().cpu().item())
    return values, aligned_mask, truncated_tokens, truncated_examples


def evaluate_critic(
    *,
    critic,
    tokenizer,
    trajectories: list[dict[str, Any]],
    config: EvalConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    total_valid = 0
    endpoint_valid_count = 0
    reward_sum = 0.0
    prompt_gap_sum = 0.0
    traj_gap_sum = 0.0
    vf_loss_token_weighted_sum = 0.0
    vf_loss_seq_weighted_sum = 0.0
    token_count = 0.0
    seq_count = 0.0
    alignment_truncated_tokens = 0
    alignment_truncated_examples = 0
    endpoint_missing_examples = 0
    per_rows: list[dict[str, Any]] = []

    for batch in tqdm(list(_chunks(trajectories, config.batch_size)), desc="Critic eval", unit="batch"):
        full_ids = [row["prompt_ids"] + row["response_ids"] for row in batch]
        response_ids = [row["response_ids"] for row in batch]
        response_masks = [row["response_mask"] for row in batch]
        prompt_lengths = torch.tensor([len(row["prompt_ids"]) for row in batch], dtype=torch.long, device=device)
        input_ids, attention_mask = _pad_2d(full_ids, pad_value=tokenizer.pad_token_id, device=device)
        _, response_present_mask = _pad_2d(response_ids, pad_value=tokenizer.pad_token_id, device=device)
        response_mask = torch.zeros_like(response_present_mask)
        for idx, mask_row in enumerate(response_masks):
            if mask_row:
                response_mask[idx, : len(mask_row)] = torch.tensor(mask_row, dtype=torch.long, device=device)
        rewards = torch.tensor([float(row["reward"]) for row in batch], dtype=torch.float32, device=device)

        with torch.inference_mode():
            full_values = critic_sequence_values(critic, input_ids=input_ids, attention_mask=attention_mask).float()
        original_response_mask = response_mask
        reward_valid_rows = original_response_mask.bool().any(dim=-1)
        valid_rows, prompt_end_values, trajectory_end_values, missing_endpoint_examples = _endpoint_values(
            full_values,
            prompt_lengths=prompt_lengths,
            response_mask=original_response_mask,
        )
        endpoint_missing_examples += missing_endpoint_examples

        values, response_mask, batch_truncated_tokens, batch_truncated_examples = _response_aligned_values(
            full_values,
            prompt_lengths=prompt_lengths,
            response_mask=original_response_mask,
        )
        alignment_truncated_tokens += batch_truncated_tokens
        alignment_truncated_examples += batch_truncated_examples
        values = values * response_mask.to(dtype=values.dtype)
        token_rewards = _token_level_rewards(rewards, response_mask)
        returns = _compute_gae_returns(
            token_level_rewards=token_rewards,
            values=values,
            response_mask=response_mask,
            gamma=config.gamma,
            lam=config.lam,
        )
        reward_values = rewards[reward_valid_rows]
        if reward_values.numel() > 0:
            total_valid += int(reward_values.numel())
            reward_sum += float(reward_values.sum().detach().cpu().item())

        endpoint_rewards = rewards[valid_rows]
        if endpoint_rewards.numel() > 0:
            prompt_gaps = torch.abs(prompt_end_values - endpoint_rewards)
            trajectory_gaps = torch.abs(trajectory_end_values - endpoint_rewards)
            endpoint_count = int(endpoint_rewards.numel())
            endpoint_valid_count += endpoint_count
            prompt_gap_sum += float(prompt_gaps.sum().detach().cpu().item())
            traj_gap_sum += float(trajectory_gaps.sum().detach().cpu().item())
        else:
            prompt_gaps = full_values.new_empty((0,), dtype=full_values.dtype)
            trajectory_gaps = full_values.new_empty((0,), dtype=full_values.dtype)

        loss_mat = torch.maximum(
            (values - returns) ** 2,
            (torch.clamp(values, values - config.cliprange_value, values + config.cliprange_value) - returns) ** 2,
        )
        batch_token_count = float(response_mask.sum().detach().cpu().item())
        vf_valid_rows = response_mask.bool().any(dim=-1)
        batch_seq_count = float(vf_valid_rows.sum().detach().cpu().item())
        token_loss = 0.5 * ((loss_mat * response_mask).sum() / torch.clamp(response_mask.sum(), min=1.0))
        seq_losses = (loss_mat * response_mask).sum(dim=-1) / torch.clamp(response_mask.sum(dim=-1), min=1)
        seq_loss = 0.5 * (seq_losses[vf_valid_rows].sum() / max(int(batch_seq_count), 1))
        vf_loss_token_weighted_sum += float(token_loss.detach().cpu().item()) * batch_token_count
        vf_loss_seq_weighted_sum += float(seq_loss.detach().cpu().item()) * batch_seq_count
        token_count += batch_token_count
        seq_count += batch_seq_count

        batch_vf_loss = _vf_loss(
            vpreds=values,
            old_values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=config.cliprange_value,
            loss_agg_mode=config.loss_agg_mode,
        )

        valid_indices = torch.where(valid_rows)[0].detach().cpu().tolist()
        prompt_values_cpu = prompt_end_values.detach().cpu().tolist()
        traj_values_cpu = trajectory_end_values.detach().cpu().tolist()
        returns_cpu = returns.detach().cpu()
        for local_out_idx, batch_idx in enumerate(valid_indices):
            row = batch[batch_idx]
            row_mask = response_mask[batch_idx].bool().detach().cpu()
            valid_returns = returns_cpu[batch_idx][row_mask]
            per_rows.append(
                {
                    "example_id": row["example"].example_id,
                    "data_source": row["example"].data_source,
                    "reward": float(row["reward"]),
                    "prompt_end_value": float(prompt_values_cpu[local_out_idx]),
                    "trajectory_end_value": float(traj_values_cpu[local_out_idx]),
                    "prompt_end_vs_reward_gap": float(prompt_gaps[local_out_idx].detach().cpu().item()),
                    "trajectory_end_vs_reward_gap": float(trajectory_gaps[local_out_idx].detach().cpu().item()),
                    "response_tokens": int(row_mask.sum().item()),
                    "return_first": float(valid_returns[0].item()) if valid_returns.numel() else math.nan,
                    "return_last": float(valid_returns[-1].item()) if valid_returns.numel() else math.nan,
                    "batch_vf_loss": batch_vf_loss,
                    "response_text": row["response_text"],
                }
            )

    metrics = {
        "num_examples": float(len(trajectories)),
        "num_valid_examples": float(total_valid),
        "reward_mean": reward_sum / max(total_valid, 1),
        "accuracy": reward_sum / max(total_valid, 1),
        "acc/mean@1": reward_sum / max(total_valid, 1),
        "critic/prompt_end_vs_reward_gap": prompt_gap_sum / max(endpoint_valid_count, 1),
        "critic/trajectory_end_vs_reward_gap": traj_gap_sum / max(endpoint_valid_count, 1),
        # Same name as training. With old_values equal to the evaluated critic's
        # current values, value clipping is inactive; for the default token-mean
        # aggregation this is 0.5 * mean((V - GAE_return)^2) over valid response tokens.
        "critic/vf_loss": vf_loss_token_weighted_sum / max(token_count, 1.0)
        if config.loss_agg_mode == "token-mean"
        else vf_loss_seq_weighted_sum / max(seq_count, 1.0),
        "critic/vf_loss_token_mean": vf_loss_token_weighted_sum / max(token_count, 1.0),
        "critic/vf_loss_seq_mean_token_mean": vf_loss_seq_weighted_sum / max(seq_count, 1.0),
        "critic/value_alignment_truncated_tokens": float(alignment_truncated_tokens),
        "critic/value_alignment_truncated_examples": float(alignment_truncated_examples),
        "critic/endpoint_valid_examples": float(endpoint_valid_count),
        "critic/endpoint_missing_examples": float(endpoint_missing_examples),
        "total_response_tokens": token_count,
        "gamma": float(config.gamma),
        "lam": float(config.lam),
        "cliprange_value": float(config.cliprange_value),
    }
    return metrics, per_rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_json(output_dir / "summary.json", {"runs": summary_rows})


def _clear_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _cleanup_vllm(device: torch.device) -> None:
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as exc:
        print(f"WARNING: vLLM cleanup skipped or partially failed: {exc}")
    _clear_cuda_cache(device)


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_decode_seed(config.seed)
    dtype = resolve_dtype(config.dtype)
    device = resolve_device(config.device)

    summary_rows: list[dict[str, Any]] = []
    for checkpoint_path in config.checkpoint_paths:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        run_name = _checkpoint_name(checkpoint_path)
        run_dir = config.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Evaluating {checkpoint_path} ===")
        actor_dir, critic_dir = _resolve_component_dirs(config, checkpoint_path, config.output_dir)
        tokenizer = load_tokenizer(actor_dir, trust_remote_code=config.trust_remote_code)
        examples = load_examples(
            config.dataset_path,
            tokenizer=tokenizer,
            prompt_key=config.prompt_key,
            response_key=config.response_key,
            start_index=config.start_index,
            max_examples=config.max_examples,
            shuffle_examples=config.shuffle_examples,
            seed=config.seed,
            pretokenize_max_length=config.max_prompt_length,
        )
        if not examples:
            raise ValueError("No examples selected for evaluation.")
        eos_token_ids = resolve_eos_token_ids(actor_dir, tokenizer)
        if config.generation_backend == "torch":
            actor = _load_actor(actor_dir, dtype=dtype, device=device, trust_remote_code=config.trust_remote_code)
            trajectories = generate_trajectories_torch(
                actor=actor,
                tokenizer=tokenizer,
                examples=examples,
                config=config,
                device=device,
                eos_token_ids=eos_token_ids,
            )
            del actor
            _clear_cuda_cache(device)
        elif config.generation_backend == "vllm":
            if device.type != "cuda":
                raise ValueError("--generation_backend=vllm requires --device to be a CUDA device.")
            llm = _load_vllm_actor(actor_dir, config=config)
            trajectories = generate_trajectories_vllm(
                llm=llm,
                tokenizer=tokenizer,
                examples=examples,
                config=config,
                eos_token_ids=eos_token_ids,
            )
            del llm
            _cleanup_vllm(device)
        else:
            raise ValueError(f"Unsupported generation backend: {config.generation_backend}")

        per_rows: list[dict[str, Any]] = []
        if _looks_like_critic_checkpoint(critic_dir, trust_remote_code=config.trust_remote_code):
            critic_tokenizer = load_tokenizer(critic_dir, trust_remote_code=config.trust_remote_code)
            critic = load_critic_model(critic_dir, dtype=dtype, device=device, trust_remote_code=config.trust_remote_code)
            metrics, per_rows = evaluate_critic(
                critic=critic,
                tokenizer=critic_tokenizer,
                trajectories=trajectories,
                config=config,
                device=device,
            )
            del critic
            _clear_cuda_cache(device)
        else:
            message = (
                f"{critic_dir} does not look like a critic/value-head checkpoint; "
                "writing actor accuracy only."
            )
            if config.require_critic:
                raise ValueError(message)
            print(f"WARNING: {message}")
            metrics = _accuracy_metrics(trajectories)
            metrics["critic_metrics_available"] = 0.0

        metadata = {
            "checkpoint_path": str(checkpoint_path),
            "actor_dir": str(actor_dir),
            "critic_dir": str(critic_dir),
            "dataset_path": str(config.dataset_path),
            "prompt_key": config.prompt_key,
            "response_key": config.response_key,
            "start_index": config.start_index,
            "max_examples": config.max_examples,
            "shuffle_examples": config.shuffle_examples,
            "seed": config.seed,
            "max_prompt_length": config.max_prompt_length,
            "max_new_tokens": config.max_new_tokens,
            "actor_sampling_mode": config.actor_sampling_mode,
            "actor_temperature": config.actor_temperature,
            "actor_top_p": config.actor_top_p,
            "actor_top_k": config.actor_top_k,
            "generation_backend": config.generation_backend,
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": config.vllm_tensor_parallel_size,
            "vllm_max_model_len": config.vllm_max_model_len,
            "vllm_max_num_seqs": config.vllm_max_num_seqs,
            "vllm_enforce_eager": config.vllm_enforce_eager,
            "loss_agg_mode": config.loss_agg_mode,
            "merged_checkpoint_root": str(config.merged_checkpoint_root) if config.merged_checkpoint_root else None,
            "require_critic": config.require_critic,
        }
        _write_json(run_dir / "metrics.json", {"metrics": metrics, "metadata": metadata})
        _write_csv(run_dir / "per_example_metrics.csv", per_rows)
        if config.save_trajectories:
            jsonl_rows = []
            for row in trajectories:
                example = row["example"]
                jsonl_rows.append(
                    {
                        "example_id": example.example_id,
                        "data_source": example.data_source,
                        "ground_truth": example.ground_truth,
                        "reward": row["reward"],
                        "response_text": row["response_text"],
                        "prompt_ids": row["prompt_ids"],
                        "response_ids": row["response_ids"],
                        "response_mask": row["response_mask"],
                    }
                )
            _write_jsonl(run_dir / "trajectories.jsonl", jsonl_rows)

        summary = {"checkpoint": str(checkpoint_path), **metrics}
        summary_rows.append(summary)
        _write_summary(config.output_dir, summary_rows)
        print(json.dumps(summary, indent=2, sort_keys=True))

    print(f"\nWrote results to {config.output_dir}")


if __name__ == "__main__":
    main()
