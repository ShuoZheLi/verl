from __future__ import annotations

import argparse
import functools
import csv
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import CPUOffload, FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from value_decoding.checkpointing import (
    ensure_merged_component_checkpoint,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
)
from verl.trainer.ppo.value_categorical import (
    extract_value_head_spec,
    unscale_scalar_values,
    value_logits_to_probs,
    value_probs_to_scaled_scalar,
)

LOSS_TYPES = ("mse", "bce", "pairwise", "hybrid")
SAMPLING_MODES = ("uniform", "prompt_balanced", "rankable_prioritized", "mixed")
MAIN_RESULTS_FIELDS = (
    "step",
    "train_loss",
    "eval_mse",
    "pearson",
    "spearman",
    "pairwise_ranking_accuracy",
    "top1_candidate_reward",
    "oracle_candidate_reward",
    "random_candidate_reward",
    "conditional_success_recovery_rate",
    "false_high_selected_rate",
    "value_gap_correct_minus_wrong",
)


@dataclass(frozen=True)
class CandidateExample:
    input_ids: tuple[int, ...]
    target_reward: float
    prompt_id: Any
    candidate_group_id: str
    search_step: int
    selected_by_collector: bool
    collector_value: float | None
    group_has_mixed_rewards: bool


class SearchInducedCriticDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        *,
        tokenizer,
        max_seq_length: int,
    ) -> None:
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.examples: list[CandidateExample] = []
        self.prompt_to_indices: dict[Any, list[int]] = defaultdict(list)
        self.group_to_indices: dict[str, list[int]] = defaultdict(list)
        self.positive_indices: list[int] = []
        self.negative_indices: list[int] = []
        self.rankable_group_ids: list[str] = []
        self.nonrankable_indices: list[int] = []
        self.num_missing_token_ids = 0
        self.num_missing_prompt_text = 0
        self._load()
        if not self.examples:
            raise ValueError(f"No examples were loaded from {self.data_path}")
        self._build_sampling_indexes()

    def _load(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        with self.data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    reward = float(row["mc_reward"])
                except KeyError as exc:
                    raise KeyError(f"Row {line_number} in {self.data_path} is missing mc_reward") from exc

                token_ids = self._row_input_ids(row, line_number=line_number)
                if not token_ids:
                    raise ValueError(f"Row {line_number} in {self.data_path} produced an empty input sequence")
                if len(token_ids) > self.max_seq_length:
                    token_ids = token_ids[-self.max_seq_length :]

                prompt_id = row.get("prompt_id", "__missing_prompt_id__")
                group_id = str(row.get("candidate_group_id") or f"prompt_{prompt_id}_step_{row.get('search_step', 0)}")
                collector_value = row.get("collector_value")
                example = CandidateExample(
                    input_ids=tuple(int(token_id) for token_id in token_ids),
                    target_reward=reward,
                    prompt_id=prompt_id,
                    candidate_group_id=group_id,
                    search_step=int(row.get("search_step", 0) or 0),
                    selected_by_collector=bool(row.get("selected_by_collector", False)),
                    collector_value=None if collector_value is None else float(collector_value),
                    group_has_mixed_rewards=bool(row.get("group_has_mixed_rewards", False)),
                )
                index = len(self.examples)
                self.examples.append(example)
                self.prompt_to_indices[prompt_id].append(index)
                self.group_to_indices[group_id].append(index)

    def _row_input_ids(self, row: dict[str, Any], *, line_number: int) -> list[int]:
        candidate_ids = row.get("candidate_prefix_token_ids")
        prompt_text = row.get("prompt_text")
        if isinstance(candidate_ids, list) and candidate_ids:
            candidate_ids = [int(token_id) for token_id in candidate_ids]
            if isinstance(prompt_text, str) and prompt_text:
                prompt_ids = self.tokenizer(
                    prompt_text,
                    return_attention_mask=False,
                )["input_ids"]
                return [int(token_id) for token_id in prompt_ids] + candidate_ids
            self.num_missing_prompt_text += 1
            return candidate_ids

        self.num_missing_token_ids += 1
        candidate_prefix_text = row.get("candidate_prefix_text")
        if not isinstance(prompt_text, str) or not isinstance(candidate_prefix_text, str):
            raise ValueError(
                f"Row {line_number} in {self.data_path} needs candidate_prefix_token_ids, or both "
                "prompt_text and candidate_prefix_text for fallback tokenization."
            )
        return [
            int(token_id)
            for token_id in self.tokenizer(
                prompt_text + candidate_prefix_text,
                return_attention_mask=False,
            )["input_ids"]
        ]

    def _build_sampling_indexes(self) -> None:
        rankable_set: set[int] = set()
        for group_id, indices in self.group_to_indices.items():
            rewards = [self.examples[index].target_reward for index in indices]
            has_mixed_rewards = len({float(reward) for reward in rewards}) > 1
            if has_mixed_rewards or any(self.examples[index].group_has_mixed_rewards for index in indices):
                self.rankable_group_ids.append(group_id)
                rankable_set.update(indices)

        for index, example in enumerate(self.examples):
            if example.target_reward >= 0.5:
                self.positive_indices.append(index)
            else:
                self.negative_indices.append(index)
            if index not in rankable_set:
                self.nonrankable_indices.append(index)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CandidateExample:
        return self.examples[index]


class SearchInducedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SearchInducedCriticDataset,
        *,
        batch_size: int,
        mode: str,
        seed: int,
        max_examples_per_prompt_per_batch: int,
        positive_fraction: float | None,
        rankable_group_fraction: float,
        drop_last: bool = False,
    ) -> None:
        if mode not in SAMPLING_MODES:
            raise ValueError(f"Unsupported batch sampling mode: {mode}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.mode = mode
        self.seed = int(seed)
        self.max_examples_per_prompt_per_batch = int(max_examples_per_prompt_per_batch)
        self.positive_fraction = positive_fraction
        self.rankable_group_fraction = float(rankable_group_fraction)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.num_batches = len(dataset) // self.batch_size
        if len(dataset) % self.batch_size and not self.drop_last:
            self.num_batches += 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1000003)
        all_indices = list(range(len(self.dataset)))
        rng.shuffle(all_indices)

        if self.mode == "uniform":
            yield from self._simple_batches(all_indices)
            return
        if self.mode == "prompt_balanced":
            yield from self._prompt_balanced_batches(all_indices, rng)
            return
        if self.mode == "rankable_prioritized":
            yield from self._rankable_prioritized_batches(rng)
            return
        if self.mode == "mixed":
            yield from self._mixed_batches(rng)
            return
        raise AssertionError(f"Unhandled sampler mode: {self.mode}")

    def _simple_batches(self, indices: list[int]) -> Iterator[list[int]]:
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def _prompt_balanced_batches(self, indices: list[int], rng: random.Random) -> Iterator[list[int]]:
        unused = list(indices)
        for _ in range(self.num_batches):
            if not unused:
                break
            prompt_counts: dict[Any, int] = defaultdict(int)
            batch: list[int] = []
            deferred: list[int] = []
            while unused and len(batch) < self.batch_size:
                index = unused.pop()
                prompt_id = self.dataset.examples[index].prompt_id
                if prompt_counts[prompt_id] >= self.max_examples_per_prompt_per_batch:
                    deferred.append(index)
                    continue
                batch.append(index)
                prompt_counts[prompt_id] += 1
            unused.extend(deferred)
            rng.shuffle(unused)
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def _rankable_prioritized_batches(self, rng: random.Random) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            batch = self._sample_rankable_examples(
                rng,
                max(1, min(self.batch_size, int(round(self.batch_size * self.rankable_group_fraction)))),
            )
            self._fill_random(batch, rng, self.batch_size)
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def _mixed_batches(self, rng: random.Random) -> Iterator[list[int]]:
        positive_fraction = 0.25 if self.positive_fraction is None else float(self.positive_fraction)
        rankable_count = max(0, int(round(self.batch_size * self.rankable_group_fraction)))
        remaining_after_rankable = max(0, self.batch_size - rankable_count)
        positive_count = max(0, int(round(remaining_after_rankable * positive_fraction)))
        negative_count = max(0, remaining_after_rankable - positive_count)
        for _ in range(self.num_batches):
            batch = self._sample_rankable_examples(rng, min(rankable_count, self.batch_size))
            self._sample_from_pool(
                batch,
                self.dataset.positive_indices,
                rng,
                max(0, min(positive_count, self.batch_size - len(batch))),
            )
            self._sample_from_pool(
                batch,
                self.dataset.negative_indices,
                rng,
                max(0, min(negative_count, self.batch_size - len(batch))),
            )
            self._fill_random(batch, rng, self.batch_size)
            rng.shuffle(batch)
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch[: self.batch_size]

    def _sample_rankable_examples(self, rng: random.Random, count: int) -> list[int]:
        if count <= 0 or not self.dataset.rankable_group_ids:
            return []
        batch: list[int] = []
        group_ids = list(self.dataset.rankable_group_ids)
        rng.shuffle(group_ids)
        while len(batch) < count and group_ids:
            group_id = group_ids.pop()
            indices = list(self.dataset.group_to_indices[group_id])
            positives = [index for index in indices if self.dataset.examples[index].target_reward >= 0.5]
            negatives = [index for index in indices if self.dataset.examples[index].target_reward < 0.5]
            rng.shuffle(positives)
            rng.shuffle(negatives)
            if positives and negatives:
                batch.append(positives[0])
                if len(batch) < count:
                    batch.append(negatives[0])
                extras = positives[1:] + negatives[1:]
                rng.shuffle(extras)
                while extras and len(batch) < count and sum(1 for idx in batch if self.dataset.examples[idx].candidate_group_id == group_id) < 4:
                    batch.append(extras.pop())
        if len(batch) < count:
            rankable_indices = [index for group_id in self.dataset.rankable_group_ids for index in self.dataset.group_to_indices[group_id]]
            self._sample_from_pool(batch, rankable_indices, rng, count - len(batch))
        return batch[:count]

    def _sample_from_pool(self, batch: list[int], pool: Sequence[int], rng: random.Random, count: int) -> None:
        if count <= 0 or not pool:
            return
        for _ in range(count):
            batch.append(int(rng.choice(pool)))

    def _fill_random(self, batch: list[int], rng: random.Random, target_size: int) -> None:
        all_indices = list(range(len(self.dataset)))
        while len(batch) < target_size:
            batch.append(int(rng.choice(all_indices)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a critic on search-induced Monte Carlo candidate-prefix data.")
    parser.add_argument("--init_critic_checkpoint_dir", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--loss_type", type=str, choices=LOSS_TYPES, default="mse")
    parser.add_argument("--rank_loss_weight", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional optimizer-step limit for smoke tests.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--trainable_scope", type=str, default="all", choices=("all", "value_head"))
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_sampling_mode", type=str, choices=SAMPLING_MODES, default="mixed")
    parser.add_argument("--max_examples_per_prompt_per_batch", type=int, default=4)
    parser.add_argument("--positive_fraction", type=float, default=None)
    parser.add_argument("--rankable_group_fraction", type=float, default=0.5)

    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--save_every_steps", type=int, default=200)
    parser.add_argument("--eval_at_start", action="store_true", help="Run held-out eval once at step 0 before training.")
    parser.add_argument("--run_end_to_end_chunk_eval", action="store_true")
    parser.add_argument("--chunk_eval_script_path", type=str, default=None)
    parser.add_argument("--chunk_eval_actor_checkpoint_dir", type=str, default=None)
    parser.add_argument("--chunk_eval_dataset_path", type=str, default=None)
    parser.add_argument("--chunk_eval_max_examples", type=int, default=None)
    parser.add_argument("--chunk_eval_num_seeds", type=int, default=3)
    parser.add_argument("--chunk_eval_generation_backend", type=str, default="vllm", choices=("torch", "vllm"))

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--distributed_backend", type=str, default="none", choices=("none", "fsdp"))
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--critic_hf_source_dir", type=str, default=None)
    parser.add_argument("--merged_root", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="value-decoding")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=("online", "offline", "disabled"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed_context(args: argparse.Namespace) -> DistributedContext:
    if args.distributed_backend == "none":
        return DistributedContext(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=resolve_device(args.device),
        )

    if args.distributed_backend != "fsdp":
        raise ValueError(f"Unsupported distributed backend: {args.distributed_backend}")
    if args.trainable_scope != "all":
        raise ValueError(
            "--distributed_backend fsdp currently requires --trainable_scope all. "
            "FSDP with frozen base parameters can fail during multi-rank backward; "
            "use --distributed_backend none for value_head-only training."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP distributed training requires CUDA devices.")

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                device_id=device,
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed_context(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        # Do not barrier during exception cleanup: if one rank failed, a final
        # barrier can hang surviving ranks. Normal success paths barrier before
        # reaching cleanup.
        dist.destroy_process_group()


def rank_print(distributed: DistributedContext, message: str) -> None:
    if distributed.is_main_process:
        print(message, flush=True)


def barrier(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.barrier()


def assert_finite_tensor(name: str, tensor: torch.Tensor, *, step: int, distributed: DistributedContext) -> None:
    tensor_detached = tensor.detach()
    if torch.isfinite(tensor_detached).all().item():
        return
    finite_mask = torch.isfinite(tensor_detached)
    finite_values = tensor_detached[finite_mask].float()
    if finite_values.numel() > 0:
        details = f"finite_min={finite_values.min().item():.6g} finite_max={finite_values.max().item():.6g}"
    else:
        details = "no finite values"
    raise FloatingPointError(f"Non-finite {name} at step={step}, rank={distributed.rank}: {details}.")


def reduce_loss_dict_for_logging(loss_dict: dict[str, Any], distributed: DistributedContext) -> dict[str, Any]:
    if not distributed.enabled:
        return loss_dict
    reduced = dict(loss_dict)
    tensor_keys = ("loss", "loss_mse", "loss_bce", "loss_rank", "value_mean", "target_mean")
    for key in tensor_keys:
        tensor = reduced.get(key)
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().float()
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
        reduced[key] = value
    for key in ("num_rankable_groups_in_batch", "num_pairs_in_batch"):
        value = torch.tensor(float(reduced.get(key, 0)), device=distributed.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        reduced[key] = int(value.item())
    return reduced


def init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("--use_wandb was set, but wandb is not installed in this environment.") from exc

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "config": config,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    if args.wandb_tags:
        init_kwargs["tags"] = list(args.wandb_tags)
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def wandb_log(
    wandb_run,
    metrics: dict[str, Any],
    *,
    step: int | None = None,
    commit: bool = True,
    mirror_path: Path | None = None,
) -> None:
    clean_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool)) and value is not None and math.isfinite(float(value))
    }
    if mirror_path is not None and clean_metrics:
        append_jsonl(
            mirror_path,
            {
                "wandb_step": None if step is None else int(step),
                "commit": bool(commit),
                "keys": sorted(clean_metrics),
                "metrics": clean_metrics,
            },
        )
    if wandb_run is None or not clean_metrics:
        return
    if step is None:
        wandb_run.log(clean_metrics, commit=commit)
    else:
        wandb_run.log(clean_metrics, step=int(step), commit=commit)


def wandb_finish(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def _module_by_dotted_name(model: torch.nn.Module, name: str):
    module = model
    for part in name.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def find_value_head_parameter_names(model: torch.nn.Module) -> set[str]:
    value_head_names = ("score", "classifier", "v_head", "v_head.summary", "prompt_prior_head")
    parameter_names: set[str] = set()
    for module_name in value_head_names:
        module = _module_by_dotted_name(model, module_name)
        if module is None:
            continue
        for parameter_name, _ in module.named_parameters(prefix=module_name, recurse=True):
            parameter_names.add(parameter_name)
    return parameter_names


def configure_trainable_parameters(model: torch.nn.Module, trainable_scope: str) -> dict[str, Any]:
    if trainable_scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif trainable_scope == "value_head":
        value_head_parameter_names = find_value_head_parameter_names(model)
        if not value_head_parameter_names:
            raise ValueError(
                "--trainable_scope value_head was requested, but no value-head parameters were found "
                "under score/classifier/v_head."
            )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in value_head_parameter_names)
    else:
        raise ValueError(f"Unsupported trainable_scope: {trainable_scope}")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not trainable_parameters:
        raise ValueError(f"No trainable parameters found for trainable_scope={trainable_scope}")
    return {
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count": int(total_parameter_count),
        "trainable_parameter_fraction": float(trainable_parameter_count / max(1, total_parameter_count)),
    }


def enable_gradient_checkpointing_if_requested(model: torch.nn.Module, enabled: bool) -> bool:
    if not enabled:
        return False
    gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(gradient_checkpointing_enable):
        raise ValueError("--gradient_checkpointing was set, but this model does not support gradient_checkpointing_enable().")
    gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = getattr(model, "config", None)
    if config is not None:
        setattr(config, "use_cache", False)
    return True


def log_cuda_memory(label: str, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    allocated_gib = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved_gib = torch.cuda.memory_reserved(device_index) / 1024**3
    total_gib = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    print(
        f"CUDA memory {label}: allocated={allocated_gib:.2f}GiB "
        f"reserved={reserved_gib:.2f}GiB total={total_gib:.2f}GiB",
        flush=True,
    )


def infer_transformer_layer_classes(model: torch.nn.Module) -> set[type[torch.nn.Module]]:
    layer_classes: set[type[torch.nn.Module]] = set()
    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name.endswith("DecoderLayer") or class_name.endswith("EncoderLayer") or class_name.endswith("Block"):
            layer_classes.add(module.__class__)
    return layer_classes


def wrap_with_fsdp_if_needed(
    model: torch.nn.Module,
    *,
    distributed: DistributedContext,
    args: argparse.Namespace,
) -> torch.nn.Module:
    if not distributed.enabled:
        return model
    transformer_layer_classes = infer_transformer_layer_classes(model)
    auto_wrap_policy = None
    if transformer_layer_classes:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_classes,
        )
        rank_print(
            distributed,
            "FSDP auto-wrap layer classes: " + ", ".join(sorted(cls.__name__ for cls in transformer_layer_classes)),
        )
    else:
        rank_print(distributed, "FSDP auto-wrap layer classes not found; wrapping whole model.")

    cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
        device_id=distributed.device,
        use_orig_params=True,
    )


def clip_grad_norm(model: torch.nn.Module, max_norm: float) -> torch.Tensor:
    if isinstance(model, FSDP):
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def finite_grad_norm_value(grad_norm: torch.Tensor | float, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(grad_norm):
        return grad_norm.detach().to(device=device, dtype=torch.float32)
    return torch.tensor(float(grad_norm), device=device, dtype=torch.float32)


def summarize_non_finite_gradients(model: torch.nn.Module, *, max_items: int = 8) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        grad_detached = grad.detach()
        finite_mask = torch.isfinite(grad_detached)
        if finite_mask.all().item():
            continue
        finite_values = grad_detached[finite_mask].float()
        summaries.append(
            {
                "name": name,
                "shape": list(grad_detached.shape),
                "dtype": str(grad_detached.dtype),
                "numel": int(grad_detached.numel()),
                "num_nan": int(torch.isnan(grad_detached).sum().item()),
                "num_posinf": int(torch.isposinf(grad_detached).sum().item()),
                "num_neginf": int(torch.isneginf(grad_detached).sum().item()),
                "finite_abs_max": None if finite_values.numel() == 0 else float(finite_values.abs().max().item()),
            }
        )
        if len(summaries) >= int(max_items):
            break
    return summaries


def log_non_finite_grad_skip(
    *,
    output_dir: Path,
    step: int,
    epoch: int,
    grad_norm: torch.Tensor,
    distributed: DistributedContext,
    model: torch.nn.Module | None = None,
) -> None:
    if not distributed.is_main_process:
        return
    row: dict[str, Any] = {
        "step": int(step),
        "epoch": int(epoch),
        "skipped_optimizer_step": True,
        "reason": "non_finite_grad_norm",
        "grad_norm": float(grad_norm.detach().cpu().item()),
        "world_size": distributed.world_size,
    }
    if model is not None:
        row["non_finite_gradients"] = summarize_non_finite_gradients(model)
    append_jsonl(output_dir / "train_log.jsonl", row)


def collate_candidates(examples: Sequence[CandidateExample], *, pad_token_id: int) -> dict[str, Any]:
    max_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
    target_reward = torch.tensor([example.target_reward for example in examples], dtype=torch.float32)
    for row_index, example in enumerate(examples):
        sequence = torch.tensor(example.input_ids, dtype=torch.long)
        input_ids[row_index, : sequence.numel()] = sequence
        attention_mask[row_index, : sequence.numel()] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_reward": target_reward,
        "prompt_id": [example.prompt_id for example in examples],
        "candidate_group_id": [example.candidate_group_id for example in examples],
        "search_step": torch.tensor([example.search_step for example in examples], dtype=torch.long),
        "selected_by_collector": torch.tensor([example.selected_by_collector for example in examples], dtype=torch.bool),
        "collector_value": [example.collector_value for example in examples],
    }


def extract_scalar_values_from_outputs(critic, outputs) -> torch.Tensor:
    if hasattr(critic, "v_head"):
        raw_values = outputs[2]
    elif hasattr(outputs, "logits"):
        raw_values = outputs.logits
    elif isinstance(outputs, tuple):
        raw_values = outputs[0]
    else:
        raise TypeError(f"Unsupported critic output type: {type(outputs).__name__}")

    if raw_values.dim() == 2:
        return raw_values.float()
    if raw_values.dim() != 3:
        raise ValueError(f"Unexpected critic value tensor shape: {tuple(raw_values.shape)}")
    if raw_values.shape[-1] == 1:
        return raw_values.squeeze(-1).float()

    spec = extract_value_head_spec(getattr(critic, "config", {}))
    probs = value_logits_to_probs(raw_values.float())
    support = spec.support(device=probs.device, dtype=probs.dtype)
    scaled_values = value_probs_to_scaled_scalar(probs, support)
    return unscale_scalar_values(scaled_values, spec).float()


def critic_last_token_values_trainable(critic, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = critic(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    sequence_values = extract_scalar_values_from_outputs(critic, outputs)
    last_indices = attention_mask.long().sum(dim=1) - 1
    if torch.any(last_indices < 0):
        raise ValueError("Each sequence must contain at least one unmasked token.")
    return sequence_values.gather(dim=1, index=last_indices[:, None]).squeeze(1)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("input_ids", "attention_mask", "target_reward", "search_step", "selected_by_collector"):
        moved[key] = batch[key].to(device)
    return moved


def compute_rank_loss(values: torch.Tensor, targets: torch.Tensor, group_ids: Sequence[str]) -> tuple[torch.Tensor, int, int]:
    group_to_positions: dict[str, list[int]] = defaultdict(list)
    for position, group_id in enumerate(group_ids):
        group_to_positions[str(group_id)].append(position)

    losses: list[torch.Tensor] = []
    num_rankable_groups = 0
    num_pairs = 0
    for positions in group_to_positions.values():
        pairs: list[tuple[int, int]] = []
        for left_offset, left_position in enumerate(positions):
            left_reward = float(targets[left_position].detach().item())
            for right_position in positions[left_offset + 1 :]:
                right_reward = float(targets[right_position].detach().item())
                if left_reward == right_reward:
                    continue
                if left_reward > right_reward:
                    pairs.append((left_position, right_position))
                else:
                    pairs.append((right_position, left_position))
        if not pairs:
            continue
        num_rankable_groups += 1
        if len(pairs) > 16:
            pairs = random.sample(pairs, 16)
        for pos, neg in pairs:
            losses.append(-F.logsigmoid(values[pos].float() - values[neg].float()))
        num_pairs += len(pairs)

    if not losses:
        return values.float().sum() * 0.0, 0, 0
    return torch.stack(losses).mean(), num_rankable_groups, num_pairs


def compute_loss(
    values: torch.Tensor,
    targets: torch.Tensor,
    batch: dict[str, Any],
    *,
    loss_type: str,
    rank_loss_weight: float,
) -> dict[str, Any]:
    values_f = values.float()
    targets_f = targets.float()
    loss_mse = F.mse_loss(values_f, targets_f)
    loss_bce = F.binary_cross_entropy_with_logits(values_f, targets_f)
    loss_rank, num_rankable_groups, num_pairs = compute_rank_loss(values_f, targets_f, batch["candidate_group_id"])

    if loss_type == "mse":
        loss = loss_mse
    elif loss_type == "bce":
        loss = loss_bce
    elif loss_type == "pairwise":
        loss = loss_rank
    elif loss_type == "hybrid":
        loss = loss_mse + float(rank_loss_weight) * loss_rank
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return {
        "loss": loss,
        "loss_mse": loss_mse.detach(),
        "loss_bce": loss_bce.detach(),
        "loss_rank": loss_rank.detach(),
        "num_rankable_groups_in_batch": num_rankable_groups,
        "num_pairs_in_batch": num_pairs,
        "value_mean": values_f.detach().mean(),
        "target_mean": targets_f.detach().mean(),
    }


def finite_mean(values: Sequence[float]) -> float | None:
    filtered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not filtered:
        return None
    return float(np.mean(np.asarray(filtered, dtype=np.float64)))


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) <= 0 or np.std(y_arr) <= 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def average_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and arr[order[end]] == arr[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson_corr(average_ranks(x).tolist(), average_ranks(y).tolist())


def brier_score(values: Sequence[float], rewards: Sequence[float]) -> float | None:
    if not values:
        return None
    probs = 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    return float(np.mean((probs - rewards_arr) ** 2))


def _metric_prefix(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def compute_value_metrics(values: Sequence[float], rewards: Sequence[float], group_ids: Sequence[str]) -> dict[str, Any]:
    values_arr = np.asarray(values, dtype=np.float64)
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    metrics: dict[str, Any] = {
        "eval_mse": float(np.mean((values_arr - rewards_arr) ** 2)) if len(values_arr) else None,
        "eval_brier": brier_score(values, rewards),
        "pearson": pearson_corr(values, rewards),
        "spearman": spearman_corr(values, rewards),
    }

    correct_mask = rewards_arr >= 0.5
    wrong_mask = rewards_arr < 0.5
    mean_correct = float(values_arr[correct_mask].mean()) if np.any(correct_mask) else None
    mean_wrong = float(values_arr[wrong_mask].mean()) if np.any(wrong_mask) else None
    metrics.update(
        {
            "mean_value_correct": mean_correct,
            "mean_value_wrong": mean_wrong,
            "value_gap_correct_minus_wrong": None
            if mean_correct is None or mean_wrong is None
            else float(mean_correct - mean_wrong),
        }
    )

    group_to_positions: dict[str, list[int]] = defaultdict(list)
    for position, group_id in enumerate(group_ids):
        group_to_positions[str(group_id)].append(position)

    oracle_rewards: list[float] = []
    random_rewards: list[float] = []
    top1_rewards: list[float] = []
    success_recoveries: list[float] = []
    false_highs: list[float] = []
    pairwise_correct = 0
    pairwise_total = 0

    for positions in group_to_positions.values():
        group_rewards = rewards_arr[positions]
        group_values = values_arr[positions]
        if len(group_rewards) == 0:
            continue
        selected_position = int(np.argmax(group_values))
        selected_reward = float(group_rewards[selected_position])
        oracle_reward = float(np.max(group_rewards))
        oracle_rewards.append(oracle_reward)
        random_rewards.append(float(np.mean(group_rewards)))
        top1_rewards.append(selected_reward)
        if oracle_reward >= 0.5:
            success_recoveries.append(float(selected_reward >= 0.5))
            false_highs.append(float(selected_reward < 0.5))

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                reward_i = float(group_rewards[i])
                reward_j = float(group_rewards[j])
                if reward_i == reward_j:
                    continue
                pairwise_total += 1
                value_i = float(group_values[i])
                value_j = float(group_values[j])
                if (reward_i > reward_j and value_i > value_j) or (reward_j > reward_i and value_j > value_i):
                    pairwise_correct += 1
                elif value_i == value_j:
                    pairwise_correct += 0.5

    metrics.update(
        {
            "mean_group_oracle_reward": finite_mean(oracle_rewards),
            "mean_group_random_reward": finite_mean(random_rewards),
            "mean_group_critic_top1_reward": finite_mean(top1_rewards),
            "top1_candidate_reward": finite_mean(top1_rewards),
            "oracle_candidate_reward": finite_mean(oracle_rewards),
            "random_candidate_reward": finite_mean(random_rewards),
            "conditional_success_recovery_rate": finite_mean(success_recoveries),
            "false_high_selected_rate": finite_mean(false_highs),
            "pairwise_ranking_accuracy": None if pairwise_total == 0 else float(pairwise_correct / pairwise_total),
            "num_pairwise_pairs": int(pairwise_total),
            "num_groups": int(len(group_to_positions)),
        }
    )
    return metrics


def evaluate_critic(critic, dataloader: DataLoader, *, device: torch.device, max_examples: int | None = None) -> dict[str, Any]:
    critic.eval()
    values: list[float] = []
    rewards: list[float] = []
    group_ids: list[str] = []
    collector_values: list[float] = []
    collector_rewards: list[float] = []
    collector_group_ids: list[str] = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = batch_to_device(batch, device)
        # Use no_grad rather than inference_mode. FSDP can fail on the next
        # training forward after parameters were materialized under
        # inference_mode, because its post-backward hook expects grad_fn access.
        with torch.no_grad():
            batch_values = critic_last_token_values_trainable(critic, batch["input_ids"], batch["attention_mask"])
        batch_values_list = batch_values.detach().cpu().float().tolist()
        batch_rewards = batch["target_reward"].detach().cpu().float().tolist()
        for value, reward, group_id, collector_value in zip(
            batch_values_list,
            batch_rewards,
            batch["candidate_group_id"],
            batch["collector_value"],
        ):
            values.append(float(value))
            rewards.append(float(reward))
            group_ids.append(str(group_id))
            if collector_value is not None:
                collector_values.append(float(collector_value))
                collector_rewards.append(float(reward))
                collector_group_ids.append(str(group_id))
            if max_examples is not None and len(values) >= int(max_examples):
                break
        if max_examples is not None and len(values) >= int(max_examples):
            break

    new_metrics = compute_value_metrics(values, rewards, group_ids)
    metrics = dict(new_metrics)
    metrics["num_eval_examples"] = len(values)
    if collector_values:
        collector_metrics = compute_value_metrics(collector_values, collector_rewards, collector_group_ids)
        metrics.update(_metric_prefix("collector_", collector_metrics))
        metrics["new_minus_collector_pairwise_accuracy"] = subtract_optional(
            metrics.get("pairwise_ranking_accuracy"), collector_metrics.get("pairwise_ranking_accuracy")
        )
        metrics["new_minus_collector_top1_reward"] = subtract_optional(
            metrics.get("mean_group_critic_top1_reward"), collector_metrics.get("mean_group_critic_top1_reward")
        )
        metrics["new_minus_collector_false_high_rate"] = subtract_optional(
            metrics.get("false_high_selected_rate"), collector_metrics.get("false_high_selected_rate")
        )
    critic.train()
    return metrics


def subtract_optional(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_main_results(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAIN_RESULTS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in MAIN_RESULTS_FIELDS})


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(critic, tokenizer, output_dir: Path, step: int, *, distributed: DistributedContext) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    if distributed.enabled:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(critic, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = critic.state_dict()
        if distributed.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = unwrap_model(critic)
            if not hasattr(model_to_save, "save_pretrained"):
                raise TypeError(f"Critic model of type {type(model_to_save).__name__} does not support save_pretrained().")
            model_to_save.save_pretrained(str(checkpoint_dir), state_dict=state_dict, safe_serialization=True)
            tokenizer.save_pretrained(str(checkpoint_dir))
        barrier(distributed)
        return checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(critic)
    if not hasattr(model_to_save, "save_pretrained"):
        raise TypeError(f"Critic model of type {type(model_to_save).__name__} does not support save_pretrained().")
    model_to_save.save_pretrained(str(checkpoint_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(checkpoint_dir))
    return checkpoint_dir


def write_plots(output_dir: Path) -> None:
    metrics_path = output_dir / "eval_metrics.jsonl"
    if not metrics_path.is_file():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return
    plot_specs = [
        ("eval_mse", "eval_mse_over_steps.png"),
        ("pairwise_ranking_accuracy", "pairwise_ranking_over_steps.png"),
        ("mean_group_critic_top1_reward", "top1_reward_over_steps.png"),
        ("false_high_selected_rate", "false_high_rate_over_steps.png"),
    ]
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric_name, filename in plot_specs:
        points = [(row.get("step"), row.get(metric_name)) for row in rows if row.get(metric_name) is not None]
        if not points:
            continue
        steps, values = zip(*points)
        plt.figure()
        plt.plot(steps, values, marker="o")
        plt.xlabel("step")
        plt.ylabel(metric_name)
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()


def maybe_run_chunk_eval(args: argparse.Namespace, checkpoint_dir: Path, step: int, output_dir: Path) -> None:
    if not args.run_end_to_end_chunk_eval:
        return
    if not args.chunk_eval_script_path:
        raise ValueError("--chunk_eval_script_path is required when --run_end_to_end_chunk_eval is enabled")
    if not args.chunk_eval_actor_checkpoint_dir:
        raise ValueError("--chunk_eval_actor_checkpoint_dir is required when --run_end_to_end_chunk_eval is enabled")
    if not args.chunk_eval_dataset_path:
        raise ValueError("--chunk_eval_dataset_path is required when --run_end_to_end_chunk_eval is enabled")

    num_seeds = max(1, int(args.chunk_eval_num_seeds))
    for seed_offset in range(num_seeds):
        eval_seed = int(args.seed) + seed_offset
        eval_output_dir = output_dir / f"chunk_guidance_eval_step_{step:06d}" / f"seed_{eval_seed}"
        cmd = [
            os.environ.get("PYTHON", "python"),
            str(args.chunk_eval_script_path),
            "--actor_checkpoint_dir",
            str(args.chunk_eval_actor_checkpoint_dir),
            "--critic_checkpoint_dir",
            str(checkpoint_dir),
            "--dataset_path",
            str(args.chunk_eval_dataset_path),
            "--output_dir",
            str(eval_output_dir),
            "--chunk_sizes",
            "128",
            "256",
            "--num_chunk_candidates_values",
            "8",
            "--generation_backend",
            str(args.chunk_eval_generation_backend),
            "--dtype",
            str(args.dtype),
            "--seed",
            str(eval_seed),
            "--trust_remote_code",
        ]
        if args.chunk_eval_max_examples is not None:
            cmd.extend(["--max_examples", str(args.chunk_eval_max_examples)])
        subprocess.run(cmd, check=True)


def make_git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        return None


def run_eval_and_log(
    *,
    critic,
    eval_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    step: int,
    train_loss: float | None,
    wandb_run,
    args: argparse.Namespace,
    distributed: DistributedContext,
    extra_fields: dict[str, Any] | None = None,
    log_to_wandb: bool = True,
) -> dict[str, Any] | None:
    metrics = evaluate_critic(
        critic,
        eval_loader,
        device=device,
        max_examples=args.max_eval_examples,
    )
    if distributed.is_main_process:
        eval_row = {"step": step, **(extra_fields or {}), **metrics}
        append_jsonl(output_dir / "eval_metrics.jsonl", eval_row)
        append_main_results(
            output_dir / "main_results.csv",
            {"step": step, "train_loss": train_loss, **metrics},
        )
        wandb_payload = {"eval/step": step, **{f"eval/{key}": value for key, value in metrics.items()}}
        for key, value in (extra_fields or {}).items():
            wandb_payload[f"eval/{key}"] = value
        if log_to_wandb:
            wandb_log(wandb_run, wandb_payload, step=step, mirror_path=output_dir / "wandb_log_payloads.jsonl")
        if not args.no_plots:
            write_plots(output_dir)
    barrier(distributed)
    return metrics if distributed.is_main_process else None


def main() -> None:
    args = parse_args()
    args.max_seq_length = 1024
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive")
    if args.eval_every_steps <= 0:
        raise ValueError("--eval_every_steps must be positive")
    if args.save_every_steps <= 0:
        raise ValueError("--save_every_steps must be positive")

    distributed = init_distributed_context(args)
    try:
        set_seed(int(args.seed) + distributed.rank)
        output_dir = Path(args.output_dir).resolve()
        if distributed.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier(distributed)

        init_checkpoint_dir = Path(args.init_critic_checkpoint_dir).resolve()
        if distributed.is_main_process:
            critic_hf_dir = ensure_merged_component_checkpoint(
                init_checkpoint_dir,
                component="critic",
                merged_root=Path(args.merged_root).resolve() if args.merged_root else None,
                hf_source_dir=Path(args.critic_hf_source_dir).resolve() if args.critic_hf_source_dir else None,
                skip_merge=bool(args.skip_merge),
            )
            critic_hf_dir_payload = [str(critic_hf_dir)]
        else:
            critic_hf_dir_payload = [None]
        if distributed.enabled:
            dist.broadcast_object_list(critic_hf_dir_payload, src=0)
        critic_hf_dir = Path(critic_hf_dir_payload[0])
        barrier(distributed)

        dtype = resolve_dtype(args.dtype)
        device = distributed.device
        tokenizer = load_tokenizer(critic_hf_dir, trust_remote_code=bool(args.trust_remote_code))
        critic = load_critic_model(
            critic_hf_dir,
            dtype=dtype,
            device=device,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if type(critic).__name__ == "PRMCriticAdapter":
            raise TypeError("Training PRM critic adapters is not supported by this script; use a value-head critic checkpoint.")
        gradient_checkpointing_enabled = enable_gradient_checkpointing_if_requested(critic, bool(args.gradient_checkpointing))
        trainable_info = configure_trainable_parameters(critic, args.trainable_scope)
        rank_print(
            distributed,
            "Trainable parameters: "
            f"{trainable_info['trainable_parameter_count']:,} / {trainable_info['total_parameter_count']:,} "
            f"({100.0 * trainable_info['trainable_parameter_fraction']:.4f}%) "
            f"scope={args.trainable_scope} gradient_checkpointing={gradient_checkpointing_enabled} "
            f"distributed={args.distributed_backend} world_size={distributed.world_size}",
        )
        critic = wrap_with_fsdp_if_needed(critic, distributed=distributed, args=args)
        critic.train()
        if distributed.is_main_process:
            log_cuda_memory("after_model_load", device)

        train_dataset = SearchInducedCriticDataset(
            args.train_data_path,
            tokenizer=tokenizer,
            max_seq_length=int(args.max_seq_length),
        )
        eval_dataset = SearchInducedCriticDataset(
            args.eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=int(args.max_seq_length),
        )
        pad_token_id = int(tokenizer.pad_token_id)
        train_sampler = SearchInducedBatchSampler(
            train_dataset,
            batch_size=int(args.batch_size),
            mode=args.batch_sampling_mode,
            seed=int(args.seed) + distributed.rank * 1009,
            max_examples_per_prompt_per_batch=int(args.max_examples_per_prompt_per_batch),
            positive_fraction=args.positive_fraction,
            rankable_group_fraction=float(args.rankable_group_fraction),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=lambda examples: collate_candidates(examples, pad_token_id=pad_token_id),
            num_workers=int(args.num_workers),
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(args.eval_batch_size or args.batch_size),
            shuffle=False,
            collate_fn=lambda examples: collate_candidates(examples, pad_token_id=pad_token_id),
            num_workers=int(args.num_workers),
        )

        config_payload = vars(args).copy()
        config_payload.update(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repo_root": str(Path(__file__).resolve().parents[1]),
                "git_commit": make_git_commit(Path(__file__).resolve().parents[1]),
                "resolved_critic_hf_dir": str(critic_hf_dir),
                "train_num_examples": len(train_dataset),
                "eval_num_examples": len(eval_dataset),
                "train_num_rankable_groups": len(train_dataset.rankable_group_ids),
                "eval_num_rankable_groups": len(eval_dataset.rankable_group_ids),
                "train_missing_token_id_rows": train_dataset.num_missing_token_ids,
                "train_missing_prompt_text_rows": train_dataset.num_missing_prompt_text,
                "eval_missing_token_id_rows": eval_dataset.num_missing_token_ids,
                "eval_missing_prompt_text_rows": eval_dataset.num_missing_prompt_text,
                "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
                "distributed_enabled": distributed.enabled,
                "world_size": distributed.world_size,
                **trainable_info,
            }
        )
        if distributed.is_main_process:
            save_json(output_dir / "config.json", config_payload)
            wandb_run = init_wandb(args, config_payload)
            if wandb_run is not None:
                wandb_run.summary["world_size"] = distributed.world_size
                wandb_run.summary["effective_batch_size"] = int(args.batch_size) * int(args.grad_accum_steps) * distributed.world_size
                wandb_run.summary["train_num_examples"] = len(train_dataset)
                wandb_run.summary["eval_num_examples"] = len(eval_dataset)
                wandb_run.summary["max_eval_examples"] = 0 if args.max_eval_examples is None else int(args.max_eval_examples)
        else:
            wandb_run = None
        wandb_mirror_path = output_dir / "wandb_log_payloads.jsonl" if distributed.is_main_process else None
        barrier(distributed)

        optimizer = AdamW(
            (parameter for parameter in critic.parameters() if parameter.requires_grad),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            eps=float(args.adam_eps),
        )
        global_step = 0
        micro_step = 0
        gradient_accumulated_batches = 0
        optimizer.zero_grad(set_to_none=True)
        last_train_log: dict[str, Any] = {"loss": None}

        if args.eval_at_start:
            run_eval_and_log(
                critic=critic,
                eval_loader=eval_loader,
                device=device,
                output_dir=output_dir,
                step=0,
                train_loss=None,
                wandb_run=wandb_run,
                args=args,
                distributed=distributed,
                extra_fields={"eval_at_start": True},
            )

        progress = tqdm(total=args.max_train_steps, desc="train steps", disable=not distributed.is_main_process) if args.max_train_steps else None
        stop_training = False
        for epoch in range(int(args.num_train_epochs)):
            train_sampler.set_epoch(epoch)
            iterator = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False, disable=not distributed.is_main_process)
            for batch in iterator:
                batch = batch_to_device(batch, device)
                values = critic_last_token_values_trainable(critic, batch["input_ids"], batch["attention_mask"])
                assert_finite_tensor("critic values", values, step=global_step + 1, distributed=distributed)
                loss_dict = compute_loss(
                    values,
                    batch["target_reward"],
                    batch,
                    loss_type=args.loss_type,
                    rank_loss_weight=float(args.rank_loss_weight),
                )
                loss = loss_dict["loss"]
                assert_finite_tensor("loss", loss, step=global_step + 1, distributed=distributed)
                loss.backward()
                micro_step += 1
                gradient_accumulated_batches += 1

                if micro_step % int(args.grad_accum_steps) != 0:
                    continue

                for parameter in critic.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(gradient_accumulated_batches))
                grad_norm = clip_grad_norm(critic, 1.0)
                grad_norm_value = finite_grad_norm_value(grad_norm, device=device)
                if not torch.isfinite(grad_norm_value).item():
                    log_non_finite_grad_skip(
                        output_dir=output_dir,
                        step=global_step + 1,
                        epoch=epoch,
                        grad_norm=grad_norm_value,
                        distributed=distributed,
                        model=critic,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    gradient_accumulated_batches = 0
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                gradient_accumulated_batches = 0
                global_step += 1
                if progress is not None:
                    progress.update(1)

                reduced_loss_dict = reduce_loss_dict_for_logging(loss_dict, distributed)
                if distributed.is_main_process:
                    last_train_log = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(reduced_loss_dict["loss"].detach().cpu().item()),
                        "loss_mse": float(reduced_loss_dict["loss_mse"].cpu().item()),
                        "loss_bce": float(reduced_loss_dict["loss_bce"].cpu().item()),
                        "loss_rank": float(reduced_loss_dict["loss_rank"].cpu().item()),
                        "grad_norm": float(grad_norm_value.detach().cpu().item()),
                        "num_rankable_groups_in_batch": int(reduced_loss_dict["num_rankable_groups_in_batch"]),
                        "num_pairs_in_batch": int(reduced_loss_dict["num_pairs_in_batch"]),
                        "value_mean": float(reduced_loss_dict["value_mean"].cpu().item()),
                        "target_mean": float(reduced_loss_dict["target_mean"].cpu().item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "world_size": distributed.world_size,
                        "effective_batch_size": int(args.batch_size) * int(args.grad_accum_steps) * distributed.world_size,
                    }
                    append_jsonl(output_dir / "train_log.jsonl", last_train_log)

                eval_metrics = None
                if global_step % int(args.eval_every_steps) == 0:
                    eval_metrics = run_eval_and_log(
                        critic=critic,
                        eval_loader=eval_loader,
                        device=device,
                        output_dir=output_dir,
                        step=global_step,
                        train_loss=last_train_log.get("loss"),
                        wandb_run=wandb_run,
                        args=args,
                        distributed=distributed,
                        log_to_wandb=False,
                    )

                if distributed.is_main_process:
                    wandb_payload = {
                        "train/step": global_step,
                        "train/loss": last_train_log["loss"],
                        "train/loss_mse": last_train_log["loss_mse"],
                        "train/loss_bce": last_train_log["loss_bce"],
                        "train/loss_rank": last_train_log["loss_rank"],
                        "train/grad_norm": last_train_log["grad_norm"],
                        "train/num_rankable_groups_in_batch": last_train_log["num_rankable_groups_in_batch"],
                        "train/num_pairs_in_batch": last_train_log["num_pairs_in_batch"],
                        "train/value_mean": last_train_log["value_mean"],
                        "train/target_mean": last_train_log["target_mean"],
                        "train/lr": last_train_log["lr"],
                        "train/epoch": epoch,
                        "train/effective_batch_size": last_train_log["effective_batch_size"],
                    }
                    if eval_metrics is not None:
                        wandb_payload.update({"eval/step": global_step, **{f"eval/{key}": value for key, value in eval_metrics.items()}})
                    wandb_log(wandb_run, wandb_payload, step=global_step, mirror_path=wandb_mirror_path)
                if global_step % int(args.save_every_steps) == 0:
                    checkpoint_dir = save_checkpoint(critic, tokenizer, output_dir, global_step, distributed=distributed)
                    if distributed.is_main_process:
                        maybe_run_chunk_eval(args, checkpoint_dir, global_step, output_dir)
                    barrier(distributed)

                if args.max_train_steps is not None and global_step >= int(args.max_train_steps):
                    stop_training = True
                stop_tensor = torch.tensor(int(stop_training), device=device)
                if distributed.enabled:
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                stop_training = bool(stop_tensor.item())
                if stop_training:
                    break
            if stop_training:
                break
            if gradient_accumulated_batches > 0:
                for parameter in critic.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(gradient_accumulated_batches))
                grad_norm = clip_grad_norm(critic, 1.0)
                grad_norm_value = finite_grad_norm_value(grad_norm, device=device)
                if not torch.isfinite(grad_norm_value).item():
                    log_non_finite_grad_skip(
                        output_dir=output_dir,
                        step=global_step + 1,
                        epoch=epoch,
                        grad_norm=grad_norm_value,
                        distributed=distributed,
                        model=critic,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    gradient_accumulated_batches = 0
                    global_step += 1
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                gradient_accumulated_batches = 0
                global_step += 1
                if progress is not None:
                    progress.update(1)
                if distributed.is_main_process:
                    remainder_log = {
                        **last_train_log,
                        "step": global_step,
                        "epoch": epoch,
                        "flushed_epoch_remainder": True,
                    }
                    append_jsonl(output_dir / "train_log.jsonl", remainder_log)
                    wandb_log(
                        wandb_run,
                        {
                            "train/step": global_step,
                            "train/flushed_epoch_remainder": True,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                        mirror_path=wandb_mirror_path,
                    )

        if progress is not None:
            progress.close()

        if global_step == 0:
            raise RuntimeError("Training finished before any optimizer step. Reduce --grad_accum_steps or check the data.")

        final_metrics = run_eval_and_log(
            critic=critic,
            eval_loader=eval_loader,
            device=device,
            output_dir=output_dir,
            step=global_step,
            train_loss=last_train_log.get("loss"),
            wandb_run=wandb_run,
            args=args,
            distributed=distributed,
            extra_fields={"final": True},
            log_to_wandb=(global_step % int(args.eval_every_steps) != 0),
        )
        final_checkpoint_dir = save_checkpoint(critic, tokenizer, output_dir, global_step, distributed=distributed)
        if distributed.is_main_process:
            final_row = {"step": global_step, "final": True, **(final_metrics or {})}
            save_json(output_dir / "final_metrics.json", final_row)
            if wandb_run is not None:
                wandb_run.summary["final_checkpoint_dir"] = str(final_checkpoint_dir)
                for key, value in (final_metrics or {}).items():
                    if isinstance(value, (int, float, bool)) and value is not None and math.isfinite(float(value)):
                        wandb_run.summary[f"final/{key}"] = value
            if not args.no_plots:
                write_plots(output_dir)
            maybe_run_chunk_eval(args, final_checkpoint_dir, global_step, output_dir)
            wandb_finish(wandb_run)
            print(json.dumps({"step": global_step, "final_checkpoint_dir": str(final_checkpoint_dir), **(final_metrics or {})}, indent=2, sort_keys=True))
        barrier(distributed)
    finally:
        cleanup_distributed_context(distributed)


if __name__ == "__main__":
    main()
