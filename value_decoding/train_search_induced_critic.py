from __future__ import annotations

import argparse
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
import torch.nn.functional as F
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
    parser.add_argument("--max_seq_length", type=int, default=4096)
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


def wandb_log(wandb_run, metrics: dict[str, Any], *, step: int) -> None:
    if wandb_run is None:
        return
    clean_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool)) and value is not None and math.isfinite(float(value))
    }
    if clean_metrics:
        wandb_run.log(clean_metrics, step=int(step))


def wandb_finish(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


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


@torch.inference_mode()
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


def save_checkpoint(critic, tokenizer, output_dir: Path, step: int) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = critic.module if hasattr(critic, "module") else critic
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


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive")
    if args.eval_every_steps <= 0:
        raise ValueError("--eval_every_steps must be positive")
    if args.save_every_steps <= 0:
        raise ValueError("--save_every_steps must be positive")

    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    init_checkpoint_dir = Path(args.init_critic_checkpoint_dir).resolve()
    critic_hf_dir = ensure_merged_component_checkpoint(
        init_checkpoint_dir,
        component="critic",
        merged_root=Path(args.merged_root).resolve() if args.merged_root else None,
        hf_source_dir=Path(args.critic_hf_source_dir).resolve() if args.critic_hf_source_dir else None,
        skip_merge=bool(args.skip_merge),
    )
    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(critic_hf_dir, trust_remote_code=bool(args.trust_remote_code))
    critic = load_critic_model(
        critic_hf_dir,
        dtype=dtype,
        device=device,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if type(critic).__name__ == "PRMCriticAdapter":
        raise TypeError("Training PRM critic adapters is not supported by this script; use a value-head critic checkpoint.")
    critic.train()

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
        seed=int(args.seed),
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
        }
    )
    save_json(output_dir / "config.json", config_payload)
    wandb_run = init_wandb(args, config_payload)
    if wandb_run is not None:
        wandb_run.define_metric("train/step")
        wandb_run.define_metric("train/*", step_metric="train/step")
        wandb_run.define_metric("eval/step")
        wandb_run.define_metric("eval/*", step_metric="eval/step")

    optimizer = AdamW(critic.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    global_step = 0
    micro_step = 0
    gradient_accumulated_batches = 0
    optimizer.zero_grad(set_to_none=True)
    last_train_log: dict[str, Any] = {"loss": None}

    if args.eval_at_start:
        eval_metrics = evaluate_critic(
            critic,
            eval_loader,
            device=device,
            max_examples=args.max_eval_examples,
        )
        eval_row = {"step": 0, "eval_at_start": True, **eval_metrics}
        append_jsonl(output_dir / "eval_metrics.jsonl", eval_row)
        append_main_results(
            output_dir / "main_results.csv",
            {"step": 0, "train_loss": None, **eval_metrics},
        )
        wandb_log(
            wandb_run,
            {"eval/step": 0, "eval/eval_at_start": True, **{f"eval/{key}": value for key, value in eval_metrics.items()}},
            step=0,
        )
        if not args.no_plots:
            write_plots(output_dir)

    progress = tqdm(total=args.max_train_steps, desc="train steps") if args.max_train_steps else None
    stop_training = False
    for epoch in range(int(args.num_train_epochs)):
        train_sampler.set_epoch(epoch)
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False):
            batch = batch_to_device(batch, device)
            values = critic_last_token_values_trainable(critic, batch["input_ids"], batch["attention_mask"])
            loss_dict = compute_loss(
                values,
                batch["target_reward"],
                batch,
                loss_type=args.loss_type,
                rank_loss_weight=float(args.rank_loss_weight),
            )
            loss = loss_dict["loss"]
            loss.backward()
            micro_step += 1
            gradient_accumulated_batches += 1

            if micro_step % int(args.grad_accum_steps) != 0:
                continue

            for parameter in critic.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(gradient_accumulated_batches))
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gradient_accumulated_batches = 0
            global_step += 1
            if progress is not None:
                progress.update(1)

            last_train_log = {
                "step": global_step,
                "epoch": epoch,
                "loss": float(loss_dict["loss"].detach().cpu().item()),
                "loss_mse": float(loss_dict["loss_mse"].cpu().item()),
                "loss_bce": float(loss_dict["loss_bce"].cpu().item()),
                "loss_rank": float(loss_dict["loss_rank"].cpu().item()),
                "num_rankable_groups_in_batch": int(loss_dict["num_rankable_groups_in_batch"]),
                "num_pairs_in_batch": int(loss_dict["num_pairs_in_batch"]),
                "value_mean": float(loss_dict["value_mean"].cpu().item()),
                "target_mean": float(loss_dict["target_mean"].cpu().item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            append_jsonl(output_dir / "train_log.jsonl", last_train_log)
            wandb_log(
                wandb_run,
                {
                    "train/step": global_step,
                    "train/loss": last_train_log["loss"],
                    "train/loss_mse": last_train_log["loss_mse"],
                    "train/loss_bce": last_train_log["loss_bce"],
                    "train/loss_rank": last_train_log["loss_rank"],
                    "train/num_rankable_groups_in_batch": last_train_log["num_rankable_groups_in_batch"],
                    "train/num_pairs_in_batch": last_train_log["num_pairs_in_batch"],
                    "train/value_mean": last_train_log["value_mean"],
                    "train/target_mean": last_train_log["target_mean"],
                    "train/lr": last_train_log["lr"],
                    "train/epoch": epoch,
                },
                step=global_step,
            )

            if global_step % int(args.eval_every_steps) == 0:
                eval_metrics = evaluate_critic(
                    critic,
                    eval_loader,
                    device=device,
                    max_examples=args.max_eval_examples,
                )
                eval_row = {"step": global_step, **eval_metrics}
                append_jsonl(output_dir / "eval_metrics.jsonl", eval_row)
                append_main_results(
                    output_dir / "main_results.csv",
                    {"step": global_step, "train_loss": last_train_log.get("loss"), **eval_metrics},
                )
                wandb_log(
                    wandb_run,
                    {"eval/step": global_step, **{f"eval/{key}": value for key, value in eval_metrics.items()}},
                    step=global_step,
                )
                if not args.no_plots:
                    write_plots(output_dir)

            if global_step % int(args.save_every_steps) == 0:
                checkpoint_dir = save_checkpoint(critic, tokenizer, output_dir, global_step)
                maybe_run_chunk_eval(args, checkpoint_dir, global_step, output_dir)

            if args.max_train_steps is not None and global_step >= int(args.max_train_steps):
                stop_training = True
                break
        if stop_training:
            break
        if gradient_accumulated_batches > 0:
            for parameter in critic.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(gradient_accumulated_batches))
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gradient_accumulated_batches = 0
            global_step += 1
            if progress is not None:
                progress.update(1)
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
            )

    if progress is not None:
        progress.close()

    if global_step == 0:
        raise RuntimeError("Training finished before any optimizer step. Reduce --grad_accum_steps or check the data.")

    final_metrics = evaluate_critic(critic, eval_loader, device=device, max_examples=args.max_eval_examples)
    final_row = {"step": global_step, "final": True, **final_metrics}
    append_jsonl(output_dir / "eval_metrics.jsonl", final_row)
    append_main_results(output_dir / "main_results.csv", {"step": global_step, "train_loss": last_train_log.get("loss"), **final_metrics})
    final_checkpoint_dir = save_checkpoint(critic, tokenizer, output_dir, global_step)
    save_json(output_dir / "final_metrics.json", final_row)
    wandb_log(
        wandb_run,
        {
            "eval/step": global_step,
            "eval/final": True,
            "eval/final_checkpoint_step": global_step,
            **{f"eval/{key}": value for key, value in final_metrics.items()},
        },
        step=global_step,
    )
    if wandb_run is not None:
        wandb_run.summary["final_checkpoint_dir"] = str(final_checkpoint_dir)
        for key, value in final_metrics.items():
            if isinstance(value, (int, float, bool)) and value is not None and math.isfinite(float(value)):
                wandb_run.summary[f"final/{key}"] = value
    if not args.no_plots:
        write_plots(output_dir)
    maybe_run_chunk_eval(args, final_checkpoint_dir, global_step, output_dir)
    wandb_finish(wandb_run)
    print(json.dumps({"step": global_step, "final_checkpoint_dir": str(final_checkpoint_dir), **final_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
