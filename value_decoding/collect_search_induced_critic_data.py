from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from value_decoding.decoding import critic_sequence_last_values, sample_token_from_actor, set_decode_seed

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
except ImportError:  # pragma: no cover - depends on runtime environment
    LLM = None
    SamplingParams = None
    TokensPrompt = None


SELECTION_MODES = ("argmax", "epsilon_greedy", "softmax_value", "actor_logprob", "random")


@dataclass(frozen=True)
class GeneratedSegment:
    token_ids: tuple[int, ...]
    text: str
    ended_with_eos: bool
    hit_max_length: bool
    logprob: float | None
    avg_logprob: float | None


@dataclass(frozen=True)
class ActorGenerationRequest:
    prompt_token_ids: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]
    max_new_tokens: int
    seed: int


@dataclass(frozen=True)
class CandidateRecord:
    candidate_index: int
    prefix_tokens_before_chunk: tuple[int, ...]
    chunk_tokens: tuple[int, ...]
    candidate_prefix_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    full_response_tokens: tuple[int, ...]
    prefix_text_before_chunk: str
    chunk_text: str
    candidate_prefix_text: str
    completion_text: str
    full_response_text: str
    candidate_chunk_has_eos: bool
    completion_ended_with_eos: bool
    completion_hit_max_length: bool
    collector_value: float
    actor_chunk_logprob: float | None
    actor_chunk_avg_logprob: float | None
    mc_reward: float


class ActorGenerator:
    def generate(
        self,
        *,
        prompt_token_ids: Sequence[int],
        prefix_token_ids: Sequence[int],
        max_new_tokens: int,
        seed: int,
    ) -> GeneratedSegment:
        raise NotImplementedError

    def generate_batch(self, requests: Sequence[ActorGenerationRequest]) -> list[GeneratedSegment]:
        return [
            self.generate(
                prompt_token_ids=request.prompt_token_ids,
                prefix_token_ids=request.prefix_token_ids,
                max_new_tokens=request.max_new_tokens,
                seed=request.seed,
            )
            for request in requests
        ]


class VLLMActorGenerator(ActorGenerator):
    def __init__(
        self,
        *,
        model_dir: Path,
        tokenizer,
        dtype_name: str,
        trust_remote_code: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_token_ids: tuple[int, ...],
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        max_model_len: int | None,
        enforce_eager: bool,
        request_logprobs: bool,
    ) -> None:
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM is not installed")
        kwargs: dict[str, Any] = {
            "model": str(model_dir),
            "tokenizer": str(model_dir),
            "dtype": _resolve_vllm_dtype_name(dtype_name),
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": bool(enforce_eager),
        }
        if max_model_len is not None and max_model_len > 0:
            kwargs["max_model_len"] = int(max_model_len)
        self.llm = LLM(**kwargs)
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.eos_token_ids = tuple(int(token_id) for token_id in eos_token_ids)
        self.request_logprobs = bool(request_logprobs)
        vocab_size = getattr(getattr(self.llm, "llm_engine", None), "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(tokenizer, "vocab_size", None)
        self.logprobs_count = 1 if request_logprobs else None
        if self.logprobs_count is not None and vocab_size is not None:
            self.logprobs_count = min(int(vocab_size), max(1, self.logprobs_count))

    def generate(
        self,
        *,
        prompt_token_ids: Sequence[int],
        prefix_token_ids: Sequence[int],
        max_new_tokens: int,
        seed: int,
    ) -> GeneratedSegment:
        return self.generate_batch(
            [
                ActorGenerationRequest(
                    prompt_token_ids=tuple(int(token_id) for token_id in prompt_token_ids),
                    prefix_token_ids=tuple(int(token_id) for token_id in prefix_token_ids),
                    max_new_tokens=int(max_new_tokens),
                    seed=int(seed),
                )
            ]
        )[0]

    def generate_batch(self, requests: Sequence[ActorGenerationRequest]) -> list[GeneratedSegment]:
        if not requests:
            return []

        active_indices: list[int] = []
        prompts: list[Any] = []
        sampling_params: list[Any] = []
        results: list[GeneratedSegment | None] = [None for _ in requests]
        for request_index, request in enumerate(requests):
            if int(request.max_new_tokens) <= 0:
                results[request_index] = GeneratedSegment((), "", False, True, 0.0, None)
                continue
            input_token_ids = [int(token_id) for token_id in (*request.prompt_token_ids, *request.prefix_token_ids)]
            prompt_payload = (
                TokensPrompt(prompt_token_ids=input_token_ids)
                if TokensPrompt is not None
                else {"prompt_token_ids": input_token_ids}
            )
            prompts.append(prompt_payload)
            sampling_params.append(
                SamplingParams(
                    n=1,
                    temperature=float(self.temperature),
                    top_p=float(self.top_p),
                    top_k=0 if self.top_k <= 0 else int(self.top_k),
                    max_tokens=int(request.max_new_tokens),
                    seed=int(request.seed),
                    stop_token_ids=list(self.eos_token_ids),
                    logprobs=self.logprobs_count,
                    skip_special_tokens=True,
                )
            )
            active_indices.append(request_index)

        if active_indices:
            request_outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
            if len(request_outputs) != len(active_indices):
                raise RuntimeError(
                    f"vLLM returned {len(request_outputs)} outputs for {len(active_indices)} batched prompts."
                )
            for request_output, request_index in zip(request_outputs, active_indices):
                output = request_output.outputs[0]
                request = requests[request_index]
                token_ids = _vllm_output_token_ids(output)
                finish_reason = _vllm_finish_reason(output)
                ended_with_eos = bool(token_ids and token_ids[-1] in self.eos_token_ids) or finish_reason == "stop"
                hit_max_length = finish_reason == "length" or (
                    int(request.max_new_tokens) > 0 and not ended_with_eos and len(token_ids) >= int(request.max_new_tokens)
                )
                logprob = _vllm_output_sequence_logprob(output, token_ids) if self.request_logprobs else None
                avg_logprob = None if logprob is None or not token_ids else float(logprob) / float(len(token_ids))
                results[request_index] = GeneratedSegment(
                    token_ids=token_ids,
                    text=_decode_token_ids(self.tokenizer, token_ids),
                    ended_with_eos=bool(ended_with_eos),
                    hit_max_length=bool(hit_max_length),
                    logprob=logprob,
                    avg_logprob=avg_logprob,
                )

        missing = [index for index, result in enumerate(results) if result is None]
        if missing:
            raise RuntimeError(f"Missing vLLM generations for request indices: {missing}")
        return [result for result in results if result is not None]


class TorchActorGenerator(ActorGenerator):
    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: torch.device,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_token_ids: tuple[int, ...],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.eos_token_ids = tuple(int(token_id) for token_id in eos_token_ids)

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt_token_ids: Sequence[int],
        prefix_token_ids: Sequence[int],
        max_new_tokens: int,
        seed: int,
    ) -> GeneratedSegment:
        if max_new_tokens <= 0:
            return GeneratedSegment((), "", False, True, 0.0, None)
        set_decode_seed(int(seed))
        input_token_ids = [int(token_id) for token_id in (*prompt_token_ids, *prefix_token_ids)]
        if not input_token_ids:
            raise ValueError("Cannot generate from an empty prompt.")
        sequence = torch.tensor([input_token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(sequence, device=self.device)
        generated: list[int] = []
        token_logprobs: list[float] = []
        ended_with_eos = False
        past_key_values = None
        use_cache = True

        for step in range(int(max_new_tokens)):
            if step == 0 or past_key_values is None:
                outputs = self.model(input_ids=sequence, attention_mask=attention_mask, use_cache=True)
            else:
                outputs = self.model(
                    input_ids=sequence[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                use_cache = False
            logits = outputs.logits[:, -1, :]
            token_id = sample_token_from_actor(
                logits,
                sampling_mode="sample",
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            token_logprob = _selected_token_logprob(
                logits,
                token_id=token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            generated.append(int(token_id))
            token_logprobs.append(float(token_logprob))
            token_tensor = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
            sequence = torch.cat([sequence, token_tensor], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(token_tensor, device=self.device)], dim=1)
            if int(token_id) in self.eos_token_ids:
                ended_with_eos = True
                break
            if not use_cache:
                past_key_values = None

        token_ids = tuple(generated)
        logprob = float(sum(token_logprobs)) if token_logprobs else 0.0
        avg_logprob = float(logprob) / float(len(token_logprobs)) if token_logprobs else None
        hit_max_length = not ended_with_eos and len(generated) >= int(max_new_tokens)
        return GeneratedSegment(
            token_ids=token_ids,
            text=_decode_token_ids(self.tokenizer, token_ids),
            ended_with_eos=bool(ended_with_eos),
            hit_max_length=bool(hit_max_length),
            logprob=logprob,
            avg_logprob=avg_logprob,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect search-induced critic training data from chunk candidates.")
    parser.add_argument("--actor_checkpoint_dir", required=True, type=str)
    parser.add_argument("--collector_critic_checkpoint_dir", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--prompt_key", default="prompt", type=str)
    parser.add_argument("--response_key", default=None, type=_none_or_str)
    parser.add_argument("--max_prompts", required=True, type=int)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--shuffle_prompts", nargs="?", const=True, default=False, type=_str_to_bool)
    parser.add_argument("--chunk_size", required=True, type=int)
    parser.add_argument("--num_chunk_candidates", required=True, type=int)
    parser.add_argument("--num_search_steps_per_prompt", required=True, type=int)
    parser.add_argument("--completion_max_new_tokens", required=True, type=int)
    parser.add_argument("--collector_selection_mode", required=True, choices=SELECTION_MODES)
    parser.add_argument("--collector_epsilon", default=0.1, type=float)
    parser.add_argument("--collector_value_temperature", default=1.0, type=float)
    parser.add_argument("--actor_temperature", default=1.0, type=float)
    parser.add_argument("--actor_top_p", default=1.0, type=float)
    parser.add_argument("--actor_top_k", default=0, type=int)
    parser.add_argument("--actor_batch_size", default=8, type=int)
    parser.add_argument("--max_prompt_length", default=2048, type=int)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--trust_remote_code", nargs="?", const=True, default=False, type=_str_to_bool)
    parser.add_argument("--save_full_text", nargs="?", const=True, default=True, type=_str_to_bool)
    parser.add_argument("--save_token_ids", nargs="?", const=True, default=True, type=_str_to_bool)
    parser.add_argument("--save_completed_responses", nargs="?", const=True, default=False, type=_str_to_bool)
    parser.add_argument("--debug_num_prompts", default=None, type=int)
    parser.add_argument("--generation_backend", default="auto", choices=("auto", "vllm", "torch"))
    parser.add_argument("--actor_device", default=None, type=str)
    parser.add_argument("--critic_device", default=None, type=str)
    parser.add_argument("--merged_root", default=None, type=str)
    parser.add_argument("--actor_hf_source_dir", default=None, type=str)
    parser.add_argument("--critic_hf_source_dir", default=None, type=str)
    parser.add_argument("--skip_merge", nargs="?", const=True, default=False, type=_str_to_bool)
    parser.add_argument("--critic_batch_size", default=8, type=int)
    parser.add_argument("--vllm_gpu_memory_utilization", default=0.6, type=float)
    parser.add_argument("--vllm_tensor_parallel_size", default=1, type=int)
    parser.add_argument("--vllm_max_model_len", default=None, type=int)
    parser.add_argument("--vllm_enforce_eager", nargs="?", const=True, default=False, type=_str_to_bool)
    args = parser.parse_args()
    _validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    set_decode_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    actor_checkpoint_dir = Path(args.actor_checkpoint_dir)
    critic_checkpoint_dir = Path(args.collector_critic_checkpoint_dir)
    merged_root = None if args.merged_root is None else Path(args.merged_root)
    actor_hf_dir = ensure_merged_component_checkpoint(
        actor_checkpoint_dir,
        component="actor",
        merged_root=merged_root,
        hf_source_dir=None if args.actor_hf_source_dir is None else Path(args.actor_hf_source_dir),
        skip_merge=bool(args.skip_merge),
    )
    critic_hf_dir = ensure_merged_component_checkpoint(
        critic_checkpoint_dir,
        component="critic",
        merged_root=merged_root,
        hf_source_dir=None if args.critic_hf_source_dir is None else Path(args.critic_hf_source_dir),
        skip_merge=bool(args.skip_merge),
    )

    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=bool(args.trust_remote_code))
    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, tokenizer)
    examples = load_examples(
        args.dataset_path,
        tokenizer=tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=int(args.start_index),
        max_examples=int(args.debug_num_prompts or args.max_prompts),
        shuffle_examples=bool(args.shuffle_prompts),
        seed=int(args.seed),
        pretokenize_max_length=int(args.max_prompt_length),
    )

    dtype = resolve_dtype(args.dtype)
    critic_device = resolve_device(args.critic_device)
    collector_critic = load_critic_model(
        critic_hf_dir,
        dtype=dtype,
        device=critic_device,
        trust_remote_code=bool(args.trust_remote_code),
    )
    collector_critic.eval()
    for parameter in collector_critic.parameters():
        parameter.requires_grad_(False)

    backend = _resolve_generation_backend(args.generation_backend)
    actor_model = None
    if backend == "vllm":
        try:
            actor_generator: ActorGenerator = VLLMActorGenerator(
                model_dir=actor_hf_dir,
                tokenizer=tokenizer,
                dtype_name=args.dtype,
                trust_remote_code=bool(args.trust_remote_code),
                temperature=float(args.actor_temperature),
                top_p=float(args.actor_top_p),
                top_k=int(args.actor_top_k),
                eos_token_ids=eos_token_ids,
                gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
                tensor_parallel_size=int(args.vllm_tensor_parallel_size),
                max_model_len=args.vllm_max_model_len,
                enforce_eager=bool(args.vllm_enforce_eager),
                request_logprobs=True,
            )
        except Exception:
            if args.generation_backend == "vllm":
                raise
            backend = "torch"
            actor_generator = _build_torch_generator(args, actor_hf_dir, tokenizer, dtype, eos_token_ids)
            actor_model = getattr(actor_generator, "model", None)
    else:
        actor_generator = _build_torch_generator(args, actor_hf_dir, tokenizer, dtype, eos_token_ids)
        actor_model = getattr(actor_generator, "model", None)
    if actor_model is not None:
        actor_model.eval()
        for parameter in actor_model.parameters():
            parameter.requires_grad_(False)

    candidate_path = output_dir / "search_induced_candidates.jsonl"
    prompt_summary_path = output_dir / "prompt_summaries.jsonl"
    all_candidate_rewards: list[float] = []
    all_candidate_values: list[float] = []
    all_group_metrics: list[dict[str, Any]] = []
    all_prompt_summaries: list[dict[str, Any]] = []

    with candidate_path.open("w", encoding="utf-8") as candidate_handle, prompt_summary_path.open(
        "w", encoding="utf-8"
    ) as prompt_handle:
        for prompt_index, example in enumerate(tqdm(examples, desc="collect prompts")):
            prompt_summary, prompt_group_metrics, prompt_rewards, prompt_values = collect_for_prompt(
                args=args,
                example=example,
                prompt_index=prompt_index,
                tokenizer=tokenizer,
                eos_token_ids=eos_token_ids,
                actor_generator=actor_generator,
                collector_critic=collector_critic,
                critic_device=critic_device,
                candidate_handle=candidate_handle,
            )
            prompt_handle.write(json.dumps(prompt_summary, ensure_ascii=False) + "\n")
            all_prompt_summaries.append(prompt_summary)
            all_group_metrics.extend(prompt_group_metrics)
            all_candidate_rewards.extend(prompt_rewards)
            all_candidate_values.extend(prompt_values)

    summary = build_summary_metrics(
        args=args,
        num_prompts=len(examples),
        candidate_rewards=all_candidate_rewards,
        candidate_values=all_candidate_values,
        group_metrics=all_group_metrics,
        prompt_summaries=all_prompt_summaries,
    )
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_readme(output_dir, args=args, actor_hf_dir=actor_hf_dir, critic_hf_dir=critic_hf_dir, backend=backend)


def collect_for_prompt(
    *,
    args: argparse.Namespace,
    example: ExampleRecord,
    prompt_index: int,
    tokenizer,
    eos_token_ids: tuple[int, ...],
    actor_generator: ActorGenerator,
    collector_critic,
    critic_device: torch.device,
    candidate_handle,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float], list[float]]:
    prompt_token_ids = _example_prompt_token_ids(example, tokenizer, max_prompt_length=int(args.max_prompt_length))
    prefix_tokens: tuple[int, ...] = ()
    committed_prefix_has_eos = False
    prompt_group_metrics: list[dict[str, Any]] = []
    prompt_rewards: list[float] = []
    prompt_values: list[float] = []
    num_steps_collected = 0
    rng = random.Random(_stable_seed(args.seed, example.example_id, 17))

    for search_step in range(1, int(args.num_search_steps_per_prompt) + 1):
        candidates: list[CandidateRecord] = []
        chunk_requests = [
            ActorGenerationRequest(
                prompt_token_ids=prompt_token_ids,
                prefix_token_ids=prefix_tokens,
                max_new_tokens=int(args.chunk_size),
                seed=_stable_seed(args.seed, example.example_id, search_step, candidate_index, 1009),
            )
            for candidate_index in range(int(args.num_chunk_candidates))
        ]
        chunk_segments = actor_generate_requests(
            actor_generator,
            chunk_requests,
            batch_size=int(args.actor_batch_size),
        )

        candidate_prefixes = [tuple((*prefix_tokens, *segment.token_ids)) for segment in chunk_segments]
        collector_values = score_candidate_prefixes(
            collector_critic=collector_critic,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            candidate_prefixes=candidate_prefixes,
            device=critic_device,
            batch_size=int(args.critic_batch_size),
        )

        completion_requests: list[ActorGenerationRequest | None] = []
        for candidate_index, chunk in enumerate(chunk_segments):
            candidate_prefix = candidate_prefixes[candidate_index]
            chunk_has_eos = bool(
                chunk.ended_with_eos or any(token_id in eos_token_ids for token_id in chunk.token_ids)
            )
            remaining_budget = max(0, int(args.completion_max_new_tokens) - len(candidate_prefix))
            if chunk_has_eos:
                completion_requests.append(None)
            else:
                completion_requests.append(
                    ActorGenerationRequest(
                        prompt_token_ids=prompt_token_ids,
                        prefix_token_ids=candidate_prefix,
                        max_new_tokens=remaining_budget,
                        seed=_stable_seed(args.seed, example.example_id, search_step, candidate_index, 7919),
                    )
                )

        active_completion_requests = [request for request in completion_requests if request is not None]
        active_completions = actor_generate_requests(
            actor_generator,
            active_completion_requests,
            batch_size=int(args.actor_batch_size),
        )
        completion_iter = iter(active_completions)
        completions: list[GeneratedSegment] = []
        for request in completion_requests:
            if request is None:
                completions.append(GeneratedSegment((), "", False, False, 0.0, None))
            else:
                completions.append(next(completion_iter))

        rewards: list[float] = []
        for candidate_index, (chunk, completion) in enumerate(zip(chunk_segments, completions, strict=True)):
            candidate_prefix = candidate_prefixes[candidate_index]
            chunk_has_eos = bool(
                chunk.ended_with_eos or any(token_id in eos_token_ids for token_id in chunk.token_ids)
            )
            full_response_tokens = tuple((*candidate_prefix, *completion.token_ids))
            full_response_text = _decode_token_ids(tokenizer, full_response_tokens)
            reward = float(score_response(example, full_response_text))
            rewards.append(reward)
            candidates.append(
                CandidateRecord(
                    candidate_index=candidate_index,
                    prefix_tokens_before_chunk=prefix_tokens,
                    chunk_tokens=chunk.token_ids,
                    candidate_prefix_tokens=candidate_prefix,
                    completion_tokens=completion.token_ids,
                    full_response_tokens=full_response_tokens,
                    prefix_text_before_chunk=_decode_token_ids(tokenizer, prefix_tokens),
                    chunk_text=_decode_token_ids(tokenizer, chunk.token_ids),
                    candidate_prefix_text=_decode_token_ids(tokenizer, candidate_prefix),
                    completion_text=_decode_token_ids(tokenizer, completion.token_ids),
                    full_response_text=full_response_text,
                    candidate_chunk_has_eos=chunk_has_eos,
                    completion_ended_with_eos=bool(
                        completion.ended_with_eos or any(token_id in eos_token_ids for token_id in completion.token_ids[-1:])
                    ),
                    completion_hit_max_length=bool(completion.hit_max_length),
                    collector_value=float(collector_values[candidate_index]),
                    actor_chunk_logprob=chunk.logprob,
                    actor_chunk_avg_logprob=chunk.avg_logprob,
                    mc_reward=reward,
                )
            )

        selected_index = select_candidate_index(
            mode=args.collector_selection_mode,
            values=[candidate.collector_value for candidate in candidates],
            actor_logprobs=[candidate.actor_chunk_logprob for candidate in candidates],
            epsilon=float(args.collector_epsilon),
            value_temperature=float(args.collector_value_temperature),
            rng=rng,
        )
        group_metrics = compute_group_metrics(
            rewards=[candidate.mc_reward for candidate in candidates],
            values=[candidate.collector_value for candidate in candidates],
        )
        group_id = f"prompt_{int(example.example_id):06d}_step_{search_step:02d}"
        group_metrics.update(
            {
                "prompt_id": int(example.example_id),
                "search_step": int(search_step),
                "candidate_group_id": group_id,
                "selected_candidate_index": int(selected_index),
            }
        )
        ranks = descending_ranks([candidate.collector_value for candidate in candidates])
        for candidate in candidates:
            row = build_candidate_row(
                args=args,
                example=example,
                search_step=search_step,
                candidate=candidate,
                selected_index=selected_index,
                collector_rank_desc=ranks[candidate.candidate_index],
                group_id=group_id,
                group_metrics=group_metrics,
            )
            candidate_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            prompt_rewards.append(candidate.mc_reward)
            prompt_values.append(candidate.collector_value)

        prompt_group_metrics.append(group_metrics)
        num_steps_collected += 1
        selected_candidate = candidates[selected_index]
        prefix_tokens = selected_candidate.candidate_prefix_tokens
        if selected_candidate.candidate_chunk_has_eos:
            committed_prefix_has_eos = True
            break

    prompt_summary = build_prompt_summary(
        prompt_id=int(example.example_id),
        num_steps_collected=num_steps_collected,
        group_metrics=prompt_group_metrics,
        final_committed_prefix_length=len(prefix_tokens),
        committed_prefix_has_eos=committed_prefix_has_eos,
    )
    return prompt_summary, prompt_group_metrics, prompt_rewards, prompt_values


def actor_generate_requests(
    actor_generator: ActorGenerator,
    requests: Sequence[ActorGenerationRequest],
    *,
    batch_size: int,
) -> list[GeneratedSegment]:
    if not requests:
        return []
    resolved_batch_size = max(1, int(batch_size))
    segments: list[GeneratedSegment] = []
    for start in range(0, len(requests), resolved_batch_size):
        segments.extend(actor_generator.generate_batch(requests[start : start + resolved_batch_size]))
    if len(segments) != len(requests):
        raise RuntimeError(f"Actor returned {len(segments)} generations for {len(requests)} requests.")
    return segments

@torch.inference_mode()
def score_candidate_prefixes(
    *,
    collector_critic,
    tokenizer,
    prompt_token_ids: Sequence[int],
    candidate_prefixes: Sequence[Sequence[int]],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    pad_token_id = int(tokenizer.pad_token_id)
    for start in range(0, len(candidate_prefixes), max(1, int(batch_size))):
        batch_prefixes = candidate_prefixes[start : start + max(1, int(batch_size))]
        sequences = [tuple(int(token_id) for token_id in (*prompt_token_ids, *prefix)) for prefix in batch_prefixes]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
        for row_index, sequence in enumerate(sequences):
            sequence_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
            input_ids[row_index, : len(sequence)] = sequence_tensor
            attention_mask[row_index, : len(sequence)] = 1
        batch_values = critic_sequence_last_values(collector_critic, input_ids, attention_mask=attention_mask)
        values.extend(float(value) for value in batch_values.detach().cpu().float().tolist())
    return values


def build_candidate_row(
    *,
    args: argparse.Namespace,
    example: ExampleRecord,
    search_step: int,
    candidate: CandidateRecord,
    selected_index: int,
    collector_rank_desc: int,
    group_id: str,
    group_metrics: dict[str, Any],
) -> dict[str, Any]:
    save_text = bool(args.save_full_text)
    save_tokens = bool(args.save_token_ids)
    save_completed = bool(args.save_completed_responses)
    row: dict[str, Any] = {
        "example_id": f"prompt_{int(example.example_id):06d}_step_{search_step:02d}_candidate_{candidate.candidate_index:02d}",
        "prompt_id": int(example.example_id),
        "search_step": int(search_step),
        "candidate_index": int(candidate.candidate_index),
        "prompt_text": example.prompt_text if save_text else None,
        "prefix_text_before_chunk": candidate.prefix_text_before_chunk if save_text else None,
        "chunk_text": candidate.chunk_text if save_text else None,
        "candidate_prefix_text": candidate.candidate_prefix_text if save_text else None,
        "completion_text": candidate.completion_text if save_text and save_completed else None,
        "full_response_text": candidate.full_response_text if save_text and save_completed else None,
        "prefix_token_ids_before_chunk": list(candidate.prefix_tokens_before_chunk) if save_tokens else None,
        "chunk_token_ids": list(candidate.chunk_tokens) if save_tokens else None,
        "candidate_prefix_token_ids": list(candidate.candidate_prefix_tokens) if save_tokens else None,
        "completion_token_ids": list(candidate.completion_tokens) if save_tokens and save_completed else None,
        "full_response_token_ids": list(candidate.full_response_tokens) if save_tokens and save_completed else None,
        "prefix_length_before_chunk": len(candidate.prefix_tokens_before_chunk),
        "chunk_length": len(candidate.chunk_tokens),
        "candidate_prefix_length": len(candidate.candidate_prefix_tokens),
        "completion_length": len(candidate.completion_tokens),
        "full_response_length": len(candidate.full_response_tokens),
        "candidate_chunk_has_eos": bool(candidate.candidate_chunk_has_eos),
        "completion_ended_with_eos": bool(candidate.completion_ended_with_eos),
        "completion_hit_max_length": bool(candidate.completion_hit_max_length),
        "collector_value": float(candidate.collector_value),
        "collector_rank_desc": int(collector_rank_desc),
        "selected_by_collector": bool(candidate.candidate_index == selected_index),
        "actor_chunk_logprob": _json_float_or_none(candidate.actor_chunk_logprob),
        "actor_chunk_avg_logprob": _json_float_or_none(candidate.actor_chunk_avg_logprob),
        "mc_reward": float(candidate.mc_reward),
        "task_score": float(candidate.mc_reward),
        "accuracy": float(candidate.mc_reward),
        "collector_selection_mode": args.collector_selection_mode,
        "collector_critic_checkpoint_dir": args.collector_critic_checkpoint_dir,
        "actor_checkpoint_dir": args.actor_checkpoint_dir,
        "candidate_group_id": group_id,
        "group_num_candidates": int(group_metrics["group_num_candidates"]),
        "group_num_successes": float(group_metrics["group_num_successes"]),
        "group_success_rate": float(group_metrics["group_success_rate"]),
        "group_has_mixed_rewards": bool(group_metrics["group_has_mixed_rewards"]),
        "group_oracle_reward": float(group_metrics["group_oracle_reward"]),
        "group_random_reward_mean": float(group_metrics["group_random_reward_mean"]),
        "group_collector_top1_reward": float(group_metrics["group_collector_top1_reward"]),
        "group_false_high_selected": bool(group_metrics["group_false_high_selected"]),
        "collector_pairwise_ranking_accuracy_in_group": _json_float_or_none(
            group_metrics["collector_pairwise_ranking_accuracy_in_group"]
        ),
        "seed": int(args.seed),
    }
    return row


def build_prompt_summary(
    *,
    prompt_id: int,
    num_steps_collected: int,
    group_metrics: Sequence[dict[str, Any]],
    final_committed_prefix_length: int,
    committed_prefix_has_eos: bool,
) -> dict[str, Any]:
    rankable = [group for group in group_metrics if group["collector_pairwise_ranking_accuracy_in_group"] is not None]
    return {
        "prompt_id": int(prompt_id),
        "num_search_steps_collected": int(num_steps_collected),
        "num_candidate_examples": int(sum(group["group_num_candidates"] for group in group_metrics)),
        "num_rankable_groups": int(len(rankable)),
        "mean_group_success_rate": _safe_mean(group["group_success_rate"] for group in group_metrics),
        "mean_group_oracle_reward": _safe_mean(group["group_oracle_reward"] for group in group_metrics),
        "mean_group_collector_top1_reward": _safe_mean(group["group_collector_top1_reward"] for group in group_metrics),
        "num_false_high_selected_groups": int(sum(bool(group["group_false_high_selected"]) for group in group_metrics)),
        "final_committed_prefix_length": int(final_committed_prefix_length),
        "committed_prefix_has_eos": bool(committed_prefix_has_eos),
    }


def compute_group_metrics(*, rewards: Sequence[float], values: Sequence[float]) -> dict[str, Any]:
    if len(rewards) != len(values) or not rewards:
        raise ValueError("Group rewards and values must have the same non-zero length.")
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    top1_index = int(np.argmax(values_arr))
    oracle_reward = float(np.max(rewards_arr))
    top1_reward = float(rewards_arr[top1_index])
    pairwise_acc = pairwise_ranking_accuracy(rewards_arr.tolist(), values_arr.tolist())
    return {
        "group_num_candidates": int(len(rewards)),
        "group_num_successes": float(np.sum(rewards_arr)),
        "group_success_rate": float(np.mean(rewards_arr)),
        "group_has_mixed_rewards": bool(len(set(float(reward) for reward in rewards)) > 1),
        "group_oracle_reward": oracle_reward,
        "group_random_reward_mean": float(np.mean(rewards_arr)),
        "group_collector_top1_reward": top1_reward,
        "group_false_high_selected": bool(top1_reward == 0.0 and oracle_reward == 1.0),
        "collector_pairwise_ranking_accuracy_in_group": pairwise_acc,
    }


def build_summary_metrics(
    *,
    args: argparse.Namespace,
    num_prompts: int,
    candidate_rewards: Sequence[float],
    candidate_values: Sequence[float],
    group_metrics: Sequence[dict[str, Any]],
    prompt_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    oracle_positive = [group for group in group_metrics if float(group["group_oracle_reward"]) == 1.0]
    rankable_groups = [group for group in group_metrics if group["collector_pairwise_ranking_accuracy_in_group"] is not None]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actor_checkpoint_dir": args.actor_checkpoint_dir,
        "collector_critic_checkpoint_dir": args.collector_critic_checkpoint_dir,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "num_prompts": int(num_prompts),
        "chunk_size": int(args.chunk_size),
        "num_chunk_candidates": int(args.num_chunk_candidates),
        "num_search_steps_per_prompt": int(args.num_search_steps_per_prompt),
        "collector_selection_mode": args.collector_selection_mode,
        "num_candidate_examples": int(len(candidate_rewards)),
        "num_candidate_groups": int(len(group_metrics)),
        "candidate_level": {
            "mean_mc_reward": _safe_mean(candidate_rewards),
            "mean_collector_value": _safe_mean(candidate_values),
            "pearson_collector_value_vs_reward": correlation(candidate_values, candidate_rewards, method="pearson"),
            "spearman_collector_value_vs_reward": correlation(candidate_values, candidate_rewards, method="spearman"),
        },
        "group_level": {
            "fraction_rankable_groups": float(len(rankable_groups) / len(group_metrics)) if group_metrics else None,
            "mean_group_success_rate": _safe_mean(group["group_success_rate"] for group in group_metrics),
            "mean_group_oracle_reward": _safe_mean(group["group_oracle_reward"] for group in group_metrics),
            "mean_group_random_reward": _safe_mean(group["group_random_reward_mean"] for group in group_metrics),
            "mean_group_collector_top1_reward": _safe_mean(
                group["group_collector_top1_reward"] for group in group_metrics
            ),
            "collector_recovery_rate": (
                _safe_mean(float(group["group_collector_top1_reward"]) == 1.0 for group in oracle_positive)
                if oracle_positive
                else None
            ),
            "false_high_selected_rate": (
                _safe_mean(bool(group["group_false_high_selected"]) for group in oracle_positive)
                if oracle_positive
                else None
            ),
            "collector_pairwise_ranking_accuracy": _safe_mean(
                group["collector_pairwise_ranking_accuracy_in_group"] for group in rankable_groups
            ),
        },
        "committed_path": {
            "mean_num_steps_collected_per_prompt": _safe_mean(
                summary["num_search_steps_collected"] for summary in prompt_summaries
            ),
            "mean_final_committed_prefix_length": _safe_mean(
                summary["final_committed_prefix_length"] for summary in prompt_summaries
            ),
            "fraction_prompts_stopped_by_eos": _safe_mean(
                bool(summary["committed_prefix_has_eos"]) for summary in prompt_summaries
            ),
        },
    }


def select_candidate_index(
    *,
    mode: str,
    values: Sequence[float],
    actor_logprobs: Sequence[float | None],
    epsilon: float,
    value_temperature: float,
    rng: random.Random,
) -> int:
    if not values:
        raise ValueError("Cannot select from empty candidates.")
    if mode == "argmax":
        return int(np.argmax(np.asarray(values, dtype=np.float64)))
    if mode == "epsilon_greedy":
        if rng.random() < float(epsilon):
            return rng.randrange(len(values))
        return int(np.argmax(np.asarray(values, dtype=np.float64)))
    if mode == "softmax_value":
        temperature = max(float(value_temperature), 1e-8)
        logits = np.asarray(values, dtype=np.float64) / temperature
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        return int(rng.choices(range(len(values)), weights=probs.tolist(), k=1)[0])
    if mode == "actor_logprob":
        resolved = [float("-inf") if logprob is None else float(logprob) for logprob in actor_logprobs]
        return int(np.argmax(np.asarray(resolved, dtype=np.float64)))
    if mode == "random":
        return rng.randrange(len(values))
    raise ValueError(f"Unsupported collector selection mode: {mode}")


def pairwise_ranking_accuracy(rewards: Sequence[float], values: Sequence[float]) -> float | None:
    correct = 0.0
    total = 0.0
    for i in range(len(rewards)):
        for j in range(i + 1, len(rewards)):
            if float(rewards[i]) == float(rewards[j]):
                continue
            total += 1.0
            reward_order = 1.0 if float(rewards[i]) > float(rewards[j]) else -1.0
            value_diff = float(values[i]) - float(values[j])
            if value_diff == 0.0:
                correct += 0.5
            elif (value_diff > 0.0 and reward_order > 0.0) or (value_diff < 0.0 and reward_order < 0.0):
                correct += 1.0
    if total == 0.0:
        return None
    return float(correct / total)


def descending_ranks(values: Sequence[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0 for _ in values]
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def correlation(x_values: Sequence[float], y_values: Sequence[float], *, method: str) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if method == "spearman":
        x = rankdata_average_ties(x)
        y = rankdata_average_ties(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def write_readme(output_dir: Path, *, args: argparse.Namespace, actor_hf_dir: Path, critic_hf_dir: Path, backend: str) -> None:
    lines = [
        "# Search-Induced Critic Data",
        "",
        "This directory was produced by `value_decoding/collect_search_induced_critic_data.py`.",
        "",
        "## Files",
        "",
        "- `config.json`: command-line configuration.",
        "- `search_induced_candidates.jsonl`: one row per sampled candidate chunk boundary.",
        "- `prompt_summaries.jsonl`: one row per prompt.",
        "- `summary_metrics.json`: aggregate collection and ranking diagnostics.",
        "",
        "## Run Summary",
        "",
        f"- Actor checkpoint: `{args.actor_checkpoint_dir}`",
        f"- Actor HF directory: `{actor_hf_dir}`",
        f"- Collector critic checkpoint: `{args.collector_critic_checkpoint_dir}`",
        f"- Collector critic HF directory: `{critic_hf_dir}`",
        f"- Dataset: `{args.dataset_path}`",
        f"- Generation backend: `{backend}`",
        f"- Selection mode: `{args.collector_selection_mode}`",
        f"- Chunk candidates per group: `{args.num_chunk_candidates}`",
        f"- Search steps per prompt: `{args.num_search_steps_per_prompt}`",
        "",
        "All candidate chunks are saved, including non-selected alternatives, so later critic training can use MC targets and ranking contrast.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_torch_generator(
    args: argparse.Namespace,
    actor_hf_dir: Path,
    tokenizer,
    dtype: torch.dtype,
    eos_token_ids: tuple[int, ...],
) -> TorchActorGenerator:
    actor_device = resolve_device(args.actor_device)
    actor_model = load_actor_model(
        actor_hf_dir,
        dtype=dtype,
        device=actor_device,
        trust_remote_code=bool(args.trust_remote_code),
    )
    return TorchActorGenerator(
        model=actor_model,
        tokenizer=tokenizer,
        device=actor_device,
        temperature=float(args.actor_temperature),
        top_p=float(args.actor_top_p),
        top_k=int(args.actor_top_k),
        eos_token_ids=eos_token_ids,
    )


def _selected_token_logprob(
    logits: torch.Tensor,
    *,
    token_id: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> float:
    if temperature <= 0.0:
        return 0.0
    scaled_logits = logits.float() / float(temperature)
    filtered_logits = _filter_logits_like_decoding(scaled_logits, top_k=top_k, top_p=top_p)
    log_probs = torch.log_softmax(filtered_logits, dim=-1)
    value = log_probs[0, int(token_id)]
    return float(value.detach().cpu()) if torch.isfinite(value) else float("-inf")


def _filter_logits_like_decoding(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0 and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(filtered, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        filtered = filtered.masked_fill(indices_to_remove, float("-inf"))
    return filtered


def _example_prompt_token_ids(example: ExampleRecord, tokenizer, *, max_prompt_length: int) -> tuple[int, ...]:
    if example.prompt_token_ids is not None:
        return tuple(int(token_id) for token_id in example.prompt_token_ids)
    tokenized = tokenizer(
        example.prompt_text,
        truncation=True,
        max_length=int(max_prompt_length),
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return tuple(int(token_id) for token_id in tokenized["input_ids"])


def _decode_token_ids(tokenizer, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=True)


def _vllm_output_token_ids(output: Any) -> tuple[int, ...]:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is None:
        token_ids = getattr(output, "output_token_ids", None)
    if token_ids is None:
        return ()
    return tuple(int(token_id) for token_id in token_ids)


def _vllm_finish_reason(output: Any) -> str | None:
    finish_reason = getattr(output, "finish_reason", None)
    return None if finish_reason is None else str(finish_reason)


def _vllm_output_sequence_logprob(output: Any, token_ids: Sequence[int]) -> float | None:
    cumulative = getattr(output, "cumulative_logprob", None)
    if cumulative is not None:
        try:
            return float(cumulative)
        except (TypeError, ValueError):
            pass
    logprobs = getattr(output, "logprobs", None)
    if not logprobs:
        return None
    total = 0.0
    count = 0
    for token_id, step_logprobs in zip(token_ids, logprobs):
        value = _extract_vllm_token_logprob(step_logprobs, int(token_id))
        if value is None:
            return None
        total += float(value)
        count += 1
    return total if count else 0.0


def _extract_vllm_token_logprob(step_logprobs: Any, token_id: int) -> float | None:
    if step_logprobs is None:
        return None
    if isinstance(step_logprobs, dict):
        item = step_logprobs.get(token_id)
        if item is None:
            item = step_logprobs.get(str(token_id))
        if item is None:
            return None
        if isinstance(item, (int, float)):
            return float(item)
        logprob = getattr(item, "logprob", None)
        return None if logprob is None else float(logprob)
    return None


def _resolve_vllm_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.lower()
    if normalized == "bf16":
        return "bfloat16"
    if normalized == "fp16":
        return "float16"
    if normalized == "fp32":
        return "float32"
    return dtype_name


def _resolve_generation_backend(requested: str) -> str:
    if requested == "torch":
        return "torch"
    if requested == "vllm":
        return "vllm"
    if LLM is not None and torch.cuda.is_available():
        return "vllm"
    return "torch"


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _none_or_str(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


def _validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "max_prompts",
        "chunk_size",
        "num_chunk_candidates",
        "num_search_steps_per_prompt",
        "completion_max_new_tokens",
        "max_prompt_length",
        "critic_batch_size",
        "actor_batch_size",
    )
    for field in positive_int_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field} must be positive, got {getattr(args, field)}")
    if args.debug_num_prompts is not None and int(args.debug_num_prompts) <= 0:
        raise ValueError("--debug_num_prompts must be positive when provided")
    if not (0.0 <= float(args.collector_epsilon) <= 1.0):
        raise ValueError("--collector_epsilon must be in [0, 1]")
    if float(args.collector_value_temperature) <= 0.0:
        raise ValueError("--collector_value_temperature must be positive")
    if float(args.actor_temperature) < 0.0:
        raise ValueError("--actor_temperature must be non-negative")
    if not (0.0 < float(args.actor_top_p) <= 1.0):
        raise ValueError("--actor_top_p must be in (0, 1]")
    if int(args.actor_top_k) < 0:
        raise ValueError("--actor_top_k must be non-negative")
    if args.generation_backend in {"auto", "vllm"} and not (0.0 < float(args.vllm_gpu_memory_utilization) <= 1.0):
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1]")


def _stable_seed(*parts: int) -> int:
    value = 0x9E3779B9
    for part in parts:
        value = (value ^ (int(part) + 0x9E3779B9 + ((value << 6) & 0xFFFFFFFF) + (value >> 2))) & 0xFFFFFFFF
    return int(value % (2**31 - 1))


def _safe_mean(values: Iterable[Any]) -> float | None:
    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        numbers.append(float(value))
    if not numbers:
        return None
    return float(np.mean(np.asarray(numbers, dtype=np.float64)))


def _json_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


if __name__ == "__main__":
    main()
