from __future__ import annotations

import argparse
from pathlib import Path

import torch

from value_decoding.checkpointing import (
    load_actor_model,
    load_critic_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_eos_token_ids,
)
from value_decoding.data import load_examples
from value_decoding.decoding import (
    ActorSamplingMode,
    DecodingMode,
    NormalizationType,
    RunSpec,
    decode_example,
)


def _assert_close(lhs: float, rhs: float, *, atol: float = 1e-5, rtol: float = 1e-5, message: str) -> None:
    if abs(lhs - rhs) <= atol + rtol * abs(rhs):
        return
    raise AssertionError(f"{message}: lhs={lhs}, rhs={rhs}, atol={atol}, rtol={rtol}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run invariance self-checks for value_decoding.")
    parser.add_argument("--actor_dir", type=str, required=True)
    parser.add_argument("--critic_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--actor_device", type=str, default=None)
    parser.add_argument("--critic_device", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_two_gpu", action="store_true")
    parser.add_argument("--check_scoring_math", action="store_true", default=True)
    return parser.parse_args()


def _decode_text(
    *,
    actor,
    critic,
    tokenizer,
    example,
    run_spec: RunSpec,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    actor_device: torch.device,
    critic_device: torch.device,
    seed: int,
) -> str:
    result = decode_example(
        actor=actor,
        critic=critic,
        tokenizer=tokenizer,
        example=example,
        run_spec=run_spec,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        actor_device=actor_device,
        critic_device=critic_device,
        seed=seed,
    )
    return str(result.example_result["generated_response"])


def main() -> int:
    args = parse_args()
    actor_dir = Path(args.actor_dir).resolve()
    critic_dir = Path(args.critic_dir).resolve()

    actor_device = resolve_device(args.actor_device)
    critic_device = resolve_device(args.critic_device) if args.critic_device else actor_device
    dtype = resolve_dtype(args.dtype)

    tokenizer = load_tokenizer(actor_dir)
    examples = load_examples(
        args.dataset_path,
        tokenizer=tokenizer,
        max_examples=args.max_examples,
        pretokenize_max_length=args.max_prompt_length,
    )
    if not examples:
        raise ValueError("No examples loaded for self-check.")

    actor = load_actor_model(actor_dir, dtype=dtype, device=actor_device)
    critic = load_critic_model(critic_dir, dtype=dtype, device=critic_device)
    eos_token_ids = resolve_eos_token_ids(actor_dir, tokenizer)

    example = examples[0]
    baseline = _decode_text(
        actor=actor,
        critic=critic,
        tokenizer=tokenizer,
        example=example,
        run_spec=RunSpec(
            config_id="baseline",
            mode=DecodingMode.ACTOR_ONLY.value,
            actor_sampling_mode=ActorSamplingMode.GREEDY.value,
        ),
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_token_ids,
        actor_device=actor_device,
        critic_device=critic_device,
        seed=args.seed,
    )
    print(f"[baseline] {baseline!r}")

    invariants = {
        "critic_only_top_k_1": RunSpec(
            config_id="critic_only_top_k_1",
            mode=DecodingMode.CRITIC_ONLY_RERANK.value,
            candidate_builder="top_k",
            candidate_size=1,
        ),
        "actor_critic_beta_0": RunSpec(
            config_id="actor_critic_beta_0",
            mode=DecodingMode.ACTOR_CRITIC_RERANK.value,
            candidate_builder="top_k",
            candidate_size=4,
            beta=0.0,
            normalization=NormalizationType.ZSCORE.value,
        ),
        "actor_critic_top_k_1": RunSpec(
            config_id="actor_critic_top_k_1",
            mode=DecodingMode.ACTOR_CRITIC_RERANK.value,
            candidate_builder="top_k",
            candidate_size=1,
            beta=1.0,
            normalization=NormalizationType.ZSCORE.value,
        ),
    }

    for name, spec in invariants.items():
        decoded = _decode_text(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            run_spec=spec,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=args.seed,
        )
        print(f"[{name}] {decoded!r}")
        if decoded != baseline:
            raise AssertionError(f"{name} did not match baseline greedy decode.")

    if args.check_scoring_math:
        critic_debug = decode_example(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            run_spec=RunSpec(
                config_id="critic_debug",
                mode=DecodingMode.CRITIC_ONLY_RERANK.value,
                candidate_builder="top_k",
                candidate_size=4,
            ),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=min(args.max_new_tokens, 4),
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=args.seed,
            debug_full_candidates=True,
        )
        for step in critic_debug.step_results:
            scores = step["candidate_scores"]
            values = step["candidate_critic_values"]
            selected_rank = step["selected_token_actor_rank_in_candidates"]
            if selected_rank is None:
                raise AssertionError("critic_only debug step unexpectedly had no selected rank.")
            if scores != values:
                raise AssertionError("critic_only scores should exactly equal candidate critic values.")
            expected_rank = max(range(len(scores)), key=lambda idx: scores[idx])
            if expected_rank != selected_rank:
                raise AssertionError(
                    f"critic_only selected rank mismatch: expected {expected_rank}, got {selected_rank}"
                )

        rerank_debug = decode_example(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            run_spec=RunSpec(
                config_id="rerank_debug",
                mode=DecodingMode.ACTOR_CRITIC_RERANK.value,
                candidate_builder="top_k",
                candidate_size=4,
                beta=1.0,
                normalization=NormalizationType.ZSCORE.value,
            ),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=min(args.max_new_tokens, 4),
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=args.seed,
            debug_full_candidates=True,
        )
        for step in rerank_debug.step_results:
            scores = step["candidate_scores"]
            logps = step["candidate_actor_logprobs"]
            norm_values = step["candidate_normalized_values"]
            selected_rank = step["selected_token_actor_rank_in_candidates"]
            if selected_rank is None:
                raise AssertionError("rerank debug step unexpectedly had no selected rank.")
            for idx, score in enumerate(scores):
                expected = logps[idx] + norm_values[idx]
                _assert_close(
                    score,
                    expected,
                    atol=1e-5,
                    rtol=1e-5,
                    message=f"rerank score mismatch at step {step['step_index']} candidate {idx}",
                )
            expected_rank = max(range(len(scores)), key=lambda idx: scores[idx])
            if expected_rank != selected_rank:
                raise AssertionError(
                    f"rerank selected rank mismatch: expected {expected_rank}, got {selected_rank}"
                )

        soft_zero = _decode_text(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            run_spec=RunSpec(
                config_id="soft_zero",
                mode=DecodingMode.ACTOR_CRITIC_SOFT_RERANK.value,
                candidate_builder="top_k",
                candidate_size=4,
                beta=1.0,
                normalization=NormalizationType.ZSCORE.value,
                rank_temperature=0.0,
            ),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=args.seed,
        )
        hard = _decode_text(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            run_spec=RunSpec(
                config_id="hard",
                mode=DecodingMode.ACTOR_CRITIC_RERANK.value,
                candidate_builder="top_k",
                candidate_size=4,
                beta=1.0,
                normalization=NormalizationType.ZSCORE.value,
            ),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=args.seed,
        )
        if soft_zero != hard:
            raise AssertionError("Soft rerank with tau=0 did not collapse to hard rerank.")

        soft_debug = decode_example(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            run_spec=RunSpec(
                config_id="soft_debug",
                mode=DecodingMode.ACTOR_CRITIC_SOFT_RERANK.value,
                candidate_builder="top_k",
                candidate_size=4,
                beta=1.0,
                normalization=NormalizationType.ZSCORE.value,
                rank_temperature=1.0,
            ),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=min(args.max_new_tokens, 4),
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=args.seed,
            debug_full_candidates=True,
        )
        for step in soft_debug.step_results:
            probs = step["candidate_selection_probabilities"]
            if probs is None:
                raise AssertionError("soft rerank debug step missing selection probabilities.")
            _assert_close(
                sum(probs),
                1.0,
                atol=1e-6,
                rtol=1e-6,
                message=f"soft rerank probabilities did not sum to 1 at step {step['step_index']}",
            )
            if any(prob < 0.0 for prob in probs):
                raise AssertionError("soft rerank produced a negative selection probability.")

    if args.check_two_gpu:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("--check_two_gpu requested, but fewer than 2 CUDA devices are available.")

        second_critic_device = torch.device("cuda:1")
        critic_two_gpu = load_critic_model(critic_dir, dtype=dtype, device=second_critic_device)
        one_gpu = _decode_text(
            actor=actor,
            critic=critic,
            tokenizer=tokenizer,
            example=example,
            run_spec=RunSpec(
                config_id="one_gpu",
                mode=DecodingMode.ACTOR_CRITIC_RERANK.value,
                candidate_builder="top_k",
                candidate_size=4,
                beta=1.0,
                normalization=NormalizationType.ZSCORE.value,
            ),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=critic_device,
            seed=args.seed,
        )
        two_gpu = _decode_text(
            actor=actor,
            critic=critic_two_gpu,
            tokenizer=tokenizer,
            example=example,
            run_spec=RunSpec(
                config_id="two_gpu",
                mode=DecodingMode.ACTOR_CRITIC_RERANK.value,
                candidate_builder="top_k",
                candidate_size=4,
                beta=1.0,
                normalization=NormalizationType.ZSCORE.value,
            ),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            actor_device=actor_device,
            critic_device=second_critic_device,
            seed=args.seed,
        )
        print(f"[one_gpu] {one_gpu!r}")
        print(f"[two_gpu] {two_gpu!r}")
        if one_gpu != two_gpu:
            raise AssertionError("Two-GPU split result did not match one-GPU result.")

    print("[ok] All self-checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
