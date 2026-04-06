from __future__ import annotations

from pathlib import Path

import torch

from value_decoding.chunk_ranking_benchmark import (
    CriticSpec,
    ExampleRecord,
    PrefixState,
    SampledActorContinuation,
    _pairwise_ranking_stats,
    aggregate_metrics,
    build_prefix_summary,
    parse_worker_layouts,
    rebuild_prefix_rows_from_candidate_rows,
    select_prefix_states,
    validate_candidate_bank,
)


def _candidate_row(
    *,
    candidate_chunk_id: int,
    chunk_token_ids: tuple[int, ...],
    final_task_score: float,
    chunk_logprob: float,
    realized_chunk_length: int,
    completed_response_length: int,
    chunk_contains_eos: bool,
    old_end: float,
    old_mean: float,
    new_end: float,
    new_mean: float,
) -> dict[str, float | int | bool]:
    return {
        "candidate_chunk_id": candidate_chunk_id,
        "chunk_token_ids": list(chunk_token_ids),
        "final_task_score": final_task_score,
        "chunk_logprob": chunk_logprob,
        "realized_chunk_length": realized_chunk_length,
        "completed_response_length": completed_response_length,
        "chunk_contains_eos": chunk_contains_eos,
        "old_critic_end_value": old_end,
        "old_critic_mean_value": old_mean,
        "new_critic_end_value": new_end,
        "new_critic_mean_value": new_mean,
    }


def test_pairwise_ranking_tie_gets_half_credit() -> None:
    stats = _pairwise_ranking_stats([0.0, 1.0], [0.25, 0.25])
    assert stats["correct_pairs"] == 0.5
    assert stats["rankable_pairs"] == 1
    assert stats["accuracy"] == 0.5


def test_select_prefix_states_uses_bucket_midpoints() -> None:
    prompt_ids = torch.tensor([[101, 102]], dtype=torch.long)
    reference_rollout = SampledActorContinuation(
        prefix_length=2,
        continuation_token_ids=tuple(range(10, 18)),
        full_sequence_token_ids=tuple([101, 102, *range(10, 18)]),
        continuation_length=8,
        continuation_text="",
        eos_emitted=False,
        max_length_hit=False,
        sum_actor_logprob=0.0,
    )

    prefix_states = select_prefix_states(prompt_ids=prompt_ids, reference_rollout=reference_rollout)

    assert [state.bucket for state in prefix_states] == ["early", "middle", "late"]
    assert [state.step_index for state in prefix_states] == [0, 3, 6]
    assert [len(state.prefix_token_ids) for state in prefix_states] == [2, 5, 8]


def test_parse_worker_layouts_supports_two_critics() -> None:
    critic_specs = [
        CriticSpec(name="old", checkpoint_dir=Path("/tmp/old"), device="cuda:1"),
        CriticSpec(name="new", checkpoint_dir=Path("/tmp/new"), device="cuda:2"),
    ]

    layouts = parse_worker_layouts(
        ["cuda:0,cuda:1,cuda:2", "cuda:3,cuda:1,cuda:2"],
        critic_specs=critic_specs,
        actor_device=None,
        default_device=None,
    )

    assert layouts == [
        ("cuda:0", ("cuda:1", "cuda:2")),
        ("cuda:3", ("cuda:1", "cuda:2")),
    ]


def test_aggregate_metrics_matches_expected_synthetic_ranking() -> None:
    example = ExampleRecord(example_id=7, prompt_text="p", data_source="math", ground_truth="gt")
    prefix_early = PrefixState(
        prefix_id=0,
        bucket="early",
        step_index=0,
        prompt_length=2,
        prefix_token_ids=(11, 12),
        reference_response_length=12,
        reference_response_eos_emitted=False,
        reference_response_max_length_hit=False,
    )
    prefix_late = PrefixState(
        prefix_id=1,
        bucket="late",
        step_index=5,
        prompt_length=2,
        prefix_token_ids=(11, 12, 13, 14, 15, 16, 17),
        reference_response_length=12,
        reference_response_eos_emitted=False,
        reference_response_max_length_hit=False,
    )

    prefix0_candidates = [
        _candidate_row(
            candidate_chunk_id=0,
            chunk_token_ids=(31, 32),
            final_task_score=0.0,
            chunk_logprob=-0.1,
            realized_chunk_length=2,
            completed_response_length=10,
            chunk_contains_eos=False,
            old_end=0.2,
            old_mean=0.15,
            new_end=0.1,
            new_mean=0.05,
        ),
        _candidate_row(
            candidate_chunk_id=1,
            chunk_token_ids=(41, 42, 43),
            final_task_score=1.0,
            chunk_logprob=-0.5,
            realized_chunk_length=3,
            completed_response_length=11,
            chunk_contains_eos=False,
            old_end=0.9,
            old_mean=0.8,
            new_end=0.8,
            new_mean=0.7,
        ),
        _candidate_row(
            candidate_chunk_id=2,
            chunk_token_ids=(99,),
            final_task_score=0.0,
            chunk_logprob=-0.6,
            realized_chunk_length=1,
            completed_response_length=8,
            chunk_contains_eos=True,
            old_end=0.4,
            old_mean=0.35,
            new_end=0.3,
            new_mean=0.25,
        ),
    ]
    prefix1_candidates = [
        _candidate_row(
            candidate_chunk_id=0,
            chunk_token_ids=(51, 52),
            final_task_score=1.0,
            chunk_logprob=-0.1,
            realized_chunk_length=2,
            completed_response_length=9,
            chunk_contains_eos=False,
            old_end=0.7,
            old_mean=0.6,
            new_end=0.4,
            new_mean=0.35,
        ),
        _candidate_row(
            candidate_chunk_id=1,
            chunk_token_ids=(61, 62, 63, 64),
            final_task_score=0.0,
            chunk_logprob=-0.3,
            realized_chunk_length=4,
            completed_response_length=13,
            chunk_contains_eos=False,
            old_end=0.6,
            old_mean=0.55,
            new_end=0.9,
            new_mean=0.85,
        ),
        _candidate_row(
            candidate_chunk_id=2,
            chunk_token_ids=(98,),
            final_task_score=1.0,
            chunk_logprob=-0.2,
            realized_chunk_length=1,
            completed_response_length=7,
            chunk_contains_eos=True,
            old_end=0.8,
            old_mean=0.75,
            new_end=0.8,
            new_mean=0.7,
        ),
    ]

    prefix_rows = [
        build_prefix_summary(
            example=example,
            prefix_state=prefix_early,
            candidate_rows=prefix0_candidates,
            critic_names=["old", "new"],
            base_seed=123,
            reference_rollout_seed=999,
        ),
        build_prefix_summary(
            example=example,
            prefix_state=prefix_late,
            candidate_rows=prefix1_candidates,
            critic_names=["old", "new"],
            base_seed=123,
            reference_rollout_seed=999,
        ),
    ]

    candidate_rows = []
    for _prefix_row, prefix_state, candidates in zip(
        prefix_rows,
        [prefix_early, prefix_late],
        [prefix0_candidates, prefix1_candidates],
        strict=True,
    ):
        for row in candidates:
            candidate_rows.append(
                {
                    "example_id": example.example_id,
                    "prompt_id": example.example_id,
                    "prompt_text": example.prompt_text,
                    "prompt_token_ids": list(prefix_state.prefix_token_ids[: prefix_state.prompt_length]),
                    "prefix_id": prefix_state.prefix_id,
                    "prefix_bucket": prefix_state.bucket,
                    "prefix_step_index": prefix_state.step_index,
                    "prompt_length": prefix_state.prompt_length,
                    "prefix_sequence_length": len(prefix_state.prefix_token_ids),
                    "prefix_generated_length": prefix_state.step_index,
                    "prefix_token_ids": list(prefix_state.prefix_token_ids),
                    "reference_rollout_seed": 999,
                    "reference_response_length": prefix_state.reference_response_length,
                    "reference_response_eos_emitted": prefix_state.reference_response_eos_emitted,
                    "reference_response_max_length_hit": prefix_state.reference_response_max_length_hit,
                    "data_source": example.data_source,
                    "ground_truth": example.ground_truth,
                    "chunk_end_sequence_length": len(prefix_state.prefix_token_ids) + len(row["chunk_token_ids"]),
                    "chunk_end_sequence_token_ids": list(prefix_state.prefix_token_ids) + list(row["chunk_token_ids"]),
                    "completed_response_eos_emitted": False,
                    "completed_response_max_length_hit": False,
                    "completed_response_token_ids": list(range(row["completed_response_length"])),
                    **row,
                }
            )

    validate_candidate_bank(candidate_rows, require_trace_fields=True)
    rebuilt_prefix_rows = rebuild_prefix_rows_from_candidate_rows(
        candidate_rows=candidate_rows,
        base_seed=123,
        critic_names=["old", "new"],
    )
    assert rebuilt_prefix_rows == prefix_rows

    metrics = aggregate_metrics(
        candidate_rows=candidate_rows,
        prefix_rows=rebuilt_prefix_rows,
        critic_names=["old", "new"],
        bootstrap_samples=50,
        bootstrap_seed=1234,
    )

    old_metrics = metrics["methods"]["old"]
    new_metrics = metrics["methods"]["new"]
    actor_metrics = metrics["methods"]["actor_logprob"]
    oracle_metrics = metrics["methods"]["oracle_best_chunk"]

    assert old_metrics["weighted_pairwise_ranking_accuracy"] == 1.0
    assert new_metrics["weighted_pairwise_ranking_accuracy"] == 0.5
    assert actor_metrics["weighted_pairwise_ranking_accuracy"] == 0.75

    assert old_metrics["top1_selected_mean_task_score"] == 1.0
    assert new_metrics["top1_selected_mean_task_score"] == 0.5
    assert actor_metrics["top1_selected_mean_task_score"] == 0.5
    assert oracle_metrics["top1_selected_mean_task_score"] == 1.0

    assert old_metrics["chunk_success_recovery_rate"] == 1.0
    assert new_metrics["chunk_success_recovery_rate"] == 0.5
    assert actor_metrics["chunk_success_recovery_rate"] == 0.5

    assert metrics["comparisons"]["new_minus_old"]["weighted_pairwise_ranking_accuracy"] == -0.5
    assert metrics["comparisons"]["new_minus_old"]["top1_selected_mean_task_score"] == -0.5
    assert metrics["comparisons"]["new_minus_actor_logprob"]["weighted_pairwise_ranking_accuracy"] == -0.25
    assert metrics["comparisons"]["new_minus_actor_logprob"]["top1_selected_mean_task_score"] == 0.0

    assert metrics["by_prefix_bucket"]["early"]["old"]["top1_selected_mean_task_score"] == 1.0
    assert metrics["by_prefix_bucket"]["late"]["new"]["top1_selected_mean_task_score"] == 0.0
