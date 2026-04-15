from __future__ import annotations

from argparse import Namespace

import pytest
import torch

import value_decoding.chunk_guidance_eval as chunk_guidance_eval_module
from value_decoding.chunk_guidance_eval import ChunkCandidate
from value_decoding.chunk_guidance_eval import (
    ChunkRunSpec,
    build_run_specs,
    build_reducer_comparisons,
    parse_value_reducer,
    reduce_chunk_values,
    reduce_end,
    reduce_mean,
    reduce_tail_exp,
    reduce_tail_mean,
    run_chunk_guided_response,
    sample_actor_chunk,
)
from value_decoding.data import ExampleRecord


def _build_args(**overrides) -> Namespace:
    args = Namespace(
        only_critic_only=True,
        skip_actor_only_baselines=False,
        include_critic_only=True,
        actor_sampling_mode="sample",
        actor_temperature=1.0,
        actor_top_p=1.0,
        actor_top_k=0,
        chunk_sizes=[32],
        num_chunk_candidates_values=[8],
        betas=[0.0],
        value_reducers=["end"],
        comparison_value_reducer=None,
        comparison_tail_h=None,
        comparison_tail_exp_alpha=None,
        include_uncertainty_only=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_parse_value_reducer_supports_tail_exp_alias_and_canonicalizes_alpha() -> None:
    reducer = parse_value_reducer("tailexp_h8_a0.85")

    assert reducer.kind == "tail_exp"
    assert reducer.tail_length == 8
    assert reducer.alpha == pytest.approx(0.85)
    assert reducer.canonical_name == "tail_exp_h8_a0p85"
    assert reducer.method_suffix == "tailexp_h8_a0p85"
    assert reducer.selected_metric_key == "selected_chunk_exp_tail_value__h8__a0p85"
    assert reducer.aggregate_metric_key == "mean_selected_chunk_exp_tail_value__h8__a0p85"


def test_tail_reducers_handle_short_chunks_with_min_tail_length() -> None:
    chunk_values = [1.0, 2.0, 3.0]

    assert reduce_end(chunk_values) == 3.0
    assert reduce_mean(chunk_values) == pytest.approx(2.0)
    assert reduce_tail_mean(chunk_values, 8) == pytest.approx(2.0)
    assert reduce_tail_exp(chunk_values, 8, 0.5) == pytest.approx(2.4285714285714284)


def test_reduce_tail_exp_gives_highest_weight_to_last_token() -> None:
    chunk_values = [1.0, 2.0, 4.0]

    expected = (1.0 * 0.25 + 2.0 * 0.5 + 4.0 * 1.0) / 1.75

    assert reduce_tail_exp(chunk_values, 3, 0.5) == pytest.approx(expected)
    assert reduce_chunk_values(chunk_values, "tail_exp_h3_a0p5") == pytest.approx(expected)


def test_build_run_specs_uses_expected_tail_reducer_naming_for_critic_only() -> None:
    specs = build_run_specs(
        _build_args(
            value_reducers=["end", "tail_mean_h4", "tail_exp_h8_a0p85"],
        )
    )

    assert [spec.config_id for spec in specs] == [
        "chunk_rerank_critic_only_endvalue__m32__k8",
        "chunk_rerank_critic_only_tailmean_h4__m32__k8",
        "chunk_rerank_critic_only_tailexp_h8_a0p85__m32__k8",
    ]
    assert [spec.value_reducer for spec in specs] == [
        "end",
        "tail_mean_h4",
        "tail_exp_h8_a0p85",
    ]
    assert [spec.comparison_value_reducer for spec in specs] == [
        None,
        "tail_mean_h4",
        "tail_mean_h8",
    ]


def test_build_run_specs_can_auto_resolve_same_h_tail_exp_comparison() -> None:
    specs = build_run_specs(
        _build_args(
            value_reducers=["tail_mean_h8", "tail_exp_h8_a0p85"],
            comparison_tail_exp_alpha=0.7,
        )
    )

    assert [spec.comparison_value_reducer for spec in specs] == [
        "tail_exp_h8_a0p7",
        "tail_exp_h8_a0p7",
    ]


def test_build_run_specs_can_include_uncertainty_selector() -> None:
    specs = build_run_specs(
        _build_args(
            only_critic_only=False,
            include_critic_only=True,
            include_uncertainty_only=True,
            betas=[0.0],
            value_reducers=["end"],
        )
    )

    assert [spec.method_name for spec in specs] == [
        "actor_only_sample",
        "chunk_rerank_actor_only",
        "chunk_rerank_uncertainty_meanentropy",
        "chunk_rerank_critic_only_endvalue",
    ]
    assert [spec.score_mode for spec in specs] == [
        "actor_only_sample",
        "actor_logprob_only",
        "uncertainty_meanentropy",
        "critic_only",
    ]


def test_parse_value_reducer_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        parse_value_reducer("tail_exp_h8_a1p0")


def test_build_reducer_comparisons_adds_end_and_same_h_tailmean_baselines() -> None:
    end_spec = ChunkRunSpec(
        config_id="chunk_rerank_critic_only_endvalue__m32__k8",
        method_name="chunk_rerank_critic_only_endvalue",
        score_mode="critic_only",
        chunk_size=32,
        num_chunk_candidates=8,
        value_reducer="end",
    )
    tail_mean_spec = ChunkRunSpec(
        config_id="chunk_rerank_critic_only_tailmean_h8__m32__k8",
        method_name="chunk_rerank_critic_only_tailmean_h8",
        score_mode="critic_only",
        chunk_size=32,
        num_chunk_candidates=8,
        value_reducer="tail_mean_h8",
        comparison_value_reducer="tail_mean_h8",
    )
    tail_exp_spec = ChunkRunSpec(
        config_id="chunk_rerank_critic_only_tailexp_h8_a0p85__m32__k8",
        method_name="chunk_rerank_critic_only_tailexp_h8_a0p85",
        score_mode="critic_only",
        chunk_size=32,
        num_chunk_candidates=8,
        value_reducer="tail_exp_h8_a0p85",
        comparison_value_reducer="tail_mean_h8",
    )

    def _result(
        example_id: int,
        *,
        task_score: float,
        response_length: int,
        end_value: float,
        reducer_value: float,
        tail_mean_h8: float,
        diff_from_end: float,
        diff_from_comparison: float | None,
    ) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "example_id": example_id,
            "task_score": task_score,
            "response_length": response_length,
            "mean_selected_chunk_end_value": end_value,
            "mean_selected_chunk_reducer_value": reducer_value,
            "mean_selected_chunk_tail_mean_h8": tail_mean_h8,
            "fraction_chunk_decisions_different_from_endvalue_winner": diff_from_end,
        }
        if diff_from_comparison is not None:
            row["fraction_chunk_decisions_different_from_comparison_winner"] = diff_from_comparison
        return row

    example_results_by_config = {
        end_spec.config_id: [
            _result(0, task_score=0.0, response_length=10, end_value=0.2, reducer_value=0.2, tail_mean_h8=0.15, diff_from_end=0.0, diff_from_comparison=None),
            _result(1, task_score=1.0, response_length=12, end_value=0.4, reducer_value=0.4, tail_mean_h8=0.35, diff_from_end=0.0, diff_from_comparison=None),
        ],
        tail_mean_spec.config_id: [
            _result(0, task_score=1.0, response_length=11, end_value=0.3, reducer_value=0.25, tail_mean_h8=0.25, diff_from_end=0.4, diff_from_comparison=0.0),
            _result(1, task_score=1.0, response_length=13, end_value=0.45, reducer_value=0.42, tail_mean_h8=0.42, diff_from_end=0.2, diff_from_comparison=0.0),
        ],
        tail_exp_spec.config_id: [
            _result(0, task_score=1.0, response_length=11, end_value=0.35, reducer_value=0.3, tail_mean_h8=0.27, diff_from_end=0.6, diff_from_comparison=0.3),
            _result(1, task_score=1.0, response_length=14, end_value=0.5, reducer_value=0.47, tail_mean_h8=0.44, diff_from_end=0.3, diff_from_comparison=0.1),
        ],
    }

    comparisons = build_reducer_comparisons(
        run_specs=[end_spec, tail_mean_spec, tail_exp_spec],
        example_results_by_config=example_results_by_config,
        bootstrap_seed=123,
        bootstrap_samples=50,
    )

    assert len(comparisons) == 3

    tail_exp_vs_end = next(
        comparison
        for comparison in comparisons
        if comparison["target_config_id"] == tail_exp_spec.config_id
        and comparison["baseline_config_id"] == end_spec.config_id
    )
    assert tail_exp_vs_end["comparison_type"] == "vs_endvalue_baseline"
    assert tail_exp_vs_end["delta_mean_accuracy"] == pytest.approx(0.5)
    assert tail_exp_vs_end["delta_mean_selected_chunk_end_value"] == pytest.approx(0.125)
    assert tail_exp_vs_end["delta_mean_selected_chunk_tail_mean_h8"] == pytest.approx(0.105)

    tail_exp_vs_tail_mean = next(
        comparison
        for comparison in comparisons
        if comparison["target_config_id"] == tail_exp_spec.config_id
        and comparison["baseline_config_id"] == tail_mean_spec.config_id
    )
    assert tail_exp_vs_tail_mean["comparison_type"] == "vs_tailmean_same_h_baseline"
    assert tail_exp_vs_tail_mean["delta_mean_selected_chunk_reducer_value"] == pytest.approx(0.05)
    assert tail_exp_vs_tail_mean["shared_comparison_value_reducer"] == "tail_mean_h8"
    assert tail_exp_vs_tail_mean["delta_mean_fraction_chunk_decisions_different_from_comparison_winner"] == pytest.approx(0.2)


def test_sample_actor_chunk_tracks_mean_entropy_from_raw_actor_distribution() -> None:
    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            return ",".join(str(token_id) for token_id in token_ids)

    class DummyActorOutput:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits
            self.past_key_values = None

    class DummyActor:
        def __init__(self) -> None:
            self._logits_by_sequence = {
                (101,): torch.tensor([0.0, 0.0], dtype=torch.float32),
                (101, 0): torch.tensor([2.0, 0.0], dtype=torch.float32),
                (101, 0, 0): torch.tensor([2.0, 0.0], dtype=torch.float32),
            }

        def __call__(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
            sequence = tuple(int(token_id) for token_id in input_ids[0].tolist())
            step_logits = self._logits_by_sequence[sequence]
            logits = torch.zeros((1, input_ids.shape[1], step_logits.shape[0]), dtype=torch.float32)
            logits[:, -1, :] = step_logits
            return DummyActorOutput(logits)

    class DummyCriticOutput:
        def __init__(self, values: torch.Tensor) -> None:
            self.logits = values

    class DummyCritic:
        def __call__(self, input_ids, attention_mask=None, use_cache=False):
            sequence = tuple(int(token_id) for token_id in input_ids[0].tolist())
            assert sequence == (101, 0, 0)
            values = torch.tensor([[[0.0], [0.25], [0.5]]], dtype=torch.float32)
            return DummyCriticOutput(values)

    candidate = sample_actor_chunk(
        actor=DummyActor(),
        critic=DummyCritic(),
        tokenizer=DummyTokenizer(),
        prefix_ids=torch.tensor([[101]], dtype=torch.long),
        actor_device=torch.device("cpu"),
        critic_device=torch.device("cpu"),
        max_chunk_len=2,
        sampling_mode="greedy",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=123,
        eos_token_ids=(999,),
        use_actor_cache=False,
        candidate_index=0,
    )

    second_step_logits = torch.tensor([2.0, 0.0], dtype=torch.float32)
    second_step_probs = torch.softmax(second_step_logits, dim=-1)
    expected_entropies = [
        float(torch.log(torch.tensor(2.0)).item()),
        float((-(second_step_probs * torch.log(second_step_probs))).sum().item()),
    ]

    assert candidate.chunk_token_ids == (0, 0)
    assert candidate.token_entropies == pytest.approx(expected_entropies)
    assert candidate.chunk_uncertainty == pytest.approx(sum(expected_entropies) / 2.0)
    assert candidate.token_logprobs == pytest.approx(
        [
            float(torch.log(torch.tensor(0.5)).item()),
            float(torch.log(second_step_probs[0]).item()),
        ]
    )
    assert candidate.chunk_logprob == pytest.approx(sum(candidate.token_logprobs))
    assert candidate.chunk_values == pytest.approx((0.25, 0.5))
    assert candidate.end_value == pytest.approx(0.5)
    assert candidate.mean_value == pytest.approx(0.375)


def test_sample_actor_chunk_can_skip_critic_values() -> None:
    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            return ",".join(str(token_id) for token_id in token_ids)

    class DummyActorOutput:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits
            self.past_key_values = None

    class DummyActor:
        def __init__(self) -> None:
            self._logits_by_sequence = {
                (101,): torch.tensor([0.0, 0.0], dtype=torch.float32),
                (101, 0): torch.tensor([2.0, 0.0], dtype=torch.float32),
                (101, 0, 0): torch.tensor([2.0, 0.0], dtype=torch.float32),
            }

        def __call__(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
            sequence = tuple(int(token_id) for token_id in input_ids[0].tolist())
            step_logits = self._logits_by_sequence[sequence]
            logits = torch.zeros((1, input_ids.shape[1], step_logits.shape[0]), dtype=torch.float32)
            logits[:, -1, :] = step_logits
            return DummyActorOutput(logits)

    candidate = sample_actor_chunk(
        actor=DummyActor(),
        critic=None,
        tokenizer=DummyTokenizer(),
        prefix_ids=torch.tensor([[101]], dtype=torch.long),
        actor_device=torch.device("cpu"),
        critic_device=None,
        max_chunk_len=2,
        sampling_mode="greedy",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=123,
        eos_token_ids=(999,),
        use_actor_cache=False,
        candidate_index=0,
    )

    assert candidate.chunk_token_ids == (0, 0)
    assert candidate.chunk_values == ()
    assert candidate.end_value is None
    assert candidate.mean_value is None
    assert candidate.chunk_uncertainty > 0.0


def test_run_chunk_guided_response_uses_tail_exp_and_logs_endvalue_disagreement(monkeypatch) -> None:
    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            return ",".join(str(token_id) for token_id in token_ids)

    candidates = [
        ChunkCandidate(
            candidate_index=0,
            chunk_token_ids=(11, 12, 13),
            chunk_text="11,12,13",
            chunk_length=3,
            chunk_logprob=-0.1,
            chunk_values=(0.1, 0.1, 1.0),
            end_value=1.0,
            mean_value=(0.1 + 0.1 + 1.0) / 3.0,
            contains_eos=True,
        ),
        ChunkCandidate(
            candidate_index=1,
            chunk_token_ids=(21, 22, 23),
            chunk_text="21,22,23",
            chunk_length=3,
            chunk_logprob=-0.2,
            chunk_values=(0.9, 0.9, 0.8),
            end_value=0.8,
            mean_value=(0.9 + 0.9 + 0.8) / 3.0,
            contains_eos=True,
        ),
    ]

    def _fake_sample_actor_chunk(**kwargs):
        return candidates[int(kwargs["candidate_index"])]

    monkeypatch.setattr(chunk_guidance_eval_module, "sample_actor_chunk", _fake_sample_actor_chunk)
    monkeypatch.setattr(chunk_guidance_eval_module, "score_response", lambda example, response_text: 1.0)

    spec = ChunkRunSpec(
        config_id="chunk_rerank_critic_only_tailexp_h3_a0p5__m3__k2",
        method_name="chunk_rerank_critic_only_tailexp_h3_a0p5",
        score_mode="critic_only",
        chunk_size=3,
        num_chunk_candidates=2,
        value_reducer="tail_exp_h3_a0p5",
        comparison_value_reducer="tail_mean_h3",
        actor_sampling_mode="sample",
    )
    example = ExampleRecord(
        example_id=0,
        prompt_text="prompt",
        data_source="math",
        ground_truth="gt",
    )

    artifacts = run_chunk_guided_response(
        actor=object(),
        critic=object(),
        tokenizer=DummyTokenizer(),
        example=example,
        prompt_ids=torch.tensor([[101, 102]], dtype=torch.long),
        spec=spec,
        actor_device=torch.device("cpu"),
        critic_device=torch.device("cpu"),
        max_new_tokens=3,
        eos_token_ids=(999,),
        normalization_eps=1e-6,
        seed=123,
        use_actor_cache=False,
        debug_full_chunk_candidates=True,
    )

    expected_candidate0 = (0.1 * 0.25 + 0.1 * 0.5 + 1.0 * 1.0) / 1.75
    expected_candidate1 = (0.9 * 0.25 + 0.9 * 0.5 + 0.8 * 1.0) / 1.75

    assert len(artifacts.chunk_decision_results) == 1
    decision = artifacts.chunk_decision_results[0]
    assert decision["candidate_chunk_reducer_values"] == pytest.approx([expected_candidate0, expected_candidate1])
    assert decision["candidate_chunk_tail_mean_h3_values"] == pytest.approx([0.4, (0.9 + 0.9 + 0.8) / 3.0])
    assert decision["candidate_normalized_chunk_reducer_values"] == pytest.approx([-1.0, 1.0], abs=1e-5)
    assert decision["endvalue_chunk_winner_index"] == 0
    assert decision["comparison_value_reducer"] == "tail_mean_h3"
    assert decision["comparison_chunk_winner_index"] == 1
    assert decision["tail_mean_h3_chunk_winner_index"] == 1
    assert decision["selected_chunk_index"] == 1
    assert decision["selected_differs_from_endvalue_winner"] is True
    assert decision["selected_differs_from_comparison_winner"] is False
    assert decision["selected_differs_from_tail_mean_h3_winner"] is False
    assert decision["selected_chunk_reducer_value"] == pytest.approx(expected_candidate1)
    assert decision["selected_chunk_tail_mean_h2"] == pytest.approx(0.85)
    assert decision["selected_chunk_tail_mean_h4"] == pytest.approx((0.9 + 0.9 + 0.8) / 3.0)
    half_gap = (expected_candidate1 - expected_candidate0) / 2.0
    expected_margin = 2.0 * half_gap / (half_gap + 1e-6)
    assert decision["selected_chunk_score_margin"] == pytest.approx(expected_margin)

    example_result = artifacts.example_result
    assert example_result["response_length"] == 3
    assert example_result["mean_selected_chunk_end_value"] == pytest.approx(0.8)
    assert example_result["mean_selected_chunk_reducer_value"] == pytest.approx(expected_candidate1)
    assert example_result["mean_selected_chunk_tail_mean_h2"] == pytest.approx(0.85)
    assert example_result["fraction_chunk_decisions_different_from_endvalue_winner"] == pytest.approx(1.0)
    assert example_result["fraction_chunk_decisions_different_from_comparison_winner"] == pytest.approx(0.0)
    assert example_result["fraction_chunk_decisions_different_from_tail_mean_h3_winner"] == pytest.approx(0.0)


def test_run_chunk_guided_response_selects_min_mean_entropy_and_logs_uncertainty_disagreement(monkeypatch) -> None:
    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            return ",".join(str(token_id) for token_id in token_ids)

    candidates = [
        ChunkCandidate(
            candidate_index=0,
            chunk_token_ids=(11, 12, 13),
            chunk_text="11,12,13",
            chunk_length=3,
            chunk_logprob=-0.5,
            chunk_values=(0.2, 0.3, 0.4),
            end_value=0.4,
            mean_value=0.3,
            contains_eos=True,
            token_logprobs=(-0.1, -0.2, -0.2),
            token_entropies=(0.1, 0.2, 0.3),
            chunk_uncertainty=0.2,
        ),
        ChunkCandidate(
            candidate_index=1,
            chunk_token_ids=(21, 22, 23),
            chunk_text="21,22,23",
            chunk_length=3,
            chunk_logprob=-0.1,
            chunk_values=(0.7, 0.8, 0.9),
            end_value=0.9,
            mean_value=0.8,
            contains_eos=True,
            token_logprobs=(-0.03, -0.03, -0.04),
            token_entropies=(0.9, 1.0, 1.1),
            chunk_uncertainty=1.0,
        ),
    ]

    def _fake_sample_actor_chunk(**kwargs):
        return candidates[int(kwargs["candidate_index"])]

    monkeypatch.setattr(chunk_guidance_eval_module, "sample_actor_chunk", _fake_sample_actor_chunk)
    monkeypatch.setattr(chunk_guidance_eval_module, "score_response", lambda example, response_text: 1.0)

    spec = ChunkRunSpec(
        config_id="chunk_rerank_uncertainty_meanentropy__m3__k2",
        method_name="chunk_rerank_uncertainty_meanentropy",
        score_mode="uncertainty_meanentropy",
        chunk_size=3,
        num_chunk_candidates=2,
        actor_sampling_mode="sample",
    )
    example = ExampleRecord(
        example_id=0,
        prompt_text="prompt",
        data_source="math",
        ground_truth="gt",
    )

    artifacts = run_chunk_guided_response(
        actor=object(),
        critic=object(),
        tokenizer=DummyTokenizer(),
        example=example,
        prompt_ids=torch.tensor([[101, 102]], dtype=torch.long),
        spec=spec,
        actor_device=torch.device("cpu"),
        critic_device=torch.device("cpu"),
        max_new_tokens=3,
        eos_token_ids=(999,),
        normalization_eps=1e-6,
        seed=123,
        use_actor_cache=False,
        debug_full_chunk_candidates=True,
    )

    assert len(artifacts.chunk_decision_results) == 1
    decision = artifacts.chunk_decision_results[0]
    assert decision["candidate_chunk_uncertainties"] == pytest.approx([0.2, 1.0])
    assert decision["candidate_normalized_chunk_uncertainties"] == pytest.approx([-1.0, 1.0], abs=1e-5)
    assert decision["uncertainty_chunk_winner_index"] == 0
    assert decision["endvalue_chunk_winner_index"] == 1
    assert decision["actor_only_chunk_winner_index"] == 1
    assert decision["selected_chunk_index"] == 0
    assert decision["selected_chunk_uncertainty"] == pytest.approx(0.2)
    assert decision["selected_chunk_entropy_horizon_mean"] == pytest.approx(0.2)
    assert decision["selected_differs_from_uncertainty_winner"] is False
    assert decision["selected_differs_from_endvalue_winner"] is True
    assert decision["selected_differs_from_actor_only_chunk_winner"] is True
    assert decision["selected_chunk_score_margin"] == pytest.approx(2.0, abs=1e-5)
    assert decision["candidate_chunk_token_entropies"] == [[0.1, 0.2, 0.3], [0.9, 1.0, 1.1]]

    example_result = artifacts.example_result
    assert example_result["response_length"] == 3
    assert example_result["mean_selected_chunk_end_value"] == pytest.approx(0.4)
    assert example_result["mean_selected_chunk_uncertainty"] == pytest.approx(0.2)
    assert example_result["mean_selected_chunk_entropy_horizon_mean"] == pytest.approx(0.2)
    assert example_result["fraction_chunk_decisions_different_from_uncertainty_winner"] == pytest.approx(0.0)
    assert example_result["fraction_chunk_decisions_different_from_endvalue_winner"] == pytest.approx(1.0)
    assert example_result["fraction_chunk_decisions_different_from_actor_only_chunk_winner"] == pytest.approx(1.0)


def test_run_chunk_guided_response_uncertainty_without_critic_skips_value_diagnostics(monkeypatch) -> None:
    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            return ",".join(str(token_id) for token_id in token_ids)

    candidates = [
        ChunkCandidate(
            candidate_index=0,
            chunk_token_ids=(11, 12, 13),
            chunk_text="11,12,13",
            chunk_length=3,
            chunk_logprob=-0.5,
            chunk_values=(),
            end_value=None,
            mean_value=None,
            contains_eos=True,
            token_logprobs=(-0.1, -0.2, -0.2),
            token_entropies=(0.1, 0.2, 0.3),
            chunk_uncertainty=0.2,
        ),
        ChunkCandidate(
            candidate_index=1,
            chunk_token_ids=(21, 22, 23),
            chunk_text="21,22,23",
            chunk_length=3,
            chunk_logprob=-0.1,
            chunk_values=(),
            end_value=None,
            mean_value=None,
            contains_eos=True,
            token_logprobs=(-0.03, -0.03, -0.04),
            token_entropies=(0.9, 1.0, 1.1),
            chunk_uncertainty=1.0,
        ),
    ]

    def _fake_sample_actor_chunk(**kwargs):
        assert kwargs["critic"] is None
        assert kwargs["critic_device"] is None
        return candidates[int(kwargs["candidate_index"])]

    monkeypatch.setattr(chunk_guidance_eval_module, "sample_actor_chunk", _fake_sample_actor_chunk)
    monkeypatch.setattr(chunk_guidance_eval_module, "score_response", lambda example, response_text: 1.0)

    spec = ChunkRunSpec(
        config_id="chunk_rerank_uncertainty_meanentropy__m3__k2",
        method_name="chunk_rerank_uncertainty_meanentropy",
        score_mode="uncertainty_meanentropy",
        chunk_size=3,
        num_chunk_candidates=2,
        actor_sampling_mode="sample",
    )
    example = ExampleRecord(
        example_id=0,
        prompt_text="prompt",
        data_source="math",
        ground_truth="gt",
    )

    artifacts = run_chunk_guided_response(
        actor=object(),
        critic=None,
        tokenizer=DummyTokenizer(),
        example=example,
        prompt_ids=torch.tensor([[101, 102]], dtype=torch.long),
        spec=spec,
        actor_device=torch.device("cpu"),
        critic_device=None,
        max_new_tokens=3,
        eos_token_ids=(999,),
        normalization_eps=1e-6,
        seed=123,
        use_actor_cache=False,
        debug_full_chunk_candidates=True,
    )

    decision = artifacts.chunk_decision_results[0]
    assert decision["candidate_chunk_end_values"] is None
    assert decision["candidate_chunk_mean_values"] is None
    assert decision["endvalue_chunk_winner_index"] is None
    assert decision["selected_chunk_end_value"] is None
    assert decision["selected_chunk_mean_value"] is None
    assert decision["selected_differs_from_endvalue_winner"] is None

    example_result = artifacts.example_result
    assert example_result["mean_selected_chunk_end_value"] is None
    assert example_result["mean_selected_chunk_mean_value"] is None
    assert example_result["fraction_chunk_decisions_different_from_endvalue_winner"] is None
