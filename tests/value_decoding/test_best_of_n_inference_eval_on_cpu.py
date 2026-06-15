from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from value_decoding.best_of_n_inference_eval import sample_vllm_actor_trajectory
from value_decoding.data import ExampleRecord


@dataclass
class _FakeLogprob:
    logprob: float


class _FakeOutput:
    def __init__(self) -> None:
        self.token_ids = (10, 11)
        self.text = "not a gsm answer"
        self.finish_reason = "length"
        self.cumulative_logprob = -1.0
        self.logprobs = [{10: _FakeLogprob(-0.25)}, {11: _FakeLogprob(-0.75)}]


class _FakeRequestOutput:
    def __init__(self) -> None:
        self.outputs = [_FakeOutput()]


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, *, prompts, sampling_params, use_tqdm):
        self.calls.append({"prompts": prompts, "sampling_params": sampling_params, "use_tqdm": use_tqdm})
        return [_FakeRequestOutput()]


class _FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeTokensPrompt:
    def __init__(self, *, prompt_token_ids) -> None:
        self.prompt_token_ids = prompt_token_ids


class _FakeTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        del skip_special_tokens
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


def test_sample_vllm_actor_trajectory_preserves_tokens_and_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    import value_decoding.best_of_n_inference_eval as module

    monkeypatch.setattr(module, "_require_vllm", lambda: (object, _FakeSamplingParams, _FakeTokensPrompt))
    monkeypatch.setattr(module, "score_response", lambda example, response_text: 0.0)
    llm = _FakeLLM()
    example = ExampleRecord(
        example_id=3,
        prompt_text="Question?",
        ground_truth="42",
        data_source="unit",
        prompt_token_ids=(1, 2, 3),
    )

    trajectory = sample_vllm_actor_trajectory(
        llm=llm,
        tokenizer=_FakeTokenizer(),
        example=example,
        prompt_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        sample_idx=5,
        seed=123,
        actor_sampling_mode="sample",
        actor_temperature=0.7,
        actor_top_p=0.9,
        actor_top_k=20,
        max_new_tokens=16,
        eos_token_ids=(99,),
    )

    assert trajectory.sample_idx == 5
    assert trajectory.sample_seed == 123
    assert trajectory.prompt_length == 3
    assert trajectory.full_sequence_token_ids == (1, 2, 3, 10, 11)
    assert trajectory.response_text == "not a gsm answer"
    assert trajectory.response_length == 2
    assert trajectory.eos_emitted is False
    assert trajectory.max_length_hit is False
    assert trajectory.actor_response_logprob == pytest.approx(-1.0)
    assert trajectory.actor_response_avg_logprob == pytest.approx(-0.5)

    call = llm.calls[0]
    assert call["prompts"][0].prompt_token_ids == [1, 2, 3]
    assert call["use_tqdm"] is False
    assert call["sampling_params"].kwargs["seed"] == 123
    assert call["sampling_params"].kwargs["logprobs"] == 1
    assert call["sampling_params"].kwargs["top_k"] == 20


def test_vllm_logprob_extraction_requires_sampled_token() -> None:
    import value_decoding.best_of_n_inference_eval as module

    with pytest.raises(RuntimeError, match="sampled token id 11"):
        module._vllm_token_logprob({10: _FakeLogprob(-0.25)}, 11)
