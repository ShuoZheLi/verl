from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput

from value_decoding.checkpointing import PRMCriticAdapter, resolve_component_checkpoint_dir
from value_decoding.decoding import critic_sequence_last_values, critic_sequence_values


class _DummyPRMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.config = type("Config", (), {"use_return_dict": True})()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput:
        del attention_mask, use_cache, return_dict
        if input_ids is None:
            raise ValueError("input_ids are required")
        positive_logits = input_ids.to(dtype=torch.float32).cumsum(dim=1)
        logits = torch.stack(
            [torch.zeros_like(positive_logits), positive_logits],
            dim=-1,
        )
        return TokenClassifierOutput(logits=logits)


def _positive_prob(logit_value: int) -> float:
    logits = torch.tensor([0.0, float(logit_value)], dtype=torch.float32)
    return float(torch.softmax(logits, dim=0)[1].item())


def _dummy_exact_prefix_prob(prefix_tokens: list[int], separator_token_id: int) -> float:
    return _positive_prob(sum(prefix_tokens) + separator_token_id)


def test_resolve_component_checkpoint_dir_accepts_direct_hf_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForProcessRewardModel"]}), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"stub")

    resolved = resolve_component_checkpoint_dir(tmp_path, component="critic")

    assert resolved == tmp_path


def test_prm_critic_adapter_sequence_values_replace_last_token_with_step_end_score() -> None:
    separator_token_id = 7
    adapter = PRMCriticAdapter(
        _DummyPRMModel(),
        separator_token_id=separator_token_id,
        pad_token_id=0,
    )
    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )

    values = adapter.sequence_values(input_ids=input_ids, attention_mask=attention_mask)
    last_values = adapter.sequence_last_values(input_ids=input_ids, attention_mask=attention_mask)

    expected_values = torch.tensor(
        [
            [
                _dummy_exact_prefix_prob([1], separator_token_id),
                _dummy_exact_prefix_prob([1, 2], separator_token_id),
                _dummy_exact_prefix_prob([1, 2, 3], separator_token_id),
            ],
            [
                _dummy_exact_prefix_prob([4], separator_token_id),
                _dummy_exact_prefix_prob([4, 5], separator_token_id),
                0.0,
            ],
        ],
        dtype=torch.float32,
    )
    expected_last_values = torch.tensor(
        [
            _dummy_exact_prefix_prob([1, 2, 3], separator_token_id),
            _dummy_exact_prefix_prob([4, 5], separator_token_id),
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(last_values, expected_last_values)


def test_decoding_helpers_use_prm_adapter_custom_value_methods() -> None:
    separator_token_id = 7
    adapter = PRMCriticAdapter(
        _DummyPRMModel(),
        separator_token_id=separator_token_id,
        pad_token_id=0,
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    sequence_values = critic_sequence_values(adapter, input_ids)
    sequence_last_values = critic_sequence_last_values(adapter, input_ids)

    torch.testing.assert_close(
        sequence_values,
        torch.tensor(
            [[
                _dummy_exact_prefix_prob([1], separator_token_id),
                _dummy_exact_prefix_prob([1, 2], separator_token_id),
                _dummy_exact_prefix_prob([1, 2, 3], separator_token_id),
            ]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        sequence_last_values,
        torch.tensor([_dummy_exact_prefix_prob([1, 2, 3], separator_token_id)], dtype=torch.float32),
    )


def test_prm_critic_adapter_continuation_values_match_exact_prefix_scores() -> None:
    separator_token_id = 7
    adapter = PRMCriticAdapter(
        _DummyPRMModel(),
        separator_token_id=separator_token_id,
        pad_token_id=0,
    )
    prefix_ids = torch.tensor([[10, 20]], dtype=torch.long)
    continuation_ids = torch.tensor([3, 4, 5], dtype=torch.long)

    values = adapter.continuation_values(prefix_ids=prefix_ids, continuation_ids=continuation_ids)

    expected = torch.tensor(
        [
            _dummy_exact_prefix_prob([10, 20, 3], separator_token_id),
            _dummy_exact_prefix_prob([10, 20, 3, 4], separator_token_id),
            _dummy_exact_prefix_prob([10, 20, 3, 4, 5], separator_token_id),
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(values, expected)


def test_resolve_component_checkpoint_dir_prefers_component_subdir_when_present(tmp_path: Path) -> None:
    component_dir = tmp_path / "critic"
    component_dir.mkdir(parents=True)
    (component_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForProcessRewardModel"]}), encoding="utf-8")
    (component_dir / "model.safetensors").write_bytes(b"stub")
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnusedRoot"]}), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"root")

    resolved = resolve_component_checkpoint_dir(tmp_path, component="critic")

    assert resolved == component_dir


def test_prm_critic_adapter_rejects_empty_sequences() -> None:
    adapter = PRMCriticAdapter(
        _DummyPRMModel(),
        separator_token_id=7,
        pad_token_id=0,
    )
    input_ids = torch.tensor([[0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="at least one unmasked token"):
        adapter.sequence_last_values(input_ids=input_ids, attention_mask=attention_mask)
