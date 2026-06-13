import pytest
import torch

from value_decoding.chunk_guidance_eval import resolve_candidate_end_value


def test_as_is_uses_last_value_for_eos_chunk():
    chunk_values = torch.tensor([1.0, 2.0, 3.0])

    assert resolve_candidate_end_value(
        chunk_values=chunk_values,
        explicit_eos_at_end=True,
        eos_end_value_mode="as_is",
    ) == pytest.approx(3.0)


def test_ppo_pre_eos_uses_previous_chunk_value_for_multi_token_eos_chunk():
    chunk_values = torch.tensor([1.0, 2.0, 3.0])

    assert resolve_candidate_end_value(
        chunk_values=chunk_values,
        explicit_eos_at_end=True,
        eos_end_value_mode="ppo_pre_eos",
    ) == pytest.approx(2.0)


def test_ppo_pre_eos_uses_prefix_value_for_single_token_eos_chunk():
    chunk_values = torch.tensor([3.0])
    prefix_value = torch.tensor(0.5)

    assert resolve_candidate_end_value(
        chunk_values=chunk_values,
        explicit_eos_at_end=True,
        eos_end_value_mode="ppo_pre_eos",
        prefix_pre_eos_value=prefix_value,
    ) == pytest.approx(0.5)


def test_ppo_pre_eos_does_not_shift_non_eos_chunk():
    chunk_values = torch.tensor([1.0, 2.0, 3.0])

    assert resolve_candidate_end_value(
        chunk_values=chunk_values,
        explicit_eos_at_end=False,
        eos_end_value_mode="ppo_pre_eos",
    ) == pytest.approx(3.0)


def test_ppo_pre_eos_requires_prefix_value_for_single_token_eos_chunk():
    with pytest.raises(ValueError, match="prefix_pre_eos_value"):
        resolve_candidate_end_value(
            chunk_values=torch.tensor([3.0]),
            explicit_eos_at_end=True,
            eos_end_value_mode="ppo_pre_eos",
        )
