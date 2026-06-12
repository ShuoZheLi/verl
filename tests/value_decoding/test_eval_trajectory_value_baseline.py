import math

import numpy as np
import torch

from value_decoding.eval_trajectory_value_baseline import (
    _auroc_binary,
    _spearman,
    response_aligned_values_from_raw,
    select_trajectory_value,
    summarize,
)


def test_response_aligned_values_match_verl_shift():
    raw = torch.arange(10, dtype=torch.float32)
    aligned = response_aligned_values_from_raw(raw, prompt_length=4, response_length=3)
    assert aligned.tolist() == [3.0, 4.0, 5.0]


def test_pre_eos_uses_verl_aligned_eos_position_not_raw_post_eos():
    # prompt length 4, response [11, EOS]. Raw index 4 is the value before EOS;
    # raw index 5 would be the post-EOS/token-hidden value and must not be selected.
    raw = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.75, -9.0], dtype=torch.float32)
    readout = select_trajectory_value(
        raw_values=raw,
        prompt_length=4,
        response_token_ids=[11, 99],
        eos_token_ids={99},
        value_position="pre_eos",
    )
    assert math.isclose(readout.value, 0.75)
    assert readout.selected_response_index == 1
    assert readout.selected_full_index == 4
    assert readout.ended_with_eos is True


def test_pre_eos_without_eos_falls_back_to_last_response_aligned_value():
    raw = torch.arange(8, dtype=torch.float32)
    readout = select_trajectory_value(
        raw_values=raw,
        prompt_length=3,
        response_token_ids=[10, 11, 12],
        eos_token_ids={99},
        value_position="pre_eos",
    )
    assert readout.value == 4.0
    assert readout.selected_response_index == 2
    assert readout.selected_full_index == 4
    assert readout.ended_with_eos is False


def test_tail_mean_excludes_eos_when_possible():
    raw = torch.tensor([0.0, 1.0, 2.0, 10.0, 20.0, 99.0], dtype=torch.float32)
    readout = select_trajectory_value(
        raw_values=raw,
        prompt_length=2,
        response_token_ids=[5, 6, 7, 99],
        eos_token_ids={99},
        value_position="tail_mean_2",
    )
    # Response-aligned values are [1, 2, 10, 20]; tail before EOS is [2, 10].
    assert math.isclose(readout.value, 6.0)
    assert readout.selected_response_index == 2


def test_summary_constant_binary_mse_and_improvement():
    rows = [
        {"reward": 1.0, "critic_value": 0.9},
        {"reward": 0.0, "critic_value": 0.2},
        {"reward": 1.0, "critic_value": 0.8},
        {"reward": 0.0, "critic_value": 0.1},
    ]
    summary = summarize(rows, value_position="pre_eos")
    assert math.isclose(summary["avg_policy_accuracy"], 0.5)
    assert math.isclose(summary["constant_mse"], 0.25)
    assert math.isclose(summary["constant_mse_binary_formula"], 0.25)
    assert summary["critic_mse"] < summary["constant_mse"]
    assert summary["relative_mse_improvement"] > 0.0
    assert summary["value_gap_correct_minus_wrong"] > 0.0


def test_rank_metrics_handle_ties():
    x = np.asarray([0.2, 0.2, 0.8, 0.9])
    y = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert _spearman(x, y) > 0.0
    assert math.isclose(_auroc_binary(x, y), 1.0)


def test_shard_sequence_round_robin_partition():
    from value_decoding.eval_trajectory_value_baseline import shard_sequence

    assert shard_sequence(list(range(7)), num_shards=3, shard_id=0) == [0, 3, 6]
    assert shard_sequence(list(range(7)), num_shards=3, shard_id=1) == [1, 4]
    assert shard_sequence(list(range(7)), num_shards=3, shard_id=2) == [2, 5]


def test_aggregate_output_dirs_recomputes_global_rbar(tmp_path):
    from value_decoding.eval_trajectory_value_baseline import aggregate_output_dirs

    shard0 = tmp_path / "shard_0"
    shard1 = tmp_path / "shard_1"
    shard0.mkdir()
    shard1.mkdir()
    (shard0 / "trajectory_values.jsonl").write_text(
        '{"trajectory_id": 0, "prompt_id": 0, "reward": 1.0, "critic_value": 0.8}\n',
        encoding="utf-8",
    )
    (shard1 / "trajectory_values.jsonl").write_text(
        '{"trajectory_id": 0, "prompt_id": 1, "reward": 0.0, "critic_value": 0.2}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "merged"
    summary = aggregate_output_dirs([shard0, shard1], output_dir, value_position="pre_eos")
    assert math.isclose(summary["avg_policy_accuracy"], 0.5)
    assert math.isclose(summary["constant_mse"], 0.25)
    rows = (output_dir / "trajectory_values.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert '"r_bar": 0.5' in rows[0]
