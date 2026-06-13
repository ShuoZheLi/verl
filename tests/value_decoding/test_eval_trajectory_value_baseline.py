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


def test_vllm_output_token_ids_supports_known_attribute_names():
    from value_decoding.eval_trajectory_value_baseline import vllm_output_token_ids

    class OutputWithTokenIds:
        token_ids = (1, 2, 3)

    class OutputWithOutputTokenIds:
        token_ids = None
        output_token_ids = [4, 5]

    assert vllm_output_token_ids(OutputWithTokenIds()) == [1, 2, 3]
    assert vllm_output_token_ids(OutputWithOutputTokenIds()) == [4, 5]


def test_maybe_append_vllm_stop_eos_when_stop_token_omitted():
    from value_decoding.eval_trajectory_value_baseline import maybe_append_vllm_stop_eos

    class Tokenizer:
        eos_token_id = 99

    class Output:
        finish_reason = "stop"
        stop_reason = 99

    assert maybe_append_vllm_stop_eos([1, 2], Output(), Tokenizer()) == [1, 2, 99]
    assert maybe_append_vllm_stop_eos([1, 2, 99], Output(), Tokenizer()) == [1, 2, 99]


def test_critic_label_from_path_is_unique():
    from pathlib import Path
    from value_decoding.eval_trajectory_value_baseline import critic_label_from_path

    used = set()
    first = critic_label_from_path(Path('/tmp/run/global_step_100'), used)
    second = critic_label_from_path(Path('/tmp/run/global_step_100'), used)
    assert first == 'run__global_step_100'
    assert second == 'run__global_step_100_1'


def test_aggregate_preserves_single_critic_metadata(tmp_path):
    import json
    from value_decoding.eval_trajectory_value_baseline import aggregate_output_dirs

    shard = tmp_path / 'shard'
    run_dir = shard / 'critic_a' / 'pre_eos'
    run_dir.mkdir(parents=True)
    row = {
        'trajectory_id': 0,
        'prompt_id': 0,
        'reward': 1.0,
        'critic_value': 0.75,
        'critic_label': 'critic_a',
        'critic_checkpoint_dir': '/ckpt/a',
    }
    (run_dir / 'trajectory_values.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    out = tmp_path / 'agg'
    summary = aggregate_output_dirs([run_dir], out, value_position='pre_eos')
    assert summary['critic_label'] == 'critic_a'
    assert summary['critic_checkpoint_dir'] == '/ckpt/a'


def test_evaluate_critic_position_writes_correct_losses_and_summary(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import torch
    import value_decoding.eval_trajectory_value_baseline as ev

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            return {"input_ids": [10, 11, 12]}

    class Critic:
        pass

    def fake_sequence_values(critic, input_ids, attention_mask=None):
        # prompt length is 3; response length is 2, so VERL-aligned response
        # values are raw[2:4] = [0.25, 0.8]. pre_eos on response [7, EOS]
        # should select 0.8, not raw EOS/post position 123.0.
        return torch.tensor([[0.0, 0.1, 0.25, 0.8, 123.0]], dtype=torch.float32)

    monkeypatch.setattr(ev, "critic_sequence_values", fake_sequence_values)
    trajectories = [
        ev.TrajectoryRecord(
            trajectory_id=0,
            prompt_id=0,
            prompt="prompt",
            response="answer",
            response_token_ids=[7, 99],
            reward=1.0,
            data_source="math",
            ground_truth="answer",
        )
    ]
    summary = ev.evaluate_critic_position(
        critic=Critic(),
        critic_dir=Path("/critic/a"),
        critic_label="critic_a",
        tokenizer=Tokenizer(),
        trajectories=trajectories,
        r_bar=1.0,
        eos_token_ids={99},
        value_position="pre_eos",
        output_dir=tmp_path,
        device=torch.device("cpu"),
        max_prompt_length=None,
        config={"test": True},
    )
    assert summary["critic_label"] == "critic_a"
    assert summary["critic_checkpoint_dir"] == "/critic/a"
    assert math.isclose(summary["critic_mse"], (0.8 - 1.0) ** 2, rel_tol=1e-6, abs_tol=1e-8)
    row = json.loads((tmp_path / "trajectory_values.jsonl").read_text(encoding="utf-8").strip())
    assert math.isclose(row["critic_value"], 0.8, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(row["critic_loss"], (0.8 - 1.0) ** 2, rel_tol=1e-6, abs_tol=1e-8)
    assert row["constant_loss"] == 0.0
    assert row["selected_full_index"] == 3
