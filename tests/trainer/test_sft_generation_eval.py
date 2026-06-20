from collections import defaultdict
from unittest.mock import patch

import numpy as np

from verl.trainer.sft_trainer import (
    SFTTrainer,
    build_generation_reward_metrics,
    get_generation_autocast_dtype,
    normalize_eval_methods,
    merge_generation_reward_results,
    repeat_non_tensor_batch,
    score_generation_outputs,
)


class DictLike(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_repeat_non_tensor_batch_interleaves_like_ppo_validation():
    batch = {
        "data_source": np.array(["math", "math"], dtype=object),
        "reward_model": np.array([{"ground_truth": "1"}, {"ground_truth": "2"}], dtype=object),
    }

    repeated = repeat_non_tensor_batch(batch, repeat_times=3)

    assert repeated["data_source"].tolist() == ["math", "math", "math", "math", "math", "math"]
    assert [item["ground_truth"] for item in repeated["reward_model"]] == ["1", "1", "1", "2", "2", "2"]


def test_score_generation_outputs_records_reward_and_acc_for_scalar_scores():
    batch = {
        "data_source": np.array(["lighteval/MATH", "lighteval/MATH"], dtype=object),
        "reward_model": np.array([{"ground_truth": "4"}, {"ground_truth": "5"}], dtype=object),
        "extra_info": np.array([{}, {}], dtype=object),
    }

    with patch("verl.trainer.sft_trainer.default_compute_score", side_effect=[1.0, 0.0]) as compute_score:
        scores, extra = score_generation_outputs(batch, ["answer 4", "answer 3"], DictLike(reward_kwargs={}))

    assert scores == [1.0, 0.0]
    assert extra["reward"] == [1.0, 0.0]
    assert extra["acc"] == [1.0, 0.0]
    assert compute_score.call_args_list[0].kwargs["ground_truth"] == "4"


def test_merge_and_build_generation_reward_metrics_match_grouped_pass_at_k_math():
    result_a = {
        "data_sources": ["math", "math"],
        "sample_uids": ["a", "a"],
        "sample_scores": [0.0, 1.0],
        "sample_response_lengths": [10, 12],
        "reward_extra_infos_dict": {"reward": [0.0, 1.0], "acc": [0.0, 1.0]},
    }
    result_b = {
        "data_sources": ["math", "math"],
        "sample_uids": ["b", "b"],
        "sample_scores": [0.0, 0.0],
        "sample_response_lengths": [8, 9],
        "reward_extra_infos_dict": defaultdict(list, {"reward": [0.0, 0.0], "acc": [0.0, 0.0]}),
    }

    merged = merge_generation_reward_results([result_a, result_b])
    metrics = build_generation_reward_metrics(merged)

    assert merged["sample_scores"] == [0.0, 1.0, 0.0, 0.0]
    assert metrics["val-core/math/acc/mean@2"] == 0.25
    assert metrics["val-core/math/acc/best@2/mean"] == 0.377


def test_normalize_eval_methods_supports_single_both_and_lists():
    assert normalize_eval_methods("loss") == ("loss",)
    assert normalize_eval_methods("generation_reward") == ("generation_reward",)
    assert normalize_eval_methods("both") == ("loss", "generation_reward")
    assert normalize_eval_methods(["loss", "generation_reward", "loss"]) == ("loss", "generation_reward")


def test_should_validate_method_uses_per_method_frequency_overrides():
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.val_dataloader = object()
    trainer.generation_val_dataloader = object()
    trainer.test_freq = 10
    trainer.loss_test_freq = 2
    trainer.generation_reward_test_freq = 3

    assert trainer._should_validate_method("loss", global_step=4, is_last_step=False)
    assert not trainer._should_validate_method("generation_reward", global_step=4, is_last_step=False)
    assert trainer._should_validate_method("generation_reward", global_step=6, is_last_step=False)
    assert trainer._should_validate_method("loss", global_step=5, is_last_step=True)
    assert trainer._should_validate_method("generation_reward", global_step=5, is_last_step=True)


def test_generation_autocast_dtype_prefers_config_then_model_dtype():
    import torch

    model = torch.nn.Linear(2, 2).to(dtype=torch.float16)
    assert get_generation_autocast_dtype(model, {}) == torch.float16
    assert get_generation_autocast_dtype(model, {"dtype": "bf16"}) == torch.bfloat16


def test_available_eval_methods_respects_selected_methods_and_dataloaders():
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.eval_methods = ("loss", "generation_reward")
    trainer.val_dataloader = object()
    trainer.generation_val_dataloader = None
    assert trainer._available_eval_methods() == ["loss"]

    trainer.generation_val_dataloader = object()
    assert trainer._available_eval_methods() == ["loss", "generation_reward"]
