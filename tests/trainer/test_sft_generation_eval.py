from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pytest

from verl.trainer.sft_trainer import (
    SFTTrainer,
    build_generation_reward_metrics,
    generate_texts_from_prompts_vllm,
    generate_texts_from_prompts,
    get_generation_autocast_dtype,
    get_vllm_sampling_params,
    normalize_eval_methods,
    merge_generation_reward_results,
    repeat_non_tensor_batch,
    score_generation_outputs,
    sync_fsdp_weights_to_vllm,
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


def test_hf_generation_accepts_quoted_hydra_null_and_bool_values():
    import torch

    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, prompts, return_tensors=None, padding=None):
            return {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.tensor([[1, 1]])}

        def batch_decode(self, response_ids, skip_special_tokens=True):
            return ["decoded"]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[5, 6, 7, 0]])

    model = Model()
    outputs, lengths = generate_texts_from_prompts(
        model=model,
        tokenizer=Tokenizer(),
        prompt_texts=["prompt"],
        device="cpu",
        generation_config=DictLike(do_sample="False", top_k="null", max_new_tokens="4"),
    )

    assert outputs == ["decoded"]
    assert lengths == [1]
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 4
    assert "top_k" not in model.kwargs
    assert "temperature" not in model.kwargs


def test_available_eval_methods_respects_selected_methods_and_dataloaders():
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.eval_methods = ("loss", "generation_reward")
    trainer.val_dataloader = object()
    trainer.generation_val_dataloader = None
    assert trainer._available_eval_methods() == ["loss"]

    trainer.generation_val_dataloader = object()
    assert trainer._available_eval_methods() == ["loss", "generation_reward"]


def test_vllm_sampling_params_uses_single_output_per_repeated_prompt():
    class Tokenizer:
        eos_token_id = 151643

    params = get_vllm_sampling_params(DictLike(n=4, do_sample=False, max_new_tokens=128), Tokenizer())

    assert params.n == 1
    assert params.max_tokens == 128
    assert params.temperature == 0.0


def test_generate_texts_from_prompts_vllm_syncs_once_per_sync_version():
    class Completion:
        text = "answer"
        token_ids = [1, 2]

    class RequestOutput:
        outputs = [Completion()]

    class Engine:
        def __init__(self):
            self.sync_calls = 0

        def generate(self, prompts, sampling_params=None, use_tqdm=False):
            return [RequestOutput() for _ in prompts]

    engine = Engine()

    def fake_sync(model, vllm_engine, generation_config, sync_version=None):
        assert vllm_engine is engine
        assert sync_version == 7
        engine.sync_calls += 1

    with patch("verl.trainer.sft_trainer.get_or_create_vllm_engine", return_value=engine), patch(
        "verl.trainer.sft_trainer.sync_fsdp_weights_to_vllm", side_effect=fake_sync
    ):
        outputs, lengths = generate_texts_from_prompts_vllm(
            model=object(),
            tokenizer=object(),
            prompt_texts=["a", "b"],
            generation_config=DictLike(n=3, do_sample=False, max_new_tokens=8),
            sync_version=7,
        )

    assert outputs == ["answer", "answer"]
    assert lengths == [2, 2]
    assert engine.sync_calls == 1


def test_vllm_sampling_params_accepts_quoted_hydra_null_and_bool_values():
    params = get_vllm_sampling_params(
        DictLike(n=1, do_sample="False", max_new_tokens=16, top_k="null"),
        object(),
    )

    assert params.temperature == 0.0
    assert params.max_tokens == 16


def test_sync_fsdp_weights_to_vllm_batches_and_caches_by_sync_version():
    import torch
    import verl.trainer.sft_trainer as sft_trainer

    class Model:
        def state_dict(self):
            return {
                "a.weight": torch.tensor([1.0]),
                "b.weight": torch.tensor([2.0]),
                "metadata": "skip",
                "c.weight": torch.tensor([3.0]),
            }

    class VllmModel:
        def __init__(self):
            self.loaded_batches = []

        def load_weights(self, weight_batch):
            self.loaded_batches.append([(name, tensor.clone()) for name, tensor in weight_batch])

    class Engine:
        def __init__(self):
            self.vllm_model = VllmModel()

        def apply_model(self, fn):
            return fn(self.vllm_model)

    previous_sync_version = sft_trainer._SFT_VLLM_SYNC_VERSION
    sft_trainer._SFT_VLLM_SYNC_VERSION = None
    try:
        engine = Engine()
        config = DictLike(vllm_weight_sync_batch_size=2, vllm_sync_weights=True)

        sync_fsdp_weights_to_vllm(Model(), engine, config, sync_version=11)
        sync_fsdp_weights_to_vllm(Model(), engine, config, sync_version=11)

        assert [[name for name, _ in batch] for batch in engine.vllm_model.loaded_batches] == [
            ["a.weight", "b.weight"],
            ["c.weight"],
        ]
    finally:
        sft_trainer._SFT_VLLM_SYNC_VERSION = previous_sync_version

def test_evaluate_generation_reward_batches_rejects_output_count_mismatch():
    import torch

    class Model(torch.nn.Module):
        pass

    class Tokenizer:
        pass

    class Config:
        trainer = DictLike(generation_eval=DictLike(backend="vllm", n=2))
        data = DictLike(generation_eval_prompt_key="prompt", apply_chat_template_kwargs={})

    batch = {
        "prompt": np.array(["p1", "p2"], dtype=object),
        "data_source": np.array(["math", "math"], dtype=object),
        "reward_model": np.array([{"ground_truth": "1"}, {"ground_truth": "2"}], dtype=object),
    }

    with patch("verl.trainer.sft_trainer.extract_prompt_texts", return_value=["p1", "p2"]), patch(
        "verl.trainer.sft_trainer.generate_texts_from_prompts_vllm", return_value=(["only one"], [1])
    ), pytest.raises(RuntimeError, match="returned 1 outputs for 4 prompts"):
        from verl.trainer.sft_trainer import evaluate_generation_reward_batches

        evaluate_generation_reward_batches(Model(), Tokenizer(), [batch], "cpu", Config())
