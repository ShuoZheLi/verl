from collections import namedtuple

import torch
from torch import nn

from verl.utils.sparse_update_mask import (
    SparseUpdateMaskManager,
    bottom_fraction_mask,
    build_masks_from_model,
    build_safe_svd_lowmag_mask_for_tensor,
    load_sparse_masks,
    save_sparse_masks,
    should_mask_param,
    top_fraction_mask,
)


def test_top_and_bottom_fraction_masks_counts():
    scores = torch.arange(10, dtype=torch.float32)
    top = top_fraction_mask(scores, 0.3)
    bottom = bottom_fraction_mask(scores, 0.2)
    assert top.sum().item() == 3
    assert bottom.sum().item() == 2
    assert torch.equal(top, torch.tensor([False, False, False, False, False, False, False, True, True, True]))
    assert torch.equal(bottom, torch.tensor([True, True, False, False, False, False, False, False, False, False]))


def test_safe_mask_matches_definition_for_full_rank_reconstruction():
    weight = torch.tensor([[1.0, 4.0], [2.0, 3.0]])
    train_mask = build_safe_svd_lowmag_mask_for_tensor(
        weight, rank_k=2, alpha_princ=0.5, alpha_low=0.25, mode="safe_svd_lowmag"
    )
    principal = top_fraction_mask(weight.abs(), 0.5).cpu()
    low = bottom_fraction_mask(weight.abs(), 0.25).cpu()
    assert torch.equal(train_mask, (~principal) | low)


def test_should_mask_param_respects_excludes_and_targets():
    assert should_mask_param("model.layers.0.self_attn.q_proj.weight", torch.empty(4, 4), ["q_proj"], [], False)
    assert not should_mask_param("model.embed_tokens.weight", torch.empty(4, 4), ["q_proj"], ["embed"], False)
    assert not should_mask_param("model.layers.0.self_attn.q_proj.bias", torch.empty(4), ["q_proj"], [], False)
    assert not should_mask_param("model.layers.0.input_layernorm.weight", torch.empty(4), ["q_proj"], ["norm"], False)


def test_sparse_mask_save_load_roundtrip(tmp_path):
    path = tmp_path / "mask.pt"
    masks = {"layers.0.q_proj.weight": torch.tensor([[True, False], [False, True]])}
    metadata = {"mode": "safe_svd_lowmag", "linear_trainable_fraction": 0.5}
    save_sparse_masks(str(path), masks, metadata)
    loaded_masks, loaded_metadata = load_sparse_masks(str(path))
    assert torch.equal(loaded_masks["layers.0.q_proj.weight"], masks["layers.0.q_proj.weight"])
    assert loaded_metadata == metadata


def test_optimizer_enforcement_restores_frozen_and_masks_adamw_state():
    torch.manual_seed(0)
    model = nn.Linear(4, 2, bias=False)
    original = model.weight.detach().clone()
    mask = torch.zeros_like(model.weight, dtype=torch.bool)
    mask[:, :2] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    config = {
        "enabled": True,
        "restore_frozen_after_step": True,
        "mask_optimizer_state": True,
        "verify_frozen_weights": True,
        "verification_interval": 1,
        "verification_tolerance": 0.0,
        "strict_load": True,
    }
    manager = SparseUpdateMaskManager(model, {"weight": mask}, config)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(3, 4)).pow(2).sum()
        loss.backward()
        manager.apply_grad_mask()
        manager.mask_optimizer_state(optimizer)
        optimizer.step()
        manager.restore_frozen_params()
        manager.mask_optimizer_state(optimizer)
        manager.verify_frozen_params()

    assert torch.equal(model.weight.detach()[:, 2:], original[:, 2:])
    state = optimizer.state[model.weight]
    assert torch.equal(state["exp_avg"][:, 2:], torch.zeros_like(state["exp_avg"][:, 2:]))
    assert torch.equal(state["exp_avg_sq"][:, 2:], torch.zeros_like(state["exp_avg_sq"][:, 2:]))
    assert not torch.equal(model.weight.detach()[:, :2], original[:, :2])


def test_manager_fails_on_local_shard_shape_mismatch():
    model = nn.Linear(4, 2, bias=False)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        model.weight.set_(model.weight.detach().reshape(-1)[:4])
    try:
        SparseUpdateMaskManager(model, {"weight": mask}, {"enabled": True, "strict_load": True})
    except ValueError as exc:
        assert "could not find FSDP shard metadata" in str(exc)
    else:
        raise AssertionError("Expected local shard/full mask shape mismatch to fail loudly")


def test_manager_supports_fsdp_original_param_local_shard():
    ParamInfo = namedtuple("ParamInfo", ["param_name", "module", "module_name"])
    ShardInfo = namedtuple(
        "ShardInfo", ["in_shard", "offset_in_shard", "numel_in_shard", "intra_param_start_idx", "intra_param_end_idx"]
    )

    model = nn.Module()
    model.q_proj = nn.Module()
    model.q_proj.weight = nn.Parameter(torch.tensor([10.0, 20.0, 30.0]))
    flat_param = nn.Parameter(torch.empty(0))
    flat_param._param_infos = (ParamInfo("weight", model.q_proj, "q_proj"),)
    flat_param._shard_param_infos = (ShardInfo(True, 0, 3, 2, 4),)
    model._handle = type("Handle", (), {"flat_param": flat_param})()

    full_mask = torch.tensor([[True, False, True], [False, True, False]])
    manager = SparseUpdateMaskManager(model, {"q_proj.weight": full_mask}, {"enabled": True, "strict_load": True})
    assert torch.equal(manager.local_masks["q_proj.weight"], torch.tensor([True, False, True]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    model.q_proj.weight.grad = torch.ones_like(model.q_proj.weight)
    manager.apply_grad_mask()
    assert torch.equal(model.q_proj.weight.grad, torch.tensor([1.0, 0.0, 1.0]))
    optimizer.step()
    manager.restore_frozen_params()
    assert model.q_proj.weight[1].item() == 20.0

    state = manager.state_dict()
    assert torch.equal(state["local_masks"]["q_proj.weight"], torch.tensor([True, False, True]))
    manager.load_state_dict(state)
    assert torch.equal(manager.local_masks["q_proj.weight"], torch.tensor([True, False, True]))


def test_random_same_density_is_deterministic_from_seed():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)

    torch.manual_seed(0)
    model = TinyModel()
    config = {
        "mode": "random_same_density",
        "rank_k": 2,
        "alpha_princ": 0.5,
        "alpha_low": 0.5,
        "target_modules": ["q_proj"],
        "exclude_keywords": [],
        "seed": 123,
    }
    masks_a = build_masks_from_model(model, config)
    masks_b = build_masks_from_model(model, config)
    assert torch.equal(masks_a["q_proj.weight"], masks_b["q_proj.weight"])
