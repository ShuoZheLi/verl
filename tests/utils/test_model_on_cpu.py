# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace  # Or use a mock object library

import pytest
import torch
from torch import nn

from verl.utils.model import _init_value_head_parameters, update_model_config


# Parametrize with different override scenarios
@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"param_a": 5, "new_param": "plain_added"},
        {"param_a": 2, "nested_params": {"sub_param_x": "updated_x", "sub_param_z": True}},
    ],
)
def test_update_model_config(override_kwargs):
    """
    Tests that update_model_config correctly updates attributes,
    handling both plain and nested overrides via parametrization.
    """
    # Create a fresh mock config object for each test case
    mock_config = SimpleNamespace(
        param_a=1, nested_params=SimpleNamespace(sub_param_x="original_x", sub_param_y=100), other_param="keep_me"
    )
    # Apply the updates using the parametrized override_kwargs
    update_model_config(mock_config, override_kwargs)

    # Assertions to check if the config was updated correctly
    if "nested_params" in override_kwargs:  # Case 2: Nested override
        override_nested = override_kwargs["nested_params"]
        assert mock_config.nested_params.sub_param_x == override_nested["sub_param_x"], "Nested sub_param_x mismatch"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert hasattr(mock_config.nested_params, "sub_param_z"), "Expected nested sub_param_z to be added"
        assert mock_config.nested_params.sub_param_z == override_nested["sub_param_z"], "Value of sub_param_z mismatch"
    else:  # Case 1: Plain override (nested params untouched)
        assert mock_config.nested_params.sub_param_x == "original_x", "Nested sub_param_x should be unchanged"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert not hasattr(mock_config.nested_params, "sub_param_z"), "Nested sub_param_z should not exist"


def test_init_value_head_parameters_supports_xavier_uniform():
    class DummyValueModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_head = nn.Linear(8, 1)

    torch.manual_seed(0)
    model = DummyValueModel()
    _init_value_head_parameters(model, mean=0.0, std=None, method="xavier_uniform")

    expected_bound = (6.0 / (model.v_head.in_features + model.v_head.out_features)) ** 0.5
    assert torch.count_nonzero(model.v_head.weight).item() > 0
    assert torch.all(model.v_head.weight.abs() <= expected_bound + 1e-6)
    assert torch.allclose(model.v_head.bias, torch.zeros_like(model.v_head.bias))


def test_init_value_head_parameters_rejects_normal_without_std():
    class DummyValueModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_head = nn.Linear(4, 1)

    with pytest.raises(ValueError, match="value_head_init_std must be set"):
        _init_value_head_parameters(DummyValueModel(), mean=0.0, std=None, method="normal")
