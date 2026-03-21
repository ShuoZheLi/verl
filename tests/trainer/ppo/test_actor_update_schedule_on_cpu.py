# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_schedule_trainer(
    *,
    critic_warmup: int,
    actor_update_interval: int = 1,
    use_zero_critic: bool = False,
) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.use_zero_critic = use_zero_critic
    trainer.global_steps = 0
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "critic_warmup": critic_warmup,
                "actor_update_interval": actor_update_interval,
            }
        }
    )
    return trainer


def test_actor_updates_every_step_after_50_critic_only_steps():
    trainer = _make_schedule_trainer(critic_warmup=51, actor_update_interval=1)

    update_steps = []
    for step in range(1, 56):
        trainer.global_steps = step
        if trainer._should_update_actor():
            update_steps.append(step)

    assert update_steps == [51, 52, 53, 54, 55]
    assert trainer._count_actor_updates_through_step(50) == 0
    assert trainer._count_actor_updates_through_step(55) == 5


def test_actor_update_interval_without_warmup_starts_on_interval_boundary():
    trainer = _make_schedule_trainer(critic_warmup=0, actor_update_interval=2)

    update_steps = []
    for step in range(1, 8):
        trainer.global_steps = step
        if trainer._should_update_actor():
            update_steps.append(step)

    assert update_steps == [2, 4, 6]
