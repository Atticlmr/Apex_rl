# Copyright (c) 2026 GitHub@Apex_rl Developer
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

"""Independent update, runner, and checkpoint tests for FlashSAC."""

import gymnasium as gym
import torch

from apexrl.agent.off_policy_runner import OffPolicyRunner
from apexrl.algorithms.flash_sac import FlashSAC, FlashSACConfig
from apexrl.envs.gym_wrapper import GymVecEnvContinuous


def make_flash_sac_config(**overrides) -> FlashSACConfig:
    """Create a lightweight FlashSAC config for CPU smoke tests."""
    values = {
        "batch_size": 8,
        "buffer_size": 128,
        "learning_starts": 0,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 1,
        "actor_hidden_dims": [32],
        "critic_hidden_dims": [32],
        "layer_norm": False,
        "log_interval": 0,
        "save_interval": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return FlashSACConfig(**values)


def populate_replay(agent: FlashSAC, steps: int = 4) -> None:
    """Populate replay with random Pendulum transitions."""
    obs = agent._to_tensor_observation(agent.env.reset())
    for _ in range(steps):
        actions = agent.sample_random_actions()
        next_obs, rewards, _, extras = agent.env.step(actions)
        next_obs = agent._to_tensor_observation(next_obs)
        terminated = extras["terminated"].to(agent.device).float()
        agent.store_transition(obs, actions, rewards, next_obs, terminated)
        obs = next_obs


def test_flash_sac_update_reports_flash_metrics():
    """FlashSAC update should execute its specialized critic stabilization path."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = make_flash_sac_config(
        reward_scale=2.0,
        target_q_clip=10.0,
        critic_feature_norm_coef=0.01,
        critic_feature_norm_target=1.0,
    )
    agent = FlashSAC(env=env, cfg=cfg, device=torch.device("cpu"))
    populate_replay(agent)

    stats = agent.update()

    assert stats["train/reward_scale"] == 2.0
    assert "train/critic_feature_norm_loss" in stats
    assert "train/update_to_data_ratio" in stats
    assert "train/actor_weight_norm" in stats
    assert agent.num_updates == 1
    env.close()


def test_off_policy_runner_can_create_flash_sac_agent():
    """OffPolicyRunner should construct and train FlashSAC by algorithm name."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = make_flash_sac_config(learning_starts=8)
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        algorithm="flash_sac",
        device=torch.device("cpu"),
    )

    result = runner.learn(total_timesteps=32)

    assert isinstance(runner.agent, FlashSAC)
    assert result["total_timesteps"] >= 32
    assert runner.agent.num_updates > 0
    runner.close()


def test_flash_sac_checkpoint_restores_training_and_replay_state(tmp_path):
    """FlashSAC checkpoints should restore networks, counters, and replay data."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = make_flash_sac_config(save_replay_buffer=True)
    agent = FlashSAC(env=env, cfg=cfg, device=torch.device("cpu"))
    populate_replay(agent)
    agent.update()
    agent.iteration = 7
    agent.total_timesteps = 24
    checkpoint = tmp_path / "flash_sac.pt"
    agent.save(str(checkpoint))

    restored_env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    restored = FlashSAC(
        env=restored_env,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    restored.load(str(checkpoint))

    assert restored.iteration == 7
    assert restored.total_timesteps == 24
    assert restored.num_updates == agent.num_updates
    assert len(restored.replay_buffer) == len(agent.replay_buffer)
    for source, loaded in zip(agent.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(source, loaded)
    env.close()
    restored_env.close()
