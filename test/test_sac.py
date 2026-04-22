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

"""Smoke tests for Soft Actor-Critic."""

import gymnasium as gym
import torch

from apexrl.agent.off_policy_runner import OffPolicyRunner
from apexrl.algorithms.sac import SAC, SACConfig
from apexrl.envs.gym_wrapper import GymVecEnvContinuous
from helpers import make_multimodal_continuous_env


def test_sac_actor_respects_action_bounds():
    """SAC actor should emit environment-scaled bounded actions."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    agent = SAC(
        env=env,
        cfg=SACConfig(batch_size=8, buffer_size=64, learning_starts=0, device="cpu"),
        device=torch.device("cpu"),
    )

    obs = agent._to_tensor_observation(env.reset())
    actions = agent.act(obs)
    action_low = torch.as_tensor(
        env.action_space_gym.low,
        dtype=torch.float32,
        device=actions.device,
    )
    action_high = torch.as_tensor(
        env.action_space_gym.high,
        dtype=torch.float32,
        device=actions.device,
    )

    assert actions.shape == (env.num_envs, env.num_actions)
    assert torch.all(actions <= action_high + 1e-5)
    assert torch.all(actions >= action_low - 1e-5)
    env.close()


def test_sac_update_smoke():
    """SAC update should produce scalar losses once replay is populated."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = SACConfig(
        batch_size=8,
        buffer_size=64,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        device="cpu",
    )
    agent = SAC(env=env, cfg=cfg, device=torch.device("cpu"))

    obs = agent._to_tensor_observation(env.reset())
    for _ in range(4):
        actions = agent.sample_random_actions()
        next_obs, rewards, _, extras = env.step(actions)
        next_obs = agent._to_tensor_observation(next_obs)
        terminated = extras["terminated"].to(agent.device).float()
        agent.store_transition(obs, actions, rewards, next_obs, terminated)
        obs = next_obs

    stats = agent.update()
    assert "train/q_loss" in stats
    assert "train/actor_loss" in stats
    assert "train/alpha" in stats
    assert agent.num_updates == 1
    env.close()


def test_sac_learn_smoke_pendulum():
    """SAC should train through the off-policy runner entrypoint."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = SACConfig(
        batch_size=8,
        buffer_size=256,
        learning_starts=8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        log_interval=16,
        save_interval=0,
        device="cpu",
    )
    agent = SAC(env=env, cfg=cfg, device=torch.device("cpu"))

    result = agent.learn(total_timesteps=64)
    assert result["total_timesteps"] >= 64
    assert agent.num_updates > 0
    env.close()


def test_off_policy_runner_can_create_sac_agent():
    """OffPolicyRunner should auto-create SAC when requested by algorithm name."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = SACConfig(
        batch_size=8,
        buffer_size=256,
        learning_starts=8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        log_interval=16,
        save_interval=0,
        device="cpu",
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        algorithm="sac",
        device=torch.device("cpu"),
    )

    result = runner.learn(total_timesteps=64)
    assert isinstance(runner.agent, SAC)
    assert result["total_timesteps"] >= 64
    assert runner.agent.num_updates > 0
    runner.close()


def test_sac_supports_multimodal_and_privileged_obs():
    """SAC should train with nested actor obs and privileged critic obs."""
    env = GymVecEnvContinuous(
        [make_multimodal_continuous_env for _ in range(2)],
        device="cpu",
    )
    cfg = SACConfig(
        batch_size=8,
        buffer_size=128,
        learning_starts=8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        log_interval=0,
        save_interval=0,
        device="cpu",
    )
    agent = SAC(env=env, cfg=cfg, device=torch.device("cpu"))

    result = agent.learn(total_timesteps=32)
    assert result["total_timesteps"] >= 32
    assert agent.num_updates > 0
    env.close()


def test_sac_supports_muon_optimizer():
    """SAC should train with the mixed Muon optimizer path."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = SACConfig(
        optimizer="muon",
        batch_size=8,
        buffer_size=128,
        learning_starts=8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        log_interval=0,
        save_interval=0,
        device="cpu",
    )
    agent = SAC(env=env, cfg=cfg, device=torch.device("cpu"))

    result = agent.learn(total_timesteps=32)
    assert result["total_timesteps"] >= 32
    assert agent.num_updates > 0
    env.close()
