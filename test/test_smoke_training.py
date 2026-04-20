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

"""Smoke tests for PPO training paths."""

import gymnasium as gym
import torch

from apexrl.agent.on_policy_runner import OnPolicyRunner
from apexrl.algorithms.ppo import PPO, PPOConfig
from apexrl.buffer.rollout_buffer import RolloutBuffer
from apexrl.envs.gym_wrapper import GymVecEnv, GymVecEnvContinuous
from apexrl.models import MLPActor, MLPCritic, MLPDiscreteActor


def test_rollout_buffer_supports_continuous_actions():
    """Continuous actions should round-trip through the rollout buffer."""
    buffer = RolloutBuffer(
        num_envs=4,
        num_steps=2,
        obs_shape=(3,),
        action_shape=(2,),
        action_dtype=torch.float32,
        device=torch.device("cpu"),
    )

    actions = torch.randn(4, 2)
    buffer.add(
        observations=torch.randn(4, 3),
        privileged_observations=None,
        actions=actions,
        rewards=torch.randn(4),
        dones=torch.zeros(4),
        values=torch.randn(4),
        log_probs=torch.randn(4),
    )

    data = buffer.get_all_data()
    assert data["actions"].shape == (8, 2)
    assert data["actions"].dtype == torch.float32


def test_gym_vecenv_reports_truncation_metadata():
    """Gym wrappers should surface timeout semantics explicitly."""
    env = GymVecEnv(
        [lambda: gym.make("CartPole-v1", max_episode_steps=1)],
        device="cpu",
    )
    _, _, dones, extras = env.step(torch.tensor([0]))

    assert dones.tolist() == [True]
    assert extras["truncated"].tolist() == [True]
    assert extras["time_outs"].tolist() == [True]
    assert extras["terminated"].tolist() == [False]
    assert extras["final_observation"].shape == env.obs_buf.shape
    env.close()


def test_on_policy_runner_smoke_cartpole():
    """Discrete PPO path should train for one iteration."""
    env = GymVecEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = PPOConfig(
        num_steps=8,
        num_epochs=1,
        minibatch_size=8,
        learning_rate_schedule="constant",
        max_iterations=1,
        device="cpu",
    )
    runner = OnPolicyRunner(
        env=env,
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
    )

    result = runner.learn(num_iterations=1)
    assert result["final_iteration"] == 0
    assert result["total_timesteps"] == cfg.num_steps * env.num_envs
    runner.close()


def test_ppo_collect_rollout_smoke_pendulum():
    """Continuous PPO rollout collection should succeed."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = PPOConfig(
        num_steps=8,
        num_epochs=1,
        minibatch_size=8,
        learning_rate_schedule="constant",
        device="cpu",
    )
    agent = PPO(
        env=env,
        cfg=cfg,
        actor_class=MLPActor,
        critic_class=MLPCritic,
        device=torch.device("cpu"),
    )

    stats = agent.collect_rollout()
    assert "rollout/mean_reward" in stats
    assert agent.rollout_buffer.get_all_data()["actions"].shape[1] == 1
    env.close()
