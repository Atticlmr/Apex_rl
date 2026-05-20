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
from apexrl.algorithms.ppo import PPO, PPOConfig, RecurrentPPO, RecurrentPPOConfig
from apexrl.buffer.rollout_buffer import RolloutBuffer
from apexrl.buffer.rollout_rnn_buffer import RolloutRNNBuffer
from apexrl.envs.gym_wrapper import GymVecEnv, GymVecEnvContinuous
from apexrl.models import (
    GRUCritic,
    GRUDiscreteActor,
    MLPActor,
    MLPCritic,
    MLPDiscreteActor,
)
from helpers import make_multimodal_discrete_env, make_uint8_multimodal_discrete_env


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


def test_rollout_buffer_supports_structured_and_privileged_obs():
    """Rollout buffer should flatten nested actor and privileged observations."""
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=2,
        obs_shape={"image": (1, 4, 4), "vector": (3,)},
        action_shape=(),
        action_dtype=torch.long,
        device=torch.device("cpu"),
        num_privileged_obs=7,
        privileged_obs_shape={"state": (5,), "context": (2,)},
    )

    for _ in range(2):
        buffer.add(
            observations={
                "image": torch.randn(2, 1, 4, 4),
                "vector": torch.randn(2, 3),
            },
            privileged_observations={
                "state": torch.randn(2, 5),
                "context": torch.randn(2, 2),
            },
            actions=torch.randint(0, 2, (2,)),
            rewards=torch.randn(2),
            dones=torch.zeros(2),
            values=torch.randn(2),
            log_probs=torch.randn(2),
        )

    data = buffer.get_all_data()
    assert data["observations"]["image"].shape == (4, 1, 4, 4)
    assert data["observations"]["vector"].shape == (4, 3)
    assert data["privileged_observations"]["state"].shape == (4, 5)
    assert data["privileged_observations"]["context"].shape == (4, 2)


def test_rollout_rnn_buffer_sequence_minibatches():
    """Recurrent rollout buffer should return contiguous sequence batches."""
    buffer = RolloutRNNBuffer(
        num_envs=2,
        num_steps=4,
        obs_shape=(3,),
        action_shape=(),
        action_dtype=torch.long,
        actor_hidden_shape=(1, 5),
        critic_hidden_shape=(1, 7),
        device=torch.device("cpu"),
    )

    for step in range(4):
        buffer.add(
            observations=torch.full((2, 3), float(step)),
            privileged_observations=None,
            actions=torch.randint(0, 2, (2,)),
            rewards=torch.randn(2),
            dones=torch.zeros(2),
            values=torch.randn(2),
            log_probs=torch.randn(2),
            actor_hidden_states=torch.randn(1, 2, 5),
            critic_hidden_states=torch.randn(1, 2, 7),
        )

    batches = list(
        buffer.get_sequence_minibatches(
            sequence_length=2,
            minibatch_size=2,
            shuffle=False,
        )
    )
    assert batches[0]["observations"].shape == (2, 2, 3)
    assert batches[0]["actions"].shape == (2, 2)
    assert batches[0]["actor_hidden_states"].shape == (1, 2, 5)


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


def test_gym_vecenv_preserves_structured_leaf_dtypes():
    """Gym wrappers should keep raw per-leaf dtypes for structured observations."""
    env = GymVecEnv(
        [make_uint8_multimodal_discrete_env for _ in range(2)],
        device="cpu",
    )

    obs = env.reset()
    assert obs["obs"]["image"].dtype == torch.uint8
    assert obs["obs"]["vector"].dtype == torch.float32
    assert obs["privileged_obs"]["state"].dtype == torch.float32
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


def test_ppo_supports_multimodal_and_privileged_obs():
    """PPO should train with nested actor obs and privileged critic obs."""
    env = GymVecEnv(
        [make_multimodal_discrete_env for _ in range(2)],
        device="cpu",
    )
    cfg = PPOConfig(
        num_steps=8,
        num_epochs=1,
        minibatch_size=8,
        learning_rate_schedule="constant",
        use_asymmetric=True,
        device="cpu",
    )
    agent = PPO(
        env=env,
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
        device=torch.device("cpu"),
    )

    rollout_stats = agent.collect_rollout()
    update_stats = agent.update()
    assert "rollout/mean_reward" in rollout_stats
    assert "train/policy_loss" in update_stats
    env.close()


def test_ppo_supports_muon_optimizer():
    """PPO should train with the integrated mixed Muon optimizer path."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = PPOConfig(
        num_steps=8,
        num_epochs=1,
        minibatch_size=8,
        learning_rate_schedule="constant",
        optimizer="muon",
        device="cpu",
    )
    agent = PPO(
        env=env,
        cfg=cfg,
        actor_class=MLPActor,
        critic_class=MLPCritic,
        device=torch.device("cpu"),
    )

    rollout_stats = agent.collect_rollout()
    update_stats = agent.update()
    assert "rollout/mean_reward" in rollout_stats
    assert "train/policy_loss" in update_stats
    env.close()


class CustomRecurrentDiscreteActor(GRUDiscreteActor):
    """Custom recurrent actor used to verify class injection."""


class CustomRecurrentCritic(GRUCritic):
    """Custom recurrent critic used to verify class injection."""


def test_recurrent_ppo_supports_custom_recurrent_classes():
    """Recurrent PPO should accept user-provided recurrent actor/critic classes."""
    env = GymVecEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = RecurrentPPOConfig(
        num_steps=8,
        sequence_length=4,
        num_epochs=1,
        recurrent_minibatch_size=2,
        rnn_hidden_size=16,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        learning_rate_schedule="constant",
        device="cpu",
    )
    agent = RecurrentPPO(
        env=env,
        cfg=cfg,
        actor_class=CustomRecurrentDiscreteActor,
        critic_class=CustomRecurrentCritic,
        device=torch.device("cpu"),
    )

    rollout_stats = agent.collect_rollout()
    update_stats = agent.update()
    assert isinstance(agent.actor, CustomRecurrentDiscreteActor)
    assert isinstance(agent.critic, CustomRecurrentCritic)
    assert "rollout/mean_reward" in rollout_stats
    assert "train/policy_loss" in update_stats
    env.close()


def test_on_policy_runner_can_create_recurrent_ppo():
    """OnPolicyRunner should auto-create recurrent PPO from algorithm name."""
    env = GymVecEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = RecurrentPPOConfig(
        num_steps=8,
        sequence_length=4,
        num_epochs=1,
        recurrent_minibatch_size=2,
        rnn_hidden_size=16,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        learning_rate_schedule="constant",
        max_iterations=1,
        device="cpu",
    )
    runner = OnPolicyRunner(
        env=env,
        cfg=cfg,
        algorithm="recurrent_ppo",
        actor_class=GRUDiscreteActor,
        critic_class=GRUCritic,
    )

    result = runner.learn(num_iterations=1)
    assert result["final_iteration"] == 0
    assert result["total_timesteps"] == cfg.num_steps * env.num_envs
    runner.close()


def test_gru_actor_single_step_call_keeps_deterministic_keyword():
    """GRU actor should still support actor.act(obs, deterministic=True)."""
    env = GymVecEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(2)],
        device="cpu",
    )
    actor = GRUDiscreteActor(
        env.observation_space_gym,
        env.action_space_gym,
        {"hidden_dims": [16], "rnn_hidden_size": 16},
    )

    obs = env.get_observations()
    actions, log_probs, hidden = actor.act(obs, deterministic=True)
    assert actions.shape == (2,)
    assert log_probs.shape == (2,)
    assert hidden.shape == (1, 2, 16)

    actions_positional, _, _ = actor.act(obs, True)
    assert actions_positional.shape == (2,)
    env.close()


def test_recurrent_ppo_carries_hidden_state_between_rollouts():
    """Recurrent PPO should not reset hidden state at rollout boundaries."""
    env = GymVecEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = RecurrentPPOConfig(
        num_steps=4,
        sequence_length=2,
        num_epochs=1,
        recurrent_minibatch_size=2,
        rnn_hidden_size=16,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        learning_rate_schedule="constant",
        device="cpu",
    )
    agent = RecurrentPPO(
        env=env,
        cfg=cfg,
        actor_class=GRUDiscreteActor,
        critic_class=GRUCritic,
        device=torch.device("cpu"),
    )

    agent.collect_rollout()
    assert agent.actor_hidden_state is not None
    assert agent.prev_dones is not None
    assert not agent.prev_dones.any()
    carried_hidden = agent.actor_hidden_state.clone()
    agent.collect_rollout()
    stored_initial_hidden = agent.rollout_buffer.actor_hidden_states[0].transpose(0, 1)
    assert torch.allclose(stored_initial_hidden, carried_hidden)
    env.close()
