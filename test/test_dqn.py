"""Tests for DQN and off-policy infrastructure."""

import gymnasium as gym
import torch

from apexrl.agent.off_policy_runner import OffPolicyRunner
from apexrl.algorithms.dqn import DQN, DQNConfig
from apexrl.buffer.replay_buffer import ReplayBuffer
from apexrl.envs.gym_wrapper import GymVecEnv
from apexrl.models import MLPQNetwork


def test_replay_buffer_supports_discrete_actions():
    """Discrete replay transitions should round-trip with scalar actions."""
    buffer = ReplayBuffer(
        capacity=8,
        obs_shape=(4,),
        action_shape=(),
        device="cpu",
    )

    buffer.add(
        observations=torch.randn(4, 4),
        actions=torch.tensor([0, 1, 0, 1]),
        rewards=torch.randn(4),
        next_observations=torch.randn(4, 4),
        dones=torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )

    batch = buffer.sample(4)
    assert batch["actions"].shape == (4,)
    assert batch["actions"].dtype == torch.long
    assert batch["observations"].shape == (4, 4)


def test_dqn_update_smoke():
    """DQN update should produce loss statistics once replay is populated."""
    env = GymVecEnv([lambda: gym.make("CartPole-v1") for _ in range(2)], device="cpu")
    cfg = DQNConfig(
        batch_size=8,
        buffer_size=64,
        learning_starts=0,
        target_update_interval=1,
        train_freq=1,
        gradient_steps=1,
    )
    agent = DQN(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )

    obs = agent._to_tensor_observation(env.reset())
    for _ in range(4):
        actions = agent.sample_random_actions()
        next_obs, rewards, dones, extras = env.step(actions)
        next_obs = agent._to_tensor_observation(next_obs)
        dones = dones.to(agent.device).bool()
        terminated = extras["terminated"].to(agent.device).float()
        final_obs = agent._to_tensor_observation(extras["final_observation"])
        next_obs_for_buffer = next_obs.clone()
        next_obs_for_buffer[dones] = final_obs[dones]
        agent.store_transition(obs, actions, rewards, next_obs_for_buffer, terminated)
        obs = next_obs

    stats = agent.update()
    assert "train/q_loss" in stats
    assert agent.num_updates == 1
    env.close()


def test_dueling_q_network_combines_value_and_advantage():
    """Dueling mode should combine value and advantage streams correctly."""
    env = GymVecEnv([lambda: gym.make("CartPole-v1")], device="cpu")
    cfg = DQNConfig(dueling=True)
    agent = DQN(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )

    obs = torch.randn(3, *env.observation_space_gym.shape)
    with torch.no_grad():
        features = agent.q_network._forward_features(obs)
        values = agent.q_network.value_head(features)
        advantages = agent.q_network.advantage_head(features)
        expected_q = values + advantages - advantages.mean(dim=-1, keepdim=True)
        actual_q = agent.q_network(obs)

    assert agent.q_network.dueling is True
    assert torch.allclose(actual_q, expected_q)
    env.close()


def test_off_policy_runner_smoke_cartpole():
    """Off-policy runner should train DQN for a short CartPole smoke run."""
    env = GymVecEnv([lambda: gym.make("CartPole-v1") for _ in range(2)], device="cpu")
    cfg = DQNConfig(
        batch_size=8,
        buffer_size=256,
        learning_starts=8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=2,
        log_interval=16,
        save_interval=0,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )

    result = runner.learn(total_timesteps=64)
    assert result["total_timesteps"] >= 64
    assert runner.agent.num_updates > 0
    runner.close()


def test_off_policy_runner_smoke_cartpole_dueling():
    """Dueling DQN should also train through the off-policy runner."""
    env = GymVecEnv([lambda: gym.make("CartPole-v1") for _ in range(2)], device="cpu")
    cfg = DQNConfig(
        dueling=True,
        batch_size=8,
        buffer_size=256,
        learning_starts=8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=2,
        log_interval=16,
        save_interval=0,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )

    result = runner.learn(total_timesteps=64)
    assert result["total_timesteps"] >= 64
    assert runner.agent.q_network.dueling is True
    assert runner.agent.num_updates > 0
    runner.close()
