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
from apexrl.algorithms.flash_sac.distribution import categorical_td_projection
from apexrl.algorithms.flash_sac.network import (
    FlashSACCategoricalQNetwork,
    UnitLinear,
)
from apexrl.envs.gym_wrapper import GymVecEnvContinuous
from helpers import make_multimodal_continuous_env


def make_flash_sac_config(**overrides) -> FlashSACConfig:
    """Create a lightweight FlashSAC config for CPU smoke tests."""
    values = {
        "batch_size": 8,
        "buffer_size": 128,
        "learning_starts": 0,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 1,
        "actor_hidden_dim": 32,
        "critic_hidden_dim": 32,
        "actor_num_blocks": 1,
        "critic_num_blocks": 1,
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
    assert stats["train/actor_updated"] == 1.0
    assert "train/critic1_cross_entropy" in stats
    assert "train/critic_feature_norm_loss" in stats
    assert "train/update_to_data_ratio" in stats
    assert "train/actor_weight_norm" in stats
    assert agent.num_updates == 1
    env.close()


def test_categorical_projection_preserves_probability_mass_and_terminals():
    """Categorical Bellman projection should preserve mass and stop bootstrapping."""
    support = torch.linspace(-2.0, 2.0, 5)
    next_log_probs = torch.log_softmax(torch.randn(2, 5), dim=-1)
    rewards = torch.tensor([0.5, -0.5])
    dones = torch.tensor([0.0, 1.0])
    projected = categorical_td_projection(
        next_log_probs,
        rewards,
        dones,
        entropy_terms=torch.zeros(2),
        gamma=0.99,
        support=support,
    )

    assert torch.allclose(projected.sum(-1), torch.ones(2), atol=1e-6)
    terminal_value = (projected[1] * support).sum()
    assert torch.allclose(terminal_value, rewards[1], atol=1e-6)


def test_flash_sac_uses_distributional_critic_and_delayed_actor_updates():
    """Default FlashSAC should use categorical critics and delayed policy updates."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = make_flash_sac_config(actor_update_period=2)
    agent = FlashSAC(env=env, cfg=cfg, device=torch.device("cpu"))
    populate_replay(agent)

    assert isinstance(agent.critic1, FlashSACCategoricalQNetwork)
    assert agent.critic1.support.shape == (cfg.critic_num_bins,)
    first_stats = agent.update()
    second_stats = agent.update()

    assert first_stats["train/actor_updated"] == 1.0
    assert second_stats["train/actor_updated"] == 0.0
    assert not torch.equal(
        agent.target_critic1.backbone.embedder.norm.running_mean,
        torch.zeros_like(agent.target_critic1.backbone.embedder.norm.running_mean),
    )
    for module in agent.critic1.modules():
        if isinstance(module, UnitLinear):
            row_norms = module.weight.norm(dim=-1)
            assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)
    env.close()


def test_flash_sac_repeats_exploration_noise():
    """Policy noise should remain fixed until its sampled repeat length expires."""
    env = GymVecEnvContinuous(
        [lambda: gym.make("Pendulum-v1") for _ in range(2)],
        device="cpu",
    )
    cfg = make_flash_sac_config(noise_repeat_max=4)
    agent = FlashSAC(env=env, cfg=cfg, device=torch.device("cpu"))
    obs = agent._to_tensor_observation(env.reset())
    agent._noise_repeat_lengths.fill_(3)
    agent._noise_repeat_counts.fill_(1)

    first_actions = agent.act(obs)
    second_actions = agent.act(obs)

    assert torch.equal(first_actions, second_actions)
    assert torch.equal(agent._noise_repeat_counts, torch.full((2,), 3))
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


def test_flash_sac_supports_multimodal_and_privileged_observations():
    """FlashSAC networks should preserve ApexRL structured observation support."""
    env = GymVecEnvContinuous(
        [make_multimodal_continuous_env for _ in range(2)],
        device="cpu",
    )
    cfg = make_flash_sac_config(learning_starts=8)
    agent = FlashSAC(env=env, cfg=cfg, device=torch.device("cpu"))

    result = agent.learn(total_timesteps=32)

    assert result["total_timesteps"] >= 32
    assert agent.num_updates > 0
    assert agent._distributional_critic
    env.close()


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
    assert restored.reward_normalizer is not None
    assert agent.reward_normalizer is not None
    assert torch.equal(
        restored.reward_normalizer.count,
        agent.reward_normalizer.count,
    )
    assert torch.equal(restored._cached_noise, agent._cached_noise)
    for source, loaded in zip(agent.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(source, loaded)
    env.close()
    restored_env.close()
