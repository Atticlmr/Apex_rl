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

"""Smoke tests for MAPPO multi-agent support."""

from __future__ import annotations

import torch
from gymnasium import spaces

from apexrl.models import MLPActor, MLPCritic
from apexrl.multiagent import (
    IPPO,
    MAPPO,
    IPPOConfig,
    MAPPOConfig,
    MultiAgentRunner,
    MultiAgentVecEnv,
)
from apexrl.multiagent.buffer import MultiAgentRolloutBuffer


class DummyDroneTransportVecEnv(MultiAgentVecEnv):
    """Small cooperative continuous-control env for MAPPO smoke tests."""

    def __init__(
        self,
        num_envs: int = 2,
        num_agents: int = 3,
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.possible_agents = [f"drone_{idx}" for idx in range(num_agents)]
        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "self": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(4,),
                        dtype=float,
                    ),
                    "payload": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(3,),
                        dtype=float,
                    ),
                }
            )
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
            for agent_id in self.possible_agents
        }
        self.state_space = spaces.Dict(
            {
                "drones": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(num_agents * 4,),
                    dtype=float,
                ),
                "payload": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(5,),
                    dtype=float,
                ),
            }
        )
        self.step_count = 0
        self.max_steps = 4
        self._obs = self._build_obs()
        self._state = self._build_state()

    def _phase(self) -> torch.Tensor:
        return torch.full(
            (self.num_envs, 1),
            float(self.step_count) / self.max_steps,
            device=self.device,
        )

    def _build_obs(self):
        phase = self._phase()
        obs = {}
        for idx, agent_id in enumerate(self.possible_agents):
            agent_offset = torch.full_like(phase, idx * 0.1)
            obs[agent_id] = {
                "self": torch.cat(
                    [phase + agent_offset, phase, -phase, phase * 0.5],
                    dim=-1,
                ),
                "payload": torch.cat([phase, phase * 0.25, -phase], dim=-1),
            }
        return obs

    def _build_state(self):
        phase = self._phase()
        return {
            "drones": torch.cat(
                [self._obs[agent_id]["self"] for agent_id in self.possible_agents],
                dim=-1,
            ),
            "payload": torch.cat(
                [phase, -phase, phase * 0.5, phase * 0.25, phase],
                dim=-1,
            ),
        }

    def reset(self):
        self.step_count = 0
        self._obs = self._build_obs()
        self._state = self._build_state()
        return self._obs, {}

    def get_observations(self):
        return self._obs

    def get_state(self):
        return self._state

    def step(self, actions):
        self.step_count += 1
        self._obs = self._build_obs()
        self._state = self._build_state()
        reward = {
            agent_id: 1.0 - actions[agent_id].pow(2).mean(dim=-1)
            for agent_id in self.possible_agents
        }
        done_mask = torch.full(
            (self.num_envs,),
            self.step_count >= self.max_steps,
            dtype=torch.bool,
            device=self.device,
        )
        terminated = {agent_id: done_mask for agent_id in self.possible_agents}
        truncated = {
            agent_id: torch.zeros_like(done_mask) for agent_id in self.possible_agents
        }
        extras = {
            "log": {
                "payload_error": torch.full(
                    (self.num_envs,),
                    0.25,
                    device=self.device,
                ),
                "nested": {
                    "formation_error": 0.5,
                },
            }
        }
        return self._obs, reward, terminated, truncated, extras


def test_multiagent_rollout_buffer_supports_agent_dicts():
    """Multi-agent rollout buffer should flatten per-agent data."""
    agents = ["drone_0", "drone_1"]
    buffer = MultiAgentRolloutBuffer(
        possible_agents=agents,
        num_envs=2,
        num_steps=2,
        obs_specs={agent_id: {"self": (3,)} for agent_id in agents},
        action_shapes={agent_id: (2,) for agent_id in agents},
        action_dtypes={agent_id: torch.float32 for agent_id in agents},
        device=torch.device("cpu"),
        state_spec={"payload": (4,)},
    )

    for _ in range(2):
        buffer.add(
            observations={agent_id: {"self": torch.randn(2, 3)} for agent_id in agents},
            state={"payload": torch.randn(2, 4)},
            actions={agent_id: torch.randn(2, 2) for agent_id in agents},
            rewards={agent_id: torch.randn(2) for agent_id in agents},
            dones={agent_id: torch.zeros(2) for agent_id in agents},
            values={agent_id: torch.randn(2) for agent_id in agents},
            log_probs={agent_id: torch.randn(2) for agent_id in agents},
        )

    buffer.compute_returns_and_advantages(
        last_values={agent_id: torch.zeros(2) for agent_id in agents},
        gamma=0.99,
        gae_lambda=0.95,
    )
    data = buffer.get_agent_data("drone_0")
    assert data["observations"]["self"].shape == (4, 3)
    assert data["states"]["payload"].shape == (4, 4)
    assert data["actions"].shape == (4, 2)


def test_mappo_collect_update_and_learn_smoke(tmp_path):
    """MAPPO should collect, update, save/load and learn for one iteration."""
    env = DummyDroneTransportVecEnv(num_envs=2, num_agents=3)
    cfg = MAPPOConfig(
        num_steps=4,
        num_epochs=1,
        minibatch_size=4,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        learning_rate_schedule="constant",
        device="cpu",
    )
    agent = MAPPO(
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

    checkpoint = tmp_path / "mappo.pt"
    agent.save(str(checkpoint))
    agent.load(str(checkpoint))

    result = agent.learn(num_iterations=1)
    assert result["final_iteration"] == 0
    expected_timesteps = cfg.num_steps * env.num_envs * len(env.possible_agents) * 2
    assert result["total_timesteps"] == expected_timesteps
    assert len(agent.episode_rewards) > 0


def test_ippo_uses_local_observations_for_critics(tmp_path):
    """IPPO should train decentralized critics from per-agent observations."""
    env = DummyDroneTransportVecEnv(num_envs=2, num_agents=2)
    cfg = IPPOConfig(
        num_steps=4,
        num_epochs=1,
        minibatch_size=4,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        learning_rate_schedule="constant",
        device="cpu",
    )
    agent = IPPO(
        env=env,
        cfg=cfg,
        actor_class=MLPActor,
        critic_class=MLPCritic,
        device=torch.device("cpu"),
    )

    assert not agent.cfg.centralized_critic
    assert agent.rollout_buffer.states is None
    for agent_id in env.possible_agents:
        assert agent.critics[agent_id].obs_space == env.observation_spaces[agent_id]

    rollout_stats = agent.collect_rollout()
    update_stats = agent.update()
    assert "rollout/mean_reward" in rollout_stats
    assert "train/policy_loss" in update_stats

    checkpoint = tmp_path / "ippo.pt"
    agent.save(str(checkpoint))
    agent.load(str(checkpoint))


def test_ippo_rejects_centralized_critic():
    """IPPO config should not allow centralized critic mode."""
    cfg = IPPOConfig(centralized_critic=True)
    try:
        IPPO(
            possible_agents=["agent_0"],
            observation_spaces={
                "agent_0": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
            },
            action_spaces={
                "agent_0": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
            },
            cfg=cfg,
            actor_class=MLPActor,
            critic_class=MLPCritic,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "centralized_critic=False" in str(exc)
    else:
        raise AssertionError("IPPO should reject centralized_critic=True")


def test_multiagent_runner_logs_and_checkpoints(tmp_path):
    """MultiAgentRunner should log TensorBoard events and save checkpoints."""
    env = DummyDroneTransportVecEnv(num_envs=2, num_agents=2)
    cfg = MAPPOConfig(
        num_steps=4,
        num_epochs=1,
        minibatch_size=4,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        learning_rate_schedule="constant",
        log_interval=1,
        save_interval=1,
        extra_log_keys=["log"],
        log_train_metrics_vs_iteration=True,
        log_episode_metrics_vs_iteration=True,
        log_detailed_rollout_stats=True,
        device="cpu",
    )
    agent = MAPPO(
        env=env,
        cfg=cfg,
        actor_class=MLPActor,
        critic_class=MLPCritic,
        device=torch.device("cpu"),
    )
    runner = MultiAgentRunner(
        env=env,
        agent=agent,
        cfg=cfg,
        log_dir=str(tmp_path / "logs"),
        save_dir=str(tmp_path / "ckpts"),
        device=torch.device("cpu"),
    )

    result = runner.learn(num_iterations=1)

    assert result["final_iteration"] == 0
    assert (tmp_path / "ckpts" / "checkpoint_0.pt").exists()
    assert (tmp_path / "ckpts" / "checkpoint_final.pt").exists()
    event_files = list((tmp_path / "logs").rglob("events.out.tfevents.*"))
    assert event_files
    assert not runner.log_buffers["/extra/log/payload_error"]
