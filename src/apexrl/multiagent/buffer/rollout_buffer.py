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

"""Rollout storage for on-policy multi-agent algorithms."""

from __future__ import annotations

from typing import Any

import torch

from apexrl.multiagent.typing import AgentID, MultiAgentAction, MultiAgentObs
from apexrl.utils import (
    Observation,
    allocate_observation_storage,
    flatten_time_env_observation,
    observation_set_index,
)


class MultiAgentRolloutBuffer:
    """Per-agent rollout buffer with optional centralized state storage."""

    def __init__(
        self,
        *,
        possible_agents: list[AgentID],
        num_envs: int,
        num_steps: int,
        obs_specs: dict[AgentID, Any],
        action_shapes: dict[AgentID, tuple[int, ...]],
        action_dtypes: dict[AgentID, torch.dtype],
        device: torch.device,
        state_spec: Any | None = None,
    ):
        """Initialize multi-agent rollout storage."""
        self.possible_agents = list(possible_agents)
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_specs = obs_specs
        self.action_shapes = action_shapes
        self.action_dtypes = action_dtypes
        self.device = device
        self.state_spec = state_spec

        self.observations = {
            agent_id: allocate_observation_storage(
                (num_steps, num_envs),
                obs_specs[agent_id],
                device=device,
                dtype=torch.float32,
            )
            for agent_id in self.possible_agents
        }
        self.states = (
            allocate_observation_storage(
                (num_steps, num_envs),
                state_spec,
                device=device,
                dtype=torch.float32,
            )
            if state_spec is not None
            else None
        )
        self.actions = {
            agent_id: torch.zeros(
                (num_steps, num_envs, *action_shapes[agent_id]),
                device=device,
                dtype=action_dtypes[agent_id],
            )
            for agent_id in self.possible_agents
        }
        self.rewards = {
            agent_id: torch.zeros(num_steps, num_envs, device=device)
            for agent_id in self.possible_agents
        }
        self.dones = {
            agent_id: torch.zeros(num_steps, num_envs, device=device)
            for agent_id in self.possible_agents
        }
        self.values = {
            agent_id: torch.zeros(num_steps, num_envs, device=device)
            for agent_id in self.possible_agents
        }
        self.log_probs = {
            agent_id: torch.zeros(num_steps, num_envs, device=device)
            for agent_id in self.possible_agents
        }
        self.advantages = {
            agent_id: torch.zeros(num_steps, num_envs, device=device)
            for agent_id in self.possible_agents
        }
        self.returns = {
            agent_id: torch.zeros(num_steps, num_envs, device=device)
            for agent_id in self.possible_agents
        }
        self.step = 0

    def add(
        self,
        *,
        observations: MultiAgentObs,
        state: Observation | None,
        actions: MultiAgentAction,
        rewards: dict[AgentID, torch.Tensor],
        dones: dict[AgentID, torch.Tensor],
        values: dict[AgentID, torch.Tensor],
        log_probs: dict[AgentID, torch.Tensor],
    ) -> None:
        """Append one vectorized multi-agent transition."""
        if self.step >= self.num_steps:
            raise ValueError(f"Rollout buffer is full (capacity: {self.num_steps})")

        for agent_id in self.possible_agents:
            observation_set_index(
                self.observations[agent_id],
                self.step,
                observations[agent_id],
            )
            self.actions[agent_id][self.step].copy_(actions[agent_id])
            self.rewards[agent_id][self.step].copy_(rewards[agent_id])
            self.dones[agent_id][self.step].copy_(dones[agent_id].float())
            self.values[agent_id][self.step].copy_(values[agent_id])
            self.log_probs[agent_id][self.step].copy_(log_probs[agent_id])

        if self.states is not None and state is not None:
            observation_set_index(self.states, self.step, state)

        self.step += 1

    def compute_returns_and_advantages(
        self,
        *,
        last_values: dict[AgentID, torch.Tensor],
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute per-agent GAE returns and advantages."""
        for agent_id in self.possible_agents:
            advantages = torch.zeros_like(self.rewards[agent_id])
            last_gae = torch.zeros(self.num_envs, device=self.device)
            rewards = self.rewards[agent_id]
            dones = self.dones[agent_id]
            values = self.values[agent_id]

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_values = last_values[agent_id]
                    next_non_terminal = 1.0 - dones[t]
                else:
                    next_values = values[t + 1]
                    next_non_terminal = 1.0 - dones[t]

                delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            self.advantages[agent_id] = advantages
            self.returns[agent_id] = advantages + values

    def normalize_advantages(self) -> None:
        """Normalize advantages across all agents and rollout samples."""
        all_advantages = torch.cat(
            [self.advantages[agent_id].reshape(-1) for agent_id in self.possible_agents]
        )
        mean = all_advantages.mean()
        std = all_advantages.std()
        for agent_id in self.possible_agents:
            self.advantages[agent_id] = (self.advantages[agent_id] - mean) / (
                std + 1e-8
            )

    def get_agent_data(self, agent_id: AgentID) -> dict[str, Any]:
        """Return flattened rollout data for a single agent."""
        total = self.num_steps * self.num_envs
        action_shape = self.action_shapes[agent_id]
        if action_shape:
            actions = self.actions[agent_id].reshape(total, *action_shape)
        else:
            actions = self.actions[agent_id].reshape(total)

        return {
            "observations": flatten_time_env_observation(self.observations[agent_id]),
            "states": (
                flatten_time_env_observation(self.states)
                if self.states is not None
                else None
            ),
            "actions": actions,
            "old_log_probs": self.log_probs[agent_id].reshape(total),
            "advantages": self.advantages[agent_id].reshape(total),
            "returns": self.returns[agent_id].reshape(total),
            "values": self.values[agent_id].reshape(total),
        }

    def clear(self) -> None:
        """Reset write position without reallocating storage."""
        self.step = 0

    def get_state_dict(self) -> dict[str, Any]:
        """Return raw storage for diagnostics and tests."""
        return {
            "observations": self.observations,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "returns": self.returns,
            "step": self.step,
        }
