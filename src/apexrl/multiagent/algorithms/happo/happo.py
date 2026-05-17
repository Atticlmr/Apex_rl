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

"""Heterogeneous-Agent PPO (HAPPO) implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces

from apexrl.models import Actor, Critic
from apexrl.multiagent.algorithms.happo.config import HAPPOConfig
from apexrl.multiagent.algorithms.mappo import MAPPO
from apexrl.multiagent.envs import MultiAgentVecEnv
from apexrl.multiagent.typing import AgentID
from apexrl.utils import observation_batch_size, observation_index


class HAPPO(MAPPO):
    """HAPPO with sequential policy updates.

    The class reuses MAPPO rollout, storage, logging and checkpointing. During
    update, agents are optimized sequentially. The policy loss for each agent is
    weighted by the product of new/old policy ratios from agents that have
    already been updated in the current order.
    """

    def __init__(
        self,
        *,
        env: MultiAgentVecEnv | None = None,
        cfg: HAPPOConfig | None = None,
        possible_agents: list[AgentID] | None = None,
        models: dict[AgentID, dict[str, nn.Module]] | None = None,
        observation_spaces: dict[AgentID, spaces.Space] | None = None,
        action_spaces: dict[AgentID, spaces.Space] | None = None,
        state_space: spaces.Space | None = None,
        actor_class: type[Actor] | None = None,
        critic_class: type[Critic] | None = None,
        actor_cfg: dict[str, Any] | None = None,
        critic_cfg: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        happo_cfg = cfg if cfg is not None else HAPPOConfig()
        if happo_cfg.share_actor:
            raise ValueError("HAPPO requires share_actor=False")
        super().__init__(
            env=env,
            cfg=happo_cfg,
            possible_agents=possible_agents,
            models=models,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            state_space=state_space,
            actor_class=actor_class,
            critic_class=critic_class,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
            device=device,
        )
        self._validate_separate_actors()

    def _validate_separate_actors(self) -> None:
        """Reject reused actor instances in HAPPO."""
        actor_ids = {id(self.actors[agent_id]) for agent_id in self.possible_agents}
        if len(actor_ids) != len(self.possible_agents):
            raise ValueError("HAPPO requires a separate actor instance per agent")

    def update(self) -> dict[str, float]:
        """Update agents sequentially with HAPPO correction factors."""
        if self.rollout_buffer is None:
            raise ValueError("No rollout buffer is available")
        for module in self._unique_modules():
            module.train()

        minibatch_size = self.cfg.get_minibatch_size(self.rollout_buffer.num_envs)
        agent_data = {
            agent_id: self.rollout_buffer.get_agent_data(agent_id)
            for agent_id in self.possible_agents
        }

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_grad_norm = 0.0
        total_correction = 0.0
        num_updates = 0
        early_stopped = False

        for _ in range(self.cfg.num_epochs):
            ordered_agents = self._agent_update_order()
            batch_size = observation_batch_size(
                agent_data[ordered_agents[0]]["observations"]
            )
            indices = torch.randperm(batch_size, device=self.device)

            for agent_position, agent_id in enumerate(ordered_agents):
                data = agent_data[agent_id]
                observations = data["observations"]
                states = data["states"]
                actions = data["actions"]
                old_log_probs = data["old_log_probs"]
                advantages = data["advantages"]
                returns = data["returns"]
                old_values = data["values"]
                previous_agents = ordered_agents[:agent_position]

                for start in range(0, batch_size, minibatch_size):
                    end = min(start + minibatch_size, batch_size)
                    mb_indices = indices[start:end]

                    mb_obs = observation_index(observations, mb_indices)
                    mb_state = (
                        observation_index(states, mb_indices)
                        if states is not None
                        else None
                    )
                    mb_actions = actions[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_old_values = old_values[mb_indices]

                    correction = self._happo_correction(
                        previous_agents,
                        agent_data,
                        mb_indices,
                    )
                    corrected_advantages = correction * mb_advantages

                    log_probs, entropy = self.actors[agent_id].evaluate(
                        mb_obs,
                        mb_actions,
                    )
                    critic_input = mb_state if self.cfg.centralized_critic else mb_obs
                    values = self.critics[agent_id].get_value(critic_input)

                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.cfg.clip_range,
                        1 + self.cfg.clip_range,
                    )
                    policy_loss = -torch.min(
                        ratio * corrected_advantages,
                        clipped_ratio * corrected_advantages,
                    ).mean()

                    if self.cfg.clip_range_vf is not None:
                        value_pred_clipped = mb_old_values + torch.clamp(
                            values - mb_old_values,
                            -self.cfg.clip_range_vf,
                            self.cfg.clip_range_vf,
                        )
                        value_loss1 = nn.functional.mse_loss(values, mb_returns)
                        value_loss2 = nn.functional.mse_loss(
                            value_pred_clipped,
                            mb_returns,
                        )
                        value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                    else:
                        value_loss = 0.5 * nn.functional.mse_loss(values, mb_returns)

                    entropy_loss = -entropy.mean()
                    loss = (
                        policy_loss
                        + self.cfg.vf_coef * value_loss
                        + self.cfg.ent_coef * entropy_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self._unique_parameters(),
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()

                    with torch.no_grad():
                        ratio_clamped = torch.clamp(ratio, min=1e-8, max=10.0)
                        approx_kl = (
                            (ratio_clamped - 1) - torch.log(ratio_clamped)
                        ).mean()
                        clip_fraction = (
                            ((ratio - 1).abs() > self.cfg.clip_range).float().mean()
                        )

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_approx_kl += approx_kl.item()
                    total_clip_fraction += clip_fraction.item()
                    total_grad_norm += float(grad_norm.item())
                    total_correction += float(correction.mean().item())
                    num_updates += 1

                    if (
                        self.cfg.target_kl is not None
                        and approx_kl.item() > self.cfg.target_kl
                    ):
                        early_stopped = True
                        break
                if early_stopped:
                    break
            if early_stopped:
                break

        if num_updates == 0:
            raise RuntimeError("HAPPO update produced no minibatches")

        return {
            "train/policy_loss": total_policy_loss / num_updates,
            "train/value_loss": total_value_loss / num_updates,
            "train/entropy_loss": total_entropy_loss / num_updates,
            "train/approx_kl": total_approx_kl / num_updates,
            "train/clip_fraction": total_clip_fraction / num_updates,
            "train/grad_norm": total_grad_norm / num_updates,
            "train/happo_correction": total_correction / num_updates,
            "train/learning_rate": self.get_current_lr(),
            "train/early_stopped": float(early_stopped),
        }

    def _agent_update_order(self) -> list[AgentID]:
        """Return the HAPPO agent update order for one epoch."""
        agents = list(self.possible_agents)
        if getattr(self.cfg, "shuffle_agent_order", True):
            permutation = torch.randperm(len(agents), device=self.device).tolist()
            return [agents[index] for index in permutation]
        return agents

    def _happo_correction(
        self,
        previous_agents: list[AgentID],
        agent_data: dict[AgentID, dict[str, Any]],
        mb_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute product of previous agents' updated policy ratios."""
        batch_size = mb_indices.shape[0]
        correction = torch.ones(batch_size, device=self.device)
        if not previous_agents:
            return correction

        with torch.no_grad():
            for previous_agent in previous_agents:
                data = agent_data[previous_agent]
                obs = observation_index(data["observations"], mb_indices)
                actions = data["actions"][mb_indices]
                old_log_probs = data["old_log_probs"][mb_indices]
                log_probs, _ = self.actors[previous_agent].evaluate(obs, actions)
                correction = correction * torch.exp(log_probs - old_log_probs)

        return correction.detach()
