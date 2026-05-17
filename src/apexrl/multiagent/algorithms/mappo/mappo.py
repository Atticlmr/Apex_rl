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

"""Multi-Agent PPO (MAPPO) implementation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces

from apexrl.models.base import Actor, ContinuousActor, Critic, DiscreteActor
from apexrl.multiagent.algorithms.mappo.config import MAPPOConfig
from apexrl.multiagent.buffer import MultiAgentRolloutBuffer
from apexrl.multiagent.envs import MultiAgentVecEnv
from apexrl.multiagent.typing import AgentID, MultiAgentAction, MultiAgentObs
from apexrl.multiagent.utils import (
    action_space_shape_dtype,
    merge_terminated_truncated,
    multiagent_to_tensor,
    observation_spec_from_space,
    state_to_tensor,
    to_agent_tensor_dict,
    validate_agent_ids,
)
from apexrl.optimizers import build_optimizer
from apexrl.utils import (
    Observation,
    observation_batch_size,
    observation_index,
)


class MAPPO:
    """Multi-Agent PPO with centralized critic support.

    The API follows the same high-level shape as skrl's multi-agent agents:
    users provide ``possible_agents`` plus per-agent ``models``,
    ``observation_spaces``, ``state_space`` and ``action_spaces``.
    """

    def __init__(
        self,
        *,
        env: MultiAgentVecEnv | None = None,
        cfg: MAPPOConfig | None = None,
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
        """Initialize MAPPO."""
        self.env = env
        self.cfg = cfg if cfg is not None else MAPPOConfig()

        if device is None:
            if self.cfg.device == "auto":
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device(self.cfg.device)
        else:
            self.device = device

        self.possible_agents = list(
            possible_agents or getattr(env, "possible_agents", None) or []
        )
        if not self.possible_agents:
            raise ValueError("possible_agents must be provided")

        self.observation_spaces = observation_spaces or getattr(
            env, "observation_spaces", None
        )
        self.action_spaces = action_spaces or getattr(env, "action_spaces", None)
        self.state_space = state_space or getattr(env, "state_space", None)
        if self.observation_spaces is None or self.action_spaces is None:
            raise ValueError("observation_spaces and action_spaces are required")

        validate_agent_ids(
            self.possible_agents,
            self.observation_spaces,
            name="observation_spaces",
        )
        validate_agent_ids(
            self.possible_agents,
            self.action_spaces,
            name="action_spaces",
        )
        self._validate_shared_model_spaces()

        self.num_envs = getattr(env, "num_envs", None)
        if self.num_envs is None and env is not None:
            raise ValueError("env.num_envs is required when env is provided")

        self.models = self._build_models(
            models=models,
            actor_class=actor_class,
            critic_class=critic_class,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
        )
        self.actors = {
            agent_id: self.models[agent_id]["policy"]
            for agent_id in self.possible_agents
        }
        self.critics = {
            agent_id: self.models[agent_id]["value"]
            for agent_id in self.possible_agents
        }
        self._validate_models()

        self.action_shapes = {}
        self.action_dtypes = {}
        for agent_id in self.possible_agents:
            shape, dtype = action_space_shape_dtype(self.action_spaces[agent_id])
            self.action_shapes[agent_id] = shape
            self.action_dtypes[agent_id] = dtype

        self.optimizer = build_optimizer(
            self.cfg.optimizer,
            lr=self.cfg.learning_rate,
            modules=self._unique_modules(),
        )
        self.rollout_buffer: MultiAgentRolloutBuffer | None = None
        if self.num_envs is not None:
            self.rollout_buffer = self._make_rollout_buffer(self.num_envs)

        self.iteration = 0
        self.total_timesteps = 0
        self.logger = None
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.episode_lengths: deque[float] = deque(maxlen=100)
        self._current_episode_rewards: torch.Tensor | None = None
        self._current_episode_lengths: torch.Tensor | None = None

    def _validate_shared_model_spaces(self) -> None:
        """Reject shared models when agent spaces are not homogeneous."""
        first_agent = self.possible_agents[0]
        first_obs_space = self.observation_spaces[first_agent]
        first_action_space = self.action_spaces[first_agent]
        for agent_id in self.possible_agents[1:]:
            if self.cfg.share_actor and (
                self.observation_spaces[agent_id] != first_obs_space
                or self.action_spaces[agent_id] != first_action_space
            ):
                raise ValueError(
                    "share_actor=True requires identical observation and action "
                    f"spaces. Agent {agent_id} differs from {first_agent}."
                )
            if (
                self.cfg.share_critic
                and not self.cfg.centralized_critic
                and self.observation_spaces[agent_id] != first_obs_space
            ):
                raise ValueError(
                    "share_critic=True with centralized_critic=False requires "
                    f"identical observation spaces. Agent {agent_id} differs from "
                    f"{first_agent}."
                )

    def _build_models(
        self,
        *,
        models: dict[AgentID, dict[str, nn.Module]] | None,
        actor_class: type[Actor] | None,
        critic_class: type[Critic] | None,
        actor_cfg: dict[str, Any] | None,
        critic_cfg: dict[str, Any] | None,
    ) -> dict[AgentID, dict[str, nn.Module]]:
        """Build or normalize per-agent policy/value models."""
        if models is not None:
            validate_agent_ids(self.possible_agents, models, name="models")
            return {
                agent_id: {
                    "policy": models[agent_id]["policy"].to(self.device),
                    "value": models[agent_id]["value"].to(self.device),
                }
                for agent_id in self.possible_agents
            }

        if actor_class is None or critic_class is None:
            raise ValueError("Provide either models or actor_class and critic_class")

        actor_cfg = self._build_actor_cfg(actor_cfg)
        critic_cfg = self._build_critic_cfg(critic_cfg)
        first_agent = self.possible_agents[0]
        built: dict[AgentID, dict[str, nn.Module]] = {}

        shared_actor = None
        shared_critic = None
        if self.cfg.share_actor:
            shared_actor = actor_class(
                obs_space=self.observation_spaces[first_agent],
                action_space=self.action_spaces[first_agent],
                cfg=actor_cfg,
            ).to(self.device)
        if self.cfg.share_critic:
            shared_critic = critic_class(
                obs_space=self._critic_space_for_agent(first_agent),
                cfg=critic_cfg,
            ).to(self.device)

        for agent_id in self.possible_agents:
            actor = shared_actor
            if actor is None:
                actor = actor_class(
                    obs_space=self.observation_spaces[agent_id],
                    action_space=self.action_spaces[agent_id],
                    cfg=actor_cfg,
                ).to(self.device)

            critic = shared_critic
            if critic is None:
                critic = critic_class(
                    obs_space=self._critic_space_for_agent(agent_id),
                    cfg=critic_cfg,
                ).to(self.device)

            built[agent_id] = {"policy": actor, "value": critic}

        return built

    def _build_actor_cfg(self, actor_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Merge config defaults into actor cfg."""
        merged = {
            "hidden_dims": list(self.cfg.actor_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
            "learn_std": not self.cfg.fixed_std,
            "init_std": self.cfg.std_value,
            "use_tanh_squash": self.cfg.use_tanh_squash,
            "min_log_std": self.cfg.min_log_std,
            "max_log_std": self.cfg.max_log_std,
        }
        if actor_cfg:
            merged.update(actor_cfg)
        return merged

    def _build_critic_cfg(self, critic_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Merge config defaults into critic cfg."""
        merged = {
            "hidden_dims": list(self.cfg.critic_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
        }
        if critic_cfg:
            merged.update(critic_cfg)
        return merged

    def _critic_space_for_agent(self, agent_id: AgentID) -> spaces.Space:
        """Return centralized or local critic input space."""
        if self.cfg.centralized_critic:
            if self.state_space is None:
                raise ValueError("state_space is required when centralized_critic=True")
            return self.state_space
        return self.observation_spaces[agent_id]

    def _validate_models(self) -> None:
        """Validate model interfaces and action space compatibility."""
        for agent_id in self.possible_agents:
            actor = self.actors[agent_id]
            critic = self.critics[agent_id]
            if not isinstance(actor, (ContinuousActor, DiscreteActor)):
                raise TypeError(
                    "MAPPO currently supports ContinuousActor or DiscreteActor, "
                    f"got {type(actor)} for agent {agent_id}"
                )
            if not isinstance(critic, Critic):
                raise TypeError(f"Value model for {agent_id} must inherit Critic")

    def _unique_modules(self) -> list[nn.Module]:
        """Return unique policy/value modules for optimizer construction."""
        modules = []
        seen = set()
        for agent_id in self.possible_agents:
            for module in (self.actors[agent_id], self.critics[agent_id]):
                module_id = id(module)
                if module_id in seen:
                    continue
                seen.add(module_id)
                modules.append(module)
        return modules

    def _make_rollout_buffer(self, num_envs: int) -> MultiAgentRolloutBuffer:
        """Create rollout buffer from spaces."""
        obs_specs = {
            agent_id: observation_spec_from_space(self.observation_spaces[agent_id])
            for agent_id in self.possible_agents
        }
        state_spec = (
            observation_spec_from_space(self.state_space)
            if self.cfg.centralized_critic and self.state_space is not None
            else None
        )
        return MultiAgentRolloutBuffer(
            possible_agents=self.possible_agents,
            num_envs=num_envs,
            num_steps=self.cfg.num_steps,
            obs_specs=obs_specs,
            action_shapes=self.action_shapes,
            action_dtypes=self.action_dtypes,
            device=self.device,
            state_spec=state_spec,
        )

    def act(
        self,
        observations: MultiAgentObs,
        *,
        deterministic: bool = False,
    ) -> tuple[MultiAgentAction, dict[AgentID, torch.Tensor]]:
        """Sample decentralized actions from per-agent policies."""
        validate_agent_ids(self.possible_agents, observations, name="observations")
        actions = {}
        log_probs = {}
        for agent_id in self.possible_agents:
            action, log_prob = self.actors[agent_id].act(
                observations[agent_id],
                deterministic=deterministic,
            )
            actions[agent_id] = action
            log_probs[agent_id] = log_prob
        return actions, log_probs

    def _value_inputs(
        self,
        observations: MultiAgentObs,
        state: Observation | None,
    ) -> dict[AgentID, Observation]:
        """Return critic inputs keyed by agent."""
        if self.cfg.centralized_critic:
            if state is None:
                raise ValueError("Centralized critic requires env.get_state()")
            return {agent_id: state for agent_id in self.possible_agents}
        return observations

    def get_values(
        self,
        observations: MultiAgentObs,
        state: Observation | None,
    ) -> dict[AgentID, torch.Tensor]:
        """Evaluate value functions for each agent."""
        value_inputs = self._value_inputs(observations, state)
        return {
            agent_id: self.critics[agent_id].get_value(value_inputs[agent_id])
            for agent_id in self.possible_agents
        }

    def collect_rollout(
        self,
        extras_callback: (
            Callable[
                [
                    dict[str, Any],
                    dict[AgentID, torch.Tensor],
                    dict[AgentID, torch.Tensor],
                    dict[AgentID, torch.Tensor],
                ],
                None,
            ]
            | None
        ) = None,
    ) -> dict[str, float]:
        """Collect one MAPPO rollout from the configured environment."""
        if self.env is None:
            raise ValueError("collect_rollout requires env")
        if self.rollout_buffer is None:
            self.rollout_buffer = self._make_rollout_buffer(self.env.num_envs)

        for module in self._unique_modules():
            module.eval()
        self.rollout_buffer.clear()
        self._ensure_episode_trackers()

        obs = multiagent_to_tensor(self.env.get_observations(), self.device)
        state = state_to_tensor(self.env.get_state(), self.device)
        episode_reward_sum = 0.0
        completed_episodes = 0

        for _ in range(self.cfg.num_steps):
            with torch.no_grad():
                actions, log_probs = self.act(obs, deterministic=False)
                values = self.get_values(obs, state)

            step_out = self.env.step(actions)
            next_obs_raw, rewards_raw, terminated_raw, truncated_raw, extras = (
                self._normalize_step_output(step_out)
            )
            next_obs = multiagent_to_tensor(next_obs_raw, self.device)
            next_state = state_to_tensor(self.env.get_state(), self.device)
            rewards = to_agent_tensor_dict(
                rewards_raw,
                self.possible_agents,
                self.device,
                dtype=torch.float32,
            )
            terminated = to_agent_tensor_dict(
                terminated_raw,
                self.possible_agents,
                self.device,
                dtype=torch.bool,
            )
            truncated = to_agent_tensor_dict(
                truncated_raw,
                self.possible_agents,
                self.device,
                dtype=torch.bool,
            )
            dones = merge_terminated_truncated(
                terminated,
                truncated,
                self.possible_agents,
                self.device,
            )

            if self.cfg.shared_reward:
                team_reward = torch.stack(
                    [rewards[agent_id] for agent_id in self.possible_agents]
                ).mean(dim=0)
                rewards = {agent_id: team_reward for agent_id in self.possible_agents}

            mean_reward = torch.stack(
                [rewards[agent_id] for agent_id in self.possible_agents]
            ).mean(dim=0)
            mean_done = torch.stack(
                [dones[agent_id] for agent_id in self.possible_agents]
            ).any(dim=0)
            completed_episodes += self._record_episode_progress(
                mean_reward,
                mean_done,
            )
            if extras_callback is not None:
                extras_callback(extras, dones, terminated, rewards)

            self.rollout_buffer.add(
                observations=obs,
                state=state,
                actions=actions,
                rewards=rewards,
                dones=dones,
                values=values,
                log_probs=log_probs,
            )
            episode_reward_sum += float(mean_reward.mean().item())

            obs = next_obs
            state = next_state
            del extras

        with torch.no_grad():
            last_values = self.get_values(obs, state)
        self.rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        if self.cfg.normalize_advantages:
            self.rollout_buffer.normalize_advantages()

        self.total_timesteps += (
            self.cfg.num_steps * self.env.num_envs * len(self.possible_agents)
        )

        stats = {
            "rollout/mean_reward": episode_reward_sum / self.cfg.num_steps,
            "rollout/mean_value": float(
                torch.stack(
                    [
                        self.rollout_buffer.values[agent_id].mean()
                        for agent_id in self.possible_agents
                    ]
                )
                .mean()
                .item()
            ),
            "rollout/completed_episodes": float(completed_episodes),
        }
        if completed_episodes > 0:
            stats["episode/mean_reward"] = sum(self.episode_rewards) / len(
                self.episode_rewards
            )
            stats["episode/mean_length"] = sum(self.episode_lengths) / len(
                self.episode_lengths
            )
        return stats

    def _ensure_episode_trackers(self) -> None:
        """Initialize per-env episode trackers lazily."""
        if self.env is None:
            return
        if (
            self._current_episode_rewards is not None
            and self._current_episode_rewards.shape[0] == self.env.num_envs
        ):
            return
        self._current_episode_rewards = torch.zeros(
            self.env.num_envs,
            device=self.device,
        )
        self._current_episode_lengths = torch.zeros(
            self.env.num_envs,
            dtype=torch.int32,
            device=self.device,
        )

    def _record_episode_progress(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> int:
        """Record completed episode returns and lengths."""
        self._ensure_episode_trackers()
        self._current_episode_rewards += rewards
        self._current_episode_lengths += 1

        completed = int(dones.sum().item())
        if completed == 0:
            return 0

        done_indices = torch.where(dones)[0]
        for index in done_indices:
            self.episode_rewards.append(
                float(self._current_episode_rewards[index].item())
            )
            self.episode_lengths.append(
                float(self._current_episode_lengths[index].item())
            )

        active = (~dones).float()
        self._current_episode_rewards *= active
        self._current_episode_lengths *= (~dones).to(
            self._current_episode_lengths.dtype
        )
        return completed

    def _normalize_step_output(
        self,
        step_out: Any,
    ) -> tuple[
        MultiAgentObs,
        dict[AgentID, torch.Tensor],
        dict[AgentID, torch.Tensor],
        dict[AgentID, torch.Tensor],
        dict[str, Any],
    ]:
        """Accept both 4-tuple and 5-tuple multi-agent step outputs."""
        if len(step_out) == 5:
            return step_out
        if len(step_out) == 4:
            obs, rewards, dones, extras = step_out
            truncated = {
                agent_id: torch.zeros_like(dones[agent_id], dtype=torch.bool)
                for agent_id in self.possible_agents
            }
            return obs, rewards, dones, truncated, extras
        raise ValueError("MultiAgentVecEnv.step must return 4 or 5 values")

    def update(self) -> dict[str, float]:
        """Update all MAPPO policies and value functions from rollout storage."""
        if self.rollout_buffer is None:
            raise ValueError("No rollout buffer is available")
        for module in self._unique_modules():
            module.train()

        minibatch_size = self.cfg.get_minibatch_size(self.rollout_buffer.num_envs)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_grad_norm = 0.0
        num_updates = 0
        early_stopped = False

        for _ in range(self.cfg.num_epochs):
            for agent_id in self.possible_agents:
                data = self.rollout_buffer.get_agent_data(agent_id)
                observations = data["observations"]
                states = data["states"]
                actions = data["actions"]
                old_log_probs = data["old_log_probs"]
                advantages = data["advantages"]
                returns = data["returns"]
                old_values = data["values"]
                batch_size = observation_batch_size(observations)
                indices = torch.randperm(batch_size, device=self.device)

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

                    log_probs, entropy = self.actors[agent_id].evaluate(
                        mb_obs,
                        mb_actions,
                    )
                    critic_input = mb_state if self.cfg.centralized_critic else mb_obs
                    values = self.critics[agent_id].get_value(critic_input)

                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1 - self.cfg.clip_range,
                            1 + self.cfg.clip_range,
                        )
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

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
            raise RuntimeError("MAPPO update produced no minibatches")

        return {
            "train/policy_loss": total_policy_loss / num_updates,
            "train/value_loss": total_value_loss / num_updates,
            "train/entropy_loss": total_entropy_loss / num_updates,
            "train/approx_kl": total_approx_kl / num_updates,
            "train/clip_fraction": total_clip_fraction / num_updates,
            "train/grad_norm": total_grad_norm / num_updates,
            "train/learning_rate": self.get_current_lr(),
            "train/early_stopped": float(early_stopped),
        }

    def _unique_parameters(self) -> list[torch.nn.Parameter]:
        """Return unique trainable parameters from all models."""
        params = []
        seen = set()
        for module in self._unique_modules():
            for param in module.parameters():
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)
        return params

    def get_current_lr(self) -> float:
        """Return current optimizer learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def adjust_learning_rate(
        self,
        current_iteration: int,
        total_iterations: int,
    ) -> None:
        """Adjust learning rate for linear or adaptive schedules."""
        if self.cfg.learning_rate_schedule == "constant":
            return
        progress = current_iteration / max(total_iterations, 1)
        if self.cfg.learning_rate_schedule == "linear":
            new_lr = self.cfg.learning_rate * (1 - progress)
        elif self.cfg.learning_rate_schedule == "adaptive":
            new_lr = self.cfg.min_learning_rate + (
                self.cfg.max_learning_rate - self.cfg.min_learning_rate
            ) * (1 - progress)
        else:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr * group.get("_apexrl_lr_scale", 1.0)

    def learn(
        self,
        total_timesteps: int | None = None,
        num_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Train through the canonical multi-agent runner."""
        if self.env is None:
            raise ValueError("learn requires env")
        from apexrl.multiagent.runner import MultiAgentRunner

        runner = MultiAgentRunner(
            agent=self,
            env=self.env,
            cfg=self.cfg,
            log_dir=None,
            save_dir=None,
            device=self.device,
        )
        return runner.learn(
            total_timesteps=total_timesteps,
            num_iterations=num_iterations,
        )

    def eval(self) -> None:
        """Switch all MAPPO models to eval mode."""
        for module in self._unique_modules():
            module.eval()

    def train(self) -> None:
        """Switch all MAPPO models to train mode."""
        for module in self._unique_modules():
            module.train()

    def save(self, path: str) -> None:
        """Save MAPPO checkpoint."""
        checkpoint = {
            "models": {
                agent_id: {
                    "policy": self.actors[agent_id].state_dict(),
                    "value": self.critics[agent_id].state_dict(),
                }
                for agent_id in self.possible_agents
            },
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "config": self.cfg,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load MAPPO checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        for agent_id in self.possible_agents:
            self.actors[agent_id].load_state_dict(
                checkpoint["models"][agent_id]["policy"]
            )
            self.critics[agent_id].load_state_dict(
                checkpoint["models"][agent_id]["value"]
            )
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iteration = checkpoint.get("iteration", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
