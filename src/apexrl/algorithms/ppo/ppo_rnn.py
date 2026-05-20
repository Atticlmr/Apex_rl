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

"""Recurrent PPO implementation.

``RecurrentPPO`` keeps actor and critic hidden state while collecting rollouts
and optimizes on fixed-length sequence minibatches. The PPO objective is the
same clipped surrogate objective used by :class:`apexrl.algorithms.ppo.PPO`;
the difference is data layout and recurrent state handling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from apexrl.algorithms.ppo.config_rnn import RecurrentPPOConfig
from apexrl.algorithms.ppo.ppo import PPO
from apexrl.buffer.rollout_rnn_buffer import RolloutRNNBuffer
from apexrl.models.base import Actor, Critic
from apexrl.utils import observation_index, space_to_spec


class RecurrentPPO(PPO):
    """PPO variant that trains actor and critic on contiguous sequences.

    Custom recurrent networks can be supplied through the same
    ``actor_class`` / ``critic_class`` constructor arguments used by ``PPO``.
    The actor must provide ``get_initial_state()``, ``hidden_state_shape``,
    ``act()`` and ``evaluate()``. The critic must provide
    ``get_initial_state()``, ``hidden_state_shape`` and ``get_value_rnn()``.
    Hidden states are carried across rollout boundaries and reset only for
    environments whose episodes end.
    """

    cfg: RecurrentPPOConfig
    rollout_buffer: RolloutRNNBuffer

    def __init__(
        self,
        *args,
        cfg: RecurrentPPOConfig | None = None,
        **kwargs,
    ):
        """Initialize recurrent PPO.

        Args:
            *args: Positional arguments forwarded to :class:`PPO`.
            cfg: Recurrent PPO configuration. Defaults to
                :class:`RecurrentPPOConfig`.
            **kwargs: Keyword arguments forwarded to :class:`PPO`, including
                ``env``, ``actor_class``, ``critic_class``, ``actor_cfg`` and
                ``critic_cfg``.
        """
        super().__init__(*args, cfg=cfg or RecurrentPPOConfig(), **kwargs)
        self._validate_recurrent_modules()
        self.rollout_buffer = RolloutRNNBuffer(
            num_envs=self.num_envs,
            num_steps=self.cfg.num_steps,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dtype=self.action_dtype,
            actor_hidden_shape=tuple(self.actor.hidden_state_shape),
            critic_hidden_shape=tuple(self.critic.hidden_state_shape),
            device=self.device,
            privileged_obs_shape=(
                space_to_spec(self.critic_obs_space)
                if self.cfg.use_asymmetric and self.critic_obs_space is not None
                else None
            ),
        )
        self.actor_hidden_state: torch.Tensor | None = None
        self.critic_hidden_state: torch.Tensor | None = None
        self.prev_dones = torch.ones(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

    def _build_actor_cfg(self, actor_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Merge recurrent PPO config defaults into actor configuration."""
        merged = super()._build_actor_cfg(actor_cfg)
        merged.update(
            {
                "rnn_hidden_size": self.cfg.rnn_hidden_size,
                "rnn_num_layers": self.cfg.rnn_num_layers,
            }
        )
        if self.cfg.rnn_feature_dim is not None:
            merged["rnn_feature_dim"] = self.cfg.rnn_feature_dim
        return merged

    def _build_critic_cfg(self, critic_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Merge recurrent PPO config defaults into critic configuration."""
        merged = super()._build_critic_cfg(critic_cfg)
        merged.update(
            {
                "rnn_hidden_size": self.cfg.rnn_hidden_size,
                "rnn_num_layers": self.cfg.rnn_num_layers,
            }
        )
        if self.cfg.rnn_feature_dim is not None:
            merged["rnn_feature_dim"] = self.cfg.rnn_feature_dim
        return merged

    def _validate_recurrent_modules(self) -> None:
        """Validate that actor and critic expose the recurrent PPO interface."""
        required_actor = ("get_initial_state", "hidden_state_shape", "act", "evaluate")
        required_critic = ("get_initial_state", "hidden_state_shape", "get_value_rnn")
        self._validate_recurrent_module(self.actor, required_actor, "actor")
        self._validate_recurrent_module(self.critic, required_critic, "critic")

    @staticmethod
    def _validate_recurrent_module(
        module: Actor | Critic,
        required: tuple[str, ...],
        name: str,
    ) -> None:
        """Validate a recurrent module and raise a clear initialization error."""
        missing = [attr for attr in required if not hasattr(module, attr)]
        if missing:
            raise TypeError(
                f"RecurrentPPO requires recurrent {name} modules; "
                f"missing {missing} on {type(module).__name__}"
            )

    def collect_rollout(
        self,
        extras_callback: (
            Callable[
                [dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor],
                None,
            ]
            | None
        ) = None,
    ) -> dict[str, float]:
        """Collect a recurrent PPO rollout and store hidden states.

        Args:
            extras_callback: Optional callback used by runners to consume
                environment extras for logging.

        Returns:
            Rollout statistics such as mean reward and mean value.
        """
        self.actor.eval()
        self.critic.eval()
        self.rollout_buffer.clear()

        obs = self._to_tensor_observation(self.env.get_observations())
        obs, privileged_obs = self._split_observations(obs)

        actor_hidden, critic_hidden = self._get_rollout_hidden_states()
        prev_dones = self.prev_dones.to(self.device)
        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, device=self.device)
        completed_episodes = 0
        total_reward = 0.0

        for _ in range(self.cfg.num_steps):
            masks = (~prev_dones).float().view(1, self.num_envs, 1)
            with torch.no_grad():
                actions, log_probs, next_actor_hidden = self.actor.act(
                    obs,
                    hidden_state=actor_hidden,
                    masks=masks,
                    deterministic=False,
                )
                critic_obs = (
                    privileged_obs
                    if self.cfg.use_asymmetric and privileged_obs is not None
                    else obs
                )
                values, next_critic_hidden, _ = self.critic.get_value_rnn(
                    critic_obs,
                    hidden_state=critic_hidden,
                    masks=masks,
                )
                values = values.squeeze(0)

            next_obs, rewards, dones, extras = self.env.step(actions)
            next_obs = self._to_tensor_observation(next_obs)
            next_obs, next_privileged_obs = self._split_observations(next_obs)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device).bool()

            episode_rewards += rewards
            episode_lengths += 1

            terminated = extras.get("terminated", None)
            truncated = extras.get("truncated", extras.get("time_outs", None))
            if terminated is None:
                terminated = dones
                if isinstance(truncated, torch.Tensor):
                    terminated = dones & ~truncated.to(self.device).bool()
            terminated = self._to_bool_tensor(terminated, dones)
            truncated = self._to_bool_tensor(truncated, dones, default=False)
            episode_ends = dones

            if truncated.any():
                final_obs = extras.get("final_observation")
                if final_obs is not None:
                    final_obs = self._to_tensor_observation(final_obs)
                    final_obs, final_privileged_obs = self._split_observations(
                        final_obs
                    )
                    critic_final_obs = (
                        final_privileged_obs
                        if self.cfg.use_asymmetric and final_privileged_obs is not None
                        else final_obs
                    )
                    with torch.no_grad():
                        final_values, _, _ = self.critic.get_value_rnn(
                            observation_index(critic_final_obs, truncated),
                            hidden_state=self._index_hidden(
                                next_critic_hidden,
                                truncated,
                            ),
                            masks=torch.ones(
                                1,
                                int(truncated.sum().item()),
                                1,
                                device=self.device,
                            ),
                        )
                    rewards = rewards.clone()
                    rewards[truncated] += self.cfg.gamma * final_values.squeeze(0)

            if extras_callback is not None:
                extras_callback(extras, episode_ends, terminated, episode_rewards)

            if episode_ends.any():
                completed_indices = torch.where(episode_ends)[0]
                for idx in completed_indices:
                    self.episode_rewards.append(episode_rewards[idx].item())
                    self.episode_lengths.append(episode_lengths[idx].item())
                    total_reward += episode_rewards[idx].item()
                    completed_episodes += 1
                episode_rewards = episode_rewards * (~episode_ends).float()
                episode_lengths = episode_lengths * (~episode_ends).float()

            self.rollout_buffer.add(
                observations=obs,
                privileged_observations=(
                    privileged_obs if self.cfg.use_asymmetric else None
                ),
                actions=actions,
                rewards=rewards,
                dones=episode_ends.float(),
                values=values,
                log_probs=log_probs,
                actor_hidden_states=actor_hidden,
                critic_hidden_states=critic_hidden,
            )

            actor_hidden = self._reset_hidden(next_actor_hidden, episode_ends)
            critic_hidden = self._reset_hidden(next_critic_hidden, episode_ends)
            prev_dones = episode_ends
            obs = next_obs
            privileged_obs = next_privileged_obs

        final_masks = (~prev_dones).float().view(1, self.num_envs, 1)
        with torch.no_grad():
            critic_obs = (
                privileged_obs
                if self.cfg.use_asymmetric and privileged_obs is not None
                else obs
            )
            last_values, _, _ = self.critic.get_value_rnn(
                critic_obs,
                hidden_state=critic_hidden,
                masks=final_masks,
            )
            last_values = last_values.squeeze(0)

        self.rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        if self.cfg.normalize_advantages:
            advantages = self.rollout_buffer.advantages
            self.rollout_buffer.advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        self.total_timesteps += self.cfg.num_steps * self.num_envs
        self.actor_hidden_state = actor_hidden.detach()
        self.critic_hidden_state = critic_hidden.detach()
        self.prev_dones = prev_dones.detach()

        stats = {
            "rollout/mean_reward": rewards.mean().item(),
            "rollout/mean_value": self.rollout_buffer.values.mean().item(),
        }
        if completed_episodes > 0:
            stats["rollout/mean_episode_reward"] = total_reward / completed_episodes
            stats["rollout/completed_episodes"] = completed_episodes
        return stats

    def update(self) -> dict[str, float]:
        """Update recurrent policy and value function on sequence minibatches.

        Returns:
            Training metrics including policy loss, value loss, entropy,
            approximate KL, clip fraction, gradient norms and learning rate.
        """
        self.actor.train()
        self.critic.train()

        minibatch_size = self.cfg.get_recurrent_minibatch_size(self.num_envs)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_actor_grad_norm = torch.tensor(0.0, device=self.device)
        total_critic_grad_norm = torch.tensor(0.0, device=self.device)
        num_updates = 0
        early_stopped = False

        for _ in range(self.cfg.num_epochs):
            batches = self.rollout_buffer.get_sequence_minibatches(
                sequence_length=self.cfg.sequence_length,
                minibatch_size=minibatch_size,
                shuffle=True,
            )
            for batch in batches:
                obs = batch["observations"]
                privileged_obs = batch["privileged_observations"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]
                masks = self._start_masks(batch["dones"])

                log_probs, entropy, _ = self.actor.evaluate(
                    obs,
                    actions,
                    hidden_state=batch["actor_hidden_states"].detach(),
                    masks=masks,
                )
                critic_obs = privileged_obs if privileged_obs is not None else obs
                values, _, _ = self.critic.get_value_rnn(
                    critic_obs,
                    hidden_state=batch["critic_hidden_states"].detach(),
                    masks=masks,
                )

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.cfg.clip_range_vf is not None:
                    value_pred_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.cfg.clip_range_vf,
                        self.cfg.clip_range_vf,
                    )
                    value_loss1 = (values - returns).pow(2)
                    value_loss2 = (value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * nn.functional.mse_loss(values, returns)

                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        float("inf"),
                    )
                    critic_grad_norm = nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        float("inf"),
                    )
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                else:
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        float("inf"),
                    )
                    critic_grad_norm = nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        float("inf"),
                    )
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        self.cfg.max_grad_norm,
                    )
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.cfg.max_grad_norm,
                    )
                    self.policy_optimizer.step()
                    self.value_optimizer.step()

                with torch.no_grad():
                    ratio_clamped = torch.clamp(ratio, min=1e-8, max=10.0)
                    approx_kl = ((ratio_clamped - 1) - torch.log(ratio_clamped)).mean()
                    clip_fraction = (
                        ((ratio - 1).abs() > self.cfg.clip_range).float().mean()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                total_actor_grad_norm += actor_grad_norm
                total_critic_grad_norm += critic_grad_norm
                num_updates += 1

                if (
                    self.cfg.target_kl is not None
                    and approx_kl.item() > self.cfg.target_kl
                ):
                    early_stopped = True
                    break
            if early_stopped:
                break

        if num_updates == 0:
            raise RuntimeError("RecurrentPPO update produced no minibatches")

        avg_actor_grad_norm = (total_actor_grad_norm / num_updates).item()
        avg_critic_grad_norm = (total_critic_grad_norm / num_updates).item()
        avg_total_grad_norm = (avg_actor_grad_norm**2 + avg_critic_grad_norm**2) ** 0.5
        return {
            "train/policy_loss": total_policy_loss / num_updates,
            "train/value_loss": total_value_loss / num_updates,
            "train/entropy_loss": total_entropy_loss / num_updates,
            "train/approx_kl": total_approx_kl / num_updates,
            "train/clip_fraction": total_clip_fraction / num_updates,
            "train/learning_rate": self.get_current_lr(),
            "train/actor_grad_norm": avg_actor_grad_norm,
            "train/critic_grad_norm": avg_critic_grad_norm,
            "train/total_grad_norm": avg_total_grad_norm,
            "train/early_stopped": float(early_stopped),
        }

    def eval(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate the recurrent policy while carrying hidden state.

        Args:
            num_episodes: Number of completed episodes to evaluate.

        Returns:
            Reward summary with mean, standard deviation, minimum and maximum.
        """
        self.actor.eval()
        self.critic.eval()

        obs = self._to_tensor_observation(self.env.reset())
        obs, _ = self._split_observations(obs)
        hidden = self.actor.get_initial_state(self.num_envs, self.device)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        episode_rewards = []
        current_rewards = torch.zeros(self.num_envs, device=self.device)
        episodes_completed = 0

        while episodes_completed < num_episodes:
            masks = (~dones).float().view(1, self.num_envs, 1)
            with torch.no_grad():
                actions, _, hidden = self.actor.act(
                    obs,
                    hidden_state=hidden,
                    masks=masks,
                    deterministic=True,
                )
            next_obs, rewards, dones, _ = self.env.step(actions)
            next_obs = self._to_tensor_observation(next_obs)
            next_obs, _ = self._split_observations(next_obs)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device).bool()
            hidden = self._reset_hidden(hidden, dones)
            current_rewards += rewards

            if dones.any():
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    if episodes_completed < num_episodes:
                        episode_rewards.append(current_rewards[idx].item())
                        episodes_completed += 1
                current_rewards = current_rewards * (~dones).float()
            obs = next_obs

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        return {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": (
                sum((reward - mean_reward) ** 2 for reward in episode_rewards)
                / len(episode_rewards)
            )
            ** 0.5,
            "eval/min_reward": min(episode_rewards),
            "eval/max_reward": max(episode_rewards),
        }

    @staticmethod
    def _reset_hidden(
        hidden_state: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Clear hidden states for environments whose episodes ended."""
        return hidden_state * (~dones).float().view(1, -1, 1)

    def _get_rollout_hidden_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return persistent actor and critic hidden states for rollout start."""
        if (
            self.actor_hidden_state is None
            or self.actor_hidden_state.shape[1] != self.num_envs
        ):
            self.actor_hidden_state = self.actor.get_initial_state(
                self.num_envs,
                self.device,
            )
        if (
            self.critic_hidden_state is None
            or self.critic_hidden_state.shape[1] != self.num_envs
        ):
            self.critic_hidden_state = self.critic.get_initial_state(
                self.num_envs,
                self.device,
            )
        return self.actor_hidden_state.detach(), self.critic_hidden_state.detach()

    @staticmethod
    def _index_hidden(
        hidden_state: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Index a GRU hidden state along the environment/batch dimension."""
        return hidden_state[:, index, :]

    @staticmethod
    def _start_masks(dones: torch.Tensor) -> torch.Tensor:
        """Build sequence masks that reset after previous-step done flags."""
        masks = torch.ones((*dones.shape, 1), device=dones.device)
        if dones.shape[0] > 1:
            masks[1:] = (~dones[:-1].bool()).float().unsqueeze(-1)
        return masks
