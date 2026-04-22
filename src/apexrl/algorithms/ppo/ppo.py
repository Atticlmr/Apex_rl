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

"""PPO (Proximal Policy Optimization) algorithm implementation.

Supports custom Actor and Critic networks defined by users.

References:
    - Original paper: https://arxiv.org/abs/1707.06347
    - rsl_rl: https://github.com/leggedrobotics/rsl_rl
    - stable-baselines3: https://github.com/DLR-RM/stable-baselines3
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces

from apexrl.algorithms.ppo.config import PPOConfig
from apexrl.buffer.rollout_buffer import RolloutBuffer
from apexrl.envs.vecenv import VecEnv
from apexrl.models.base import Actor, ContinuousActor, Critic, DiscreteActor
from apexrl.optimizers import get_optimizer
from apexrl.utils import (
    actor_space_from_observation_space,
    clone_observation,
    critic_space_from_observation_space,
    observation_batch_size,
    observation_index,
    observation_to_tensor,
    space_to_spec,
    split_actor_critic_observations,
)
from apexrl.utils.logger import Logger


class PPO:
    """PPO algorithm for vectorized environments with custom networks.

    Users provide custom Actor and Critic classes, and PPO handles the
    training loop, advantage computation, and optimization.

    Example:
        >>> from apexrl.algorithms.ppo import PPO, PPOConfig
        >>> from apexrl.envs.vecenv import DummyVecEnv
        >>> from apexrl.models.mlp import MLPActor, MLPCritic
        >>> from gymnasium import spaces
        >>>
        >>> # Create environment
        >>> env = DummyVecEnv(num_envs=4096, num_obs=48, num_actions=12)
        >>>
        >>> # Define observation and action spaces
        >>> obs_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(48,))
        >>> action_space = spaces.Box(low=-1, high=1, shape=(12,))
        >>>
        >>> # Configure PPO
        >>> cfg = PPOConfig(num_steps=24, learning_rate=3e-4)
        >>>
        >>> # Create agent with custom networks
        >>> agent = PPO(
        ...     env=env,
        ...     cfg=cfg,
        ...     actor_class=MLPActor,
        ...     critic_class=MLPCritic,
        ...     obs_space=obs_space,
        ...     action_space=action_space,
        ...     actor_cfg={"hidden_dims": [256, 256, 256]},
        ...     critic_cfg={"hidden_dims": [256, 256, 256]},
        ... )
        >>>
        >>> # Train
        >>> agent.learn(total_timesteps=10_000_000)
    """

    def __init__(
        self,
        env: VecEnv,
        cfg: PPOConfig | None = None,
        actor_class: type[Actor] | None = None,
        critic_class: type[Critic] | None = None,
        obs_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        actor_cfg: dict[str, Any] | None = None,
        critic_cfg: dict[str, Any] | None = None,
        actor: Actor | None = None,
        critic: Critic | None = None,
        log_dir: str | None = None,
        device: torch.device | None = None,
    ):
        """Initialize PPO algorithm.

        Args:
            env: Vectorized environment implementing VecEnv interface.
            cfg: PPO configuration. Uses defaults if None.
            actor_class: Actor class to instantiate (e.g., MLPActor, CNNActor).
            critic_class: Critic class to instantiate (e.g., MLPCritic, CNNCritic).
            obs_space: Observation space (gymnasium.Space). Required if not providing
                pre-instantiated actor/critic.
            action_space: Action space (gymnasium.Space). Required if not providing
                pre-instantiated actor/critic.
            actor_cfg: Configuration dict passed to actor_class constructor.
            critic_cfg: Configuration dict passed to critic_class constructor.
            actor: Pre-instantiated actor network (alternative to actor_class).
            critic: Pre-instantiated critic network (alternative to critic_class).
            log_dir: Directory for TensorBoard logs.
            device: Device for training. Auto-detects if None.

        Note:
            Either provide (actor_class, critic_class, obs_space, action_space) OR
            provide (actor, critic) directly.
        """
        self.env = env
        self.cfg = cfg if cfg is not None else PPOConfig()

        # Device setup
        if device is None:
            if self.cfg.device == "auto":
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device(self.cfg.device)
        else:
            self.device = device

        # Get environment info
        self.num_envs = env.num_envs

        # Initialize actor and critic
        if actor is not None and critic is not None:
            # Use pre-instantiated networks
            self.actor = actor.to(self.device)
            self.critic = critic.to(self.device)
            self.obs_space = obs_space or getattr(actor, "obs_space", None)
            self.action_space = getattr(actor, "action_space", None)
        elif actor_class is not None and critic_class is not None:
            # Instantiate from classes
            if obs_space is None:
                obs_space = getattr(env, "observation_space_gym", None)
            if action_space is None:
                action_space = getattr(env, "action_space_gym", None)
            if obs_space is None or action_space is None:
                raise ValueError(
                    "obs_space and action_space are required when using "
                    "actor_class/critic_class"
                )

            self.full_obs_space = obs_space
            self.obs_space = actor_space_from_observation_space(obs_space)
            self.action_space = action_space
            actor_cfg = self._build_actor_cfg(actor_cfg)
            critic_cfg = self._build_critic_cfg(critic_cfg)

            # Create actor
            self.actor = actor_class(
                obs_space=self.obs_space,
                action_space=action_space,
                cfg=actor_cfg,
            ).to(self.device)

            # Create critic (may use privileged obs if asymmetric)
            critic_obs_space = (
                critic_space_from_observation_space(obs_space)
                or getattr(env, "privileged_obs_space", None)
                or self.obs_space
                if self.cfg.use_asymmetric
                else self.obs_space
            )
            self.critic_obs_space = critic_obs_space
            self.critic = critic_class(
                obs_space=critic_obs_space,
                cfg=critic_cfg,
            ).to(self.device)
        else:
            raise ValueError(
                "Must provide either (actor, critic) OR "
                "(actor_class, critic_class, obs_space, action_space)"
            )
        if not hasattr(self, "full_obs_space"):
            self.full_obs_space = self.obs_space
        if not hasattr(self, "critic_obs_space"):
            self.critic_obs_space = getattr(self.critic, "obs_space", self.obs_space)

        # Verify actor type
        if not isinstance(self.actor, (ContinuousActor, DiscreteActor)):
            raise TypeError(
                "PPO currently only supports ContinuousActor or "
                f"DiscreteActor, got {type(self.actor)}"
            )

        # Get action dimension
        if isinstance(self.action_space, spaces.Box):
            self.action_dim = (
                self.action_space.shape[0] if len(self.action_space.shape) > 0 else 1
            )
            self.action_shape = self.action_space.shape
            self.action_dtype = torch.float32
        elif isinstance(self.action_space, spaces.Discrete):
            self.action_dim = 1  # Discrete actions are single integers
            self.action_shape = ()
            self.action_dtype = torch.long
        else:
            raise NotImplementedError(
                f"Action space {type(self.action_space)} not supported"
            )

        # Initialize optimizer
        optimizer_cls = get_optimizer(self.cfg.optimizer)

        if self.cfg.use_policy_optimizer:
            # Separate optimizers for policy and value
            self.policy_optimizer = optimizer_cls(
                self.actor.parameters(),
                lr=self.cfg.policy_learning_rate,
            )
            self.value_optimizer = optimizer_cls(
                self.critic.parameters(),
                lr=self.cfg.value_learning_rate,
            )
            self.optimizer = None
        else:
            # Joint optimizer
            self.optimizer = optimizer_cls(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.cfg.learning_rate,
            )
            self.policy_optimizer = None
            self.value_optimizer = None

        # Initialize rollout buffer
        # Infer obs shape from obs_space
        self.obs_shape = self._get_obs_shape(self.obs_space)

        self.rollout_buffer = RolloutBuffer(
            num_envs=self.num_envs,
            num_steps=self.cfg.num_steps,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dtype=self.action_dtype,
            device=self.device,
            num_privileged_obs=getattr(env, "num_privileged_obs", 0)
            if self.cfg.use_asymmetric
            else 0,
            privileged_obs_shape=(
                space_to_spec(self.critic_obs_space)
                if self.cfg.use_asymmetric and self.critic_obs_space is not None
                else None
            ),
        )

        # Logging
        self.log_dir = log_dir
        self.logger = None
        if log_dir:
            self.logger = Logger.create(
                backend=self.cfg.logger_backend,
                experiment_name="ppo",
                log_dir=log_dir,
                **self.cfg.logger_kwargs,
            )
        self.iteration = 0
        self.total_timesteps = 0

        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []

        # Loss history vs iteration (fixed size deque to prevent memory leak)
        self.loss_history: dict[str, deque] | None = None
        self.loss_history_maxlen: int = 1000

    def _get_obs_shape(self, obs_space: spaces.Space | None) -> Any:
        """Get observation shape from space."""
        if obs_space is None:
            obs_buf = getattr(self.env, "obs_buf", None)
            if obs_buf is not None:
                return getattr(obs_buf, "shape", None) or (self.env.num_obs,)
            # Try to infer from env
            if hasattr(self.env, "num_obs"):
                return (self.env.num_obs,)
            return (1,)  # Fallback

        return space_to_spec(obs_space)

    def _split_observations(self, obs: Any) -> tuple[Any, Any | None]:
        """Split normalized observations into actor and critic branches."""
        actor_obs, critic_obs = split_actor_critic_observations(obs)
        return actor_obs, critic_obs

    def _build_actor_cfg(self, actor_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Merge PPO config defaults into actor configuration."""
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
        """Merge PPO config defaults into critic configuration."""
        merged = {
            "hidden_dims": list(self.cfg.critic_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
        }
        if critic_cfg:
            merged.update(critic_cfg)
        return merged

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
        """Collect a rollout from the environment.

        Args:
            extras_callback: Optional callback function called after each step with
                (extras, dones, true_dones, episode_rewards) to process environment
                extras such as reward components and custom metrics.

        Returns:
            Dictionary of rollout statistics.
        """
        self.actor.eval()
        self.critic.eval()
        self.rollout_buffer.clear()

        # Get initial observations
        obs = self._to_tensor_observation(self.env.get_observations())
        obs, privileged_obs = self._split_observations(obs)

        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, device=self.device)
        completed_episodes = 0
        total_reward = 0.0

        for step in range(self.cfg.num_steps):
            # Sample actions
            with torch.no_grad():
                actions, log_probs = self.actor.act(obs, deterministic=False)
                values = self.critic.get_value(
                    privileged_obs if self.cfg.use_asymmetric and privileged_obs is not None else obs
                )

            # Step environment
            next_obs, rewards, dones, extras = self.env.step(actions)
            next_obs = self._to_tensor_observation(next_obs)
            next_obs, next_privileged_obs = self._split_observations(next_obs)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device).bool()

            # Track episode stats
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
                    final_obs, final_privileged_obs = self._split_observations(final_obs)
                    critic_final_obs = (
                        final_privileged_obs
                        if self.cfg.use_asymmetric and final_privileged_obs is not None
                        else final_obs
                    )
                    with torch.no_grad():
                        final_values = self.critic.get_value(
                            observation_index(critic_final_obs, truncated)
                        )
                    rewards = rewards.clone()
                    rewards[truncated] += self.cfg.gamma * final_values

            # Call extras callback if provided (for reward components logging)
            if extras_callback is not None:
                extras_callback(extras, episode_ends, terminated, episode_rewards)

            # Log completed episodes
            if episode_ends.any():
                completed_indices = torch.where(episode_ends)[0]
                for idx in completed_indices:
                    self.episode_rewards.append(episode_rewards[idx].item())
                    self.episode_lengths.append(episode_lengths[idx].item())
                    total_reward += episode_rewards[idx].item()
                    completed_episodes += 1

                # Reset episode tracking for completed envs
                episode_rewards = episode_rewards * (~episode_ends).float()
                episode_lengths = episode_lengths * (~episode_ends).float()

            # Store transition
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
            )

            obs = next_obs
            privileged_obs = next_privileged_obs

        # Compute returns and advantages
        with torch.no_grad():
            last_values = self.critic.get_value(
                privileged_obs if self.cfg.use_asymmetric and privileged_obs is not None else obs
            )

        self.rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        # Normalize advantages if enabled
        if self.cfg.normalize_advantages:
            advantages = self.rollout_buffer.advantages
            self.rollout_buffer.advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        self.total_timesteps += self.cfg.num_steps * self.num_envs

        # Return rollout stats
        stats = {
            "rollout/mean_reward": rewards.mean().item(),
            "rollout/mean_value": self.rollout_buffer.values.mean().item(),
        }

        if completed_episodes > 0:
            stats["rollout/mean_episode_reward"] = total_reward / completed_episodes
            stats["rollout/completed_episodes"] = completed_episodes

        return stats

    def _to_tensor_observation(self, obs: Any) -> Any:
        """Normalize environment observations to tensors on the training device."""
        return observation_to_tensor(obs, dtype=torch.float32, device=self.device)

    def _to_bool_tensor(
        self,
        value: Any,
        reference: torch.Tensor,
        default: bool = True,
    ) -> torch.Tensor:
        """Normalize boolean masks from environment extras."""
        if value is None:
            return torch.full_like(reference, default, dtype=torch.bool)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device)
        return value.to(self.device).bool()

    def update(self) -> dict[str, float]:
        """Update policy and value function using collected rollout data.

        Returns:
            Dictionary of training statistics.
        """
        self.actor.train()
        self.critic.train()

        # Get all rollout data
        data = self.rollout_buffer.get_all_data()
        observations = data["observations"]
        privileged_observations = data["privileged_observations"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]
        old_values = data["values"]

        batch_size = observation_batch_size(observations)
        minibatch_size = self.cfg.get_minibatch_size(self.num_envs)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        # Update for multiple epochs
        early_stopped = False
        for epoch in range(self.cfg.num_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            # Minibatch updates
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)

                mb_indices = indices[start:end]

                mb_obs = observation_index(observations, mb_indices)
                mb_privileged_obs = (
                    observation_index(privileged_observations, mb_indices)
                    if privileged_observations is not None
                    else None
                )
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]

                # Evaluate actions
                log_probs, entropy = self.actor.evaluate(mb_obs, mb_actions)
                values = self.critic.get_value(
                    mb_privileged_obs if mb_privileged_obs is not None else mb_obs
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.cfg.clip_range_vf is not None:
                    # Clip value function
                    value_pred_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values,
                        -self.cfg.clip_range_vf,
                        self.cfg.clip_range_vf,
                    )
                    value_loss1 = nn.functional.mse_loss(values, mb_returns)
                    value_loss2 = nn.functional.mse_loss(value_pred_clipped, mb_returns)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                else:
                    value_loss = 0.5 * nn.functional.mse_loss(values, mb_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                # Optimization step with gradient norm tracking (before clipping)
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Compute unclipped gradient norm for logging
                    actor_grad_norm_unclipped = nn.utils.clip_grad_norm_(
                        self.actor.parameters(), float("inf")
                    )
                    critic_grad_norm_unclipped = nn.utils.clip_grad_norm_(
                        self.critic.parameters(), float("inf")
                    )
                    # Apply actual gradient clipping
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                else:
                    # Separate optimizers
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    loss.backward()
                    # Compute unclipped gradient norm for logging
                    actor_grad_norm_unclipped = nn.utils.clip_grad_norm_(
                        self.actor.parameters(), float("inf")
                    )
                    critic_grad_norm_unclipped = nn.utils.clip_grad_norm_(
                        self.critic.parameters(), float("inf")
                    )
                    # Apply actual gradient clipping
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.cfg.max_grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.cfg.max_grad_norm
                    )
                    self.policy_optimizer.step()
                    self.value_optimizer.step()

                # Metrics
                with torch.no_grad():
                    # Clamp ratio for numerical stability in KL computation
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
                num_updates += 1

                # Accumulate gradient norms
                if num_updates == 1:
                    total_actor_grad_norm = actor_grad_norm_unclipped
                    total_critic_grad_norm = critic_grad_norm_unclipped
                else:
                    total_actor_grad_norm += actor_grad_norm_unclipped
                    total_critic_grad_norm += critic_grad_norm_unclipped

                # Early stopping on KL divergence
                if (
                    self.cfg.target_kl is not None
                    and approx_kl.item() > self.cfg.target_kl
                ):
                    early_stopped = True
                    break

            if early_stopped:
                break

        # Average metrics
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_approx_kl = total_approx_kl / num_updates
        avg_clip_fraction = total_clip_fraction / num_updates

        # Average gradient norms
        avg_actor_grad_norm = (total_actor_grad_norm / num_updates).item()
        avg_critic_grad_norm = (total_critic_grad_norm / num_updates).item()
        avg_total_grad_norm = (avg_actor_grad_norm**2 + avg_critic_grad_norm**2) ** 0.5

        stats = {
            "train/policy_loss": avg_policy_loss,
            "train/value_loss": avg_value_loss,
            "train/entropy_loss": avg_entropy_loss,
            "train/approx_kl": avg_approx_kl,
            "train/clip_fraction": avg_clip_fraction,
            "train/learning_rate": self.get_current_lr(),
            "train/actor_grad_norm": avg_actor_grad_norm,
            "train/critic_grad_norm": avg_critic_grad_norm,
            "train/total_grad_norm": avg_total_grad_norm,
            "train/early_stopped": float(early_stopped),
        }

        return stats

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        else:
            return self.policy_optimizer.param_groups[0]["lr"]

    def adjust_learning_rate(
        self, current_iteration: int, total_iterations: int
    ) -> None:
        """Adjust learning rate based on schedule."""
        if self.cfg.learning_rate_schedule == "constant":
            return

        if self.cfg.learning_rate_schedule == "linear":
            progress = current_iteration / total_iterations
            new_lr = self.cfg.learning_rate * (1 - progress)
        elif self.cfg.learning_rate_schedule == "adaptive":
            progress = current_iteration / total_iterations
            new_lr = self.cfg.min_learning_rate + (
                self.cfg.max_learning_rate - self.cfg.min_learning_rate
            ) * (1 - progress)
        else:
            return

        # Update optimizer learning rates
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
        else:
            for param_group in self.policy_optimizer.param_groups:
                param_group["lr"] = new_lr * (
                    self.cfg.policy_learning_rate / self.cfg.learning_rate
                )
            for param_group in self.value_optimizer.param_groups:
                param_group["lr"] = new_lr * (
                    self.cfg.value_learning_rate / self.cfg.learning_rate
                )

    def learn(self, total_timesteps: int | None = None) -> dict[str, Any]:
        """Train through the canonical OnPolicyRunner entrypoint."""
        from apexrl.agent.on_policy_runner import OnPolicyRunner

        runner = OnPolicyRunner(
            agent=self,
            env=self.env,
            cfg=self.cfg,
            log_dir=None,
            save_dir=self.log_dir,
            device=self.device,
        )
        return runner.learn(total_timesteps=total_timesteps)

    def _log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Log a batch of scalar metrics when available."""
        if self.logger and scalars:
            self.logger.log_scalars(scalars, step)

    def _get_rollout_metrics_for_logging(
        self, rollout_stats: dict[str, float]
    ) -> dict[str, float]:
        """Drop rollout metrics that are duplicated elsewhere in the same log step."""
        metrics = dict(rollout_stats)
        if self.episode_rewards:
            metrics.pop("rollout/mean_episode_reward", None)
        return metrics

    def _get_train_iteration_metrics(
        self, update_stats: dict[str, float]
    ) -> dict[str, float]:
        """Return optional train/* metrics on an iteration-based x-axis."""
        if not self.cfg.log_train_metrics_vs_iteration:
            return {}
        return {
            key.replace("train/", "train_vs_iter/"): value
            for key, value in update_stats.items()
            if key.startswith("train/")
        }

    def _get_episode_iteration_metrics(
        self, mean_reward: float, mean_length: float
    ) -> dict[str, float]:
        """Return optional episode/* metrics on an iteration-based x-axis."""
        if not self.cfg.log_episode_metrics_vs_iteration:
            return {}
        return {
            "episode_vs_iter/mean_reward": mean_reward,
            "episode_vs_iter/mean_length": mean_length,
        }

    def _get_detailed_rollout_stats(self) -> dict[str, float]:
        """Return optional rollout distribution statistics."""
        if not self.cfg.log_detailed_rollout_stats:
            return {}

        advantages = self.rollout_buffer.advantages
        values = self.rollout_buffer.values
        returns = self.rollout_buffer.returns
        return {
            "advantage/mean": advantages.mean().item(),
            "advantage/std": advantages.std().item(),
            "value/mean": values.mean().item(),
            "value/std": values.std().item(),
            "returns/mean": returns.mean().item(),
            "returns/std": returns.std().item(),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "policy_optimizer_state_dict": (
                self.policy_optimizer.state_dict() if self.policy_optimizer else None
            ),
            "value_optimizer_state_dict": (
                self.value_optimizer.state_dict() if self.value_optimizer else None
            ),
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "config": self.cfg,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        # PyTorch 2.6+ changed weights_only default to True
        # We need to handle both old and new behavior
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.policy_optimizer and checkpoint.get("policy_optimizer_state_dict"):
            self.policy_optimizer.load_state_dict(
                checkpoint["policy_optimizer_state_dict"]
            )
        if self.value_optimizer and checkpoint.get("value_optimizer_state_dict"):
            self.value_optimizer.load_state_dict(
                checkpoint["value_optimizer_state_dict"]
            )

        self.iteration = checkpoint.get("iteration", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)

    def eval(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate the agent."""
        self.actor.eval()
        self.critic.eval()

        obs = self._to_tensor_observation(self.env.reset())
        obs, _ = self._split_observations(obs)

        episode_rewards = []
        current_rewards = torch.zeros(self.num_envs, device=self.device)
        episodes_completed = 0

        while episodes_completed < num_episodes:
            with torch.no_grad():
                actions, _ = self.actor.act(obs, deterministic=True)

            next_obs, rewards, dones, _ = self.env.step(actions)
            next_obs = self._to_tensor_observation(next_obs)
            next_obs, _ = self._split_observations(next_obs)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            current_rewards += rewards

            if dones.any():
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    if episodes_completed < num_episodes:
                        episode_rewards.append(current_rewards[idx].item())
                        episodes_completed += 1
                current_rewards = current_rewards * (~dones).float()

            obs = next_obs

        return {
            "eval/mean_reward": sum(episode_rewards) / len(episode_rewards),
            "eval/std_reward": (
                sum(
                    (r - sum(episode_rewards) / len(episode_rewards)) ** 2
                    for r in episode_rewards
                )
                / len(episode_rewards)
            )
            ** 0.5,
            "eval/min_reward": min(episode_rewards),
            "eval/max_reward": max(episode_rewards),
        }
