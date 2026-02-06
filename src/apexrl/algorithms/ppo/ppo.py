"""PPO (Proximal Policy Optimization) algorithm implementation.

Supports custom Actor and Critic networks defined by users.

References:
    - Original paper: https://arxiv.org/abs/1707.06347
    - rsl_rl: https://github.com/leggedrobotics/rsl_rl
    - stable-baselines3: https://github.com/DLR-RM/stable-baselines3
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

from apexrl.algorithms.ppo.config import PPOConfig
from apexrl.buffer.rollout_buffer import RolloutBuffer
from apexrl.envs.vecenv import VecEnv
from apexrl.models.base import Actor, ContinuousActor, Critic, DiscreteActor
from apexrl.optimizers import get_optimizer


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
        cfg: Optional[PPOConfig] = None,
        actor_class: Optional[Type[Actor]] = None,
        critic_class: Optional[Type[Critic]] = None,
        obs_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        actor_cfg: Optional[Dict[str, Any]] = None,
        critic_cfg: Optional[Dict[str, Any]] = None,
        actor: Optional[Actor] = None,
        critic: Optional[Critic] = None,
        log_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
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
            if obs_space is None or action_space is None:
                raise ValueError(
                    "obs_space and action_space are required when using actor_class/critic_class"
                )

            self.obs_space = obs_space
            self.action_space = action_space

            # Create actor
            self.actor = actor_class(
                obs_space=obs_space,
                action_space=action_space,
                cfg=actor_cfg or {},
            ).to(self.device)

            # Create critic (may use privileged obs if asymmetric)
            critic_obs_space = (
                getattr(env, "privileged_obs_space", None) or obs_space
                if self.cfg.use_asymmetric
                else obs_space
            )
            self.critic = critic_class(
                obs_space=critic_obs_space,
                cfg=critic_cfg or {},
            ).to(self.device)
        else:
            raise ValueError(
                "Must provide either (actor, critic) OR "
                "(actor_class, critic_class, obs_space, action_space)"
            )

        # Verify actor type
        if not isinstance(self.actor, (ContinuousActor, DiscreteActor)):
            raise TypeError(
                f"PPO currently only supports ContinuousActor or DiscreteActor, got {type(self.actor)}"
            )

        # Get action dimension
        if isinstance(self.action_space, spaces.Box):
            self.action_dim = (
                self.action_space.shape[0] if len(self.action_space.shape) > 0 else 1
            )
        elif isinstance(self.action_space, spaces.Discrete):
            self.action_dim = 1  # Discrete actions are single integers
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
        self.obs_shape = self._get_obs_shape(obs_space)

        self.rollout_buffer = RolloutBuffer(
            num_envs=self.num_envs,
            num_steps=self.cfg.num_steps,
            obs_shape=self.obs_shape,
            device=self.device,
            num_privileged_obs=getattr(env, "num_privileged_obs", 0)
            if self.cfg.use_asymmetric
            else 0,
        )

        # Logging
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.iteration = 0
        self.total_timesteps = 0

        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []

    def _get_obs_shape(self, obs_space: Optional[spaces.Space]) -> Tuple[int, ...]:
        """Get observation shape from space."""
        if obs_space is None:
            # Try to infer from env
            if hasattr(self.env, "num_obs"):
                return (self.env.num_obs,)
            return (1,)  # Fallback

        if isinstance(obs_space, spaces.Box):
            return obs_space.shape
        else:
            raise NotImplementedError(
                f"Observation space {type(obs_space)} not supported"
            )

    def collect_rollout(self) -> Dict[str, float]:
        """Collect a rollout from the environment.

        Returns:
            Dictionary of rollout statistics.
        """
        self.actor.eval()
        self.critic.eval()
        self.rollout_buffer.clear()

        # Get initial observations
        obs = self.env.get_observations()
        if hasattr(obs, "get"):  # TensorDict
            obs = obs["obs"]
        elif isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.to(self.device)

        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, device=self.device)
        completed_episodes = 0
        total_reward = 0.0

        for step in range(self.cfg.num_steps):
            # Get privileged obs if using asymmetric critic
            if self.cfg.use_asymmetric:
                privileged_obs = self.env.get_privileged_observations()
                privileged_obs = (
                    privileged_obs.to(self.device)
                    if privileged_obs is not None
                    else None
                )
            else:
                privileged_obs = None

            # Sample actions
            with torch.no_grad():
                actions, log_probs = self.actor.act(obs, deterministic=False)
                values = self.critic.get_value(
                    privileged_obs if privileged_obs is not None else obs
                )

            # Step environment
            next_obs, rewards, dones, extras = self.env.step(actions)
            if hasattr(next_obs, "get"):  # TensorDict
                next_obs = next_obs["obs"]
            elif isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            next_obs = next_obs.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            # Track episode stats
            episode_rewards += rewards
            episode_lengths += 1

            # Check for timeouts
            time_outs = extras.get("time_outs", torch.zeros_like(dones))
            true_dones = dones & ~time_outs

            # Log completed episodes
            if true_dones.any():
                completed_indices = torch.where(true_dones)[0]
                for idx in completed_indices:
                    self.episode_rewards.append(episode_rewards[idx].item())
                    self.episode_lengths.append(episode_lengths[idx].item())
                    total_reward += episode_rewards[idx].item()
                    completed_episodes += 1

                # Reset episode tracking for completed envs
                episode_rewards = episode_rewards * (~true_dones).float()
                episode_lengths = episode_lengths * (~true_dones).float()

            # Store transition
            self.rollout_buffer.add(
                observations=obs,
                privileged_observations=privileged_obs,
                actions=actions,
                rewards=rewards,
                dones=dones.float(),
                values=values,
                log_probs=log_probs,
            )

            obs = next_obs

        # Compute returns and advantages
        with torch.no_grad():
            last_values = self.critic.get_value(
                self.env.get_privileged_observations().to(self.device)
                if self.cfg.use_asymmetric
                and self.env.get_privileged_observations() is not None
                else obs
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

    def update(self) -> Dict[str, float]:
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

        batch_size = observations.shape[0]
        minibatch_size = self.cfg.get_minibatch_size(self.num_envs)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        # Update for multiple epochs
        for epoch in range(self.cfg.num_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            # Minibatch updates
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if end > batch_size:
                    continue

                mb_indices = indices[start:end]

                mb_obs = observations[mb_indices]
                mb_privileged_obs = (
                    privileged_observations[mb_indices]
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

                # Optimization step
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
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
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_fraction = (
                        ((ratio - 1).abs() > self.cfg.clip_range).float().mean()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                num_updates += 1

        # Average metrics
        stats = {
            "train/policy_loss": total_policy_loss / num_updates,
            "train/value_loss": total_value_loss / num_updates,
            "train/entropy_loss": total_entropy_loss / num_updates,
            "train/approx_kl": total_approx_kl / num_updates,
            "train/clip_fraction": total_clip_fraction / num_updates,
            "train/learning_rate": self.get_current_lr(),
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

    def learn(self, total_timesteps: int) -> None:
        """Train the agent.

        Args:
            total_timesteps: Total number of timesteps to train for.
        """
        num_iterations = total_timesteps // (self.cfg.num_steps * self.num_envs)
        start_time = time.time()

        for iteration in range(num_iterations):
            self.iteration = iteration

            # Collect rollout
            rollout_stats = self.collect_rollout()

            # Update policy
            update_stats = self.update()

            # Adjust learning rate
            self.adjust_learning_rate(iteration, num_iterations)

            # Logging
            if self.writer and iteration % self.cfg.log_interval == 0:
                fps = (
                    self.cfg.num_steps
                    * self.num_envs
                    * self.cfg.log_interval
                    / (time.time() - start_time)
                )
                start_time = time.time()

                self.writer.add_scalar("time/fps", fps, self.total_timesteps)
                self.writer.add_scalar(
                    "time/iterations", iteration, self.total_timesteps
                )

                for key, value in rollout_stats.items():
                    self.writer.add_scalar(key, value, self.total_timesteps)

                for key, value in update_stats.items():
                    self.writer.add_scalar(key, value, self.total_timesteps)

                # Log episode stats
                if self.episode_rewards:
                    mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                    mean_length = sum(self.episode_lengths) / len(self.episode_lengths)
                    self.writer.add_scalar(
                        "episode/mean_reward", mean_reward, self.total_timesteps
                    )
                    self.writer.add_scalar(
                        "episode/mean_length", mean_length, self.total_timesteps
                    )
                    self.episode_rewards.clear()
                    self.episode_lengths.clear()

                print(
                    f"Iteration {iteration}/{num_iterations} | "
                    f"Timesteps {self.total_timesteps} | "
                    f"FPS {fps:.0f} | "
                    f"Policy Loss {update_stats['train/policy_loss']:.4f} | "
                    f"Value Loss {update_stats['train/value_loss']:.4f}"
                )

            # Save checkpoint
            if iteration % self.cfg.save_interval == 0:
                self.save(f"checkpoint_{iteration}.pt")

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

    def eval(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent."""
        self.actor.eval()
        self.critic.eval()

        obs = self.env.reset()
        if hasattr(obs, "get"):
            obs = obs["obs"]
        elif isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.to(self.device)

        episode_rewards = []
        current_rewards = torch.zeros(self.num_envs, device=self.device)
        episodes_completed = 0

        while episodes_completed < num_episodes:
            with torch.no_grad():
                actions, _ = self.actor.act(obs, deterministic=True)

            next_obs, rewards, dones, _ = self.env.step(actions)
            if hasattr(next_obs, "get"):
                next_obs = next_obs["obs"]
            elif isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            next_obs = next_obs.to(self.device)
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
