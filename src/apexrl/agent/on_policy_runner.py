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

"""On-policy runner for PPO and similar algorithms.

This module provides a runner class that handles the training loop,
logging, checkpointing, and environment interaction for on-policy RL algorithms.
"""

from __future__ import annotations

import collections
import os
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type

import torch
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

from apexrl.envs.vecenv import VecEnv


class OnPolicyRunner:
    """Runner for on-policy RL algorithms (primarily PPO).

    Handles the training loop, logging, checkpointing, and environment interaction.

    Features:
    - Default PPO algorithm, extensible to other on-policy algorithms
    - Automatic logging of reward components and environment metrics from extras
    - Flexible callback system for custom training logic
    - Built-in checkpointing and TensorBoard integration

    Usage - Minimal (Auto-create PPO agent):
        >>> from apexrl.agent.on_policy_runner import OnPolicyRunner
        >>> from apexrl.envs.vecenv import DummyVecEnv
        >>> from apexrl.models.mlp import MLPActor, MLPCritic
        >>>
        >>> env = DummyVecEnv(num_envs=4096, num_obs=48, num_actions=12)
        >>> runner = OnPolicyRunner(
        ...     env=env,
        ...     algorithm="ppo",  # or just omit, defaults to "ppo"
        ...     actor_class=MLPActor,
        ...     critic_class=MLPCritic,
        ...     log_dir="./logs",
        ... )
        >>> runner.learn(total_timesteps=10_000_000)

    Usage - With Pre-configured Agent:
        >>> from apexrl.algorithms.ppo import PPO, PPOConfig
        >>>
        >>> cfg = PPOConfig(learning_rate=3e-4)
        >>> agent = PPO(env=env, cfg=cfg, actor_class=MLPActor, critic_class=MLPCritic, ...)
        >>> runner = OnPolicyRunner(agent=agent, env=env, cfg=cfg)
        >>> runner.learn(total_timesteps=10_000_000)

    Environment Extras Format:
        The runner automatically extracts and logs metrics from environment extras.
        Environments should return extras in step() like this:

        >>> extras = {
        ...     "time_outs": time_outs,  # Required: bool tensor (num_envs,)
        ...
        ...     # Optional: Reward components (auto-accumulated per episode)
        ...     "reward_components": {
        ...         "velocity": velocity_reward,      # (num_envs,) - step-level reward
        ...         "energy": -energy_penalty,        # (num_envs,)
        ...         "stability": stability_reward,    # (num_envs,)
        ...     },
        ...
        ...     # Optional: Custom metrics (logged directly to tensorboard)
        ...     "log": {
        ...         "/reward/velocity_mean": velocity_reward.mean().item(),
        ...         "/robot/height_mean": robot_height.mean().item(),
        ...         "/episode/length_mean": episode_length.float().mean().item(),
        ...     },
        ... }
    """

    # Registry of supported algorithms
    ALGORITHMS: Dict[str, Any] = {}

    def __init__(
        self,
        env: VecEnv,
        cfg: Optional[Any] = None,
        # Algorithm selection (mutually exclusive with agent)
        algorithm: str = "ppo",
        actor_class: Optional[Type] = None,
        critic_class: Optional[Type] = None,
        obs_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        actor_cfg: Optional[Dict[str, Any]] = None,
        critic_cfg: Optional[Dict[str, Any]] = None,
        # Or provide pre-created agent
        agent: Optional[Any] = None,
        # Logging and saving
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        # Configuration
        log_reward_components: bool = True,
        log_interval: int = 10,
        save_interval: int = 100,
    ):
        """Initialize the on-policy runner.

        Args:
            env: Vectorized environment.
            cfg: Algorithm configuration. If None and algorithm="ppo", uses PPOConfig().

            # Algorithm selection (when agent is not provided):
            algorithm: Algorithm name ("ppo"). Defaults to "ppo".
            actor_class: Actor network class. Required if agent not provided.
            critic_class: Critic network class. Required if agent not provided.
            obs_space: Observation space. Required if agent not provided.
            action_space: Action space. Required if agent not provided.
            actor_cfg: Actor network configuration.
            critic_cfg: Critic network configuration.

            # Or provide pre-created agent:
            agent: Pre-configured algorithm instance (e.g., PPO).

            # Logging:
            log_dir: Directory for TensorBoard logs.
            save_dir: Directory for checkpoints. Defaults to log_dir.
            device: Device for training. Auto-detects if None.
            log_reward_components: Whether to log reward components from extras.
            log_interval: Logging interval in iterations.
            save_interval: Checkpoint saving interval.

        Raises:
            ValueError: If neither agent nor (actor_class, critic_class, spaces) provided.
        """
        self.env = env
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create or use provided agent
        if agent is not None:
            self.agent = agent
            self.cfg = cfg or getattr(agent, "cfg", None)
            if self.cfg is None:
                raise ValueError("Must provide cfg when agent doesn't have one")
        else:
            # Auto-create agent based on algorithm name
            self.agent, self.cfg = self._create_agent(
                algorithm=algorithm,
                env=env,
                cfg=cfg,
                actor_class=actor_class,
                critic_class=critic_class,
                obs_space=obs_space,
                action_space=action_space,
                actor_cfg=actor_cfg,
                critic_cfg=critic_cfg,
                device=self.device,
            )

        # Logging setup
        self.log_dir = log_dir
        self.save_dir = save_dir or log_dir

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir) if self.log_dir else None

        # Configuration
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_reward_components = log_reward_components

        # Training state
        self.iteration = 0
        self.total_timesteps = 0
        self.start_time = None

        # Episode-level reward component tracking
        self.reward_components: Dict[str, List[float]] = {}
        self.current_reward_components: Dict[str, torch.Tensor] = {}

        # Environment metrics log buffers (from extras["log"])
        self.log_buffers: Dict[str, Deque[float]] = {}
        self.log_buffer_maxlen = 1000

        # Loss history
        self.loss_history: Optional[Dict[str, Deque]] = None
        self.loss_history_maxlen = 1000

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "pre_iteration": [],
            "post_iteration": [],
            "pre_rollout": [],
            "post_rollout": [],
            "pre_update": [],
            "post_update": [],
        }

        # Sync agent's writer with runner
        if hasattr(self.agent, "writer") and self.agent.writer is None:
            self.agent.writer = self.writer

    @classmethod
    def register_algorithm(cls, name: str, agent_class: Type, config_class: Type):
        """Register a new algorithm for use with the runner.

        Args:
            name: Algorithm name (e.g., "ppo", "a2c").
            agent_class: The algorithm class (e.g., PPO).
            config_class: The configuration class (e.g., PPOConfig).
        """
        cls.ALGORITHMS[name.lower()] = {
            "agent_class": agent_class,
            "config_class": config_class,
        }

    def _create_agent(
        self,
        algorithm: str,
        env: VecEnv,
        cfg: Optional[Any],
        actor_class: Optional[Type],
        critic_class: Optional[Type],
        obs_space: Optional[spaces.Space],
        action_space: Optional[spaces.Space],
        actor_cfg: Optional[Dict],
        critic_cfg: Optional[Dict],
        device: torch.device,
    ) -> Tuple[Any, Any]:
        """Create algorithm agent based on name."""
        # Lazy import and register PPO if not already registered
        if not self.ALGORITHMS:
            from apexrl.algorithms.ppo import PPO, PPOConfig

            self.register_algorithm("ppo", PPO, PPOConfig)

        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Registered: {list(self.ALGORITHMS.keys())}"
            )

        algo_info = self.ALGORITHMS[algorithm]
        agent_class = algo_info["agent_class"]
        config_class = algo_info["config_class"]

        # Use default config if not provided
        if cfg is None:
            cfg = config_class()

        # Validate required arguments
        if actor_class is None or critic_class is None:
            raise ValueError(
                f"actor_class and critic_class are required when creating {algorithm} agent"
            )
        if obs_space is None or action_space is None:
            raise ValueError(
                f"obs_space and action_space are required when creating {algorithm} agent"
            )

        # Create agent
        agent = agent_class(
            env=env,
            cfg=cfg,
            actor_class=actor_class,
            critic_class=critic_class,
            obs_space=obs_space,
            action_space=action_space,
            actor_cfg=actor_cfg or {},
            critic_cfg=critic_cfg or {},
            log_dir=None,  # Runner handles logging
            device=device,
        )

        return agent, cfg

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add a callback for a specific event.

        Args:
            event: Event name. One of:
                - "pre_iteration": Called before each training iteration
                - "post_iteration": Called after each training iteration
                - "pre_rollout": Called before collecting rollout
                - "post_rollout": Called after collecting rollout
                - "pre_update": Called before policy update
                - "post_update": Called after policy update
            callback: Callback function. Signature depends on event.

        Raises:
            ValueError: If event name is invalid.
        """
        if event not in self.callbacks:
            raise ValueError(
                f"Unknown event: {event}. Must be one of {list(self.callbacks.keys())}"
            )
        self.callbacks[event].append(callback)

    def _call_callbacks(self, event: str, *args) -> None:
        """Call all callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                print(f"Warning: Callback failed for event '{event}': {e}")

    def collect_rollout(self) -> Dict[str, float]:
        """Collect a rollout from the environment.

        Uses the agent's collect_rollout method with an extras callback
        to capture environment metrics.
        """
        self._call_callbacks("pre_rollout", self)

        # Collect rollout with extras callback for logging
        stats = self.agent.collect_rollout(extras_callback=self._process_extras)
        self.total_timesteps = self.agent.total_timesteps

        self._call_callbacks("post_rollout", self, stats)
        return stats

    def _process_extras(
        self,
        extras: Dict[str, Any],
        dones: torch.Tensor,
        true_dones: torch.Tensor,
        episode_rewards: torch.Tensor,
    ) -> None:
        """Process environment extras to extract and log metrics.

        This is called after each environment step during rollout collection.
        Extracts:
        1. Custom metrics from extras["log"] -> directly logged to tensorboard
        2. Reward components from extras["reward_components"] -> accumulated per episode

        Args:
            extras: Extras dict from env.step().
            dones: All done flags (including timeouts).
            true_dones: True done flags (excluding timeouts).
            episode_rewards: Current episode reward accumulator.
        """
        # === Process custom metrics from extras["log"] ===
        # These are logged directly (not accumulated)
        log_dict = extras.get("log", {})
        for key, value in log_dict.items():
            # Normalize key format (ensure starts with "/")
            if not key.startswith("/"):
                key = f"/{key}"

            # Convert tensor to float
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()

            # Store in buffer for batch logging
            if key not in self.log_buffers:
                self.log_buffers[key] = collections.deque(maxlen=self.log_buffer_maxlen)
            self.log_buffers[key].append(float(value))

        # === Process reward components from extras["reward_components"] ===
        # These are accumulated per episode and logged at episode end
        if self.log_reward_components:
            reward_components = extras.get("reward_components")
            if reward_components:
                self._accumulate_reward_components(reward_components, true_dones)

    def _accumulate_reward_components(
        self,
        components: Dict[str, torch.Tensor],
        true_dones: torch.Tensor,
    ) -> None:
        """Accumulate reward components per episode.

        For each reward component:
        1. Accumulate step-level values during the episode
        2. When episode ends (true_dones), store the total
        3. Reset accumulator for completed episodes

        Args:
            components: Dict of {component_name: step_reward_tensor}.
            true_dones: Bool tensor indicating completed episodes.
        """
        for name, values in components.items():
            # Ensure tensor
            if not isinstance(values, torch.Tensor):
                continue
            values = values.to(self.device)

            # Initialize accumulator for this component if needed
            key = f"/reward_component/{name}"
            if key not in self.current_reward_components:
                self.current_reward_components[key] = torch.zeros(
                    self.env.num_envs, device=self.device
                )

            # Accumulate step rewards
            self.current_reward_components[key] += values

            # Handle completed episodes
            if true_dones.any():
                completed_indices = torch.where(true_dones)[0]
                completed_values = self.current_reward_components[key][
                    completed_indices
                ]

                # Store completed episode totals
                if key not in self.reward_components:
                    self.reward_components[key] = []

                for val in completed_values.cpu().numpy():
                    self.reward_components[key].append(float(val))

                # Reset accumulators for completed episodes
                self.current_reward_components[key] *= (~true_dones).float()

    def update(self) -> Dict[str, float]:
        """Update policy using collected rollout."""
        self._call_callbacks("pre_update", self)
        stats = self.agent.update()
        self._call_callbacks("post_update", self, stats)
        return stats

    def learn(
        self,
        total_timesteps: Optional[int] = None,
        num_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the agent.

        Priority for determining training length:
        1. num_iterations argument (if provided)
        2. total_timesteps argument (if provided)
        3. cfg.max_iterations (if set in config)
        4. Raise error if none provided

        Args:
            total_timesteps: Total environment steps to train for.
            num_iterations: Number of policy iterations. Overrides total_timesteps.

        Returns:
            Training history dict with keys: iterations, timesteps, episode_rewards, etc.
        """
        # Determine iterations (priority: num_iterations > total_timesteps > cfg.max_iterations)
        transitions_per_iter = self.cfg.num_steps * self.env.num_envs

        if num_iterations is not None:
            total_iters = num_iterations
        elif total_timesteps is not None:
            total_iters = total_timesteps // transitions_per_iter
        elif (
            hasattr(self.cfg, "max_iterations") and self.cfg.max_iterations is not None
        ):
            total_iters = self.cfg.max_iterations
        else:
            raise ValueError(
                "Must provide one of:\n"
                "  - num_iterations argument to learn()\n"
                "  - total_timesteps argument to learn()\n"
                "  - cfg.max_iterations in config (e.g., PPOConfig(max_iterations=100))"
            )

        print(
            f"Training for {total_iters} iterations ({total_iters * transitions_per_iter:,} steps)"
        )
        print(f"  Batch size: {transitions_per_iter:,} transitions/iteration")

        self.start_time = time.time()
        last_log_time = self.start_time

        # Initialize history tracking
        self.loss_history = {
            "iterations": collections.deque(maxlen=self.loss_history_maxlen),
            "policy_loss": collections.deque(maxlen=self.loss_history_maxlen),
            "value_loss": collections.deque(maxlen=self.loss_history_maxlen),
            "entropy_loss": collections.deque(maxlen=self.loss_history_maxlen),
        }

        history = {
            "iterations": [],
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
        }

        try:
            for iteration in range(total_iters):
                self.iteration = iteration
                self.agent.iteration = iteration

                self._call_callbacks("pre_iteration", self)

                # Collect rollout and update
                rollout_stats = self.collect_rollout()
                update_stats = self.update()

                # Learning rate schedule
                if hasattr(self.agent, "adjust_learning_rate"):
                    self.agent.adjust_learning_rate(iteration, total_iters)

                # Record history
                self._update_history(iteration, rollout_stats, update_stats, history)

                # Logging
                if iteration % self.log_interval == 0:
                    self._log_iteration(
                        iteration,
                        total_iters,
                        rollout_stats,
                        update_stats,
                        last_log_time,
                    )
                    last_log_time = time.time()

                # Checkpointing
                if self.save_dir and iteration % self.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{iteration}.pt")

                self._call_callbacks(
                    "post_iteration", self, {**rollout_stats, **update_stats}
                )

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            if self.save_dir:
                self.save_checkpoint("checkpoint_final.pt")
            if self.writer:
                self.writer.close()

        print(f"\nTraining complete! Total steps: {self.total_timesteps:,}")
        return {
            "history": history,
            "final_iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
        }

    def _update_history(
        self,
        iteration: int,
        rollout_stats: Dict[str, float],
        update_stats: Dict[str, float],
        history: Dict[str, List],
    ) -> None:
        """Update training history buffers."""
        history["iterations"].append(iteration)
        history["timesteps"].append(self.total_timesteps)
        history["policy_losses"].append(update_stats.get("train/policy_loss", 0))
        history["value_losses"].append(update_stats.get("train/value_loss", 0))

        self.loss_history["iterations"].append(iteration)
        self.loss_history["policy_loss"].append(
            update_stats.get("train/policy_loss", 0)
        )
        self.loss_history["value_loss"].append(update_stats.get("train/value_loss", 0))
        self.loss_history["entropy_loss"].append(
            update_stats.get("train/entropy_loss", 0)
        )

        if self.agent.episode_rewards:
            mean_reward = sum(self.agent.episode_rewards) / len(
                self.agent.episode_rewards
            )
            mean_length = sum(self.agent.episode_lengths) / len(
                self.agent.episode_lengths
            )
            history["episode_rewards"].append(mean_reward)
            history["episode_lengths"].append(mean_length)

    def _log_iteration(
        self,
        iteration: int,
        total_iters: int,
        rollout_stats: Dict[str, float],
        update_stats: Dict[str, float],
        last_log_time: float,
    ) -> None:
        """Log training progress for current iteration."""
        # Calculate FPS
        elapsed = time.time() - last_log_time
        fps = (
            (self.cfg.num_steps * self.env.num_envs * self.log_interval / elapsed)
            if elapsed > 0
            else 0
        )

        # Console output
        msg = (
            f"Iter {iteration}/{total_iters} | "
            f"Steps {self.total_timesteps:,} | "
            f"FPS {fps:.0f} | "
            f"Policy Loss {update_stats.get('train/policy_loss', 0):.4f} | "
            f"Value Loss {update_stats.get('train/value_loss', 0):.4f} | "
            f"KL {update_stats.get('train/approx_kl', 0):.4f}"
        )
        if self.agent.episode_rewards:
            mean_reward = sum(self.agent.episode_rewards) / len(
                self.agent.episode_rewards
            )
            msg += f" | Reward {mean_reward:.2f}"
        print(msg)

        # TensorBoard logging
        if not self.writer:
            return

        # Time metrics
        self.writer.add_scalar("time/fps", fps, self.total_timesteps)
        self.writer.add_scalar("time/iteration", iteration, self.total_timesteps)

        # Rollout stats (vs timesteps)
        for key, value in rollout_stats.items():
            self.writer.add_scalar(key, value, self.total_timesteps)

        # Training stats (vs both timesteps and iteration)
        for key, value in update_stats.items():
            self.writer.add_scalar(key, value, self.total_timesteps)
            # Also log with iteration for easier analysis
            iter_key = key.replace("train/", "train_vs_iter/")
            self.writer.add_scalar(iter_key, value, iteration)

        # Distribution stats
        advantages = self.agent.rollout_buffer.advantages
        values = self.agent.rollout_buffer.values
        returns = self.agent.rollout_buffer.returns

        self.writer.add_scalar(
            "stats/advantage_mean", advantages.mean().item(), iteration
        )
        self.writer.add_scalar(
            "stats/advantage_std", advantages.std().item(), iteration
        )
        self.writer.add_scalar("stats/value_mean", values.mean().item(), iteration)
        self.writer.add_scalar("stats/returns_mean", returns.mean().item(), iteration)

        # Episode stats
        if self.agent.episode_rewards:
            mean_reward = sum(self.agent.episode_rewards) / len(
                self.agent.episode_rewards
            )
            mean_length = sum(self.agent.episode_lengths) / len(
                self.agent.episode_lengths
            )

            self.writer.add_scalar(
                "episode/mean_reward", mean_reward, self.total_timesteps
            )
            self.writer.add_scalar(
                "episode/mean_length", mean_length, self.total_timesteps
            )
            self.writer.add_scalar(
                "episode_vs_iter/mean_reward", mean_reward, iteration
            )
            self.writer.add_scalar(
                "episode_vs_iter/mean_length", mean_length, iteration
            )

            self.agent.episode_rewards.clear()
            self.agent.episode_lengths.clear()

        # Log reward components
        self._log_reward_components()

        # Log environment metrics from extras
        self._log_environment_metrics()

    def _log_reward_components(self) -> None:
        """Log accumulated reward components to TensorBoard."""
        for key, values in self.reward_components.items():
            if values:
                mean_val = sum(values) / len(values)
                # Convert "/reward_component/name" to tensorboard key
                tb_key = f"reward_components/{key.replace('/reward_component/', '')}"
                self.writer.add_scalar(tb_key, mean_val, self.iteration)
                values.clear()

    def _log_environment_metrics(self) -> None:
        """Log environment-provided metrics from log buffers."""
        for key, buffer in self.log_buffers.items():
            if buffer:
                mean_val = sum(buffer) / len(buffer)
                # Remove leading "/" for tensorboard key
                tb_key = key[1:] if key.startswith("/") else key
                self.writer.add_scalar(f"env/{tb_key}", mean_val, self.iteration)
                buffer.clear()

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        if not self.save_dir:
            return
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        path = (
            filename
            if os.path.isabs(filename)
            else os.path.join(self.save_dir or ".", filename)
        )
        self.agent.load(path)
        self.iteration = getattr(self.agent, "iteration", 0)
        self.total_timesteps = getattr(self.agent, "total_timesteps", 0)
        print(f"Loaded checkpoint: {path}")

    def eval(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent."""
        return self.agent.eval(num_episodes)

    def close(self) -> None:
        """Close runner and release resources."""
        if self.writer:
            self.writer.close()
        if hasattr(self.env, "close"):
            self.env.close()
