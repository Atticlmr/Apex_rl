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

"""Off-policy runner for DQN and similar algorithms."""

from __future__ import annotations

import collections
import os
import time
from typing import Any

import torch
from gymnasium import spaces

from apexrl.envs.vecenv import VecEnv
from apexrl.utils.logger import Logger


class OffPolicyRunner:
    """Runner for off-policy algorithms such as DQN."""

    ALGORITHMS: dict[str, Any] = {}

    def __init__(
        self,
        env: VecEnv,
        cfg: Any | None = None,
        algorithm: str = "dqn",
        q_network_class: type | None = None,
        obs_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        q_network_cfg: dict[str, Any] | None = None,
        agent: Any | None = None,
        log_dir: str | None = None,
        save_dir: str | None = None,
        device: torch.device | None = None,
        log_interval: int | None = None,
        save_interval: int | None = None,
        actor_class: type | None = None,
        critic_class: type | None = None,
        actor_cfg: dict[str, Any] | None = None,
        critic_cfg: dict[str, Any] | None = None,
    ):
        """Initialize the off-policy runner."""
        self.env = env
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if agent is not None:
            self.agent = agent
            self.cfg = cfg or getattr(agent, "cfg", None)
            if self.cfg is None:
                raise ValueError("Must provide cfg when agent doesn't have one")
        else:
            self.agent, self.cfg = self._create_agent(
                algorithm=algorithm,
                env=env,
                cfg=cfg,
                q_network_class=q_network_class,
                obs_space=obs_space,
                action_space=action_space,
                q_network_cfg=q_network_cfg,
                actor_class=actor_class,
                critic_class=critic_class,
                actor_cfg=actor_cfg,
                critic_cfg=critic_cfg,
                device=self.device,
            )

        self.log_dir = log_dir
        self.save_dir = save_dir or log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.logger = None
        if self.log_dir:
            logger_backend = getattr(self.cfg, "logger_backend", "tensorboard")
            logger_kwargs = getattr(self.cfg, "logger_kwargs", None) or {}
            self.logger = Logger.create(
                backend=logger_backend,
                experiment_name="off_policy_runner",
                log_dir=self.log_dir,
                **logger_kwargs,
            )
        elif hasattr(self.agent, "logger") and self.agent.logger is not None:
            self.logger = self.agent.logger
        if hasattr(self.agent, "logger") and self.agent.logger is None:
            self.agent.logger = self.logger

        self.log_interval = (
            log_interval
            if log_interval is not None
            else getattr(self.cfg, "log_interval", 1_000)
        )
        self.save_interval = (
            save_interval
            if save_interval is not None
            else getattr(self.cfg, "save_interval", 10_000)
        )

        self.iteration = 0
        self.total_timesteps = getattr(self.agent, "total_timesteps", 0)
        self.start_time: float | None = None
        self.episode_rewards: collections.deque[float] = collections.deque(maxlen=100)
        self.episode_lengths: collections.deque[float] = collections.deque(maxlen=100)
        self.current_episode_rewards = torch.zeros(
            self.env.num_envs, device=self.device
        )
        self.current_episode_lengths = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.device
        )

    @classmethod
    def register_algorithm(
        cls, name: str, agent_class: type, config_class: type
    ) -> None:
        """Register an off-policy algorithm for auto-creation."""
        cls.ALGORITHMS[name.lower()] = {
            "agent_class": agent_class,
            "config_class": config_class,
        }

    def _create_agent(
        self,
        algorithm: str,
        env: VecEnv,
        cfg: Any | None,
        q_network_class: type | None,
        obs_space: spaces.Space | None,
        action_space: spaces.Space | None,
        q_network_cfg: dict[str, Any] | None,
        actor_class: type | None,
        critic_class: type | None,
        actor_cfg: dict[str, Any] | None,
        critic_cfg: dict[str, Any] | None,
        device: torch.device,
    ) -> tuple[Any, Any]:
        """Create algorithm agent based on name."""
        if not self.ALGORITHMS:
            from apexrl.algorithms.dqn import DQN, DQNConfig
            from apexrl.algorithms.sac import SAC, SACConfig

            self.register_algorithm("dqn", DQN, DQNConfig)
            self.register_algorithm("sac", SAC, SACConfig)

        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Registered: {list(self.ALGORITHMS.keys())}"
            )
        algo_info = self.ALGORITHMS[algorithm]
        agent_class = algo_info["agent_class"]
        config_class = algo_info["config_class"]

        if cfg is None:
            cfg = config_class()

        obs_space = obs_space or getattr(env, "observation_space_gym", None)
        action_space = action_space or getattr(env, "action_space_gym", None)
        if obs_space is None or action_space is None:
            raise ValueError(
                "obs_space and action_space are required when creating "
                f"{algorithm} agent"
            )

        if algorithm == "dqn":
            if q_network_class is None:
                raise ValueError("q_network_class is required when creating dqn agent")
            agent = agent_class(
                env=env,
                cfg=cfg,
                q_network_class=q_network_class,
                obs_space=obs_space,
                action_space=action_space,
                q_network_cfg=q_network_cfg or {},
                log_dir=None,
                device=device,
            )
        elif algorithm == "sac":
            agent = agent_class(
                env=env,
                cfg=cfg,
                actor_class=actor_class,
                critic_class=critic_class,
                obs_space=obs_space,
                action_space=action_space,
                actor_cfg=actor_cfg or {},
                critic_cfg=critic_cfg or {},
                log_dir=None,
                device=device,
            )
        else:
            raise ValueError(
                f"Unsupported off-policy auto-construction path for {algorithm}"
            )
        return agent, cfg

    def _to_bool_tensor(
        self,
        value: torch.Tensor | None,
        fallback: torch.Tensor,
        default: bool = False,
    ) -> torch.Tensor:
        """Normalize metadata masks to boolean tensors on the runner device."""
        if value is None:
            return torch.full_like(fallback, default, dtype=torch.bool)
        return value.to(self.device).bool()

    def _log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Log a batch of scalar metrics when a logger is available."""
        if self.logger and scalars:
            self.logger.log_scalars(scalars, step)

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

    def eval(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate the agent."""
        return self.agent.eval(num_episodes)

    def learn(self, total_timesteps: int | None = None) -> dict[str, Any]:
        """Train the agent for a number of environment transitions."""
        if total_timesteps is None:
            total_timesteps = getattr(self.cfg, "max_timesteps", None)
        if total_timesteps is None:
            raise ValueError(
                "Must provide total_timesteps or set cfg.max_timesteps "
                "for off-policy training"
            )

        obs = self.agent._to_tensor_observation(self.env.reset())
        self.start_time = time.time()
        history = {
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "q_losses": [],
        }
        last_update_stats: dict[str, float] = {}

        print(f"Training for {total_timesteps:,} timesteps")
        print(f"  Parallel environments: {self.env.num_envs}")

        try:
            while self.total_timesteps < total_timesteps:
                self.iteration += 1
                self.agent.iteration = self.iteration

                if self.total_timesteps < self.cfg.learning_starts:
                    actions = self.agent.sample_random_actions()
                else:
                    epsilon = self.agent.get_epsilon(self.total_timesteps)
                    actions = self.agent.act(obs, epsilon=epsilon)

                next_obs, rewards, dones, extras = self.env.step(actions)
                next_obs = self.agent._to_tensor_observation(next_obs)
                rewards = rewards.to(self.device, dtype=torch.float32)
                dones = dones.to(self.device).bool()

                truncated = self._to_bool_tensor(
                    extras.get("truncated", extras.get("time_outs")),
                    dones,
                    default=False,
                )
                terminated = extras.get("terminated")
                if terminated is None:
                    terminated = dones & ~truncated
                terminated = self._to_bool_tensor(terminated, dones, default=False)

                next_obs_for_buffer = next_obs.clone()
                final_obs = extras.get("final_observation")
                if final_obs is not None:
                    final_obs = self.agent._to_tensor_observation(final_obs)
                    next_obs_for_buffer[dones] = final_obs[dones]

                self.agent.store_transition(
                    observations=obs,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_obs_for_buffer,
                    dones=terminated.float(),
                )

                self.current_episode_rewards += rewards
                self.current_episode_lengths += 1
                if dones.any():
                    done_indices = torch.where(dones)[0]
                    for idx in done_indices:
                        self.episode_rewards.append(
                            self.current_episode_rewards[idx].item()
                        )
                        self.episode_lengths.append(
                            float(self.current_episode_lengths[idx].item())
                        )
                    self.current_episode_rewards[dones] = 0.0
                    self.current_episode_lengths[dones] = 0

                obs = next_obs
                self.total_timesteps += self.env.num_envs
                self.agent.total_timesteps = self.total_timesteps

                if (
                    self.total_timesteps >= self.cfg.learning_starts
                    and self.iteration % self.cfg.train_freq == 0
                ):
                    updates = []
                    for _ in range(self.cfg.gradient_steps):
                        stats = self.agent.update()
                        if stats:
                            updates.append(stats)
                    if updates:
                        last_update_stats = {
                            key: sum(update[key] for update in updates) / len(updates)
                            for key in updates[0]
                        }
                        history["q_losses"].append(last_update_stats["train/q_loss"])

                should_log = (
                    self.log_interval > 0
                    and self.total_timesteps % self.log_interval < self.env.num_envs
                )
                if should_log:
                    elapsed = max(time.time() - self.start_time, 1e-6)
                    fps = int(self.total_timesteps / elapsed)
                    epsilon = self.agent.get_epsilon(self.total_timesteps)
                    scalars = {
                        "time/fps": fps,
                        "exploration/epsilon": epsilon,
                        "buffer/size": float(len(self.agent.replay_buffer)),
                        "rollout/mean_reward": rewards.mean().item(),
                    }
                    scalars.update(last_update_stats)
                    if self.episode_rewards:
                        mean_reward = sum(self.episode_rewards) / len(
                            self.episode_rewards
                        )
                        mean_length = sum(self.episode_lengths) / len(
                            self.episode_lengths
                        )
                        scalars["episode/mean_reward"] = mean_reward
                        scalars["episode/mean_length"] = mean_length
                        history["episode_rewards"].append(mean_reward)
                        history["episode_lengths"].append(mean_length)
                    self._log_scalars(scalars, self.total_timesteps)
                    print(
                        f"Steps {self.total_timesteps:,} | FPS {fps} | "
                        f"Epsilon {epsilon:.3f} | "
                        "Q Loss "
                        f"{last_update_stats.get('train/q_loss', float('nan')):.4f}"
                    )

                should_save = (
                    self.save_interval > 0
                    and self.total_timesteps % self.save_interval < self.env.num_envs
                )
                if should_save:
                    self.save_checkpoint(f"checkpoint_{self.total_timesteps}.pt")

                history["timesteps"].append(self.total_timesteps)

            if self.save_dir:
                self.save_checkpoint("checkpoint_final.pt")
        finally:
            if self.logger:
                self.logger.close()

        return {
            "final_iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "history": history,
        }

    def close(self) -> None:
        """Close runner and release resources."""
        if self.logger:
            self.logger.close()
        if hasattr(self.env, "close"):
            self.env.close()
