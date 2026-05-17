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

"""Runner for multi-agent on-policy algorithms."""

from __future__ import annotations

import collections
import os
import time
from collections.abc import Callable
from typing import Any

import torch

from apexrl.multiagent.envs import MultiAgentVecEnv
from apexrl.utils.logger import Logger


class MultiAgentRunner:
    """Training loop, logging and checkpointing for multi-agent algorithms."""

    def __init__(
        self,
        *,
        env: MultiAgentVecEnv,
        agent: Any,
        cfg: Any | None = None,
        log_dir: str | None = None,
        save_dir: str | None = None,
        device: torch.device | None = None,
        log_interval: int | None = None,
        save_interval: int | None = None,
    ):
        """Initialize multi-agent runner."""
        self.env = env
        self.agent = agent
        self.cfg = cfg or getattr(agent, "cfg", None)
        if self.cfg is None:
            raise ValueError("cfg is required when agent does not expose one")

        self.device = device or getattr(agent, "device", torch.device("cpu"))
        self.log_dir = log_dir
        self.save_dir = save_dir or log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.logger = None
        if self.log_dir:
            self.logger = Logger.create(
                backend=getattr(self.cfg, "logger_backend", "tensorboard"),
                experiment_name="multiagent_runner",
                log_dir=self.log_dir,
                **(getattr(self.cfg, "logger_kwargs", None) or {}),
            )
        elif getattr(agent, "logger", None) is not None:
            self.logger = agent.logger
        if getattr(agent, "logger", None) is None:
            agent.logger = self.logger

        self.log_interval = (
            log_interval
            if log_interval is not None
            else getattr(self.cfg, "log_interval", 10)
        )
        self.save_interval = (
            save_interval
            if save_interval is not None
            else getattr(self.cfg, "save_interval", 100)
        )
        self.iteration = getattr(agent, "iteration", 0)
        self.total_timesteps = getattr(agent, "total_timesteps", 0)
        self.start_time: float | None = None
        self.loss_history: dict[str, collections.deque] = {
            "iterations": collections.deque(maxlen=1000),
            "policy_loss": collections.deque(maxlen=1000),
            "value_loss": collections.deque(maxlen=1000),
            "entropy_loss": collections.deque(maxlen=1000),
        }
        self.extra_log_keys = list(getattr(self.cfg, "extra_log_keys", []))
        self.log_buffers: dict[str, collections.deque[float]] = {}
        self.log_buffer_maxlen = 1000
        self.callbacks: dict[str, list[Callable]] = {
            "pre_iteration": [],
            "post_iteration": [],
            "pre_rollout": [],
            "post_rollout": [],
            "pre_update": [],
            "post_update": [],
        }

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add a callback for a runner event."""
        if event not in self.callbacks:
            raise ValueError(
                f"Unknown event: {event}. Must be one of {list(self.callbacks.keys())}"
            )
        self.callbacks[event].append(callback)

    def _call_callbacks(self, event: str, *args) -> None:
        """Call all callbacks registered for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as exc:
                print(f"Warning: Callback failed for event '{event}': {exc}")

    def learn(
        self,
        *,
        total_timesteps: int | None = None,
        num_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Run collect-update training loop."""
        total_iters = self._resolve_total_iterations(
            total_timesteps=total_timesteps,
            num_iterations=num_iterations,
        )
        transitions_per_iter = self._transitions_per_iteration()

        print(
            "Training for "
            f"{total_iters} iterations "
            f"({total_iters * transitions_per_iter:,} agent-steps)"
        )
        print(f"  Batch size: {transitions_per_iter:,} agent-transitions/iteration")
        print(f"  Agents: {len(self.agent.possible_agents)}")
        print(f"  Envs: {self.env.num_envs}")

        history = {
            "iterations": [],
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
        }
        self.start_time = time.time()
        last_log_time = self.start_time
        last_stats: dict[str, float] = {}

        try:
            for iteration in range(total_iters):
                self.iteration = iteration
                self.agent.iteration = iteration
                self._call_callbacks("pre_iteration", self)
                if hasattr(self.agent, "adjust_learning_rate"):
                    self.agent.adjust_learning_rate(iteration, total_iters)

                rollout_stats = self.collect_rollout()
                update_stats = self.update()
                self.total_timesteps = self.agent.total_timesteps
                last_stats = {**rollout_stats, **update_stats}
                self._update_history(iteration, rollout_stats, update_stats, history)

                if iteration % self.log_interval == 0:
                    self._log_iteration(
                        iteration,
                        total_iters,
                        rollout_stats,
                        update_stats,
                        last_log_time,
                    )
                    last_log_time = time.time()

                if self.save_dir and iteration % self.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{iteration}.pt")
                self._call_callbacks("post_iteration", self, last_stats)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            if self.save_dir:
                self.save_checkpoint("checkpoint_final.pt")
            if self.logger:
                self.logger.close()

        print(f"\nTraining complete! Total steps: {self.total_timesteps:,}")
        return {
            "history": history,
            "final_iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "stats": last_stats,
        }

    def collect_rollout(self) -> dict[str, float]:
        """Collect rollout and process configured environment extras."""
        self._call_callbacks("pre_rollout", self)
        stats = self.agent.collect_rollout(extras_callback=self._process_extras)
        self.total_timesteps = self.agent.total_timesteps
        self._call_callbacks("post_rollout", self, stats)
        return stats

    def update(self) -> dict[str, float]:
        """Update agent parameters."""
        self._call_callbacks("pre_update", self)
        stats = self.agent.update()
        self._call_callbacks("post_update", self, stats)
        return stats

    def _process_extras(
        self,
        extras: dict[str, Any],
        dones: dict[str, torch.Tensor],
        terminated: dict[str, torch.Tensor],
        rewards: dict[str, torch.Tensor],
    ) -> None:
        """Process selected environment extras for logging."""
        del dones, terminated, rewards
        for extra_key in self.extra_log_keys:
            if extra_key not in extras:
                continue
            for key, value in self._flatten_extra_metrics(
                extras[extra_key],
                str(extra_key).strip("/"),
            ).items():
                self._append_log_buffer(f"/extra/{key}", value)

    def _append_log_buffer(self, key: str, value: Any) -> None:
        """Append a numeric metric value to a rolling log buffer."""
        metric = self._to_scalar(value)
        if metric is None:
            return
        if key not in self.log_buffers:
            self.log_buffers[key] = collections.deque(maxlen=self.log_buffer_maxlen)
        self.log_buffers[key].append(metric)

    def _to_scalar(self, value: Any) -> float | None:
        """Convert scalar-like values or tensors to float."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.float().mean().item())
        if isinstance(value, bool | int | float):
            return float(value)
        return None

    def _flatten_extra_metrics(
        self,
        value: Any,
        prefix: str = "",
    ) -> dict[str, float]:
        """Flatten scalar-like extras and skip bulky payloads."""
        if not prefix and not isinstance(value, dict):
            return {}
        skip_keys = {"final_observation"}
        metrics: dict[str, float] = {}
        if isinstance(value, dict):
            for key, sub_value in value.items():
                if not prefix and key in skip_keys:
                    continue
                clean_key = str(key).strip("/")
                next_prefix = f"{prefix}/{clean_key}" if prefix else clean_key
                metrics.update(self._flatten_extra_metrics(sub_value, next_prefix))
            return metrics

        scalar = self._to_scalar(value)
        if scalar is not None and prefix:
            metrics[prefix] = scalar
        return metrics

    def _resolve_total_iterations(
        self,
        *,
        total_timesteps: int | None,
        num_iterations: int | None,
    ) -> int:
        """Resolve number of training iterations."""
        if num_iterations is not None:
            return int(num_iterations)
        if getattr(self.cfg, "max_iterations", None) is not None:
            return int(self.cfg.max_iterations)
        if total_timesteps is not None:
            return max(1, int(total_timesteps) // self._transitions_per_iteration())
        raise ValueError(
            "Provide total_timesteps, num_iterations or cfg.max_iterations"
        )

    def _transitions_per_iteration(self) -> int:
        """Return agent transitions per rollout/update iteration."""
        return int(
            self.cfg.num_steps * self.env.num_envs * len(self.agent.possible_agents)
        )

    def _update_history(
        self,
        iteration: int,
        rollout_stats: dict[str, float],
        update_stats: dict[str, float],
        history: dict[str, list],
    ) -> None:
        """Update in-memory training history."""
        history["iterations"].append(iteration)
        history["timesteps"].append(self.total_timesteps)
        history["policy_losses"].append(update_stats.get("train/policy_loss", 0.0))
        history["value_losses"].append(update_stats.get("train/value_loss", 0.0))

        self.loss_history["iterations"].append(iteration)
        self.loss_history["policy_loss"].append(
            update_stats.get("train/policy_loss", 0.0)
        )
        self.loss_history["value_loss"].append(
            update_stats.get("train/value_loss", 0.0)
        )
        self.loss_history["entropy_loss"].append(
            update_stats.get("train/entropy_loss", 0.0)
        )

        if "episode/mean_reward" in rollout_stats:
            history["episode_rewards"].append(rollout_stats["episode/mean_reward"])
            history["episode_lengths"].append(rollout_stats["episode/mean_length"])

    def _log_iteration(
        self,
        iteration: int,
        total_iters: int,
        rollout_stats: dict[str, float],
        update_stats: dict[str, float],
        last_log_time: float,
    ) -> None:
        """Print and write metrics for one iteration."""
        elapsed = time.time() - last_log_time
        fps = (
            self._transitions_per_iteration() * self.log_interval / elapsed
            if elapsed > 0
            else 0.0
        )
        msg = (
            f"Iter {iteration}/{total_iters} | "
            f"Steps {self.total_timesteps:,} | "
            f"FPS {fps:.0f} | "
            f"Reward {rollout_stats.get('rollout/mean_reward', 0):.3f} | "
            f"Policy Loss {update_stats.get('train/policy_loss', 0):.4f} | "
            f"Value Loss {update_stats.get('train/value_loss', 0):.4f} | "
            f"KL {update_stats.get('train/approx_kl', 0):.5f}"
        )
        if "episode/mean_reward" in rollout_stats:
            msg += f" | Episode Reward {rollout_stats['episode/mean_reward']:.3f}"
        print(msg)

        if not self.logger:
            return
        self._log_scalars({"time/fps": fps}, self.total_timesteps)
        self._log_scalars(
            self._get_rollout_metrics_for_logging(rollout_stats),
            self.total_timesteps,
        )
        self._log_scalars(update_stats, self.total_timesteps)
        self._log_scalars(
            self._get_train_iteration_metrics(update_stats),
            iteration,
        )
        self._log_scalars(
            self._get_episode_iteration_metrics(rollout_stats),
            iteration,
        )
        self._log_scalars(self._get_detailed_rollout_stats(), iteration)
        self._log_environment_metrics()

    def _log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Log scalar metrics if a backend logger exists."""
        if self.logger and scalars:
            self.logger.log_scalars(scalars, step)

    def _log_environment_metrics(self) -> None:
        """Log environment metrics accumulated from extras."""
        env_metrics = {}
        for key, buffer in self.log_buffers.items():
            if not buffer:
                continue
            mean_value = sum(buffer) / len(buffer)
            log_key = key[1:] if key.startswith("/") else key
            metric_key = log_key if log_key.startswith("extra/") else f"env/{log_key}"
            env_metrics[metric_key] = mean_value
            buffer.clear()
        self._log_scalars(env_metrics, self.iteration)

    def _get_rollout_metrics_for_logging(
        self,
        rollout_stats: dict[str, float],
    ) -> dict[str, float]:
        """Return rollout metrics to log on timestep axis."""
        return dict(rollout_stats)

    def _get_train_iteration_metrics(
        self,
        update_stats: dict[str, float],
    ) -> dict[str, float]:
        """Return optional train metrics on iteration axis."""
        if not getattr(self.cfg, "log_train_metrics_vs_iteration", False):
            return {}
        return {
            key.replace("train/", "train_vs_iter/"): value
            for key, value in update_stats.items()
            if key.startswith("train/")
        }

    def _get_episode_iteration_metrics(
        self,
        rollout_stats: dict[str, float],
    ) -> dict[str, float]:
        """Return optional episode metrics on iteration axis."""
        if not getattr(self.cfg, "log_episode_metrics_vs_iteration", False):
            return {}
        if "episode/mean_reward" not in rollout_stats:
            return {}
        return {
            "episode_vs_iter/mean_reward": rollout_stats["episode/mean_reward"],
            "episode_vs_iter/mean_length": rollout_stats["episode/mean_length"],
        }

    def _get_detailed_rollout_stats(self) -> dict[str, float]:
        """Return optional rollout distribution stats."""
        if not getattr(self.cfg, "log_detailed_rollout_stats", False):
            return {}
        buffer = getattr(self.agent, "rollout_buffer", None)
        if buffer is None:
            return {}
        advantages = torch.cat(
            [
                buffer.advantages[agent_id].reshape(-1)
                for agent_id in buffer.possible_agents
            ]
        )
        values = torch.cat(
            [buffer.values[agent_id].reshape(-1) for agent_id in buffer.possible_agents]
        )
        returns = torch.cat(
            [
                buffer.returns[agent_id].reshape(-1)
                for agent_id in buffer.possible_agents
            ]
        )
        return {
            "advantage/mean": advantages.mean().item(),
            "advantage/std": advantages.std().item(),
            "value/mean": values.mean().item(),
            "value/std": values.std().item(),
            "returns/mean": returns.mean().item(),
            "returns/std": returns.std().item(),
        }

    def save_checkpoint(self, filename: str) -> str:
        """Save a checkpoint under save_dir."""
        if not self.save_dir:
            raise ValueError("save_dir is not configured")
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        print(f"  Saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        self.agent.load(path)
        self.iteration = getattr(self.agent, "iteration", 0)
        self.total_timesteps = getattr(self.agent, "total_timesteps", 0)
        print(f"Loaded checkpoint: {path}")
