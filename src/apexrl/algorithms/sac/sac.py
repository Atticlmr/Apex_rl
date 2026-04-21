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

"""Soft Actor-Critic implementation for continuous-control tasks.

This module provides a minimal, practical SAC implementation that fits the
existing ApexRL off-policy stack:

- replay-buffer based training
- twin critics with target networks
- temperature tuning via entropy regularization
- runner-compatible ``act`` / ``update`` / ``learn`` methods
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from gymnasium import spaces

from apexrl.algorithms.sac.config import SACConfig
from apexrl.buffer.replay_buffer import ReplayBuffer
from apexrl.models import MLPContinuousQNetwork, MLPSquashedGaussianActor
from apexrl.optimizers import get_optimizer


class SAC:
    """Soft Actor-Critic for continuous-action environments.

    SAC jointly trains:

    - an actor that samples entropy-regularized actions
    - two critics that estimate ``Q(s, a)``
    - target critics for stable bootstrapping
    - an optional temperature parameter ``alpha`` controlling exploration
    """

    def __init__(
        self,
        env: Any,
        cfg: SACConfig | None = None,
        actor_class: type | None = None,
        critic_class: type | None = None,
        obs_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        actor_cfg: dict[str, Any] | None = None,
        critic_cfg: dict[str, Any] | None = None,
        actor: torch.nn.Module | None = None,
        critic1: torch.nn.Module | None = None,
        critic2: torch.nn.Module | None = None,
        log_dir: str | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the SAC agent."""
        self.env = env
        self.cfg = cfg or SACConfig()
        self.log_dir = log_dir
        self.logger = None

        if device is None:
            if self.cfg.device == "auto":
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device(self.cfg.device)
        else:
            self.device = device

        self.obs_space = obs_space or getattr(env, "observation_space_gym", None)
        self.action_space = action_space or getattr(env, "action_space_gym", None)
        if self.obs_space is None or self.action_space is None:
            raise ValueError("SAC requires obs_space and action_space")
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError(
                "SAC only supports Box action spaces, "
                f"got {type(self.action_space)}"
            )
        if not isinstance(self.obs_space, spaces.Box):
            raise ValueError(
                "SAC currently only supports Box observation spaces, "
                f"got {type(self.obs_space)}"
            )

        self.num_envs = env.num_envs
        self.action_dim = (
            self.action_space.shape[0] if len(self.action_space.shape) > 0 else 1
        )

        actor_class = actor_class or MLPSquashedGaussianActor
        critic_class = critic_class or MLPContinuousQNetwork

        if actor is not None and critic1 is not None and critic2 is not None:
            self.actor = actor.to(self.device)
            self.critic1 = critic1.to(self.device)
            self.critic2 = critic2.to(self.device)
        else:
            actor_cfg = self._build_actor_cfg(actor_cfg)
            critic_cfg = self._build_critic_cfg(critic_cfg)
            self.actor = actor_class(
                self.obs_space,
                self.action_space,
                actor_cfg,
            ).to(self.device)
            self.critic1 = critic_class(
                self.obs_space,
                self.action_space,
                critic_cfg,
            ).to(self.device)
            self.critic2 = critic_class(
                self.obs_space,
                self.action_space,
                critic_cfg,
            ).to(self.device)

        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        for param in self.target_critic1.parameters():
            param.requires_grad_(False)
        for param in self.target_critic2.parameters():
            param.requires_grad_(False)
        self.target_critic1.eval()
        self.target_critic2.eval()

        optimizer_cls = get_optimizer(self.cfg.optimizer)
        self.actor_optimizer = optimizer_cls(
            self.actor.parameters(),
            lr=self.cfg.actor_learning_rate,
        )
        self.critic1_optimizer = optimizer_cls(
            self.critic1.parameters(),
            lr=self.cfg.critic_learning_rate,
        )
        self.critic2_optimizer = optimizer_cls(
            self.critic2.parameters(),
            lr=self.cfg.critic_learning_rate,
        )

        self.auto_alpha = self.cfg.auto_alpha
        if self.auto_alpha:
            init_log_alpha = math.log(self.cfg.init_alpha)
            self.log_alpha = torch.tensor(
                init_log_alpha,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=self.cfg.alpha_learning_rate,
            )
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = float(self.cfg.init_alpha)

        self.target_entropy = (
            self.cfg.target_entropy
            if self.cfg.target_entropy is not None
            else -float(self.action_dim)
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            obs_shape=tuple(self.obs_space.shape),
            action_shape=self.action_space.shape,
            device=self.device,
            obs_dtype=torch.float32,
            action_dtype=torch.float32,
        )

        self.iteration = 0
        self.total_timesteps = 0
        self.num_updates = 0

    def _build_actor_cfg(
        self,
        actor_cfg: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge SAC defaults into actor configuration."""
        merged = {
            "hidden_dims": list(self.cfg.actor_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
            "use_tanh_squash": self.cfg.use_tanh_squash,
            "min_log_std": self.cfg.min_log_std,
            "max_log_std": self.cfg.max_log_std,
        }
        if actor_cfg:
            merged.update(actor_cfg)
        return merged

    def _build_critic_cfg(
        self,
        critic_cfg: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge SAC defaults into critic configuration."""
        merged = {
            "hidden_dims": list(self.cfg.critic_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
        }
        if critic_cfg:
            merged.update(critic_cfg)
        return merged

    def _to_tensor_observation(self, obs: Any) -> torch.Tensor:
        """Convert environment observations to float tensors on the agent device."""
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            elif len(obs) == 1:
                obs = next(iter(obs.values()))
            else:
                raise ValueError(
                    "SAC requires a single tensor observation or a dict with key 'obs'"
                )
        if hasattr(obs, "get"):
            obs = obs["obs"]
        elif isinstance(obs, tuple):
            obs = obs[0]

        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device, dtype=torch.float32)
        return obs

    def get_alpha(self) -> torch.Tensor:
        """Return the current entropy temperature on the training device."""
        if self.auto_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha, dtype=torch.float32, device=self.device)

    def get_epsilon(self, total_timesteps: int) -> float:
        """Runner compatibility shim for off-policy logging.

        SAC does not use epsilon-greedy exploration, so this always returns
        ``0.0``. The actual exploration strength is controlled by ``alpha``.
        """

        del total_timesteps
        return 0.0

    def sample_random_actions(self) -> torch.Tensor:
        """Sample uniformly random actions from the environment bounds."""
        action_low = torch.as_tensor(
            self.action_space.low,
            dtype=torch.float32,
            device=self.device,
        )
        action_high = torch.as_tensor(
            self.action_space.high,
            dtype=torch.float32,
            device=self.device,
        )
        return action_low + torch.rand(
            (self.num_envs, self.action_dim),
            device=self.device,
        ) * (action_high - action_low)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """Sample or select actions from the current policy.

        ``epsilon`` is accepted only for compatibility with ``OffPolicyRunner``
        and is ignored.
        """

        del epsilon
        obs = self._to_tensor_observation(obs)
        with torch.no_grad():
            actions, _ = self.actor.act(obs, deterministic=deterministic)
        return actions

    def store_transition(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Store a batch of transitions in replay."""
        self.replay_buffer.add(
            observations=self._to_tensor_observation(observations),
            actions=actions.to(self.device, dtype=torch.float32),
            rewards=rewards.to(self.device, dtype=torch.float32),
            next_observations=self._to_tensor_observation(next_observations),
            dones=dones.to(self.device, dtype=torch.float32),
        )

    def _clip_gradients(self, parameters: Any) -> float:
        """Clip gradients when enabled and return the unclipped norm estimate."""
        parameters = list(parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            float("inf"),
        )
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                self.cfg.max_grad_norm,
            )
        return float(grad_norm)

    def _soft_update_targets(self) -> None:
        """Apply Polyak averaging to target critics."""
        if self.num_updates % self.cfg.target_update_interval != 0:
            return
        tau = self.cfg.tau
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.lerp_(param.data, tau)
        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.lerp_(param.data, tau)

    def update(self) -> dict[str, float]:
        """Run one SAC update from a replay batch.

        Update flow:

        1. Sample replay transitions.
        2. Build target values with target critics and entropy regularization.
        3. Regress both critics to the shared TD target.
        4. Update the actor against the conservative ``min(Q1, Q2)`` estimate.
        5. Optionally tune ``alpha`` toward the target entropy.
        6. Soft-update target critics.
        """

        if len(self.replay_buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {}

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        alpha = self.get_alpha()

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.act(next_observations)
            target_q1 = self.target_critic1(next_observations, next_actions)
            target_q2 = self.target_critic2(next_observations, next_actions)
            # SAC target:
            # y = r + gamma * (1 - d) * (min(Q1', Q2') - alpha * log pi(a'|s'))
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)
        # Critic losses:
        # L_Qi = E[(Qi(s, a) - y)^2]
        critic1_loss = F.mse_loss(current_q1, td_target)
        critic2_loss = F.mse_loss(current_q2, td_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_grad_norm = self._clip_gradients(self.critic1.parameters())
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_grad_norm = self._clip_gradients(self.critic2.parameters())
        self.critic2_optimizer.step()

        policy_actions, log_probs = self.actor.act(observations)
        q1_pi = self.critic1(observations, policy_actions)
        q2_pi = self.critic2(observations, policy_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        # Actor loss:
        # L_pi = E[alpha * log pi(a|s) - min(Q1(s, a), Q2(s, a))]
        actor_loss = (alpha * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor.parameters())
        self.actor_optimizer.step()

        if self.auto_alpha:
            # Temperature loss:
            # L_alpha = -E[log_alpha * (log pi(a|s) + target_entropy)]
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_value = self.get_alpha()
            alpha_loss_value = alpha_loss.item()
        else:
            alpha_value = alpha
            alpha_loss_value = 0.0

        self.num_updates += 1
        self._soft_update_targets()

        q_loss = 0.5 * (critic1_loss.item() + critic2_loss.item())
        return {
            "train/q_loss": q_loss,
            "train/critic1_loss": critic1_loss.item(),
            "train/critic2_loss": critic2_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/alpha_loss": alpha_loss_value,
            "train/alpha": alpha_value.item(),
            "train/entropy": (-log_probs).mean().item(),
            "train/mean_q": min_q_pi.mean().item(),
            "train/td_target_mean": td_target.mean().item(),
            "train/actor_grad_norm": actor_grad_norm,
            "train/critic1_grad_norm": critic1_grad_norm,
            "train/critic2_grad_norm": critic2_grad_norm,
            "train/learning_rate_actor": self.actor_optimizer.param_groups[0]["lr"],
            "train/learning_rate_critic": self.critic1_optimizer.param_groups[0]["lr"],
        }

    def learn(self, total_timesteps: int | None = None) -> dict[str, Any]:
        """Train through the canonical off-policy runner entrypoint."""
        from apexrl.agent.off_policy_runner import OffPolicyRunner

        runner = OffPolicyRunner(
            agent=self,
            env=self.env,
            cfg=self.cfg,
            log_dir=None,
            save_dir=self.log_dir,
            device=self.device,
        )
        return runner.learn(total_timesteps=total_timesteps)

    def save(self, path: str) -> None:
        """Save model, optimizer, and replay state."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "alpha_optimizer_state_dict": (
                self.alpha_optimizer.state_dict() if self.alpha_optimizer else None
            ),
            "log_alpha": (
                self.log_alpha.detach().cpu() if self.log_alpha is not None else None
            ),
            "alpha": self.alpha if not self.auto_alpha else None,
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "num_updates": self.num_updates,
            "config": self.cfg,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model, optimizer, and replay state."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(
            checkpoint["critic1_optimizer_state_dict"]
        )
        self.critic2_optimizer.load_state_dict(
            checkpoint["critic2_optimizer_state_dict"]
        )
        if self.alpha_optimizer and checkpoint.get("alpha_optimizer_state_dict"):
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )
        if self.auto_alpha and checkpoint.get("log_alpha") is not None:
            self.log_alpha.data.copy_(
                checkpoint["log_alpha"].to(self.device, dtype=torch.float32)
            )
        elif checkpoint.get("alpha") is not None:
            self.alpha = float(checkpoint["alpha"])

        if checkpoint.get("replay_buffer_state_dict"):
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])

        self.iteration = checkpoint.get("iteration", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.num_updates = checkpoint.get("num_updates", 0)

    def eval(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate the deterministic policy on the current environment."""
        obs = self._to_tensor_observation(self.env.reset())
        episode_rewards: list[float] = []
        current_rewards = torch.zeros(self.num_envs, device=self.device)
        episodes_completed = 0

        while episodes_completed < num_episodes:
            actions = self.act(obs, deterministic=True)
            next_obs, rewards, dones, _ = self.env.step(actions)
            obs = self._to_tensor_observation(next_obs)
            rewards = rewards.to(self.device, dtype=torch.float32)
            dones = dones.to(self.device).bool()

            current_rewards += rewards
            if dones.any():
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    if episodes_completed < num_episodes:
                        episode_rewards.append(current_rewards[idx].item())
                        episodes_completed += 1
                current_rewards[dones] = 0.0

        rewards_tensor = torch.as_tensor(episode_rewards, dtype=torch.float32)
        return {
            "eval/mean_reward": rewards_tensor.mean().item(),
            "eval/std_reward": rewards_tensor.std(unbiased=False).item(),
            "eval/min_reward": rewards_tensor.min().item(),
            "eval/max_reward": rewards_tensor.max().item(),
        }
