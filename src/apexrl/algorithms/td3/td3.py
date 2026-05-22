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

"""Twin Delayed Deep Deterministic Policy Gradient implementation.

TD3 is an off-policy actor-critic algorithm for continuous action spaces. It
uses twin critics to reduce over-estimation, delayed actor updates, and clipped
target policy smoothing noise.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from gymnasium import spaces

from apexrl.algorithms.td3.config import TD3Config
from apexrl.buffer.replay_buffer import ReplayBuffer
from apexrl.models import MLPContinuousQNetwork, MLPDeterministicActor
from apexrl.optimizers import build_optimizer
from apexrl.utils import (
    actor_space_from_observation_space,
    critic_space_from_observation_space,
    observation_to_tensor,
    space_to_spec,
    split_actor_critic_observations,
)


class TD3:
    """Twin Delayed DDPG for continuous-action environments.

    The actor is deterministic and returns bounded continuous actions. The
    critics estimate ``Q(s, a)``. During training, targets use the minimum of
    two target critics and a clipped noise perturbation on target actions.
    """

    def __init__(
        self,
        env: Any,
        cfg: TD3Config | None = None,
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
        """Initialize the TD3 agent."""
        self.env = env
        self.cfg = cfg or TD3Config()
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

        full_obs_space = obs_space or getattr(env, "observation_space_gym", None)
        self.action_space = action_space or getattr(env, "action_space_gym", None)
        if full_obs_space is None or self.action_space is None:
            raise ValueError("TD3 requires obs_space and action_space")
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError(
                f"TD3 only supports Box action spaces, got {type(self.action_space)}"
            )

        self.full_obs_space = full_obs_space
        self.obs_space = actor_space_from_observation_space(full_obs_space)
        self.critic_obs_space = (
            critic_space_from_observation_space(full_obs_space) or self.obs_space
        )
        self.num_envs = env.num_envs
        self.action_dim = (
            self.action_space.shape[0] if len(self.action_space.shape) > 0 else 1
        )
        self.action_low = torch.as_tensor(
            self.action_space.low,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_high = torch.as_tensor(
            self.action_space.high,
            dtype=torch.float32,
            device=self.device,
        )

        actor_class = actor_class or MLPDeterministicActor
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
                self.critic_obs_space,
                self.action_space,
                critic_cfg,
            ).to(self.device)
            self.critic2 = critic_class(
                self.critic_obs_space,
                self.action_space,
                critic_cfg,
            ).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        for module in (self.target_actor, self.target_critic1, self.target_critic2):
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()

        self.actor_optimizer = build_optimizer(
            self.cfg.optimizer,
            lr=self.cfg.actor_learning_rate,
            modules=self.actor,
        )
        self.critic1_optimizer = build_optimizer(
            self.cfg.optimizer,
            lr=self.cfg.critic_learning_rate,
            modules=self.critic1,
        )
        self.critic2_optimizer = build_optimizer(
            self.cfg.optimizer,
            lr=self.cfg.critic_learning_rate,
            modules=self.critic2,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            obs_shape=space_to_spec(self.obs_space),
            action_shape=self.action_space.shape,
            device=self.device,
            obs_dtype=torch.float32,
            action_dtype=torch.float32,
            critic_obs_shape=(
                space_to_spec(self.critic_obs_space)
                if self.critic_obs_space is not self.obs_space
                else None
            ),
        )

        self.iteration = 0
        self.total_timesteps = 0
        self.num_updates = 0

    def _build_actor_cfg(self, actor_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Merge TD3 defaults into actor configuration."""
        merged = {
            "hidden_dims": list(self.cfg.actor_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
        }
        if actor_cfg:
            merged.update(actor_cfg)
        return merged

    def _build_critic_cfg(self, critic_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Merge TD3 defaults into critic configuration."""
        merged = {
            "hidden_dims": list(self.cfg.critic_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
        }
        if critic_cfg:
            merged.update(critic_cfg)
        return merged

    def _to_tensor_observation(self, obs: Any) -> Any:
        """Convert environment observations to tensors on the agent device."""
        return observation_to_tensor(obs, device=self.device)

    def _split_observations(self, obs: Any) -> tuple[Any, Any | None]:
        """Split full observations into actor and critic branches."""
        return split_actor_critic_observations(obs)

    def get_epsilon(self, total_timesteps: int) -> float:
        """Runner compatibility shim for off-policy logging."""
        del total_timesteps
        return 0.0

    def sample_random_actions(self) -> torch.Tensor:
        """Sample uniformly random actions from environment bounds."""
        return self.action_low + torch.rand(
            (self.num_envs, self.action_dim),
            device=self.device,
        ) * (self.action_high - self.action_low)

    def _actor_forward(
        self,
        obs: Any,
        actor: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Return deterministic actions from an actor module."""
        actor = actor or self.actor
        output = actor(obs)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def act(
        self,
        obs: Any,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """Return actions from the deterministic policy.

        During training, Gaussian exploration noise is added unless
        ``deterministic`` is true. ``epsilon`` is accepted for runner
        compatibility and ignored.
        """
        del epsilon
        actor_obs, _ = self._split_observations(self._to_tensor_observation(obs))
        with torch.no_grad():
            actions = self._actor_forward(actor_obs)
            if not deterministic and self.cfg.exploration_noise > 0:
                actions = (
                    actions + torch.randn_like(actions) * self.cfg.exploration_noise
                )
            actions = self._clip_actions(actions)
        return actions

    def store_transition(
        self,
        observations: Any,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Any,
        dones: torch.Tensor,
    ) -> None:
        """Store a batch of transitions in replay."""
        observations = self._to_tensor_observation(observations)
        next_observations = self._to_tensor_observation(next_observations)
        actor_observations, critic_observations = self._split_observations(observations)
        next_actor_observations, next_critic_observations = self._split_observations(
            next_observations
        )
        self.replay_buffer.add(
            observations=actor_observations,
            actions=actions.to(self.device, dtype=torch.float32),
            rewards=rewards.to(self.device, dtype=torch.float32),
            next_observations=next_actor_observations,
            dones=dones.to(self.device, dtype=torch.float32),
            critic_observations=critic_observations,
            next_critic_observations=next_critic_observations,
        )

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip actions to environment bounds."""
        return torch.max(torch.min(actions, self.action_high), self.action_low)

    def _target_policy_actions(self, next_observations: Any) -> torch.Tensor:
        """Build target actions with clipped smoothing noise."""
        next_actions = self._actor_forward(next_observations, self.target_actor)
        if self.cfg.target_policy_noise > 0:
            noise = torch.randn_like(next_actions) * self.cfg.target_policy_noise
            if self.cfg.target_noise_clip > 0:
                noise = noise.clamp(
                    -self.cfg.target_noise_clip,
                    self.cfg.target_noise_clip,
                )
            next_actions = next_actions + noise
        return self._clip_actions(next_actions)

    def _clip_gradients(self, parameters: Any) -> float:
        """Clip gradients when enabled and return the unclipped norm estimate."""
        parameters = list(parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.cfg.max_grad_norm)
        return float(grad_norm)

    def _soft_update_targets(self) -> None:
        """Apply Polyak averaging to actor and critic target networks."""
        tau = self.cfg.tau
        for target, source in (
            (self.target_actor, self.actor),
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.lerp_(param.data, tau)

    def update(self) -> dict[str, float]:
        """Run one TD3 replay update."""
        if len(self.replay_buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {}

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]
        critic_observations = batch.get("critic_observations", observations)
        next_critic_observations = batch.get(
            "next_critic_observations",
            next_observations,
        )

        with torch.no_grad():
            next_actions = self._target_policy_actions(next_observations)
            target_q1 = self.target_critic1(next_critic_observations, next_actions)
            target_q2 = self.target_critic2(next_critic_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q1 = self.critic1(critic_observations, actions)
        current_q2 = self.critic2(critic_observations, actions)
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

        actor_loss_value = 0.0
        actor_grad_norm = 0.0
        actor_updated = (self.num_updates + 1) % self.cfg.policy_delay == 0
        if actor_updated:
            policy_actions = self._actor_forward(observations)
            actor_loss = -self.critic1(critic_observations, policy_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = self._clip_gradients(self.actor.parameters())
            self.actor_optimizer.step()
            self._soft_update_targets()
            actor_loss_value = actor_loss.item()

        self.num_updates += 1
        q_loss = 0.5 * (critic1_loss.item() + critic2_loss.item())
        return {
            "train/q_loss": q_loss,
            "train/critic1_loss": critic1_loss.item(),
            "train/critic2_loss": critic2_loss.item(),
            "train/actor_loss": actor_loss_value,
            "train/actor_updated": float(actor_updated),
            "train/mean_q": torch.min(current_q1, current_q2).mean().item(),
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
        """Save model, optimizer, and optionally replay state."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "num_updates": self.num_updates,
            "config": self.cfg,
        }
        if getattr(self.cfg, "save_replay_buffer", False):
            checkpoint["replay_buffer_state_dict"] = self.replay_buffer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model, optimizer, and optionally replay state."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(
            checkpoint["critic1_optimizer_state_dict"]
        )
        self.critic2_optimizer.load_state_dict(
            checkpoint["critic2_optimizer_state_dict"]
        )
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
