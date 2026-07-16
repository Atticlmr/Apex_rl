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

"""FlashSAC implementation for high-throughput continuous control.

FlashSAC is a SAC-style algorithm tuned for large batches, large parallel
rollouts, low update-to-data ratios, and critic stabilization in
high-dimensional robot-control settings.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from gymnasium import spaces

from apexrl.algorithms.flash_sac.config import FlashSACConfig
from apexrl.algorithms.flash_sac.distribution import categorical_td_projection
from apexrl.algorithms.flash_sac.network import (
    FlashSACActor,
    FlashSACCategoricalQNetwork,
)
from apexrl.algorithms.flash_sac.reward_normalization import RewardNormalizer
from apexrl.algorithms.sac.sac import SAC


class FlashSAC(SAC):
    """FlashSAC-style variant of Soft Actor-Critic.

    The algorithm keeps SAC's entropy-regularized actor-critic objective while
    adding practical support for high-throughput off-policy training:
    reward scaling, larger default networks, feature-norm regularization,
    optional weight-norm clamping, and explicit update-to-data logging.
    """

    def __init__(
        self,
        env: Any,
        cfg: FlashSACConfig | None = None,
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
        """Initialize FlashSAC."""
        cfg = cfg or FlashSACConfig()
        if cfg.target_entropy is None:
            resolved_action_space = action_space or env.action_space_gym
            action_dim = int(resolved_action_space.shape[0])
            action_low = torch.as_tensor(resolved_action_space.low)
            action_high = torch.as_tensor(resolved_action_space.high)
            action_scale = (action_high - action_low) / 2.0
            cfg.target_entropy = (
                0.5
                * action_dim
                * math.log(2.0 * math.pi * math.e * cfg.target_sigma**2)
                + torch.log(action_scale.abs()).sum().item()
            )

        default_actor = actor_class is None and actor is None
        default_critics = critic_class is None and critic1 is None and critic2 is None
        actor_cfg = {
            "hidden_dim": cfg.actor_hidden_dim,
            "num_blocks": cfg.actor_num_blocks,
            "min_log_std": cfg.min_log_std,
            "max_log_std": cfg.max_log_std,
            **(actor_cfg or {}),
        }
        critic_cfg = {
            "hidden_dim": cfg.critic_hidden_dim,
            "num_blocks": cfg.critic_num_blocks,
            "num_bins": cfg.critic_num_bins,
            "min_value": cfg.critic_min_value,
            "max_value": cfg.critic_max_value,
            **(critic_cfg or {}),
        }
        super().__init__(
            env=env,
            cfg=cfg,
            actor_class=actor_class or FlashSACActor,
            critic_class=critic_class or FlashSACCategoricalQNetwork,
            obs_space=obs_space,
            action_space=action_space,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            log_dir=log_dir,
            device=device,
        )
        self._distributional_critic = bool(
            cfg.use_distributional_critic
            and default_critics
            and hasattr(self.critic1, "distribution")
            and hasattr(self.critic2, "distribution")
        )
        self._official_actor = bool(
            default_actor and hasattr(self.actor, "act_with_noise")
        )
        self.accepts_episode_ends = True
        self.reward_normalizer = (
            RewardNormalizer(
                num_envs=self.num_envs,
                gamma=cfg.gamma,
                max_return=cfg.normalized_return_max,
                device=self.device,
            )
            if cfg.normalize_rewards
            else None
        )
        repeat_lengths = torch.arange(1, cfg.noise_repeat_max + 1, device=self.device)
        probabilities = repeat_lengths.float().pow(-cfg.noise_repeat_zeta)
        self._noise_repeat_cdf = (probabilities / probabilities.sum()).cumsum(0)
        self._cached_noise = torch.randn(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )
        self._noise_repeat_counts = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._noise_repeat_lengths = torch.ones_like(self._noise_repeat_counts)
        self._normalize_module_parameters(self.actor)
        self._normalize_module_parameters(self.critic1)
        self._normalize_module_parameters(self.critic2)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    @torch.no_grad()
    def _normalize_module_parameters(self, module: torch.nn.Module) -> None:
        if self.cfg.unit_weight_normalization and hasattr(
            module, "normalize_parameters"
        ):
            module.normalize_parameters()

    def _sample_noise_repeat_lengths(self, count: int) -> torch.Tensor:
        samples = torch.rand(count, device=self.device)
        return torch.searchsorted(self._noise_repeat_cdf, samples).long() + 1

    def act(
        self,
        obs: Any,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """Select actions with FlashSAC's temporally repeated policy noise."""
        del epsilon
        obs, _ = self._split_observations(self._to_tensor_observation(obs))
        previous_training = self.actor.training
        self.actor.eval()
        try:
            with torch.no_grad():
                if deterministic or not self._official_actor:
                    actions, _ = self.actor.act(obs, deterministic=deterministic)
                    return actions

                batch_size = (
                    next(iter(obs.values())).shape[0]
                    if isinstance(obs, dict)
                    else obs.shape[0]
                )
                if batch_size != self._cached_noise.shape[0]:
                    self._cached_noise = torch.randn(
                        batch_size, self.action_dim, device=self.device
                    )
                    self._noise_repeat_counts = torch.zeros(
                        batch_size, dtype=torch.long, device=self.device
                    )
                    self._noise_repeat_lengths = torch.ones_like(
                        self._noise_repeat_counts
                    )
                refresh = (self._noise_repeat_counts == 0) | (
                    self._noise_repeat_counts >= self._noise_repeat_lengths
                )
                refresh_count = int(refresh.sum().item())
                if refresh_count:
                    self._cached_noise[refresh] = torch.randn(
                        refresh_count, self.action_dim, device=self.device
                    )
                    self._noise_repeat_lengths[refresh] = (
                        self._sample_noise_repeat_lengths(refresh_count)
                    )
                    self._noise_repeat_counts[refresh] = 0
                actions = self.actor.act_with_noise(obs, self._cached_noise)
                self._noise_repeat_counts += 1
                return actions
        finally:
            self.actor.train(previous_training)

    def store_transition(
        self,
        observations: Any,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Any,
        dones: torch.Tensor,
        episode_ends: torch.Tensor | None = None,
    ) -> None:
        """Store replay data and update discounted-return reward statistics."""
        super().store_transition(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )
        if self.reward_normalizer is not None:
            ends = dones if episode_ends is None else episode_ends
            self.reward_normalizer.update(
                rewards.to(self.device, dtype=torch.float32),
                ends.to(self.device).bool(),
            )

    def _scaled_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.normalize(rewards)
        return rewards * self.cfg.reward_scale

    def _module_l2_norm(self, module: torch.nn.Module) -> torch.Tensor:
        """Return the L2 norm of all trainable parameters in a module."""
        norms = [
            param.detach().float().norm(2)
            for param in module.parameters()
            if param.requires_grad
        ]
        if not norms:
            return torch.zeros((), device=self.device)
        return torch.stack(norms).norm(2)

    def _clamp_module_weight_norm(
        self,
        module: torch.nn.Module,
        max_norm: float | None,
    ) -> float:
        """Scale trainable parameters down when their global norm is too large."""
        weight_norm = self._module_l2_norm(module)
        if max_norm is not None and weight_norm.item() > max_norm:
            scale = max_norm / (weight_norm.item() + 1e-12)
            with torch.no_grad():
                for param in module.parameters():
                    if param.requires_grad:
                        param.mul_(scale)
            weight_norm = self._module_l2_norm(module)
        return float(weight_norm.item())

    def _critic_forward_with_features(
        self,
        critic: torch.nn.Module,
        observations: Any,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a critic and return optional penultimate features."""
        if hasattr(critic, "forward_with_features"):
            return critic.forward_with_features(observations, actions)
        return critic(observations, actions), None

    def _feature_norm_loss(
        self,
        features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float]:
        """Return critic feature-norm regularization loss and metric value."""
        if features is None:
            return torch.zeros((), device=self.device), 0.0
        feature_norm = features.float().norm(2, dim=-1).mean()
        target = self.cfg.critic_feature_norm_target
        if target is None:
            penalty = feature_norm.square()
        else:
            penalty = F.relu(feature_norm - target).square()
        return self.cfg.critic_feature_norm_coef * penalty, float(feature_norm.item())

    def _distributional_actor_update(
        self,
        observations: Any,
        critic_observations: Any,
        alpha: torch.Tensor,
    ) -> dict[str, float]:
        self.actor.train()
        critic1_training = self.critic1.training
        critic2_training = self.critic2.training
        self.critic1.eval()
        self.critic2.eval()
        self.critic1.requires_grad_(False)
        self.critic2.requires_grad_(False)
        policy_actions, log_probs = self.actor.act(observations)
        q1_pi = self.critic1(critic_observations, policy_actions)
        q2_pi = self.critic2(critic_observations, policy_actions)
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (alpha.detach() * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor.parameters())
        self.actor_optimizer.step()
        self._normalize_module_parameters(self.actor)
        self.critic1.requires_grad_(True)
        self.critic2.requires_grad_(True)
        self.critic1.train(critic1_training)
        self.critic2.train(critic2_training)

        alpha_loss_value = 0.0
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = alpha_loss.item()

        return {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_value,
            "entropy": (-log_probs).mean().item(),
            "mean_q": min_q_pi.mean().item(),
            "actor_grad_norm": actor_grad_norm,
        }

    def _update_distributional(self) -> dict[str, float]:
        batch = self.replay_buffer.sample(self.cfg.batch_size)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = self._scaled_rewards(batch["rewards"])
        next_observations = batch["next_observations"]
        dones = batch["dones"]
        critic_observations = batch.get("critic_observations", observations)
        next_critic_observations = batch.get(
            "next_critic_observations", next_observations
        )
        alpha = self.get_alpha()
        actor_updated = self.num_updates % self.cfg.actor_update_period == 0
        actor_stats = {
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "entropy": 0.0,
            "mean_q": 0.0,
            "actor_grad_norm": 0.0,
        }
        if actor_updated:
            actor_stats = self._distributional_actor_update(
                observations,
                critic_observations,
                alpha,
            )
            alpha = self.get_alpha()

        actor_training = self.actor.training
        self.actor.eval()
        self.target_critic1.train()
        self.target_critic2.train()
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.act(next_observations)
            target_q1, target_log_probs1, _ = self.target_critic1.distribution(
                next_critic_observations,
                next_actions,
            )
            target_q2, target_log_probs2, _ = self.target_critic2.distribution(
                next_critic_observations,
                next_actions,
            )
            select_first = target_q1 <= target_q2
            selected_log_probs = torch.where(
                select_first[:, None],
                target_log_probs1,
                target_log_probs2,
            )
            target_probs = categorical_td_projection(
                next_log_probs=selected_log_probs,
                rewards=rewards,
                dones=dones,
                entropy_terms=alpha * next_log_probs,
                gamma=self.cfg.gamma,
                support=self.target_critic1.support,
            )
            target_values = (target_probs * self.target_critic1.support).sum(-1)
        self.actor.train(actor_training)

        self.critic1.train()
        self.critic2.train()
        _, current_log_probs1, features1 = self.critic1.distribution(
            critic_observations,
            actions,
        )
        _, current_log_probs2, features2 = self.critic2.distribution(
            critic_observations,
            actions,
        )
        critic1_loss = -(target_probs * current_log_probs1).sum(-1).mean()
        critic2_loss = -(target_probs * current_log_probs2).sum(-1).mean()

        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        critic1_grad_norm = self._clip_gradients(self.critic1.parameters())
        self.critic1_optimizer.step()
        self._normalize_module_parameters(self.critic1)

        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        critic2_grad_norm = self._clip_gradients(self.critic2.parameters())
        self.critic2_optimizer.step()
        self._normalize_module_parameters(self.critic2)

        self.num_updates += 1
        self._soft_update_targets()
        critic1_weight_norm = float(self._module_l2_norm(self.critic1).item())
        critic2_weight_norm = float(self._module_l2_norm(self.critic2).item())
        actor_weight_norm = float(self._module_l2_norm(self.actor).item())
        feature_norm1 = features1.float().norm(2, dim=-1).mean().item()
        feature_norm2 = features2.float().norm(2, dim=-1).mean().item()
        utd = float(self.cfg.gradient_steps * self.cfg.batch_size) / max(
            float(self.num_envs), 1.0
        )
        return {
            "train/q_loss": 0.5 * (critic1_loss.item() + critic2_loss.item()),
            "train/critic1_loss": critic1_loss.item(),
            "train/critic2_loss": critic2_loss.item(),
            "train/critic1_cross_entropy": critic1_loss.item(),
            "train/critic2_cross_entropy": critic2_loss.item(),
            "train/critic_feature_norm_loss": 0.0,
            "train/critic1_feature_norm": feature_norm1,
            "train/critic2_feature_norm": feature_norm2,
            "train/actor_loss": actor_stats["actor_loss"],
            "train/actor_updated": float(actor_updated),
            "train/alpha_loss": actor_stats["alpha_loss"],
            "train/alpha": self.get_alpha().item(),
            "train/entropy": actor_stats["entropy"],
            "train/mean_q": actor_stats["mean_q"],
            "train/td_target_mean": target_values.mean().item(),
            "train/reward_scale": float(self.cfg.reward_scale),
            "train/reward_normalizer_scale": (
                float(torch.sqrt(self.reward_normalizer.variance).item())
                if self.reward_normalizer is not None
                else 1.0
            ),
            "train/update_to_data_ratio": utd,
            "train/actor_grad_norm": actor_stats["actor_grad_norm"],
            "train/critic1_grad_norm": critic1_grad_norm,
            "train/critic2_grad_norm": critic2_grad_norm,
            "train/actor_weight_norm": actor_weight_norm,
            "train/critic1_weight_norm": critic1_weight_norm,
            "train/critic2_weight_norm": critic2_weight_norm,
            "train/learning_rate_actor": self.actor_optimizer.param_groups[0]["lr"],
            "train/learning_rate_critic": self.critic1_optimizer.param_groups[0]["lr"],
        }

    def update(self) -> dict[str, float]:
        """Run one FlashSAC replay update."""
        if len(self.replay_buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {}

        if self._distributional_critic:
            return self._update_distributional()

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = self._scaled_rewards(batch["rewards"])
        next_observations = batch["next_observations"]
        dones = batch["dones"]
        critic_observations = batch.get("critic_observations", observations)
        next_critic_observations = batch.get(
            "next_critic_observations",
            next_observations,
        )

        alpha = self.get_alpha()

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.act(next_observations)
            target_q1 = self.target_critic1(next_critic_observations, next_actions)
            target_q2 = self.target_critic2(next_critic_observations, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * target_q
            if self.cfg.target_q_clip is not None:
                td_target = td_target.clamp(
                    -self.cfg.target_q_clip,
                    self.cfg.target_q_clip,
                )

        current_q1, critic1_features = self._critic_forward_with_features(
            self.critic1,
            critic_observations,
            actions,
        )
        current_q2, critic2_features = self._critic_forward_with_features(
            self.critic2,
            critic_observations,
            actions,
        )
        critic1_feature_loss, critic1_feature_norm = self._feature_norm_loss(
            critic1_features
        )
        critic2_feature_loss, critic2_feature_norm = self._feature_norm_loss(
            critic2_features
        )
        critic1_mse_loss = F.mse_loss(current_q1, td_target)
        critic2_mse_loss = F.mse_loss(current_q2, td_target)
        critic1_loss = critic1_mse_loss + critic1_feature_loss
        critic2_loss = critic2_mse_loss + critic2_feature_loss

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_grad_norm = self._clip_gradients(self.critic1.parameters())
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_grad_norm = self._clip_gradients(self.critic2.parameters())
        self.critic2_optimizer.step()

        critic1_weight_norm = self._clamp_module_weight_norm(
            self.critic1,
            self.cfg.critic_weight_norm_max,
        )
        critic2_weight_norm = self._clamp_module_weight_norm(
            self.critic2,
            self.cfg.critic_weight_norm_max,
        )

        policy_actions, log_probs = self.actor.act(observations)
        q1_pi = self.critic1(critic_observations, policy_actions)
        q2_pi = self.critic2(critic_observations, policy_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor.parameters())
        self.actor_optimizer.step()
        actor_weight_norm = self._clamp_module_weight_norm(
            self.actor,
            self.cfg.actor_weight_norm_max,
        )

        if self.auto_alpha:
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

        q_loss = 0.5 * (critic1_mse_loss.item() + critic2_mse_loss.item())
        samples_per_update = max(float(self.num_envs), 1.0)
        utd = float(self.cfg.gradient_steps * self.cfg.batch_size) / samples_per_update
        return {
            "train/q_loss": q_loss,
            "train/critic1_loss": critic1_loss.item(),
            "train/critic2_loss": critic2_loss.item(),
            "train/critic1_mse_loss": critic1_mse_loss.item(),
            "train/critic2_mse_loss": critic2_mse_loss.item(),
            "train/critic_feature_norm_loss": 0.5
            * (critic1_feature_loss.item() + critic2_feature_loss.item()),
            "train/critic1_feature_norm": critic1_feature_norm,
            "train/critic2_feature_norm": critic2_feature_norm,
            "train/actor_loss": actor_loss.item(),
            "train/alpha_loss": alpha_loss_value,
            "train/alpha": alpha_value.item(),
            "train/entropy": (-log_probs).mean().item(),
            "train/mean_q": min_q_pi.mean().item(),
            "train/td_target_mean": td_target.mean().item(),
            "train/reward_scale": float(self.cfg.reward_scale),
            "train/update_to_data_ratio": utd,
            "train/actor_grad_norm": actor_grad_norm,
            "train/critic1_grad_norm": critic1_grad_norm,
            "train/critic2_grad_norm": critic2_grad_norm,
            "train/actor_weight_norm": actor_weight_norm,
            "train/critic1_weight_norm": critic1_weight_norm,
            "train/critic2_weight_norm": critic2_weight_norm,
            "train/learning_rate_actor": self.actor_optimizer.param_groups[0]["lr"],
            "train/learning_rate_critic": self.critic1_optimizer.param_groups[0]["lr"],
        }

    def save(self, path: str) -> None:
        """Save SAC state plus FlashSAC normalization and exploration state."""
        super().save(path)
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        checkpoint["flash_sac_state"] = {
            "reward_normalizer": (
                self.reward_normalizer.state_dict()
                if self.reward_normalizer is not None
                else None
            ),
            "cached_noise": self._cached_noise,
            "noise_repeat_counts": self._noise_repeat_counts,
            "noise_repeat_lengths": self._noise_repeat_lengths,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Restore SAC state plus FlashSAC normalization and exploration state."""
        super().load(path)
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("flash_sac_state", {})
        normalizer_state = state.get("reward_normalizer")
        if self.reward_normalizer is not None and normalizer_state is not None:
            self.reward_normalizer.load_state_dict(normalizer_state)
        if state.get("cached_noise") is not None:
            self._cached_noise = state["cached_noise"].to(self.device)
            self._noise_repeat_counts = state["noise_repeat_counts"].to(self.device)
            self._noise_repeat_lengths = state["noise_repeat_lengths"].to(self.device)
