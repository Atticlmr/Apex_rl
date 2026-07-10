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

from typing import Any

import torch
import torch.nn.functional as F
from gymnasium import spaces

from apexrl.algorithms.flash_sac.config import FlashSACConfig
from apexrl.algorithms.sac.sac import SAC
from apexrl.models import MLPFeatureQNetwork, MLPSquashedGaussianActor


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
        super().__init__(
            env=env,
            cfg=cfg or FlashSACConfig(),
            actor_class=actor_class or MLPSquashedGaussianActor,
            critic_class=critic_class or MLPFeatureQNetwork,
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

    def update(self) -> dict[str, float]:
        """Run one FlashSAC replay update."""
        if len(self.replay_buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {}

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"] * self.cfg.reward_scale
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
