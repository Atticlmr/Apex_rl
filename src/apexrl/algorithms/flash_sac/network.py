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

"""FlashSAC network components adapted to ApexRL model interfaces."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from apexrl.models.base import ContinuousActor, ContinuousQNetwork
from apexrl.utils import flatten_observation, space_to_spec, spec_numel


class UnitLinear(nn.Linear):
    """Bias-free linear layer whose output rows are projected to unit norm."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.weight, gain=1.0)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """Batch normalization with jointly normalized scale and bias parameters."""

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        training = self.training and value.shape[0] > 1
        return F.batch_norm(
            value,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        dimension = self.weight.shape[-1]
        squared_norm = (self.weight.square() + self.bias.square()).sum()
        factor = math.sqrt(dimension) * torch.rsqrt(squared_norm + 1e-8)
        self.weight.mul_(factor)
        self.bias.mul_(factor)


class UnitRMSNorm(nn.Module):
    """RMS normalization with a constrained gain vector."""

    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            value,
            self.weight.shape,
            self.weight,
            eps=self.eps,
        )

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        dimension = self.weight.shape[-1]
        factor = math.sqrt(dimension) * torch.rsqrt(self.weight.square().sum() + 1e-8)
        self.weight.mul_(factor)


class FlashSACEmbedder(nn.Module):
    """Normalized observation projection used by actor and critics."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.projection = UnitLinear(input_dim, hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(value))


class FlashSACResidualBlock(nn.Module):
    """Expanded residual block with normalization after each projection."""

    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        expanded_dim = hidden_dim * expansion
        self.input_projection = UnitLinear(hidden_dim, expanded_dim)
        self.input_norm = UnitBatchNorm(expanded_dim)
        self.output_projection = UnitLinear(expanded_dim, hidden_dim)
        self.output_norm = UnitBatchNorm(hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.relu(self.input_norm(self.input_projection(value)))
        value = F.relu(self.output_norm(self.output_projection(value)))
        return value + residual


class _FlashSACBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            FlashSACResidualBlock(hidden_dim) for _ in range(num_blocks)
        )
        self.post_norm = UnitRMSNorm(hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.embedder(value)
        for block in self.blocks:
            value = block(value)
        return self.post_norm(value)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "normalize_parameters"):
                module.normalize_parameters()


class FlashSACActor(ContinuousActor):
    """Residual Gaussian actor with bounded parameters and environment scaling."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Box,
        cfg: dict[str, Any] | None = None,
    ):
        cfg = cfg or {}
        cfg.setdefault("use_tanh_squash", True)
        super().__init__(obs_space, action_space, cfg)
        obs_dim = spec_numel(space_to_spec(obs_space))
        hidden_dim = int(cfg.get("hidden_dim", 128))
        num_blocks = int(cfg.get("num_blocks", 2))
        self.backbone = _FlashSACBackbone(obs_dim, hidden_dim, num_blocks)
        self.mean_head = UnitLinear(hidden_dim, self.action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(self.action_dim))
        self.std_head = UnitLinear(hidden_dim, self.action_dim)
        self.std_bias = nn.Parameter(torch.zeros(self.action_dim))
        self.min_log_std = float(cfg.get("min_log_std", -10.0))
        self.max_log_std = float(cfg.get("max_log_std", 2.0))

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self.normalize_parameters()

    def _distribution_params(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(flatten_observation(obs))
        mean = F.linear(features, self.mean_head.weight, self.mean_bias)
        raw_log_std = F.linear(features, self.std_head.weight, self.std_bias)
        log_std = self.min_log_std + (
            (self.max_log_std - self.min_log_std)
            * 0.5
            * (1.0 + torch.tanh(raw_log_std))
        )
        return mean, torch.exp(log_std)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _ = self._distribution_params(obs)
        return mean

    def get_action_dist(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.distributions.Normal:
        mean, std = self._distribution_params(obs)
        return torch.distributions.Normal(mean, std)

    def _squash_and_scale(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def _log_prob(
        self,
        distribution: torch.distributions.Normal,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = distribution.log_prob(raw_action)
        log_jacobian = 2.0 * (
            math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action)
        )
        log_prob = (log_prob - log_jacobian).sum(dim=-1)
        return log_prob - torch.log(self.action_scale.abs() + 1e-6).sum()

    def act(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.get_action_dist(obs)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = self._squash_and_scale(raw_action)
        if deterministic:
            return action, torch.zeros(action.shape[0], device=action.device)
        return action, self._log_prob(distribution, raw_action)

    def act_with_noise(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        distribution = self.get_action_dist(obs)
        return self._squash_and_scale(distribution.mean + distribution.stddev * noise)

    def evaluate(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.get_action_dist(obs)
        squashed = (actions - self.action_bias) / (self.action_scale + 1e-6)
        raw_action = torch.atanh(torch.clamp(squashed, -0.999, 0.999))
        return self._log_prob(distribution, raw_action), distribution.entropy().sum(-1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.backbone.normalize_parameters()
        self.mean_head.normalize_parameters()
        self.std_head.normalize_parameters()


class FlashSACCategoricalQNetwork(ContinuousQNetwork):
    """Categorical Q critic with a fixed value support."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Box,
        cfg: dict[str, Any] | None = None,
    ):
        super().__init__(obs_space, action_space, cfg)
        cfg = cfg or {}
        obs_dim = spec_numel(space_to_spec(obs_space))
        hidden_dim = int(cfg.get("hidden_dim", 256))
        num_blocks = int(cfg.get("num_blocks", 2))
        self.num_bins = int(cfg.get("num_bins", 101))
        min_value = float(cfg.get("min_value", -5.0))
        max_value = float(cfg.get("max_value", 5.0))
        self.backbone = _FlashSACBackbone(
            obs_dim + self.action_dim,
            hidden_dim,
            num_blocks,
        )
        self.value_head = UnitLinear(hidden_dim, self.num_bins)
        self.value_bias = nn.Parameter(torch.zeros(self.num_bins))
        self.register_buffer(
            "support",
            torch.linspace(min_value, max_value, self.num_bins),
        )
        self.normalize_parameters()

    def distribution(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = flatten_observation(obs)
        actions = actions.reshape(actions.shape[0], -1)
        features = self.backbone(torch.cat([obs, actions], dim=-1))
        logits = F.linear(features, self.value_head.weight, self.value_bias)
        log_probs = F.log_softmax(logits, dim=-1)
        values = (log_probs.exp() * self.support).sum(dim=-1)
        return values, log_probs, features

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        values, _, _ = self.distribution(obs, actions)
        return values

    def forward_with_features(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, _, features = self.distribution(obs, actions)
        return values, features

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.backbone.normalize_parameters()
        self.value_head.normalize_parameters()
