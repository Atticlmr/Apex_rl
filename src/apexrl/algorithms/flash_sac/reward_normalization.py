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

"""Discounted-return reward normalization used by FlashSAC."""

from __future__ import annotations

from typing import Any

import torch


class RewardNormalizer:
    """Track discounted return variance and bound normalized return magnitude."""

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        max_return: float,
        device: torch.device,
        epsilon: float = 1e-8,
    ):
        self.gamma = gamma
        self.max_return = max_return
        self.epsilon = epsilon
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.max_abs_return = torch.zeros((), dtype=torch.float32, device=device)
        self.mean = torch.zeros((), dtype=torch.float32, device=device)
        self.variance = torch.ones((), dtype=torch.float32, device=device)
        self.count = torch.zeros((), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, rewards: torch.Tensor, episode_ends: torch.Tensor) -> None:
        self.returns = (
            self.gamma * (~episode_ends.bool()).float() * self.returns + rewards
        )
        self.max_abs_return = torch.maximum(
            self.max_abs_return,
            self.returns.abs().max(),
        )
        batch_mean = self.returns.mean()
        batch_variance = self.returns.var(unbiased=False)
        batch_count = float(self.returns.numel())
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        ratio = batch_count / total_count
        second_moment = (
            self.variance * self.count
            + batch_variance * batch_count
            + delta.square() * self.count * ratio
        )
        self.mean = self.mean + delta * ratio
        self.variance = second_moment / total_count
        self.count = total_count

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        variance_scale = torch.sqrt(self.variance + self.epsilon)
        bound_scale = self.max_abs_return / self.max_return
        return rewards / torch.maximum(variance_scale, bound_scale)

    def state_dict(self) -> dict[str, Any]:
        return {
            "returns": self.returns,
            "max_abs_return": self.max_abs_return,
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.returns.copy_(state["returns"].to(self.returns.device))
        self.max_abs_return.copy_(state["max_abs_return"].to(self.returns.device))
        self.mean.copy_(state["mean"].to(self.returns.device))
        self.variance.copy_(state["variance"].to(self.returns.device))
        self.count.copy_(state["count"].to(self.returns.device))
