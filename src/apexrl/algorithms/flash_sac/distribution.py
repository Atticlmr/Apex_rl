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

"""Categorical value projection for FlashSAC critics."""

from __future__ import annotations

import torch


def categorical_td_projection(
    next_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    entropy_terms: torch.Tensor,
    gamma: float,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project entropy-regularized Bellman targets onto a fixed support."""
    min_value = support[0]
    max_value = support[-1]
    bin_width = (max_value - min_value) / (support.numel() - 1)
    target_values = rewards[:, None] + gamma * (
        support[None, :] - entropy_terms[:, None]
    ) * (1.0 - dones[:, None])
    target_values = target_values.clamp(min_value, max_value)

    positions = (target_values - min_value) / bin_width
    lower = positions.floor().long()
    upper = torch.clamp(lower + 1, max=support.numel() - 1)
    upper_weight = positions - lower.float()
    probabilities = next_log_probs.exp()

    projected = torch.zeros_like(probabilities)
    projected.scatter_add_(1, lower, probabilities * (1.0 - upper_weight))
    projected.scatter_add_(1, upper, probabilities * upper_weight)
    return projected
