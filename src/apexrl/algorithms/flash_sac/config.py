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

"""Configuration class for FlashSAC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apexrl.algorithms.sac.config import SACConfig


@dataclass
class FlashSACConfig(SACConfig):
    """Configuration for FlashSAC-style high-throughput SAC training.

    FlashSAC keeps the SAC objective but changes the practical operating point:
    larger batches and networks, lower update-to-data ratio, and optional critic
    norm constraints for high-dimensional continuous-control tasks.
    """

    # High-throughput off-policy defaults
    batch_size: int = 2_048
    buffer_size: int = 2_000_000
    learning_starts: int = 32_768
    gradient_steps: int = 1

    # Optimizers
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 1e-4
    max_grad_norm: float = 1.0

    # Reward and target scaling
    reward_scale: float = 1.0
    target_q_clip: float | None = None

    # Network architecture
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    activation: str = "elu"
    layer_norm: bool = True
    use_tanh_squash: bool = True

    # Critic stabilization
    critic_feature_norm_coef: float = 0.0
    critic_feature_norm_target: float | None = None
    critic_weight_norm_max: float | None = None
    actor_weight_norm_max: float | None = None

    # Logging
    log_interval: int = 16_384
    save_interval: int = 1_000_000
    logger_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate FlashSAC configuration."""
        super().__post_init__()
        assert self.reward_scale > 0, "reward_scale must be positive"
        if self.target_q_clip is not None:
            assert self.target_q_clip > 0, "target_q_clip must be positive"
        assert self.critic_feature_norm_coef >= 0, (
            "critic_feature_norm_coef must be non-negative"
        )
        if self.critic_feature_norm_target is not None:
            assert self.critic_feature_norm_target > 0, (
                "critic_feature_norm_target must be positive"
            )
        if self.critic_weight_norm_max is not None:
            assert self.critic_weight_norm_max > 0, (
                "critic_weight_norm_max must be positive"
            )
        if self.actor_weight_norm_max is not None:
            assert self.actor_weight_norm_max > 0, (
                "actor_weight_norm_max must be positive"
            )
