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

    The defaults follow the public FlashSAC implementation's core operating
    regime: large replay batches, infrequent updates, residual normalized
    networks, categorical critics, bounded weights, return normalization, and
    temporally correlated exploration.
    """

    # High-throughput off-policy defaults
    batch_size: int = 2_048
    buffer_size: int = 10_000_000
    learning_starts: int = 10_000
    gradient_steps: int = 1

    # Optimizers
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    tau: float = 0.01
    init_alpha: float = 0.01

    # Reward and target scaling
    reward_scale: float = 1.0
    target_q_clip: float | None = None
    normalize_rewards: bool = True
    normalized_return_max: float = 5.0

    # Network architecture
    actor_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    activation: str = "relu"
    layer_norm: bool = False
    use_tanh_squash: bool = True
    min_log_std: float = -10.0
    max_log_std: float = 2.0

    # Distributional critic
    use_distributional_critic: bool = True
    critic_num_bins: int = 101
    critic_min_value: float = -5.0
    critic_max_value: float = 5.0

    # Actor, temperature and exploration
    actor_update_period: int = 2
    target_sigma: float = 0.15
    noise_repeat_zeta: float = 2.0
    noise_repeat_max: int = 16
    unit_weight_normalization: bool = True

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
        assert self.normalized_return_max > 0, "normalized_return_max must be positive"
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
        assert self.actor_num_blocks > 0, "actor_num_blocks must be positive"
        assert self.actor_hidden_dim > 0, "actor_hidden_dim must be positive"
        assert self.critic_num_blocks > 0, "critic_num_blocks must be positive"
        assert self.critic_hidden_dim > 0, "critic_hidden_dim must be positive"
        assert self.critic_num_bins >= 2, "critic_num_bins must be at least 2"
        assert self.critic_min_value < self.critic_max_value, (
            "critic_min_value must be less than critic_max_value"
        )
        assert self.actor_update_period > 0, "actor_update_period must be positive"
        assert self.target_sigma > 0, "target_sigma must be positive"
        assert self.noise_repeat_zeta > 1, "noise_repeat_zeta must be greater than 1"
        assert self.noise_repeat_max > 0, "noise_repeat_max must be positive"
