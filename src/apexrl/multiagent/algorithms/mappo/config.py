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

"""Configuration for Multi-Agent PPO (MAPPO)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO.

    MAPPO follows centralized training and decentralized execution: each policy
    consumes local observations, while value networks can consume a centralized
    state when ``centralized_critic`` is enabled.
    """

    num_steps: int = 24
    num_epochs: int = 5
    batch_size: int | None = None
    minibatch_size: int | None = None
    max_iterations: int | None = None

    learning_rate: float = 3e-4
    learning_rate_schedule: str = "constant"
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0
    target_kl: float | None = None

    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    layer_norm: bool = False
    fixed_std: bool = True
    std_value: float = 1.0
    use_tanh_squash: bool = False
    min_log_std: float = -5.0
    max_log_std: float = 2.0

    centralized_critic: bool = True
    share_actor: bool = True
    share_critic: bool = True
    shared_reward: bool = False

    optimizer: str = "adam"
    normalize_advantages: bool = True

    log_interval: int = 10
    save_interval: int = 100
    extra_log_keys: list[str] = field(default_factory=list)
    log_train_metrics_vs_iteration: bool = False
    log_episode_metrics_vs_iteration: bool = False
    log_detailed_rollout_stats: bool = False
    logger_backend: str | list[str] = "tensorboard"
    logger_kwargs: dict[str, Any] | None = None

    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate MAPPO configuration."""
        assert self.num_steps > 0, "num_steps must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 <= self.gae_lambda <= 1, "gae_lambda must be in [0, 1]"
        assert self.clip_range >= 0, "clip_range must be non-negative"
        assert self.vf_coef >= 0, "vf_coef must be non-negative"
        assert self.ent_coef >= 0, "ent_coef must be non-negative"
        assert self.max_grad_norm >= 0, "max_grad_norm must be non-negative"
        assert self.optimizer in ["adam", "adamw", "muon"], (
            f"optimizer must be one of 'adam', 'adamw', 'muon', got {self.optimizer}"
        )
        if self.logger_kwargs is None:
            self.logger_kwargs = {}

    def get_batch_size(self, num_envs: int) -> int:
        """Get effective per-agent batch size."""
        if self.batch_size is None:
            return self.num_steps * num_envs
        return self.batch_size

    def get_minibatch_size(self, num_envs: int) -> int:
        """Get effective per-agent minibatch size."""
        if self.minibatch_size is None:
            return self.get_batch_size(num_envs)
        return self.minibatch_size
