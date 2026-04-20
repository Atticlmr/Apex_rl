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

"""Configuration class for DQN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DQNConfig:
    """Configuration for Deep Q-Network training."""

    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 250
    tau: float = 1.0
    max_grad_norm: float = 10.0
    optimizer: str = "adam"
    double_dqn: bool = True
    dueling: bool = False
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    max_timesteps: int | None = None

    network_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    layer_norm: bool = False

    log_interval: int = 1_000
    save_interval: int = 10_000
    logger_backend: str | list[str] = "tensorboard"
    logger_kwargs: dict[str, Any] | None = None

    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate DQN configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.buffer_size > 0, "buffer_size must be positive"
        assert self.learning_starts >= 0, "learning_starts must be non-negative"
        assert self.train_freq > 0, "train_freq must be positive"
        assert self.gradient_steps > 0, "gradient_steps must be positive"
        assert self.target_update_interval > 0, (
            "target_update_interval must be positive"
        )
        assert 0 < self.tau <= 1.0, "tau must be in (0, 1]"
        assert self.max_grad_norm >= 0, "max_grad_norm must be non-negative"
        assert self.optimizer in ["adam", "adamw", "muon"], (
            f"optimizer must be one of 'adam', 'adamw', 'muon', got '{self.optimizer}'"
        )
        assert 0 <= self.epsilon_end <= self.epsilon_start <= 1.0, (
            "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1"
        )
        assert self.epsilon_decay_steps > 0, "epsilon_decay_steps must be positive"
        if self.logger_kwargs is None:
            self.logger_kwargs = {}
