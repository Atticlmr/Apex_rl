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

"""Configuration class for Soft Actor-Critic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class SACConfig:
    """Configuration for Soft Actor-Critic training.

    Attributes:
        gamma: Discount factor.
        tau: Soft target update coefficient.
        batch_size: Replay batch size.
        buffer_size: Replay buffer capacity.
        learning_starts: Number of environment steps collected before updates start.
        train_freq: Environment steps between training updates.
        gradient_steps: Number of gradient updates per training trigger.
        target_update_interval: Update interval for target critics.
        max_timesteps: Optional total training steps for runner-driven training.

        actor_learning_rate: Learning rate for policy optimizer.
        critic_learning_rate: Learning rate for Q-network optimizer.
        alpha_learning_rate: Learning rate for entropy temperature optimizer.
        optimizer: Optimizer type.
        max_grad_norm: Gradient clipping threshold.

        auto_alpha: Whether to learn the entropy temperature automatically.
        init_alpha: Initial entropy temperature.
        target_entropy: Entropy target. If None, infer from action dimension.

        actor_hidden_dims: Hidden layer dimensions for actor network.
        critic_hidden_dims: Hidden layer dimensions for critic networks.
        activation: Activation function.
        layer_norm: Whether to use layer normalization.
        use_tanh_squash: Whether the actor outputs tanh-squashed actions.
        min_log_std: Lower clamp for actor log standard deviation.
        max_log_std: Upper clamp for actor log standard deviation.

        log_interval: Logging interval in environment steps.
        save_interval: Checkpoint interval in environment steps.
        logger_backend: Logging backend.
        logger_kwargs: Additional logger keyword arguments.
        device: Device selection policy.
    """

    # Core training
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1
    max_timesteps: Optional[int] = None

    # Optimizers
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    optimizer: str = "adam"
    max_grad_norm: float = 10.0

    # Entropy temperature
    auto_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: Optional[float] = None

    # Network architecture
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    layer_norm: bool = False
    use_tanh_squash: bool = True
    min_log_std: float = -20.0
    max_log_std: float = 2.0

    # Logging
    log_interval: int = 1_000
    save_interval: int = 10_000
    logger_backend: Union[str, List[str]] = "tensorboard"
    logger_kwargs: Optional[Dict[str, Any]] = None

    # Device
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 < self.tau <= 1, "tau must be in (0, 1]"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.buffer_size > 0, "buffer_size must be positive"
        assert self.learning_starts >= 0, "learning_starts must be non-negative"
        assert self.train_freq > 0, "train_freq must be positive"
        assert self.gradient_steps > 0, "gradient_steps must be positive"
        assert self.target_update_interval > 0, (
            "target_update_interval must be positive"
        )
        assert self.actor_learning_rate > 0, "actor_learning_rate must be positive"
        assert self.critic_learning_rate > 0, "critic_learning_rate must be positive"
        assert self.alpha_learning_rate > 0, "alpha_learning_rate must be positive"
        assert self.init_alpha > 0, "init_alpha must be positive"
        assert self.max_grad_norm >= 0, "max_grad_norm must be non-negative"
        assert self.min_log_std <= self.max_log_std, (
            "min_log_std must be <= max_log_std"
        )
        assert self.optimizer in ["adam", "adamw", "muon"], (
            f"optimizer must be one of 'adam', 'adamw', 'muon', got '{self.optimizer}'"
        )
        if self.logger_kwargs is None:
            self.logger_kwargs = {}
