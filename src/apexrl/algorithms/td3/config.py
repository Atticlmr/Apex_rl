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

"""Configuration class for TD3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TD3Config:
    """Configuration for Twin Delayed DDPG training.

    Args:
        gamma: Discount factor.
        tau: Polyak averaging coefficient for target networks.
        batch_size: Replay batch size.
        buffer_size: Replay buffer capacity.
        learning_starts: Number of environment steps collected before updates.
        train_freq: Environment steps between training triggers.
        gradient_steps: Number of gradient updates per training trigger.
        policy_delay: Number of critic updates per actor/target update.
        max_timesteps: Optional total training steps for runner-driven training.
        actor_learning_rate: Learning rate for actor optimizer.
        critic_learning_rate: Learning rate for critic optimizers.
        optimizer: Optimizer type.
        max_grad_norm: Gradient clipping threshold. Set to ``0`` to disable.
        exploration_noise: Standard deviation of action noise used for
            environment exploration.
        target_policy_noise: Standard deviation of clipped target policy
            smoothing noise.
        target_noise_clip: Absolute clip value for target policy noise.
        actor_hidden_dims: Hidden layer dimensions for the actor.
        critic_hidden_dims: Hidden layer dimensions for both critics.
        activation: Activation function.
        layer_norm: Whether to use layer normalization.
        log_interval: Logging interval in environment steps.
        save_interval: Checkpoint interval in environment steps.
        save_replay_buffer: Whether checkpoints include replay buffer contents.
        extra_log_keys: Environment extras keys to record through the runner.
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
    policy_delay: int = 2
    max_timesteps: int | None = None

    # Optimizers
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    optimizer: str = "adam"
    max_grad_norm: float = 10.0

    # Exploration and target smoothing
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

    # Network architecture
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    layer_norm: bool = False

    # Logging
    log_interval: int = 1_000
    save_interval: int = 10_000
    save_replay_buffer: bool = False
    extra_log_keys: list[str] = field(default_factory=list)
    logger_backend: str | list[str] = "tensorboard"
    logger_kwargs: dict[str, Any] | None = None

    # Device
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 < self.tau <= 1, "tau must be in (0, 1]"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.buffer_size > 0, "buffer_size must be positive"
        assert self.learning_starts >= 0, "learning_starts must be non-negative"
        assert self.train_freq > 0, "train_freq must be positive"
        assert self.gradient_steps > 0, "gradient_steps must be positive"
        assert self.policy_delay > 0, "policy_delay must be positive"
        assert self.actor_learning_rate > 0, "actor_learning_rate must be positive"
        assert self.critic_learning_rate > 0, "critic_learning_rate must be positive"
        assert self.max_grad_norm >= 0, "max_grad_norm must be non-negative"
        assert self.exploration_noise >= 0, "exploration_noise must be non-negative"
        assert self.target_policy_noise >= 0, "target_policy_noise must be non-negative"
        assert self.target_noise_clip >= 0, "target_noise_clip must be non-negative"
        assert self.optimizer in ["adam", "adamw", "muon"], (
            f"optimizer must be one of 'adam', 'adamw', 'muon', got '{self.optimizer}'"
        )
        if self.logger_kwargs is None:
            self.logger_kwargs = {}
