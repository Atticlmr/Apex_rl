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

"""Configuration classes for PPO algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm.

    Attributes:
        # Training parameters
        num_steps: Number of steps per environment per rollout (n_steps).
        num_epochs: Number of epochs to update policy for each rollout.
        batch_size: Batch size for policy updates.
        minibatch_size: Minibatch size for SGD. If None, uses batch_size.

        # Learning rates
        learning_rate: Learning rate for optimizer.
        learning_rate_schedule: Learning rate schedule ("constant", "linear", "adaptive").
        max_learning_rate: Maximum learning rate for adaptive schedule.
        min_learning_rate: Minimum learning rate for adaptive schedule.

        # PPO hyperparameters
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_range: Clipping parameter for surrogate loss.
        clip_range_vf: Clipping parameter for value function loss (None = no clipping).
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.

        # Network architecture
        actor_hidden_dims: Hidden layer dimensions for actor network.
        critic_hidden_dims: Hidden layer dimensions for critic network.
        activation: Activation function ("relu", "elu", "tanh").
        layer_norm: Whether to use layer normalization.
        fixed_std: Whether to use fixed standard deviation for action distribution.
        std_value: Fixed standard deviation value.
        use_asymmetric: Whether to use asymmetric actor-critic.

        # Optimizer
        use_policy_optimizer: Whether to use separate optimizer for policy.
        policy_learning_rate: Learning rate for policy (if separate optimizer).
        value_learning_rate: Learning rate for value function (if separate optimizer).

        # Normalization
        normalize_observations: Whether to normalize observations.
        normalize_advantages: Whether to normalize advantages.
        normalize_rewards: Whether to normalize rewards.
        reward_gamma: Gamma for reward normalization running mean.

        # Logging
        log_interval: Interval for logging (in iterations).
        save_interval: Interval for saving checkpoints (in iterations).

        # Device
        device: Device to use ("cuda", "cpu", "auto").

        # Advanced
        use_torch_compile: Whether to use torch.compile for optimization.
        compile_mode: Compilation mode for torch.compile ("default", "reduce-overhead").
    """

    # Training parameters
    num_steps: int = 24
    num_epochs: int = 5
    batch_size: Optional[int] = None  # Defaults to num_steps * num_envs
    minibatch_size: Optional[int] = None  # Defaults to batch_size
    max_iterations: Optional[int] = (
        None  # Max policy updates, overrides total_timesteps
    )

    # Learning rates
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "adaptive"  # "constant", "linear", "adaptive"
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # None = no clipping
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0
    target_kl: Optional[float] = (
        None  # Target KL divergence for early stopping (None = disabled)
    )

    # Network architecture
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    layer_norm: bool = False
    fixed_std: bool = True
    std_value: float = 1.0
    use_asymmetric: bool = False

    # Optimizer
    optimizer: str = (
        "adam"  # Optimizer type: "adam", "adamw", "muon" muon is under testing
    )
    use_policy_optimizer: bool = False
    policy_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4

    # Normalization
    normalize_observations: bool = False
    normalize_advantages: bool = True
    normalize_rewards: bool = False
    reward_gamma: float = 0.99

    # Logging
    log_interval: int = 10
    save_interval: int = 100

    # Device
    device: str = "auto"

    # Advanced
    use_torch_compile: bool = False
    compile_mode: str = "default"

    def __post_init__(self):
        """Validate configuration."""
        assert self.num_steps > 0, "num_steps must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 <= self.gae_lambda <= 1, "gae_lambda must be in [0, 1]"
        assert self.clip_range >= 0, "clip_range must be non-negative"
        assert self.vf_coef >= 0, "vf_coef must be non-negative"
        assert self.ent_coef >= 0, "ent_coef must be non-negative"
        assert self.max_grad_norm >= 0, "max_grad_norm must be non-negative"
        assert self.optimizer in ["adam", "adamw", "muon"], (
            f"optimizer must be one of 'adam', 'adamw', 'muon', got '{self.optimizer}'"
        )

    def get_batch_size(self, num_envs: int) -> int:
        """Get effective batch size."""
        if self.batch_size is None:
            return self.num_steps * num_envs
        return self.batch_size

    def get_minibatch_size(self, num_envs: int) -> int:
        """Get effective minibatch size."""
        if self.minibatch_size is None:
            return self.get_batch_size(num_envs)
        return self.minibatch_size


@dataclass
class PPOStorageConfig:
    """Configuration for PPO storage/rollout buffer.

    Attributes:
        num_steps: Number of steps to store per rollout.
        num_envs: Number of parallel environments.
        obs_shape: Shape of observations (excluding batch dims).
        device: Device for storage.
        num_privileged_obs: Dimension of privileged observations.
    """

    num_steps: int
    num_envs: int
    obs_shape: tuple
    device: str = "cuda"
    num_privileged_obs: int = 0
