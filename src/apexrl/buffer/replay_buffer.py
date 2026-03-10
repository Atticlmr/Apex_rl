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

"""Replay buffer for Off-policy RL algorithms.

Supports multi-dimensional observations (e.g., images) and flexible storage.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class ReplayBuffer:
    """Buffer for storing replay data during Off-policy RL training.

    Stores transitions (observations, actions, rewards, dones, values, log_probs)
    and computes advantages using Generalized Advantage Estimation (GAE).

    Supports multi-dimensional observations (images, vectors, etc.).

    Attributes:
        num_envs (int): Number of parallel environments.
        capacity (int): Maximum number of transitions to store in the buffer.
        obs_shape (tuple): Shape of observations (can be multi-dimensional).
        action_dim (int): Dimension of the action space.
        device (torch.device): Device for tensors.
        is_priority (bool): Whether to use prioritized experience replay.
        full (bool): Whether the buffer is full.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: torch.device,
        is_priority: bool = False,
    ):
        """Initialize the replay buffer.

        Args:
            num_envs: Number of parallel environments.
            capacity: Maximum number of transitions to store in the buffer.
            obs_shape: Shape of observations (e.g., (48,) for vectors, (3, 84, 84) for images).
            action_dim: Dimension of the action space.
            device: Device for tensors.
            is_priority: Whether to use prioritized experience replay.
        """
        self.num_envs = num_envs
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.is_priority = is_priority

        # Buffers for replay data
        # Shape: (capacity, *obs_shape)
        self.observations = torch.zeros(
            (capacity, *obs_shape), device=device, dtype=torch.float32,
        )

        self.next_observations = torch.zeros(
            (capacity, *obs_shape), dtype=torch.float32, device=device,
        )

        self.actions = torch.zeros(
            (capacity, action_dim), dtype=torch.float32, device=device,
        )

        self.rewards = torch.zeros(
            capacity, device=device, dtype=torch.float32, 
        )

        self.dones = torch.zeros(
            capacity, device=device, dtype=torch.float32, 
        )

        # Buffer pointer & size
        self.pos = 0
        self.size = 0
        self.full = False

        if is_priority:
            self.priorities = torch.zeros(
                capacity, device=device, dtype=torch.float32,
            )
            self.alpha = 0.6
            self.beta = 0.4
            self.beta_increment = 1e-4
            self.eps = 1e-6

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add a transition to the replay buffer.

        Args:
            observations: Observations. Shape: (num_envs, *obs_shape)
            actions: Actions. Shape: (num_envs, action_dim) or (num_envs,)
            rewards: Rewards. Shape: (num_envs,)
            next_observations: Next observations. Shape: (num_envs, *obs_shape)
            dones: Done flags. Shape: (num_envs,)
        """
        n = self.num_envs
        start = self.pos
        end = self.pos + n
        
        if self.step >= self.num_steps:
            raise ValueError(f"Rollout buffer is full (capacity: {self.num_steps})")

        self.observations[self.step].copy_(observations)
        if (
            self.privileged_observations is not None
            and privileged_observations is not None
        ):
            self.privileged_observations[self.step].copy_(privileged_observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.log_probs[self.step].copy_(log_probs)

        self.step += 1

    def clear(self) -> None:
        """Clear the buffer."""

        self.pos = 0
        self.size = 0
        self.full = False

        if self.is_priority:
            self.priorities.zero_()

    def __len__(self) -> int:
        """Return the number of transitions stored."""
        return self.step * self.num_envs

    def to(self, device: torch.device) -> "ReplayBuffer":
        """Move all tensors to a new device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        self.observations = self.observations.to(device)
        if self.privileged_observations is not None:
            self.privileged_observations = self.privileged_observations.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.log_probs = self.log_probs.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        return self
