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

"""Replay buffer for off-policy algorithms such as DQN."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


class ReplayBuffer:
    """Circular replay buffer storing transitions on a single device."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...] = (),
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.long,
    ):
        """Initialize replay buffer storage.

        Args:
            capacity: Maximum number of transitions stored.
            obs_shape: Observation shape without batch dimension.
            action_shape: Action shape. Empty tuple means scalar actions.
            device: Device used for storage and sampling.
            obs_dtype: Observation dtype.
            action_dtype: Action dtype.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype

        self.observations = torch.zeros(
            (capacity, *self.obs_shape),
            dtype=obs_dtype,
            device=self.device,
        )
        self.next_observations = torch.zeros_like(self.observations)
        self.actions = torch.zeros(
            (capacity, *self.action_shape),
            dtype=action_dtype,
            device=self.device,
        )
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=self.device)

        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        """Return number of valid transitions currently stored."""
        return self.capacity if self.full else self.pos

    def __len__(self) -> int:
        """Return number of valid transitions currently stored."""
        return self.size

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Append a batch of transitions to the replay buffer."""
        batch_size = observations.shape[0]
        if batch_size <= 0:
            return
        if batch_size > self.capacity:
            raise ValueError(
                f"batch size {batch_size} exceeds replay capacity {self.capacity}"
            )

        observations = observations.to(self.device, dtype=self.obs_dtype)
        next_observations = next_observations.to(self.device, dtype=self.obs_dtype)
        actions = actions.to(self.device, dtype=self.action_dtype)
        rewards = rewards.to(self.device, dtype=torch.float32)
        dones = dones.to(self.device, dtype=torch.float32)

        end = self.pos + batch_size
        if end <= self.capacity:
            sl = slice(self.pos, end)
            self.observations[sl].copy_(observations)
            self.actions[sl].copy_(actions)
            self.rewards[sl].copy_(rewards)
            self.next_observations[sl].copy_(next_observations)
            self.dones[sl].copy_(dones)
        else:
            first = self.capacity - self.pos
            second = batch_size - first
            self.observations[self.pos :].copy_(observations[:first])
            self.actions[self.pos :].copy_(actions[:first])
            self.rewards[self.pos :].copy_(rewards[:first])
            self.next_observations[self.pos :].copy_(next_observations[:first])
            self.dones[self.pos :].copy_(dones[:first])

            self.observations[:second].copy_(observations[first:])
            self.actions[:second].copy_(actions[first:])
            self.rewards[:second].copy_(rewards[first:])
            self.next_observations[:second].copy_(next_observations[first:])
            self.dones[:second].copy_(dones[first:])

        self.pos = end % self.capacity
        if batch_size and end >= self.capacity:
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from buffer of size {self.size}"
            )

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        batch = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }
        if not self.action_shape:
            batch["actions"] = batch["actions"].reshape(batch_size)
        return batch

    def clear(self) -> None:
        """Reset buffer pointers without reallocating storage."""
        self.pos = 0
        self.full = False

    def state_dict(self) -> Dict[str, Any]:
        """Serialize replay buffer state for checkpointing."""
        size = self.size
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "obs_dtype": self.obs_dtype,
            "action_dtype": self.action_dtype,
            "pos": self.pos,
            "full": self.full,
            "size": size,
            "observations": self.observations[:size].clone(),
            "actions": self.actions[:size].clone(),
            "rewards": self.rewards[:size].clone(),
            "next_observations": self.next_observations[:size].clone(),
            "dones": self.dones[:size].clone(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore replay buffer state from checkpoint."""
        size = int(state_dict.get("size", 0))
        if state_dict["capacity"] != self.capacity:
            raise ValueError(
                "ReplayBuffer capacity mismatch: "
                f"{state_dict['capacity']} != {self.capacity}"
            )
        if tuple(state_dict["obs_shape"]) != self.obs_shape:
            raise ValueError("ReplayBuffer obs_shape mismatch")
        if tuple(state_dict["action_shape"]) != self.action_shape:
            raise ValueError("ReplayBuffer action_shape mismatch")

        self.clear()
        if size > 0:
            self.observations[:size].copy_(
                state_dict["observations"].to(self.device, dtype=self.obs_dtype)
            )
            self.actions[:size].copy_(
                state_dict["actions"].to(self.device, dtype=self.action_dtype)
            )
            self.rewards[:size].copy_(
                state_dict["rewards"].to(self.device, dtype=torch.float32)
            )
            self.next_observations[:size].copy_(
                state_dict["next_observations"].to(
                    self.device, dtype=self.obs_dtype
                )
            )
            self.dones[:size].copy_(
                state_dict["dones"].to(self.device, dtype=torch.float32)
            )

        self.pos = int(state_dict.get("pos", size % self.capacity))
        self.full = bool(state_dict.get("full", size == self.capacity))
