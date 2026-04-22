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

# Copyright (c) 2026 GitHub@Apex_rl Developer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay buffer for off-policy algorithms such as DQN."""

from __future__ import annotations

from typing import Any

import torch

from apexrl.utils import (
    allocate_observation_storage,
    clone_observation,
    observation_batch_size,
    observation_index,
    observation_set_index,
    observation_to_device,
)


class ReplayBuffer:
    """Circular replay buffer storing transitions on a single device."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Any,
        action_shape: tuple[int, ...] = (),
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.long,
        critic_obs_shape: Any | None = None,
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
        self.obs_shape = obs_shape
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype
        self.critic_obs_shape = critic_obs_shape

        self.observations = allocate_observation_storage(
            (capacity,),
            self.obs_shape,
            device=self.device,
            dtype=obs_dtype,
        )
        self.next_observations = allocate_observation_storage(
            (capacity,),
            self.obs_shape,
            device=self.device,
            dtype=obs_dtype,
        )
        if self.critic_obs_shape is not None:
            self.critic_observations = allocate_observation_storage(
                (capacity,),
                self.critic_obs_shape,
                device=self.device,
                dtype=obs_dtype,
            )
            self.next_critic_observations = allocate_observation_storage(
                (capacity,),
                self.critic_obs_shape,
                device=self.device,
                dtype=obs_dtype,
            )
        else:
            self.critic_observations = None
            self.next_critic_observations = None
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
        observations: Any,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Any,
        dones: torch.Tensor,
        critic_observations: Any | None = None,
        next_critic_observations: Any | None = None,
    ) -> None:
        """Append a batch of transitions to the replay buffer."""
        batch_size = observation_batch_size(observations)
        if batch_size <= 0:
            return
        if batch_size > self.capacity:
            raise ValueError(
                f"batch size {batch_size} exceeds replay capacity {self.capacity}"
            )
        observations = observation_to_device(observations, self.device)
        next_observations = observation_to_device(next_observations, self.device)
        critic_observations = observation_to_device(critic_observations, self.device)
        next_critic_observations = observation_to_device(
            next_critic_observations,
            self.device,
        )
        actions = actions.to(self.device, dtype=self.action_dtype)
        rewards = rewards.to(self.device, dtype=torch.float32)
        dones = dones.to(self.device, dtype=torch.float32)

        end = self.pos + batch_size
        if end <= self.capacity:
            sl = slice(self.pos, end)
            observation_set_index(self.observations, sl, observations)
            self.actions[sl].copy_(actions)
            self.rewards[sl].copy_(rewards)
            observation_set_index(self.next_observations, sl, next_observations)
            self.dones[sl].copy_(dones)
            if self.critic_observations is not None and critic_observations is not None:
                observation_set_index(self.critic_observations, sl, critic_observations)
                observation_set_index(
                    self.next_critic_observations,
                    sl,
                    next_critic_observations,
                )
        else:
            first = self.capacity - self.pos
            second = batch_size - first
            observation_set_index(
                self.observations,
                slice(self.pos, None),
                observation_index(observations, slice(None, first)),
            )
            self.actions[self.pos :].copy_(actions[:first])
            self.rewards[self.pos :].copy_(rewards[:first])
            observation_set_index(
                self.next_observations,
                slice(self.pos, None),
                observation_index(next_observations, slice(None, first)),
            )
            self.dones[self.pos :].copy_(dones[:first])
            if self.critic_observations is not None and critic_observations is not None:
                observation_set_index(
                    self.critic_observations,
                    slice(self.pos, None),
                    observation_index(critic_observations, slice(None, first)),
                )
                observation_set_index(
                    self.next_critic_observations,
                    slice(self.pos, None),
                    observation_index(next_critic_observations, slice(None, first)),
                )

            observation_set_index(
                self.observations,
                slice(None, second),
                observation_index(observations, slice(first, None)),
            )
            self.actions[:second].copy_(actions[first:])
            self.rewards[:second].copy_(rewards[first:])
            observation_set_index(
                self.next_observations,
                slice(None, second),
                observation_index(next_observations, slice(first, None)),
            )
            self.dones[:second].copy_(dones[first:])
            if self.critic_observations is not None and critic_observations is not None:
                observation_set_index(
                    self.critic_observations,
                    slice(None, second),
                    observation_index(critic_observations, slice(first, None)),
                )
                observation_set_index(
                    self.next_critic_observations,
                    slice(None, second),
                    observation_index(next_critic_observations, slice(first, None)),
                )

        self.pos = end % self.capacity
        if batch_size and end >= self.capacity:
            self.full = True

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a random batch of transitions."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size < batch_size:
            raise ValueError(
                "cannot sample "
                f"{batch_size} transitions from buffer of size {self.size}"
            )

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        batch = {
            "observations": observation_index(self.observations, indices),
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": observation_index(self.next_observations, indices),
            "dones": self.dones[indices],
        }
        if self.critic_observations is not None:
            batch["critic_observations"] = observation_index(
                self.critic_observations,
                indices,
            )
            batch["next_critic_observations"] = observation_index(
                self.next_critic_observations,
                indices,
            )
        if not self.action_shape:
            batch["actions"] = batch["actions"].reshape(batch_size)
        return batch

    def clear(self) -> None:
        """Reset buffer pointers without reallocating storage."""
        self.pos = 0
        self.full = False

    def state_dict(self) -> dict[str, Any]:
        """Serialize replay buffer state for checkpointing."""
        size = self.size
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "obs_dtype": self.obs_dtype,
            "action_dtype": self.action_dtype,
            "critic_obs_shape": self.critic_obs_shape,
            "pos": self.pos,
            "full": self.full,
            "size": size,
            "observations": clone_observation(
                observation_index(self.observations, slice(None, size))
            ),
            "actions": self.actions[:size].clone(),
            "rewards": self.rewards[:size].clone(),
            "next_observations": clone_observation(
                observation_index(self.next_observations, slice(None, size))
            ),
            "dones": self.dones[:size].clone(),
            "critic_observations": (
                clone_observation(
                    observation_index(self.critic_observations, slice(None, size))
                )
                if self.critic_observations is not None
                else None
            ),
            "next_critic_observations": (
                clone_observation(
                    observation_index(
                        self.next_critic_observations,
                        slice(None, size),
                    )
                )
                if self.next_critic_observations is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore replay buffer state from checkpoint."""
        size = int(state_dict.get("size", 0))
        if state_dict["capacity"] != self.capacity:
            raise ValueError(
                "ReplayBuffer capacity mismatch: "
                f"{state_dict['capacity']} != {self.capacity}"
            )
        if state_dict["obs_shape"] != self.obs_shape:
            raise ValueError("ReplayBuffer obs_shape mismatch")
        if tuple(state_dict["action_shape"]) != self.action_shape:
            raise ValueError("ReplayBuffer action_shape mismatch")
        if state_dict.get("critic_obs_shape") != self.critic_obs_shape:
            raise ValueError("ReplayBuffer critic_obs_shape mismatch")

        self.clear()
        if size > 0:
            observation_set_index(
                self.observations,
                slice(None, size),
                observation_to_device(
                    state_dict["observations"],
                    self.device,
                    self.obs_dtype,
                ),
            )
            self.actions[:size].copy_(
                state_dict["actions"].to(self.device, dtype=self.action_dtype)
            )
            self.rewards[:size].copy_(
                state_dict["rewards"].to(self.device, dtype=torch.float32)
            )
            observation_set_index(
                self.next_observations,
                slice(None, size),
                observation_to_device(
                    state_dict["next_observations"],
                    self.device,
                    self.obs_dtype,
                ),
            )
            self.dones[:size].copy_(
                state_dict["dones"].to(self.device, dtype=torch.float32)
            )
            if self.critic_observations is not None:
                observation_set_index(
                    self.critic_observations,
                    slice(None, size),
                    observation_to_device(
                        state_dict["critic_observations"],
                        self.device,
                        self.obs_dtype,
                    ),
                )
                observation_set_index(
                    self.next_critic_observations,
                    slice(None, size),
                    observation_to_device(
                        state_dict["next_critic_observations"],
                        self.device,
                        self.obs_dtype,
                    ),
                )

        self.pos = int(state_dict.get("pos", size % self.capacity))
        self.full = bool(state_dict.get("full", size == self.capacity))
