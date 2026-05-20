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

"""Sequence rollout storage for recurrent on-policy algorithms.

The buffer keeps the leading ``(time, env)`` dimensions intact so recurrent
policies can be trained with contiguous sequence minibatches. Hidden states are
stored at the beginning of each environment step and returned at the start of
each sampled sequence.
"""

from __future__ import annotations

from typing import Any

import torch

from apexrl.utils import (
    Observation,
    allocate_observation_storage,
    observation_index,
    observation_set_index,
    observation_to_device,
)


class RolloutRNNBuffer:
    """Rollout buffer that keeps time order and recurrent initial states.

    Stored transition tensors use ``(num_steps, num_envs, ...)`` layout.
    Sequence minibatches use ``(sequence_length, batch, ...)`` layout, where
    ``batch`` is the number of sampled sequences. Hidden states use GRU-style
    ``(num_layers, batch, hidden_size)`` layout when returned to the algorithm.
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_shape: Any,
        action_shape: tuple[int, ...],
        action_dtype: torch.dtype,
        actor_hidden_shape: tuple[int, ...],
        critic_hidden_shape: tuple[int, ...],
        device: torch.device,
        privileged_obs_shape: Any | None = None,
    ):
        """Initialize recurrent rollout storage.

        Args:
            num_envs: Number of parallel environments.
            num_steps: Number of environment steps stored per rollout.
            obs_shape: Observation storage spec, usually produced from the
                observation space.
            action_shape: Action tensor shape excluding time/env dimensions.
                Use ``()`` for scalar discrete actions.
            action_dtype: Tensor dtype used for stored actions.
            actor_hidden_shape: Actor hidden state shape excluding the batch
                dimension. For GRU this is ``(num_layers, hidden_size)``.
            critic_hidden_shape: Critic hidden state shape excluding the batch
                dimension.
            device: Device used for all storage tensors.
            privileged_obs_shape: Optional critic-only observation storage spec.
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.actor_hidden_shape = actor_hidden_shape
        self.critic_hidden_shape = critic_hidden_shape
        self.device = device
        self.privileged_obs_shape = privileged_obs_shape

        self.observations = allocate_observation_storage(
            (num_steps, num_envs),
            obs_shape,
            device=device,
            dtype=torch.float32,
        )
        if privileged_obs_shape is not None:
            self.privileged_observations = allocate_observation_storage(
                (num_steps, num_envs),
                privileged_obs_shape,
                device=device,
                dtype=torch.float32,
            )
        else:
            self.privileged_observations = None

        self.actions = torch.zeros(
            (num_steps, num_envs, *action_shape),
            device=device,
            dtype=action_dtype,
        )
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.actor_hidden_states = torch.zeros(
            (num_steps, num_envs, *actor_hidden_shape),
            device=device,
        )
        self.critic_hidden_states = torch.zeros(
            (num_steps, num_envs, *critic_hidden_shape),
            device=device,
        )
        self.step = 0

    def add(
        self,
        observations: Observation,
        privileged_observations: Observation | None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        actor_hidden_states: torch.Tensor,
        critic_hidden_states: torch.Tensor,
    ) -> None:
        """Add one vectorized recurrent transition.

        Args:
            observations: Actor observations for all environments.
            privileged_observations: Optional critic-only observations.
            actions: Actions sampled from the policy.
            rewards: Rewards with shape ``(num_envs,)``.
            dones: Episode end flags with shape ``(num_envs,)``.
            values: Value predictions with shape ``(num_envs,)``.
            log_probs: Action log probabilities with shape ``(num_envs,)``.
            actor_hidden_states: Actor hidden state at the start of the step,
                shaped ``(layers, num_envs, hidden)``.
            critic_hidden_states: Critic hidden state at the start of the step,
                shaped ``(layers, num_envs, hidden)``.
        """
        if self.step >= self.num_steps:
            raise ValueError(f"Rollout buffer is full (capacity: {self.num_steps})")

        observation_set_index(self.observations, self.step, observations)
        if (
            self.privileged_observations is not None
            and privileged_observations is not None
        ):
            observation_set_index(
                self.privileged_observations,
                self.step,
                privileged_observations,
            )
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.log_probs[self.step].copy_(log_probs)
        self.actor_hidden_states[self.step].copy_(
            self._batch_first_hidden(actor_hidden_states)
        )
        self.critic_hidden_states[self.step].copy_(
            self._batch_first_hidden(critic_hidden_states)
        )
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute returns and advantages over the preserved time dimension.

        Args:
            last_values: Bootstrap values for the last observations, shaped
                ``(num_envs,)``.
            gamma: Discount factor.
            gae_lambda: Generalized Advantage Estimation lambda.
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values

    def get_sequence_minibatches(
        self,
        sequence_length: int,
        minibatch_size: int,
        *,
        shuffle: bool = True,
    ):
        """Yield minibatches of contiguous sequences.

        ``minibatch_size`` counts sequences, not individual transitions.
        Returned observations keep the shape ``(sequence_length, batch, ...)``.

        Args:
            sequence_length: Number of contiguous time steps in each sequence.
            minibatch_size: Number of sequences per yielded minibatch.
            shuffle: Whether to shuffle sequence order before batching.

        Yields:
            Dictionaries containing observations, actions, log probabilities,
            advantages, returns, values, dones and initial recurrent states.
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.num_steps % sequence_length != 0:
            raise ValueError(
                "num_steps must be divisible by sequence_length for recurrent PPO"
            )

        num_chunks = self.num_steps // sequence_length
        total_sequences = num_chunks * self.num_envs
        order = (
            torch.randperm(total_sequences, device=self.device)
            if shuffle
            else torch.arange(total_sequences, device=self.device)
        )

        for start in range(0, total_sequences, minibatch_size):
            seq_indices = order[start : start + minibatch_size]
            env_indices = seq_indices % self.num_envs
            chunk_indices = seq_indices // self.num_envs
            time_starts = chunk_indices * sequence_length
            time_offsets = torch.arange(sequence_length, device=self.device)
            time_indices = time_starts.unsqueeze(0) + time_offsets.unsqueeze(1)
            env_grid = env_indices.unsqueeze(0).expand(sequence_length, -1)

            obs = observation_index(self.observations, (time_indices, env_grid))
            privileged_obs = (
                observation_index(
                    self.privileged_observations,
                    (time_indices, env_grid),
                )
                if self.privileged_observations is not None
                else None
            )
            actions = self.actions[time_indices, env_grid]
            if not self.action_shape:
                actions = actions.reshape(sequence_length, -1)

            yield {
                "observations": obs,
                "privileged_observations": privileged_obs,
                "actions": actions,
                "old_log_probs": self.log_probs[time_indices, env_grid],
                "advantages": self.advantages[time_indices, env_grid],
                "returns": self.returns[time_indices, env_grid],
                "values": self.values[time_indices, env_grid],
                "dones": self.dones[time_indices, env_grid],
                "actor_hidden_states": self._hidden_batch_first(
                    self.actor_hidden_states[time_starts, env_indices]
                ),
                "critic_hidden_states": self._hidden_batch_first(
                    self.critic_hidden_states[time_starts, env_indices]
                ),
            }

    def clear(self) -> None:
        """Clear the buffer and reset step counter."""
        self.step = 0

    def __len__(self) -> int:
        """Return the number of transitions stored."""
        return self.step * self.num_envs

    def to(self, device: torch.device) -> RolloutRNNBuffer:
        """Move all storage tensors to a new device.

        Args:
            device: Target torch device.

        Returns:
            The buffer itself.
        """
        self.device = device
        self.observations = observation_to_device(self.observations, device)
        if self.privileged_observations is not None:
            self.privileged_observations = observation_to_device(
                self.privileged_observations,
                device,
            )
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.log_probs = self.log_probs.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        self.actor_hidden_states = self.actor_hidden_states.to(device)
        self.critic_hidden_states = self.critic_hidden_states.to(device)
        return self

    @staticmethod
    def _batch_first_hidden(hidden: torch.Tensor) -> torch.Tensor:
        """Convert ``(layers, batch, hidden)`` state to batch-first storage."""
        return hidden.transpose(0, 1).contiguous()

    @staticmethod
    def _hidden_batch_first(hidden: torch.Tensor) -> torch.Tensor:
        """Convert batch-first storage back to ``(layers, batch, hidden)``."""
        return hidden.transpose(0, 1).contiguous()
