"""Rollout buffer for PPO algorithm.

Supports multi-dimensional observations (e.g., images) and flexible storage.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class RolloutBuffer:
    """Buffer for storing rollout data during PPO training.

    Stores transitions (observations, actions, rewards, dones, values, log_probs)
    and computes advantages using Generalized Advantage Estimation (GAE).

    Supports multi-dimensional observations (images, vectors, etc.).

    Attributes:
        num_envs (int): Number of parallel environments.
        num_steps (int): Number of steps per rollout (horizon).
        obs_shape (tuple): Shape of observations (can be multi-dimensional).
        device (torch.device): Device for tensors.
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        num_privileged_obs: int = 0,
    ):
        """Initialize the rollout buffer.

        Args:
            num_envs: Number of parallel environments.
            num_steps: Number of steps per rollout (n_steps in PPO).
            obs_shape: Shape of observations (e.g., (48,) for vectors, (3, 84, 84) for images).
            device: Device for tensors.
            num_privileged_obs: Dimension of privileged observations (for asymmetric critic).
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_shape = obs_shape
        self.device = device
        self.num_privileged_obs = num_privileged_obs

        # Buffers for rollout data
        # Shape: (num_steps, num_envs, *obs_shape)
        self.observations = torch.zeros(
            (num_steps, num_envs, *obs_shape),
            device=device,
            dtype=torch.float32,
        )

        # Privileged observations buffer (if using asymmetric critic)
        if num_privileged_obs > 0:
            self.privileged_observations = torch.zeros(
                (num_steps, num_envs, num_privileged_obs),
                device=device,
                dtype=torch.float32,
            )
        else:
            self.privileged_observations = None

        # Action buffer (supports multi-dimensional actions)
        # Shape will be determined by action_dim
        self.actions: torch.Tensor = torch.zeros(
            num_steps, num_envs, device=device, dtype=torch.float32
        )
        self.rewards = torch.zeros(
            num_steps, num_envs, device=device, dtype=torch.float32
        )
        self.dones = torch.zeros(
            num_steps, num_envs, device=device, dtype=torch.float32
        )
        self.values = torch.zeros(
            num_steps, num_envs, device=device, dtype=torch.float32
        )
        self.log_probs = torch.zeros(
            num_steps, num_envs, device=device, dtype=torch.float32
        )

        # Advantages and returns (computed after rollout)
        self.advantages = torch.zeros(
            num_steps, num_envs, device=device, dtype=torch.float32
        )
        self.returns = torch.zeros(
            num_steps, num_envs, device=device, dtype=torch.float32
        )

        # Current step index
        self.step = 0

    def add(
        self,
        observations: torch.Tensor,
        privileged_observations: Optional[torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observations: Observations. Shape: (num_envs, *obs_shape)
            privileged_observations: Privileged observations. Shape: (num_envs, priv_obs_dim) or None
            actions: Actions. Shape: (num_envs, action_dim) or (num_envs,)
            rewards: Rewards. Shape: (num_envs,)
            dones: Done flags. Shape: (num_envs,)
            values: Value estimates. Shape: (num_envs,)
            log_probs: Log probabilities of actions. Shape: (num_envs,)
        """
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

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_values: Value estimates for the last observations. Shape: (num_envs,)
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter for advantage estimation.
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)

        # Compute GAE backwards
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            # TD error: delta = r + gamma * V(s') - V(s)
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )

            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Store advantages and returns
        self.advantages = advantages
        self.returns = advantages + self.values

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """Get all data from the buffer (flattened).

        Returns:
            Dictionary containing all rollout data with flattened batch dimensions.
        """
        # Total number of transitions
        total_transitions = self.num_steps * self.num_envs

        # Flatten time and env dimensions for observations
        # (num_steps, num_envs, *obs_shape) -> (total_transitions, *obs_shape)
        flat_obs = self.observations.reshape(total_transitions, *self.obs_shape)

        # Flatten privileged observations
        if self.privileged_observations is not None:
            flat_privileged_obs = self.privileged_observations.reshape(
                total_transitions, self.num_privileged_obs
            )
        else:
            flat_privileged_obs = None

        # Flatten other tensors (they have shape (num_steps, num_envs, ...))
        # Handle actions which might be multi-dimensional
        if self.actions.dim() > 2:
            # Multi-dimensional actions: (num_steps, num_envs, action_dim)
            flat_actions = self.actions.reshape(total_transitions, -1)
        else:
            # Single-dimensional actions: (num_steps, num_envs)
            flat_actions = self.actions.reshape(total_transitions)

        return {
            "observations": flat_obs,
            "privileged_observations": flat_privileged_obs,
            "actions": flat_actions,
            "old_log_probs": self.log_probs.reshape(total_transitions),
            "advantages": self.advantages.reshape(total_transitions),
            "returns": self.returns.reshape(total_transitions),
            "values": self.values.reshape(total_transitions),
        }

    def get_minibatch(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Get a random minibatch from the buffer.

        Args:
            batch_size: Size of the minibatch.

        Returns:
            Tuple containing:
                - observations: (batch_size, *obs_shape)
                - privileged_observations: (batch_size, priv_obs_dim) or None
                - actions: (batch_size,) or (batch_size, action_dim)
                - old_log_probs: (batch_size,)
                - advantages: (batch_size,)
                - returns: (batch_size,)
                - values: (batch_size,)
        """
        # Get all flattened data
        data = self.get_all_data()
        total_transitions = self.num_steps * self.num_envs

        # Random indices
        indices = torch.randint(0, total_transitions, (batch_size,), device=self.device)

        obs = data["observations"][indices]
        privileged_obs = (
            data["privileged_observations"][indices]
            if data["privileged_observations"] is not None
            else None
        )
        actions = data["actions"][indices]
        old_log_probs = data["old_log_probs"][indices]
        advantages = data["advantages"][indices]
        returns = data["returns"][indices]
        values = data["values"][indices]

        return obs, privileged_obs, actions, old_log_probs, advantages, returns, values

    def clear(self) -> None:
        """Clear the buffer and reset step counter."""
        self.step = 0

    def __len__(self) -> int:
        """Return the number of transitions stored."""
        return self.step * self.num_envs

    def to(self, device: torch.device) -> "RolloutBuffer":
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
