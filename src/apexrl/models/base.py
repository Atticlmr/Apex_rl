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

"""Base classes for policy (actor) and value (critic) networks.

Users can implement custom Actor and Critic by inheriting from these
base classes and implementing the required abstract methods.

The framework supports:
- Continuous actions with Gaussian distribution
- Discrete actions (to be implemented)
- Flexible observation spaces (images, vectors, dicts)
- Flexible network architectures (MLP, CNN, RNN, etc.)

Example for continuous actions with custom CNN encoder:
    >>> from apexrl.models.base import ContinuousActor, Critic
    >>> import torch.nn as nn
    >>>
    >>> class CustomContinuousActor(ContinuousActor):
    ...     def __init__(self, obs_space, action_space, cfg):
    ...         super().__init__(obs_space, action_space, cfg)
    ...         # Custom network: CNN + MLP
    ...         self.encoder = nn.Sequential(
    ...             nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
    ...             nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
    ...             nn.Flatten(),
    ...         )
    ...         # Compute feature dim after conv
    ...         with torch.no_grad():
    ...             dummy = torch.zeros(1, *obs_space.shape)
    ...             feat_dim = self.encoder(dummy).shape[1]
    ...         self.head = nn.Sequential(
    ...             nn.Linear(feat_dim, 256), nn.ReLU(),
    ...             nn.Linear(256, self.action_dim),
    ...         )
    ...         self.log_std = nn.Parameter(torch.zeros(self.action_dim))
    ...
    ...     def forward(self, obs):
    ...         # Return action mean
    ...         features = self.encoder(obs)
    ...         return self.head(features)
    ...
    ...     def get_action_dist(self, obs):
    ...         mean = self.forward(obs)
    ...         std = torch.exp(self.log_std)
    ...         return torch.distributions.Normal(mean, std)
    ...
    ...     def act(self, obs, deterministic=False):
    ...         dist = self.get_action_dist(obs)
    ...         if deterministic:
    ...             return dist.mean, torch.zeros(...)
    ...         action = dist.sample()
    ...         return action, dist.log_prob(action).sum(-1)
    ...
    ...     def evaluate(self, obs, actions):
    ...         dist = self.get_action_dist(obs)
    ...         log_probs = dist.log_prob(actions).sum(-1)
    ...         entropy = dist.entropy().sum(-1)
    ...         return log_probs, entropy
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from gymnasium import spaces


class Actor(nn.Module, ABC):
    """Base abstract class for all policy networks.

    This is the most general interface. For specific action types,
    use ContinuousActor or DiscreteActor.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Actor.

        Args:
            obs_space: Observation space from gymnasium.
            action_space: Action space from gymnasium.
            cfg: Optional configuration dictionary for custom parameters.
        """
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.action_space = action_space
        self.cfg = cfg or {}

        # Store space information
        self.obs_shape = obs_space.shape
        self.action_shape = action_space.shape

    @abstractmethod
    def forward(
        self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass to get action distribution parameters.

        Args:
            obs: Observations from environment. Can be:
                - torch.Tensor: Flat or multi-dimensional observations
                - Dict[str, torch.Tensor]: Dictionary of observations

        Returns:
            Action distribution parameters. Shape depends on implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.

        Args:
            obs: Observations from environment.
            deterministic: If True, return deterministic action.

        Returns:
            Tuple of:
                - actions: Sampled actions.
                - log_probs: Log probabilities of actions.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions for computing loss.

        Args:
            obs: Observations from environment.
            actions: Actions to evaluate.

        Returns:
            Tuple of:
                - log_probs: Log probabilities of actions.
                - entropy: Entropy of action distribution.
        """
        raise NotImplementedError


class ContinuousActor(Actor, ABC):
    """Abstract base class for continuous action policies.

    Uses Gaussian (Normal) distribution for action sampling.

    Users must implement:
        - forward(): Returns action mean
        - get_action_dist(): Returns torch.distributions.Normal
        - act(): Sample actions (or use default implementation)
        - evaluate(): Evaluate actions (or use default implementation)

    Example:
        >>> class MyActor(ContinuousActor):
        ...     def __init__(self, obs_space, action_space, cfg):
        ...         super().__init__(obs_space, action_space, cfg)
        ...         # Your custom network here
        ...         self.net = nn.Sequential(...)
        ...         self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        ...
        ...     def forward(self, obs):
        ...         return self.net(obs)
        ...
        ...     def get_action_dist(self, obs):
        ...         mean = self.forward(obs)
        ...         std = torch.exp(self.log_std)
        ...         return torch.distributions.Normal(mean, std)
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Box,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Continuous Actor.

        Args:
            obs_space: Observation space.
            action_space: Continuous action space (Box).
            cfg: Optional configuration.
        """
        assert isinstance(action_space, spaces.Box), (
            f"ContinuousActor requires Box action space, got {type(action_space)}"
        )

        super().__init__(obs_space, action_space, cfg)

        # Continuous action info
        self.action_dim = action_space.shape[0] if len(action_space.shape) > 0 else 1

        # Tanh squashing flag (default True for bounded actions)
        self.use_tanh_squash = cfg.get("use_tanh_squash", True) if cfg else True

    @abstractmethod
    def get_action_dist(
        self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.distributions.Normal:
        """Get Gaussian action distribution.

        Args:
            obs: Observations.

        Returns:
            torch.distributions.Normal distribution.
        """
        raise NotImplementedError

    def act(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample continuous actions from Gaussian distribution.

        Args:
            obs: Observations.
            deterministic: If True, return mean without sampling.

        Returns:
            Tuple of (actions, log_probs).
            Actions are squashed with tanh if use_tanh_squash is True.
        """
        dist = self.get_action_dist(obs)

        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        if self.use_tanh_squash:
            # Tanh squashing for bounded actions
            # experimental for SAC(Soft Actor-Critic)
            action = torch.tanh(raw_action)
            # Correction for tanh squashing in log_prob
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

        return action, log_prob

    def evaluate(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate continuous actions.

        Args:
            obs: Observations.
            actions: Actions (potentially squashed with tanh).

        Returns:
            Tuple of (log_probs, entropy).
        """
        dist = self.get_action_dist(obs)

        if self.use_tanh_squash:
            # Inverse tanh to get raw actions
            # Clamp to avoid numerical issues
            clamped_actions = torch.clamp(actions, -0.999, 0.999)
            raw_actions = torch.atanh(clamped_actions)
            log_prob = dist.log_prob(raw_actions).sum(dim=-1)
            log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)
        else:
            log_prob = dist.log_prob(actions).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def to(self, device: torch.device) -> "ContinuousActor":
        """Move to device and update action bounds."""
        super().to(device)
        return self


class DiscreteActor(Actor, ABC):
    """Abstract base class for discrete action policies.

    Uses Categorical distribution for action sampling.

    Users must implement:
        - forward(): Returns action logits
        - get_action_dist(): Returns torch.distributions.Categorical

    Note: This is a placeholder for future implementation.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Discrete,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Discrete Actor.

        Args:
            obs_space: Observation space.
            action_space: Discrete action space.
            cfg: Optional configuration.
        """
        assert isinstance(action_space, spaces.Discrete), (
            f"DiscreteActor requires Discrete action space, got {type(action_space)}"
        )

        super().__init__(obs_space, action_space, cfg)

        # Discrete action info
        self.num_actions = action_space.n

    @abstractmethod
    def get_action_dist(
        self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.distributions.Categorical:
        """Get Categorical action distribution.

        Args:
            obs: Observations.

        Returns:
            torch.distributions.Categorical distribution.
        """
        raise NotImplementedError

    def act(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample discrete actions from Categorical distribution.

        Args:
            obs: Observations.
            deterministic: If True, return argmax action.

        Returns:
            Tuple of (actions, log_probs).
            Actions are integer indices.
        """
        dist = self.get_action_dist(obs)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate discrete actions.

        Args:
            obs: Observations.
            actions: Action indices.

        Returns:
            Tuple of (log_probs, entropy).
        """
        dist = self.get_action_dist(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module, ABC):
    """Abstract base class for value networks.

    The critic estimates the value function V(s).
    Supports asymmetric actor-critic (different observations).

    Users must implement:
        - forward(): Returns value estimates
        - get_value(): Returns value estimates (can call forward)

    The network structure is completely flexible - can use any architecture
    that takes observations and outputs a scalar value.

    Example:
        >>> class MyCritic(Critic):
        ...     def __init__(self, obs_space, cfg):
        ...         super().__init__(obs_space, cfg)
        ...         # Your custom network
        ...         self.encoder = nn.Sequential(...)
        ...         self.value_head = nn.Linear(feature_dim, 1)
        ...
        ...     def forward(self, obs):
        ...         features = self.encoder(obs)
        ...         return self.value_head(features).squeeze(-1)
        ...
        ...     def get_value(self, obs):
        ...         return self.forward(obs)
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Critic.

        Args:
            obs_space: Observation space from gymnasium.
            cfg: Optional configuration dictionary.
        """
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.cfg = cfg or {}
        self.obs_shape = obs_space.shape

    @abstractmethod
    def forward(
        self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass to get value estimates.

        Args:
            obs: Observations. Can be:
                - torch.Tensor: Observations (any shape)
                - Dict[str, torch.Tensor]: Dictionary of observations

        Returns:
            Value estimates. Shape: (batch_size,)
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(
        self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Get value estimates.

        This is typically the same as forward(), but allows for special
        handling if needed (e.g., target networks).

        Args:
            obs: Observations.

        Returns:
            Value estimates. Shape: (batch_size,)
        """
        raise NotImplementedError
