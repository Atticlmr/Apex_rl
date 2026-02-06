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

"""MLP-based Actor and Critic implementations.

These are reference implementations using multi-layer perceptrons.
Users can use these directly or as templates for custom implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from apexrl.models.base import ContinuousActor, Critic, DiscreteActor


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "elu",
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Args:
        input_dim: Input dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Output dimension.
        activation: Activation function name ("relu", "elu", "tanh").
        layer_norm: Whether to use layer normalization.

    Returns:
        Sequential MLP module.
    """
    layers = []
    prev_dim = input_dim

    # Get activation function
    activation_fn = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }.get(activation.lower(), nn.ELU)

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_fn())
        prev_dim = hidden_dim

    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


class MLPActor(ContinuousActor):
    """MLP-based continuous action actor.

    Simple MLP that outputs Gaussian action distribution.

    Example:
        >>> from gymnasium import spaces
        >>> import torch
        >>>
        >>> obs_space = spaces.Box(low=-1, high=1, shape=(4,))
        >>> action_space = spaces.Box(low=-1, high=1, shape=(2,))
        >>>
        >>> actor = MLPActor(obs_space, action_space, {
        ...     "hidden_dims": [256, 256],
        ...     "activation": "elu",
        ...     "learn_std": True,
        ... })
        >>>
        >>> obs = torch.randn(32, 4)  # batch_size=32
        >>> actions, log_probs = actor.act(obs)
        >>> print(actions.shape)  # [32, 2]
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Box,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MLP Actor.

        Args:
            obs_space: Observation space.
            action_space: Continuous action space.
            cfg: Configuration dict with keys:
                - hidden_dims: List of hidden layer sizes (default: [256, 256])
                - activation: Activation function (default: "elu")
                - layer_norm: Use layer norm (default: False)
                - learn_std: Learn std dev (default: True)
                - init_std: Initial std value (default: 1.0)
        """
        super().__init__(obs_space, action_space, cfg)

        # Get config with defaults
        hidden_dims = cfg.get("hidden_dims", [256, 256]) if cfg else [256, 256]
        activation = cfg.get("activation", "elu") if cfg else "elu"
        layer_norm = cfg.get("layer_norm", False) if cfg else False
        learn_std = cfg.get("learn_std", True) if cfg else True
        init_std = cfg.get("init_std", 1.0) if cfg else 1.0

        # Build network
        # Support different observation types
        if isinstance(obs_space, spaces.Box):
            obs_dim = int(torch.prod(torch.tensor(obs_space.shape)))
        else:
            raise NotImplementedError(
                f"MLPActor only supports Box obs space, got {type(obs_space)}"
            )

        self.mlp = build_mlp(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            activation=activation,
            layer_norm=layer_norm,
        )

        # Std dev parameter
        if learn_std:
            init_log_std = torch.log(torch.tensor(init_std))
            self.log_std = nn.Parameter(torch.ones(self.action_dim) * init_log_std)
            self.std = None
        else:
            self.register_buffer("std", torch.ones(self.action_dim) * init_std)
            self.log_std = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action mean.

        Args:
            obs: Observations. Shape: (batch_size, *obs_shape)

        Returns:
            Action mean. Shape: (batch_size, action_dim)
        """
        # Flatten if needed
        if obs.dim() > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.mlp(obs)

    def get_action_dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Get Gaussian action distribution.

        Args:
            obs: Observations.

        Returns:
            Normal distribution.
        """
        mean = self.forward(obs)
        if self.log_std is not None:
            std = torch.exp(self.log_std)
        else:
            std = self.std
        return torch.distributions.Normal(mean, std)


class MLPCritic(Critic):
    """MLP-based value network.

    Simple MLP that estimates state values.

    Example:
        >>> from gymnasium import spaces
        >>> import torch
        >>>
        >>> obs_space = spaces.Box(low=-1, high=1, shape=(4,))
        >>> critic = MLPCritic(obs_space, {
        ...     "hidden_dims": [256, 256],
        ...     "activation": "elu",
        ... })
        >>>
        >>> obs = torch.randn(32, 4)
        >>> values = critic.get_value(obs)
        >>> print(values.shape)  # [32]
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MLP Critic.

        Args:
            obs_space: Observation space.
            cfg: Configuration dict with keys:
                - hidden_dims: List of hidden layer sizes (default: [256, 256])
                - activation: Activation function (default: "elu")
                - layer_norm: Use layer norm (default: False)
        """
        super().__init__(obs_space, cfg)

        # Get config with defaults
        hidden_dims = cfg.get("hidden_dims", [256, 256]) if cfg else [256, 256]
        activation = cfg.get("activation", "elu") if cfg else "elu"
        layer_norm = cfg.get("layer_norm", False) if cfg else False

        # Build network
        if isinstance(obs_space, spaces.Box):
            obs_dim = int(torch.prod(torch.tensor(obs_space.shape)))
        else:
            raise NotImplementedError(
                f"MLPCritic only supports Box obs space, got {type(obs_space)}"
            )

        self.mlp = build_mlp(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass to get value estimates.

        Args:
            obs: Observations. Shape: (batch_size, *obs_shape)

        Returns:
            Value estimates. Shape: (batch_size,)
        """
        # Flatten if needed
        if obs.dim() > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.mlp(obs).squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimates.

        Args:
            obs: Observations.

        Returns:
            Value estimates. Shape: (batch_size,)
        """
        return self.forward(obs)


class CNNActor(ContinuousActor):
    """CNN-based actor for image observations.

    Example for Atari-style images or camera inputs:
        >>> from gymnasium import spaces
        >>> import torch
        >>>
        >>> obs_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
        >>> action_space = spaces.Box(low=-1, high=1, shape=(4,))
        >>>
        >>> actor = CNNActor(obs_space, action_space, {
        ...     "conv_channels": [32, 64, 64],
        ...     "conv_kernels": [8, 4, 3],
        ...     "conv_strides": [4, 2, 1],
        ...     "hidden_dims": [512],
        ... })
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Box,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CNN Actor.

        Args:
            obs_space: Image observation space (Box with 3D shape).
            action_space: Continuous action space.
            cfg: Configuration dict with keys:
                - conv_channels: List of conv channel sizes (default: [32, 64, 64])
                - conv_kernels: List of kernel sizes (default: [8, 4, 3])
                - conv_strides: List of strides (default: [4, 2, 1])
                - hidden_dims: MLP hidden dims after conv (default: [512])
                - activation: Activation function (default: "relu")
                - learn_std: Learn std dev (default: True)
        """
        super().__init__(obs_space, action_space, cfg)

        cfg = cfg or {}
        conv_channels = cfg.get("conv_channels", [32, 64, 64])
        conv_kernels = cfg.get("conv_kernels", [8, 4, 3])
        conv_strides = cfg.get("conv_strides", [4, 2, 1])
        hidden_dims = cfg.get("hidden_dims", [512])
        activation = cfg.get("activation", "relu")
        learn_std = cfg.get("learn_std", True)
        init_std = cfg.get("init_std", 1.0)

        assert isinstance(obs_space, spaces.Box), "CNNActor requires Box obs space"
        assert len(obs_space.shape) == 3, (
            f"CNNActor expects 3D obs (C, H, W), got {obs_space.shape}"
        )

        in_channels = obs_space.shape[0]

        # Build CNN encoder
        conv_layers = []
        prev_channels = in_channels
        for out_channels, kernel, stride in zip(
            conv_channels, conv_kernels, conv_strides
        ):
            conv_layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel, stride=stride)
            )
            conv_layers.append(nn.ReLU())
            prev_channels = out_channels
        conv_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*conv_layers)

        # Compute feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape)
            feature_dim = self.encoder(dummy_input).shape[1]

        # Build head MLP
        self.head = build_mlp(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            activation=activation,
        )

        # Std dev
        if learn_std:
            init_log_std = torch.log(torch.tensor(init_std))
            self.log_std = nn.Parameter(torch.ones(self.action_dim) * init_log_std)
            self.std = None
        else:
            self.register_buffer("std", torch.ones(self.action_dim) * init_std)
            self.log_std = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Image observations. Shape: (batch_size, C, H, W)

        Returns:
            Action mean. Shape: (batch_size, action_dim)
        """
        # Normalize image to [0, 1] or [-1, 1]
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0

        features = self.encoder(obs)
        return self.head(features)

    def get_action_dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Get Gaussian action distribution."""
        mean = self.forward(obs)
        if self.log_std is not None:
            std = torch.exp(self.log_std)
        else:
            std = self.std
        return torch.distributions.Normal(mean, std)


class CNNCritic(Critic):
    """CNN-based critic for image observations."""

    def __init__(
        self,
        obs_space: spaces.Space,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CNN Critic.

        Args:
            obs_space: Image observation space.
            cfg: Configuration dict (same as CNNActor).
        """
        super().__init__(obs_space, cfg)

        cfg = cfg or {}
        conv_channels = cfg.get("conv_channels", [32, 64, 64])
        conv_kernels = cfg.get("conv_kernels", [8, 4, 3])
        conv_strides = cfg.get("conv_strides", [4, 2, 1])
        hidden_dims = cfg.get("hidden_dims", [512])
        activation = cfg.get("activation", "relu")

        assert isinstance(obs_space, spaces.Box), "CNNCritic requires Box obs space"
        assert len(obs_space.shape) == 3, (
            f"CNNCritic expects 3D obs, got {obs_space.shape}"
        )

        in_channels = obs_space.shape[0]

        # Build CNN encoder
        conv_layers = []
        prev_channels = in_channels
        for out_channels, kernel, stride in zip(
            conv_channels, conv_kernels, conv_strides
        ):
            conv_layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel, stride=stride)
            )
            conv_layers.append(nn.ReLU())
            prev_channels = out_channels
        conv_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*conv_layers)

        # Compute feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape)
            feature_dim = self.encoder(dummy_input).shape[1]

        # Build head MLP
        self.head = build_mlp(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass to get value estimates.

        Args:
            obs: Image observations. Shape: (batch_size, C, H, W)

        Returns:
            Value estimates. Shape: (batch_size,)
        """
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0

        features = self.encoder(obs)
        return self.head(features).squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimates."""
        return self.forward(obs)


class MLPDiscreteActor(DiscreteActor):
    """MLP-based discrete action actor.

    Simple MLP that outputs Categorical action distribution.

    Example:
        >>> from gymnasium import spaces
        >>> import torch
        >>>
        >>> obs_space = spaces.Box(low=-1, high=1, shape=(4,))
        >>> action_space = spaces.Discrete(2)
        >>>
        >>> actor = MLPDiscreteActor(obs_space, action_space, {
        ...     "hidden_dims": [256, 256],
        ...     "activation": "elu",
        ... })
        >>>
        >>> obs = torch.randn(32, 4)  # batch_size=32
        >>> actions, log_probs = actor.act(obs)
        >>> print(actions.shape)  # [32]
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Discrete,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MLP Discrete Actor.

        Args:
            obs_space: Observation space.
            action_space: Discrete action space.
            cfg: Configuration dict with keys:
                - hidden_dims: List of hidden layer sizes (default: [256, 256])
                - activation: Activation function (default: "elu")
                - layer_norm: Use layer norm (default: False)
        """
        super().__init__(obs_space, action_space, cfg)

        # Get config with defaults
        hidden_dims = cfg.get("hidden_dims", [256, 256]) if cfg else [256, 256]
        activation = cfg.get("activation", "elu") if cfg else "elu"
        layer_norm = cfg.get("layer_norm", False) if cfg else False

        # Build network
        if isinstance(obs_space, spaces.Box):
            obs_dim = int(torch.prod(torch.tensor(obs_space.shape)))
        else:
            raise NotImplementedError(
                f"MLPDiscreteActor only supports Box obs space, got {type(obs_space)}"
            )

        self.mlp = build_mlp(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=self.num_actions,
            activation=activation,
            layer_norm=layer_norm,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action logits.

        Args:
            obs: Observations. Shape: (batch_size, *obs_shape)

        Returns:
            Action logits. Shape: (batch_size, num_actions)
        """
        # Flatten if needed
        if obs.dim() > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.mlp(obs)

    def get_action_dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Get Categorical action distribution.

        Args:
            obs: Observations.

        Returns:
            Categorical distribution.
        """
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)
