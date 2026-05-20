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

"""GRU actor and critic implementations for recurrent PPO.

The classes in this module are reference recurrent networks for
``RecurrentPPO``. They accept either single-step observations with leading batch
dimension ``(batch, ...)`` or sequence observations with leading dimensions
``(time, batch, ...)``. Recurrent hidden states use PyTorch GRU layout
``(num_layers, batch, hidden_size)``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces

from apexrl.models.base import ContinuousActor, Critic, DiscreteActor
from apexrl.models.mlp import _orthogonal_init, build_mlp
from apexrl.utils.observation import (
    Observation,
    ObservationSpec,
    TensorDict,
    TensorLeafSpec,
    space_to_spec,
    spec_numel,
)


def _obs_space_numel(obs_space: spaces.Space) -> int:
    """Return flattened feature size for a Gymnasium observation space."""
    return spec_numel(space_to_spec(obs_space))


def _is_observation_mapping(value: Any) -> bool:
    """Return whether a value behaves like a structured observation mapping."""
    return isinstance(value, TensorDict) or isinstance(value, dict)


def _flatten_recurrent_observation(observation: Observation) -> torch.Tensor:
    """Flatten observation leaves while preserving leading time/batch dims."""
    if _is_observation_mapping(observation):
        parts = [
            _flatten_recurrent_observation(observation[key])
            for key in observation.keys()
        ]
        if not parts:
            raise ValueError("Observation mapping is empty")
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    observation = observation.to(dtype=torch.float32)
    if observation.dim() == 1:
        return observation.unsqueeze(-1)
    return observation.reshape(*observation.shape[:2], -1)


def _to_sequence_observation(
    observation: Observation,
    obs_spec: ObservationSpec,
) -> tuple[Observation, bool]:
    """Ensure observations have leading ``(time, batch)`` dimensions."""
    first = _first_tensor(observation)
    first_spec = _first_leaf_spec(obs_spec)
    if first is None:
        raise ValueError("Observation tree does not contain tensor leaves")
    if first.dim() == len(first_spec.shape) + 2:
        return observation, False
    if first.dim() != len(first_spec.shape) + 1:
        raise ValueError(
            "Recurrent observations must have batch or time-batch dimensions; "
            f"got tensor shape {tuple(first.shape)} for leaf shape {first_spec.shape}"
        )
    return _unsqueeze_time(observation), True


def _first_leaf_spec(spec: ObservationSpec) -> TensorLeafSpec:
    """Return the first leaf spec in a structured observation spec."""
    if isinstance(spec, dict):
        for value in spec.values():
            return _first_leaf_spec(value)
        raise ValueError("Observation spec is empty")
    if isinstance(spec, TensorLeafSpec):
        return spec
    return TensorLeafSpec(tuple(spec), torch.float32)


def _first_tensor(value: Any) -> torch.Tensor | None:
    """Return the first tensor leaf in a structured observation tree."""
    if _is_observation_mapping(value):
        for sub_value in value.values():
            tensor = _first_tensor(sub_value)
            if tensor is not None:
                return tensor
        return None
    return value


def _unsqueeze_time(value: Observation) -> Observation:
    """Add a leading time dimension to every observation leaf."""
    if _is_observation_mapping(value):
        return TensorDict(
            {key: _unsqueeze_time(sub_value) for key, sub_value in value.items()},
            batch_size=None,
        )
    return value.unsqueeze(0)


def _apply_sequence_masks(
    rnn: nn.GRU,
    features: torch.Tensor,
    hidden_state: torch.Tensor,
    masks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a GRU sequence and reset hidden states where masks are zero."""
    if masks is None:
        return rnn(features, hidden_state)

    outputs = []
    hidden = hidden_state
    for t in range(features.shape[0]):
        mask = masks[t].reshape(1, features.shape[1], 1).to(features.device)
        hidden = hidden * mask
        out, hidden = rnn(features[t : t + 1], hidden)
        outputs.append(out)
    return torch.cat(outputs, dim=0), hidden


class _GRUBase(nn.Module):
    """Shared encoder and GRU helpers for recurrent actor/critic modules."""

    def _init_gru_base(
        self,
        *,
        obs_space: spaces.Space,
        cfg: dict[str, Any],
        default_activation: str = "elu",
    ) -> None:
        """Build the flattened observation encoder and GRU core."""
        hidden_dims = cfg.get("hidden_dims", [256])
        activation = cfg.get("activation", default_activation)
        layer_norm = cfg.get("layer_norm", False)
        self.rnn_hidden_size = int(cfg.get("rnn_hidden_size", 256))
        self.rnn_num_layers = int(cfg.get("rnn_num_layers", 1))
        self.feature_dim = int(
            cfg.get("rnn_feature_dim", hidden_dims[-1] if hidden_dims else 256)
        )
        self.obs_spec = space_to_spec(obs_space)

        obs_dim = _obs_space_numel(obs_space)
        self.encoder = build_mlp(
            input_dim=obs_dim,
            hidden_dims=hidden_dims[:-1] if hidden_dims else [],
            output_dim=self.feature_dim,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
        )
        self.encoder.apply(lambda module: _orthogonal_init(module, math.sqrt(2.0)))
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return a zero GRU hidden state with shape ``(layers, batch, hidden)``."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            self.rnn_num_layers,
            batch_size,
            self.rnn_hidden_size,
            device=device,
        )

    @property
    def hidden_state_shape(self) -> tuple[int, int]:
        """Return hidden state shape without the batch dimension."""
        return (self.rnn_num_layers, self.rnn_hidden_size)

    def _forward_sequence(
        self,
        obs: Observation,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Encode observations and run the GRU core.

        Args:
            obs: Single-step or sequence observation tree.
            hidden_state: Optional initial GRU hidden state.
            masks: Optional sequence mask with shape ``(time, batch, 1)``. A
                zero mask clears hidden state before the corresponding step.

        Returns:
            Tuple of GRU outputs, next hidden state, and a flag indicating
            whether the input was a single-step observation.
        """
        obs, was_single_step = _to_sequence_observation(obs, self.obs_spec)
        flat_obs = _flatten_recurrent_observation(obs)
        seq_len, batch_size = flat_obs.shape[:2]
        flat_features = self.encoder(flat_obs.reshape(seq_len * batch_size, -1))
        features = flat_features.reshape(seq_len, batch_size, self.feature_dim)
        if hidden_state is None:
            hidden_state = self.get_initial_state(batch_size, features.device)
        output, next_hidden = _apply_sequence_masks(
            self.rnn,
            features,
            hidden_state,
            masks,
        )
        return output, next_hidden, was_single_step


class GRUActor(_GRUBase, ContinuousActor):
    """GRU-based Gaussian actor for recurrent PPO.

    Config keys:
        hidden_dims: Feed-forward encoder dimensions. The last value is used as
            the GRU input feature size unless ``rnn_feature_dim`` is set.
        activation: Encoder activation name.
        layer_norm: Whether to apply layer normalization inside the encoder.
        rnn_hidden_size: GRU hidden size.
        rnn_num_layers: Number of GRU layers.
        rnn_feature_dim: Optional explicit encoder output dimension.
        learn_std: Whether to learn Gaussian standard deviation.
        init_std: Initial Gaussian standard deviation.
    """

    is_recurrent = True

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Box,
        cfg: dict[str, Any] | None = None,
    ):
        """Initialize the recurrent continuous actor.

        Args:
            obs_space: Actor observation space.
            action_space: Continuous action space.
            cfg: Optional network configuration dictionary.
        """
        cfg = cfg or {}
        ContinuousActor.__init__(self, obs_space, action_space, cfg)
        self._init_gru_base(obs_space=obs_space, cfg=cfg)
        learn_std = cfg.get("learn_std", True)
        init_std = cfg.get("init_std", 1.0)
        self.mean_head = nn.Linear(self.rnn_hidden_size, self.action_dim)
        _orthogonal_init(self.mean_head, 0.01)
        if learn_std:
            init_log_std = torch.log(torch.tensor(init_std))
            self.log_std = nn.Parameter(torch.ones(self.action_dim) * init_log_std)
            self.std = None
        else:
            self.register_buffer("std", torch.ones(self.action_dim) * init_std)
            self.log_std = None

    def forward(self, obs: Observation) -> torch.Tensor:
        """Return continuous action means.

        Args:
            obs: Observation tree with shape ``(batch, ...)`` or
                ``(time, batch, ...)``.

        Returns:
            Action mean tensor with shape ``(batch, action_dim)`` for single
            steps or ``(time, batch, action_dim)`` for sequences.
        """
        output, _, was_single_step = self._forward_sequence(obs)
        mean = self.mean_head(output)
        return mean.squeeze(0) if was_single_step else mean

    def get_action_dist(self, obs: Observation) -> torch.distributions.Normal:
        """Return the Gaussian action distribution without carrying state."""
        mean = self.forward(obs)
        return torch.distributions.Normal(mean, self._std())

    def get_action_dist_rnn(
        self,
        obs: Observation,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor, bool]:
        """Return action distribution and next hidden state.

        Args:
            obs: Single-step or sequence observation tree.
            hidden_state: Initial GRU hidden state.
            masks: Optional reset masks with shape ``(time, batch, 1)``.

        Returns:
            Tuple of action distribution, next hidden state, and single-step
            input flag.
        """
        output, next_hidden, was_single_step = self._forward_sequence(
            obs,
            hidden_state,
            masks,
        )
        mean = self.mean_head(output)
        dist = torch.distributions.Normal(mean, self._std())
        return dist, next_hidden, was_single_step

    def act(
        self,
        obs: Observation,
        deterministic: bool = False,
        *,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or return deterministic continuous actions.

        Args:
            obs: Single-step or sequence observation tree.
            deterministic: If ``True``, return distribution means.
            hidden_state: Optional initial GRU hidden state.
            masks: Optional reset masks with shape ``(time, batch, 1)``.

        Returns:
            Tuple of actions, log probabilities and next hidden state.
        """
        dist, next_hidden, was_single_step = self.get_action_dist_rnn(
            obs,
            hidden_state,
            masks,
        )
        raw_action = dist.mean if deterministic else dist.rsample()
        if self.use_tanh_squash:
            action = torch.tanh(raw_action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
        if was_single_step:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob, next_hidden

    def evaluate(
        self,
        obs: Observation,
        actions: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under the recurrent policy.

        Args:
            obs: Single-step or sequence observation tree.
            actions: Actions with matching leading batch or time/batch
                dimensions.
            hidden_state: Optional initial GRU hidden state.
            masks: Optional reset masks with shape ``(time, batch, 1)``.

        Returns:
            Tuple of log probabilities, entropy and next hidden state.
        """
        dist, next_hidden, was_single_step = self.get_action_dist_rnn(
            obs,
            hidden_state,
            masks,
        )
        if was_single_step:
            actions = actions.unsqueeze(0)
        if self.use_tanh_squash:
            clamped_actions = torch.clamp(actions, -0.999, 0.999)
            raw_actions = torch.atanh(clamped_actions)
            log_prob = dist.log_prob(raw_actions).sum(dim=-1)
            log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)
        else:
            log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        if was_single_step:
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
        return log_prob, entropy, next_hidden

    def _std(self) -> torch.Tensor:
        """Return the current Gaussian standard deviation tensor."""
        if self.log_std is None:
            return self.std
        min_log_std = self.cfg.get("min_log_std", -5.0)
        max_log_std = self.cfg.get("max_log_std", 2.0)
        return torch.exp(torch.clamp(self.log_std, min_log_std, max_log_std))


class GRUDiscreteActor(_GRUBase, DiscreteActor):
    """GRU-based categorical actor for recurrent PPO.

    The observation and hidden-state conventions match :class:`GRUActor`, but
    the policy distribution is categorical and actions are integer indices.
    """

    is_recurrent = True

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Discrete,
        cfg: dict[str, Any] | None = None,
    ):
        """Initialize the recurrent discrete actor.

        Args:
            obs_space: Actor observation space.
            action_space: Discrete action space.
            cfg: Optional network configuration dictionary.
        """
        cfg = cfg or {}
        DiscreteActor.__init__(self, obs_space, action_space, cfg)
        self._init_gru_base(obs_space=obs_space, cfg=cfg)
        self.logits_head = nn.Linear(self.rnn_hidden_size, self.num_actions)
        _orthogonal_init(self.logits_head, 0.01)

    def forward(self, obs: Observation) -> torch.Tensor:
        """Return categorical action logits."""
        output, _, was_single_step = self._forward_sequence(obs)
        logits = self.logits_head(output)
        return logits.squeeze(0) if was_single_step else logits

    def get_action_dist(self, obs: Observation) -> torch.distributions.Categorical:
        """Return the categorical action distribution without carrying state."""
        return torch.distributions.Categorical(logits=self.forward(obs))

    def get_action_dist_rnn(
        self,
        obs: Observation,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor, bool]:
        """Return categorical distribution and next hidden state."""
        output, next_hidden, was_single_step = self._forward_sequence(
            obs,
            hidden_state,
            masks,
        )
        dist = torch.distributions.Categorical(logits=self.logits_head(output))
        return dist, next_hidden, was_single_step

    def act(
        self,
        obs: Observation,
        deterministic: bool = False,
        *,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or return deterministic discrete actions.

        Returns:
            Tuple of action indices, log probabilities and next hidden state.
        """
        dist, next_hidden, was_single_step = self.get_action_dist_rnn(
            obs,
            hidden_state,
            masks,
        )
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        if was_single_step:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob, next_hidden

    def evaluate(
        self,
        obs: Observation,
        actions: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate discrete actions under the recurrent policy."""
        dist, next_hidden, was_single_step = self.get_action_dist_rnn(
            obs,
            hidden_state,
            masks,
        )
        if was_single_step:
            actions = actions.unsqueeze(0)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if was_single_step:
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
        return log_prob, entropy, next_hidden


class GRUCritic(_GRUBase, Critic):
    """GRU value network for recurrent PPO.

    The critic accepts single-step or sequence observations and returns scalar
    value predictions with matching leading dimensions.
    """

    is_recurrent = True

    def __init__(
        self,
        obs_space: spaces.Space,
        cfg: dict[str, Any] | None = None,
    ):
        """Initialize the recurrent critic.

        Args:
            obs_space: Critic observation space.
            cfg: Optional network configuration dictionary.
        """
        cfg = cfg or {}
        Critic.__init__(self, obs_space, cfg)
        self._init_gru_base(obs_space=obs_space, cfg=cfg)
        self.value_head = nn.Linear(self.rnn_hidden_size, 1)
        _orthogonal_init(self.value_head, 1.0)

    def forward(self, obs: Observation) -> torch.Tensor:
        """Return value predictions without explicitly carrying hidden state."""
        values, _, was_single_step = self.get_value_rnn(obs)
        return values.squeeze(0) if was_single_step else values

    def get_value(self, obs: Observation) -> torch.Tensor:
        """Return value predictions for PPO-compatible critic calls."""
        return self.forward(obs)

    def get_value_rnn(
        self,
        obs: Observation,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Return values and next critic hidden state.

        Args:
            obs: Single-step or sequence observation tree.
            hidden_state: Optional initial GRU hidden state.
            masks: Optional reset masks with shape ``(time, batch, 1)``.

        Returns:
            Tuple of values, next hidden state and single-step input flag.
        """
        output, next_hidden, was_single_step = self._forward_sequence(
            obs,
            hidden_state,
            masks,
        )
        values = self.value_head(output).squeeze(-1)
        return values, next_hidden, was_single_step
