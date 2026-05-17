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

"""Utility functions for multi-agent data normalization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from gymnasium import spaces

from apexrl.multiagent.typing import AgentID, MultiAgentObs
from apexrl.utils import Observation, observation_to_tensor, space_to_spec


def validate_agent_ids(
    possible_agents: Iterable[AgentID],
    values: dict[AgentID, Any],
    *,
    name: str,
) -> None:
    """Validate that a per-agent mapping contains every configured agent."""
    missing = [agent_id for agent_id in possible_agents if agent_id not in values]
    if missing:
        raise KeyError(f"{name} is missing agents: {missing}")


def multiagent_to_tensor(
    observations: dict[AgentID, Any],
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> MultiAgentObs:
    """Convert a per-agent observation mapping to tensors on the target device."""
    return {
        agent_id: observation_to_tensor(obs, device=device, dtype=dtype)
        for agent_id, obs in observations.items()
    }


def state_to_tensor(
    state: Any | None,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> Observation | None:
    """Convert a centralized state to tensors when one is available."""
    if state is None:
        return None
    return observation_to_tensor(state, device=device, dtype=dtype)


def to_agent_tensor_dict(
    values: dict[AgentID, Any],
    possible_agents: Iterable[AgentID],
    device: torch.device | str,
    *,
    dtype: torch.dtype,
) -> dict[AgentID, torch.Tensor]:
    """Convert scalar/tensor per-agent values to tensors on the target device."""
    validate_agent_ids(possible_agents, values, name="values")
    result = {}
    for agent_id in possible_agents:
        value = values[agent_id]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device)
        result[agent_id] = value.to(device=device, dtype=dtype)
    return result


def merge_terminated_truncated(
    terminated: dict[AgentID, Any],
    truncated: dict[AgentID, Any],
    possible_agents: Iterable[AgentID],
    device: torch.device | str,
) -> dict[AgentID, torch.Tensor]:
    """Merge per-agent terminated and truncated masks into done masks."""
    validate_agent_ids(possible_agents, terminated, name="terminated")
    validate_agent_ids(possible_agents, truncated, name="truncated")
    dones = {}
    for agent_id in possible_agents:
        term = terminated[agent_id]
        trunc = truncated[agent_id]
        if not isinstance(term, torch.Tensor):
            term = torch.as_tensor(term, device=device)
        if not isinstance(trunc, torch.Tensor):
            trunc = torch.as_tensor(trunc, device=device)
        dones[agent_id] = term.to(device=device).bool() | trunc.to(device=device).bool()
    return dones


def action_space_shape_dtype(
    space: spaces.Space,
) -> tuple[tuple[int, ...], torch.dtype]:
    """Return storage shape and dtype for a supported action space."""
    if isinstance(space, spaces.Box):
        return tuple(space.shape), torch.float32
    if isinstance(space, spaces.Discrete):
        return (), torch.long
    raise NotImplementedError(f"Unsupported action space type: {type(space)}")


def observation_spec_from_space(space: spaces.Space) -> Any:
    """Return ApexRL recursive storage spec for an observation/state space."""
    return space_to_spec(space)
