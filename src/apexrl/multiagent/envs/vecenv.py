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

"""Vectorized multi-agent environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from gymnasium import spaces

from apexrl.multiagent.typing import AgentID, MultiAgentAction, MultiAgentObs
from apexrl.utils import Observation


class MultiAgentVecEnv(ABC):
    """Abstract vectorized environment interface for multi-agent algorithms.

    Observations, actions, rewards and done masks are keyed by agent id. A
    centralized state is optional and is intended for CTDE algorithms such as
    MAPPO.
    """

    num_envs: int
    possible_agents: list[AgentID]
    observation_spaces: dict[AgentID, spaces.Space]
    action_spaces: dict[AgentID, spaces.Space]
    state_space: spaces.Space | None = None
    device: torch.device | str = "cpu"

    @abstractmethod
    def reset(self) -> tuple[MultiAgentObs, dict[str, Any]] | MultiAgentObs:
        """Reset all vectorized environments and return per-agent observations."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        actions: MultiAgentAction,
    ) -> (
        tuple[
            MultiAgentObs,
            dict[AgentID, torch.Tensor],
            dict[AgentID, torch.Tensor],
            dict[AgentID, torch.Tensor],
            dict[str, Any],
        ]
        | tuple[
            MultiAgentObs,
            dict[AgentID, torch.Tensor],
            dict[AgentID, torch.Tensor],
            dict[str, Any],
        ]
    ):
        """Step the environment with per-agent actions."""
        raise NotImplementedError

    def get_observations(self) -> MultiAgentObs:
        """Return the current per-agent observations.

        Environments can override this to avoid an implicit reset before rollout.
        """
        reset_out = self.reset()
        if isinstance(reset_out, tuple):
            return reset_out[0]
        return reset_out

    def get_state(self) -> Observation | None:
        """Return the current centralized state for CTDE critics, if available."""
        return None

    def close(self) -> None:
        """Close environment resources."""
