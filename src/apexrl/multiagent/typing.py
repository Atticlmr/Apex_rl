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

"""Common type aliases for multi-agent reinforcement learning."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch
from gymnasium import spaces

from apexrl.utils import Observation

AgentID: TypeAlias = str
MultiAgentObs: TypeAlias = dict[AgentID, Observation]
MultiAgentAction: TypeAlias = dict[AgentID, torch.Tensor]
MultiAgentReward: TypeAlias = dict[AgentID, torch.Tensor]
MultiAgentDone: TypeAlias = dict[AgentID, torch.Tensor]
MultiAgentInfo: TypeAlias = dict[str, Any]
MultiAgentSpaces: TypeAlias = dict[AgentID, spaces.Space]
