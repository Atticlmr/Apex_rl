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

"""Configuration for Heterogeneous-Agent PPO (HAPPO)."""

from __future__ import annotations

from dataclasses import dataclass

from apexrl.multiagent.algorithms.mappo import MAPPOConfig


@dataclass
class HAPPOConfig(MAPPOConfig):
    """Configuration for HAPPO.

    HAPPO uses sequential per-agent policy updates with a correction factor
    from agents updated earlier in the current update order.
    """

    share_actor: bool = False
    shuffle_agent_order: bool = True
