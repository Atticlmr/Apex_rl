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

"""Independent PPO (IPPO) implementation."""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from torch import nn

from apexrl.models import Actor, Critic
from apexrl.multiagent.algorithms.ippo.config import IPPOConfig
from apexrl.multiagent.algorithms.mappo import MAPPO
from apexrl.multiagent.envs import MultiAgentVecEnv
from apexrl.multiagent.typing import AgentID


class IPPO(MAPPO):
    """Independent PPO with decentralized value functions.

    The implementation reuses MAPPO's rollout, buffer, optimizer and logging
    machinery while forcing ``centralized_critic=False`` so critics are built
    from per-agent observation spaces instead of a global state space.
    """

    def __init__(
        self,
        *,
        env: MultiAgentVecEnv | None = None,
        cfg: IPPOConfig | None = None,
        possible_agents: list[AgentID] | None = None,
        models: dict[AgentID, dict[str, nn.Module]] | None = None,
        observation_spaces: dict[AgentID, spaces.Space] | None = None,
        action_spaces: dict[AgentID, spaces.Space] | None = None,
        state_space: spaces.Space | None = None,
        actor_class: type[Actor] | None = None,
        critic_class: type[Critic] | None = None,
        actor_cfg: dict[str, Any] | None = None,
        critic_cfg: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        ippo_cfg = cfg if cfg is not None else IPPOConfig()
        if ippo_cfg.centralized_critic:
            raise ValueError("IPPO requires centralized_critic=False")
        super().__init__(
            env=env,
            cfg=ippo_cfg,
            possible_agents=possible_agents,
            models=models,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            state_space=state_space,
            actor_class=actor_class,
            critic_class=critic_class,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
            device=device,
        )
