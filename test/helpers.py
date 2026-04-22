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

"""Test helpers for multimodal and privileged observation environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class _BaseDictObsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 5):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Dict(
                    {
                        "image": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(1, 4, 4),
                            dtype=np.float32,
                        ),
                        "vector": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "privileged_obs": spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-2.0,
                            high=2.0,
                            shape=(5,),
                            dtype=np.float32,
                        ),
                        "context": spaces.Box(
                            low=-2.0,
                            high=2.0,
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

    def _get_obs(self) -> dict[str, Any]:
        phase = np.float32((self.step_count % self.max_steps) / max(self.max_steps, 1))
        image = np.full((1, 4, 4), fill_value=phase, dtype=np.float32)
        vector = np.array([phase, phase + 0.1, phase + 0.2], dtype=np.float32)
        privileged_state = np.array(
            [phase, phase + 0.25, phase + 0.5, phase + 0.75, phase + 1.0],
            dtype=np.float32,
        )
        privileged_context = np.array([phase - 0.2, phase + 0.3], dtype=np.float32)
        return {
            "obs": {
                "image": image,
                "vector": vector,
            },
            "privileged_obs": {
                "state": privileged_state,
                "context": privileged_context,
            },
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.step_count = 0
        return self._get_obs(), {}


class DictObsDiscreteEnv(_BaseDictObsEnv):
    """Discrete-action env with nested actor obs and privileged critic obs."""

    def __init__(self, max_steps: int = 5):
        super().__init__(max_steps=max_steps)
        self.action_space = spaces.Discrete(2)

    def step(self, action: int):
        reward = 1.0 if int(action) == (self.step_count % 2) else -0.25
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, False, {}


class DictObsContinuousEnv(_BaseDictObsEnv):
    """Continuous-action env with nested actor obs and privileged critic obs."""

    def __init__(self, max_steps: int = 5):
        super().__init__(max_steps=max_steps)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        target = np.float32((self.step_count % self.max_steps) / max(self.max_steps, 1))
        reward = 1.0 - abs(float(action[0]) - float(target))
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, False, {}


def make_multimodal_discrete_env() -> DictObsDiscreteEnv:
    """Factory for a nested discrete observation env."""
    return DictObsDiscreteEnv()


def make_multimodal_continuous_env() -> DictObsContinuousEnv:
    """Factory for a nested continuous observation env."""
    return DictObsContinuousEnv()
