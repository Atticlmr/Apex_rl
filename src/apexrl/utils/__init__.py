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

"""Utility functions and classes for ApexRL."""

from apexrl.utils.logger import (
    BaseLogger,
    Logger,
    MultiLogger,
    SwanLabLogger,
    TensorBoardLogger,
    WandbLogger,
    get_logger,
)
from apexrl.utils.observation import (
    TensorDict,
    TensorLeafSpec,
    actor_space_from_observation_space,
    allocate_observation_storage,
    clone_observation,
    critic_space_from_observation_space,
    flatten_observation,
    flatten_time_env_observation,
    observation_batch_size,
    observation_index,
    observation_set_index,
    observation_to_device,
    observation_to_tensor,
    space_to_spec,
    spec_numel,
    split_actor_critic_observations,
    stack_observations,
    zeros_like_observation,
)

__all__ = [
    "BaseLogger",
    "Logger",
    "TensorBoardLogger",
    "WandbLogger",
    "SwanLabLogger",
    "MultiLogger",
    "get_logger",
    "TensorDict",
    "TensorLeafSpec",
    "actor_space_from_observation_space",
    "critic_space_from_observation_space",
    "space_to_spec",
    "spec_numel",
    "observation_to_tensor",
    "stack_observations",
    "zeros_like_observation",
    "clone_observation",
    "flatten_observation",
    "observation_batch_size",
    "observation_index",
    "observation_set_index",
    "flatten_time_env_observation",
    "allocate_observation_storage",
    "observation_to_device",
    "split_actor_critic_observations",
]
