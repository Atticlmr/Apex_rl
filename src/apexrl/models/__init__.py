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

"""Models module for ApexRL."""

from apexrl.models.base import Actor, ContinuousActor, Critic, DiscreteActor
from apexrl.models.mlp import (
    CNNActor,
    CNNCritic,
    MLPActor,
    MLPCritic,
    MLPDiscreteActor,
    build_mlp,
)

__all__ = [
    # Base classes
    "Actor",
    "ContinuousActor",
    "DiscreteActor",
    "Critic",
    # MLP implementations
    "MLPActor",
    "MLPCritic",
    "MLPDiscreteActor",
    "build_mlp",
    # CNN implementations
    "CNNActor",
    "CNNCritic",
]
