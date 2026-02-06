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

"""Custom optimizers for reinforcement learning.

This module provides optimized implementations of various optimizers
including Adam, AdamW, and Muon optimizers.
"""

from __future__ import annotations

from torch.optim import Adam, AdamW

from apexrl.optimizers.muon import Muon


__all__ = ["Adam", "AdamW", "Muon"]


def get_optimizer(name: str):
    """Get optimizer class by name.

    Args:
        name: Name of the optimizer ("adam", "adamw", "muon").

    Returns:
        Optimizer class.

    Raises:
        ValueError: If optimizer name is not recognized.
    """
    name_lower = name.lower()
    if name_lower == "adam":
        return Adam
    elif name_lower == "adamw":
        return AdamW
    elif name_lower == "muon":
        return Muon
    else:
        raise ValueError(f"Unknown optimizer: {name}. Supported: adam, adamw, muon")
