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
