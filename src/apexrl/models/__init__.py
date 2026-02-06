"""Models module for ApexRL."""

from apexrl.models.base import Actor, ContinuousActor, Critic, DiscreteActor
from apexrl.models.mlp import (
    CNNActor,
    CNNCritic,
    MLPActor,
    MLPCritic,
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
    "build_mlp",
    # CNN implementations
    "CNNActor",
    "CNNCritic",
]
