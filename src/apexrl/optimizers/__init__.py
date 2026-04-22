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

from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam, AdamW

from apexrl.optimizers.muon import (
    Muon,
    MuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
)

__all__ = ["Adam", "AdamW", "Muon", "build_optimizer", "get_optimizer"]


def get_optimizer(name: str):
    """Get optimizer class by name.

    Args:
        name: Name of the optimizer ("adam", "adamw", "muon(remain testing)").

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


def _iter_unique_named_parameters(
    modules: Iterable[nn.Module],
) -> list[tuple[str, torch.nn.Parameter, bool]]:
    """Return unique parameters together with an output-layer exclusion hint."""
    unique: list[tuple[str, torch.nn.Parameter, bool]] = []
    seen: set[int] = set()

    for module in modules:
        linear_or_conv_weights = [
            id(submodule.weight)
            for _, submodule in module.named_modules()
            if isinstance(submodule, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
            and hasattr(submodule, "weight")
        ]
        output_weight_ids = (
            {linear_or_conv_weights[-1]} if linear_or_conv_weights else set()
        )

        for name, param in module.named_parameters():
            param_id = id(param)
            if not param.requires_grad or param_id in seen:
                continue
            seen.add(param_id)
            unique.append((name, param, param_id in output_weight_ids))

    return unique


def _should_use_muon(
    name: str,
    param: torch.nn.Parameter,
    is_output_weight: bool,
) -> bool:
    """Heuristically decide whether a parameter should use Muon updates."""
    if param.ndim < 2 or is_output_weight:
        return False

    lowered = name.lower()
    excluded_tokens = (
        "embed",
        "embedding",
        "token",
        "position",
        "positional",
        "log_std",
        "mean_head",
        "value_head",
        "advantage_head",
        "q_head",
        "action_scale",
        "action_bias",
    )
    return not any(token in lowered for token in excluded_tokens)


def _split_muon_parameter_groups(
    modules: nn.Module | Iterable[nn.Module] | None = None,
    params: Iterable[torch.nn.Parameter] | None = None,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition parameters into Muon-compatible and Adam fallback groups."""
    if modules is not None:
        if isinstance(modules, nn.Module):
            modules = [modules]
        named_params = _iter_unique_named_parameters(modules)
        muon_params = [
            param
            for name, param, is_output_weight in named_params
            if _should_use_muon(name, param, is_output_weight)
        ]
        muon_param_ids = {id(param) for param in muon_params}
        aux_params = [
            param
            for _, param, _ in named_params
            if id(param) not in muon_param_ids
        ]
        return muon_params, aux_params

    if params is None:
        raise ValueError("Either modules or params must be provided")

    unique_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in params:
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        unique_params.append(param)

    muon_params = [param for param in unique_params if param.ndim >= 2]
    muon_param_ids = {id(param) for param in muon_params}
    aux_params = [param for param in unique_params if id(param) not in muon_param_ids]
    return muon_params, aux_params


def _resolve_parameter_iterable(
    modules: nn.Module | Iterable[nn.Module] | None = None,
    params: Iterable[torch.nn.Parameter] | None = None,
) -> list[torch.nn.Parameter]:
    """Normalize modules/params input into a unique parameter list."""
    if params is not None:
        return list(params)
    if modules is None:
        raise ValueError("Either modules or params must be provided")
    if isinstance(modules, nn.Module):
        return list(modules.parameters())

    resolved: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            resolved.append(param)
    return resolved


def build_optimizer(
    name: str,
    *,
    lr: float,
    modules: nn.Module | Iterable[nn.Module] | None = None,
    params: Iterable[torch.nn.Parameter] | None = None,
    weight_decay: float = 0.0,
    muon_momentum: float = 0.95,
    muon_aux_learning_rate: float | None = None,
    muon_aux_betas: tuple[float, float] = (0.9, 0.95),
    muon_aux_eps: float = 1e-10,
) -> torch.optim.Optimizer:
    """Build an optimizer, including mixed Muon/Adam parameter groups."""
    name_lower = name.lower()
    if name_lower != "muon":
        optimizer_cls = get_optimizer(name_lower)
        optimizer = optimizer_cls(
            _resolve_parameter_iterable(modules=modules, params=params),
            lr=lr,
            weight_decay=weight_decay,
        )
        for group in optimizer.param_groups:
            group["_apexrl_lr_scale"] = 1.0
        return optimizer

    muon_params, aux_params = _split_muon_parameter_groups(
        modules=modules,
        params=params,
    )
    aux_lr = muon_aux_learning_rate if muon_aux_learning_rate is not None else lr

    param_groups = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "lr": lr,
                "momentum": muon_momentum,
                "weight_decay": weight_decay,
                "use_muon": True,
            }
        )
    if aux_params:
        param_groups.append(
            {
                "params": aux_params,
                "lr": aux_lr,
                "betas": muon_aux_betas,
                "eps": muon_aux_eps,
                "weight_decay": weight_decay,
                "use_muon": False,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters were provided to build_optimizer")

    use_distributed_muon = (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    optimizer_cls = (
        MuonWithAuxAdam if use_distributed_muon else SingleDeviceMuonWithAuxAdam
    )
    optimizer = optimizer_cls(param_groups)
    for group in optimizer.param_groups:
        group["_apexrl_lr_scale"] = group["lr"] / lr if lr > 0 else 1.0
    return optimizer
