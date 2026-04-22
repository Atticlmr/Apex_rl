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

"""Utilities for TensorDict-based structured observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from gymnasium import spaces

try:
    from tensordict import TensorDict as _TensorDict
except ImportError:  # pragma: no cover

    class _TensorDict(dict):
        """Minimal TensorDict-compatible fallback used in tests and no-dep setups."""

        def __init__(self, source: dict[str, Any], batch_size: Any = None):
            super().__init__(source)
            self.batch_size = batch_size

        def clone(self) -> _TensorDict:
            return _TensorDict(dict(self), batch_size=self.batch_size)

        def to(self, device: torch.device | str, dtype: torch.dtype | None = None):
            converted = {}
            for key, value in self.items():
                if isinstance(value, _TensorDict):
                    converted[key] = value.to(device=device, dtype=dtype)
                elif isinstance(value, torch.Tensor):
                    kwargs = {"device": device}
                    if dtype is not None and value.is_floating_point():
                        kwargs["dtype"] = dtype
                    converted[key] = value.to(**kwargs)
                else:
                    converted[key] = value
            return _TensorDict(converted, batch_size=self.batch_size)


TensorDict = _TensorDict
Observation: TypeAlias = torch.Tensor | TensorDict


@dataclass(frozen=True)
class TensorLeafSpec:
    """Leaf tensor metadata for structured observation storage."""

    shape: tuple[int, ...]
    dtype: torch.dtype


ObservationSpec: TypeAlias = (
    TensorLeafSpec | tuple[int, ...] | dict[str, "ObservationSpec"]
)


def _is_tensordict(value: Any) -> bool:
    return isinstance(value, TensorDict)


def _is_observation_mapping(value: Any) -> bool:
    return _is_tensordict(value) or isinstance(value, dict)


def _wrap_tensordict(
    source: dict[str, Any],
    batch_size: Any = None,
) -> TensorDict:
    return TensorDict(source, batch_size=batch_size)


def _normalized_batch_size(batch_size: Any) -> tuple[int, ...] | None:
    if batch_size is None:
        return None
    if isinstance(batch_size, torch.Size):
        return tuple(int(dim) for dim in batch_size)
    if isinstance(batch_size, int):
        return (int(batch_size),)
    return tuple(int(dim) for dim in batch_size)


def _first_tensor(value: Observation | None) -> torch.Tensor | None:
    if value is None:
        return None
    if _is_observation_mapping(value):
        for sub_value in value.values():
            tensor = _first_tensor(sub_value)
            if tensor is not None:
                return tensor
        return None
    return value


def infer_batch_size(value: Observation | None) -> tuple[int, ...] | None:
    """Infer batch size from the first tensor leaf in an observation tree."""
    tensor = _first_tensor(value)
    if tensor is None:
        return None
    return (int(tensor.shape[0]),)


def observation_batch_size(value: Observation) -> int:
    """Return the leading batch size of an observation tree."""
    tensor = _first_tensor(value)
    if tensor is None:
        raise ValueError("Observation tree does not contain tensor leaves")
    return int(tensor.shape[0])


def _space_dtype_to_torch(space_dtype: Any) -> torch.dtype:
    """Convert a Gymnasium / NumPy dtype to the matching torch dtype."""
    np_dtype = np.dtype(space_dtype)
    return torch.from_numpy(np.zeros((), dtype=np_dtype)).dtype


def space_to_spec(space: spaces.Space) -> ObservationSpec:
    """Convert a Gymnasium space to a recursive observation spec."""
    if isinstance(space, spaces.Dict):
        return {key: space_to_spec(subspace) for key, subspace in space.spaces.items()}
    if isinstance(space, spaces.Discrete):
        return TensorLeafSpec((1,), torch.long)
    if getattr(space, "shape", None) is not None:
        shape = tuple(space.shape)
        return TensorLeafSpec(
            shape if shape else (1,),
            _space_dtype_to_torch(getattr(space, "dtype", np.float32)),
        )
    raise NotImplementedError(f"Unsupported observation space type: {type(space)}")


def spec_numel(spec: ObservationSpec) -> int:
    """Return the flattened number of elements in a recursive observation spec."""
    if isinstance(spec, dict):
        return sum(spec_numel(value) for value in spec.values())
    if isinstance(spec, TensorLeafSpec):
        return int(torch.tensor(spec.shape).prod().item()) if spec.shape else 1
    return int(torch.tensor(spec).prod().item()) if spec else 1


def actor_space_from_observation_space(obs_space: spaces.Space) -> spaces.Space:
    """Return the default actor observation space from a possibly grouped space."""
    if isinstance(obs_space, spaces.Dict) and "obs" in obs_space.spaces:
        return obs_space.spaces["obs"]
    return obs_space


def critic_space_from_observation_space(obs_space: spaces.Space) -> spaces.Space | None:
    """Return the default critic-only observation space if one is declared."""
    if not isinstance(obs_space, spaces.Dict):
        return None
    if "privileged_obs" in obs_space.spaces:
        return obs_space.spaces["privileged_obs"]
    if "critic_obs" in obs_space.spaces:
        return obs_space.spaces["critic_obs"]
    return None


def observation_to_tensor(
    observation: Any,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> Observation:
    """Recursively convert observations to tensors or TensorDicts."""
    if isinstance(observation, tuple):
        observation = observation[0]
    if _is_tensordict(observation):
        return _wrap_tensordict(
            {
                key: observation_to_tensor(value, device=device, dtype=dtype)
                for key, value in observation.items()
            },
            batch_size=getattr(observation, "batch_size", None),
        )
    if isinstance(observation, dict):
        converted = {
            key: observation_to_tensor(value, device=device, dtype=dtype)
            for key, value in observation.items()
        }
        return _wrap_tensordict(converted, batch_size=[])
    if not isinstance(observation, torch.Tensor):
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.as_tensor(observation, **kwargs)
    kwargs = {"device": device}
    if dtype is not None and observation.is_floating_point():
        kwargs["dtype"] = dtype
    return observation.to(**kwargs)


def stack_observations(observations: list[Observation]) -> Observation:
    """Stack a list of observations along a new batch dimension."""
    first = observations[0]
    if _is_observation_mapping(first):
        stacked = {
            key: stack_observations([obs[key] for obs in observations]) for key in first
        }
        batch_size = _normalized_batch_size(getattr(first, "batch_size", None))
        if batch_size is None:
            batch_size = ()
        batch_size = (len(observations), *batch_size)
        return _wrap_tensordict(stacked, batch_size=batch_size)
    return torch.stack(observations)


def zeros_like_observation(observation: Observation) -> Observation:
    """Create a zero-filled observation tree with the same structure."""
    if _is_observation_mapping(observation):
        return _wrap_tensordict(
            {
                key: zeros_like_observation(value)
                for key, value in observation.items()
            },
            batch_size=getattr(observation, "batch_size", None),
        )
    return torch.zeros_like(observation)


def clone_observation(observation: Observation) -> Observation:
    """Recursively clone an observation tree."""
    if _is_observation_mapping(observation):
        return _wrap_tensordict(
            {key: clone_observation(value) for key, value in observation.items()},
            batch_size=getattr(observation, "batch_size", None),
        )
    return observation.clone()


def observation_index(observation: Observation, index: Any) -> Observation:
    """Apply an index operation to each tensor in an observation tree."""
    if _is_observation_mapping(observation):
        return _wrap_tensordict(
            {
                key: observation_index(value, index)
                for key, value in observation.items()
            },
            batch_size=None,
        )
    return observation[index]


def observation_set_index(
    destination: Observation,
    index: Any,
    value: Observation,
) -> None:
    """Assign into an observation tree at the provided index."""
    if _is_observation_mapping(destination):
        if not _is_observation_mapping(value):
            raise TypeError("Observation tree structures must match")
        for key in destination:
            observation_set_index(destination[key], index, value[key])
        return
    destination[index] = value


def flatten_time_env_observation(observation: Observation) -> Observation:
    """Flatten leading ``(time, env)`` dimensions of an observation tree."""
    if _is_observation_mapping(observation):
        flattened = {
            key: flatten_time_env_observation(value)
            for key, value in observation.items()
        }
        first_leaf = next(iter(flattened.values()), None)
        batch_size = infer_batch_size(first_leaf)
        return _wrap_tensordict(flattened, batch_size=batch_size)
    leading = observation.shape[:2]
    total = int(leading[0] * leading[1])
    return observation.reshape(total, *observation.shape[2:])


def flatten_observation(observation: Observation) -> torch.Tensor:
    """Flatten recursive observation trees into feature tensors."""
    if _is_observation_mapping(observation):
        flat_parts = [flatten_observation(observation[key]) for key in observation]
        if not flat_parts:
            raise ValueError("Observation TensorDict is empty")
        if len(flat_parts) == 1:
            return flat_parts[0]
        return torch.cat(flat_parts, dim=-1)

    observation = observation.to(dtype=torch.float32)
    if observation.dim() == 0:
        return observation.reshape(1, 1)
    if observation.dim() == 1:
        return observation.unsqueeze(-1)
    return observation.reshape(observation.shape[0], -1)


def allocate_observation_storage(
    prefix_shape: tuple[int, ...],
    spec: ObservationSpec,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Observation:
    """Allocate zero-filled storage matching a recursive observation spec."""
    if isinstance(spec, dict):
        return _wrap_tensordict(
            {
                key: allocate_observation_storage(prefix_shape, value, device, dtype)
                for key, value in spec.items()
            },
            batch_size=prefix_shape,
        )
    if isinstance(spec, TensorLeafSpec):
        return torch.zeros(
            (*prefix_shape, *spec.shape),
            device=device,
            dtype=spec.dtype,
        )
    return torch.zeros((*prefix_shape, *spec), device=device, dtype=dtype)


def observation_to_device(
    observation: Observation | None,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> Observation | None:
    """Move an observation tree to a target device."""
    if observation is None:
        return None
    if _is_observation_mapping(observation):
        return _wrap_tensordict(
            {
                key: observation_to_device(value, device, dtype)
                for key, value in observation.items()
            },
            batch_size=getattr(observation, "batch_size", None),
        )
    kwargs = {"device": device}
    if dtype is not None and observation.is_floating_point():
        kwargs["dtype"] = dtype
    return observation.to(**kwargs)


def split_actor_critic_observations(
    observation: Observation,
) -> tuple[Observation, Observation | None]:
    """Split a full observation tree into actor and critic groups."""
    if _is_observation_mapping(observation) and "obs" in observation.keys():
        critic_observation = observation.get("privileged_obs")
        if critic_observation is None:
            critic_observation = observation.get("critic_obs")
        return observation["obs"], critic_observation
    return observation, None
