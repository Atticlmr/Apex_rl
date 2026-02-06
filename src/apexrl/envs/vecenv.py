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

"""Vectorized environment base class for efficient RL training.

This module provides an abstract base class for vectorized environments,
designed for high-performance RL algorithm implementations.
References:
    - rsl_rl: https://github.com/leggedrobotics/rsl_rl
    - skrl: https://github.com/Toni-SM/skrl
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import torch
from tensordict import TensorDict


class VecEnv(ABC):
    """Abstract base class for vectorized environments.

    The vectorized environment is a collection of environments that are synchronized.
    This means that the same type of action is applied to all environments and the
    same type of observation is returned from all environments.

    This design is optimized for GPU-accelerated simulation (e.g., Isaac Gym, Isaac Sim)
    where all environments run in parallel on the GPU.

    Attributes:
        num_envs (int): Number of parallel environments.
        num_actions (int): Dimensionality of action space.
        num_obs (int): Dimensionality of observation space.
        num_privileged_obs (int, optional): Dimensionality of privileged observations
            (e.g., for asymmetric actor-critic). Defaults to 0.
        max_episode_length (int or torch.Tensor): Maximum episode length.
            Can be a scalar (same for all envs) or a tensor (per-env).
        device (torch.device or str): Device to use for tensors.

        # Internal buffers (should be maintained by subclasses)
        obs_buf (torch.Tensor): Buffer for observations. Shape: (num_envs, num_obs)
        privileged_obs_buf (torch.Tensor, optional): Buffer for privileged observations.
            Shape: (num_envs, num_privileged_obs)
        rew_buf (torch.Tensor): Buffer for rewards. Shape: (num_envs,)
        reset_buf (torch.Tensor): Buffer for reset flags. Shape: (num_envs,)
        episode_length_buf (torch.Tensor): Current episode lengths. Shape: (num_envs,)
        extras (dict): Extra information and metrics.
    """

    num_envs: int
    """Number of parallel environments."""

    num_actions: int
    """Dimensionality of action space."""

    num_obs: int
    """Dimensionality of observation space."""

    num_privileged_obs: int = 0
    """Dimensionality of privileged observations (for asymmetric actor-critic)."""

    max_episode_length: Union[int, torch.Tensor]
    """Maximum episode length. Can be scalar or per-environment tensor."""

    device: Union[torch.device, str]
    """Device to use for tensors (typically 'cuda' for GPU simulators)."""

    # Buffers (to be initialized by subclasses)
    obs_buf: torch.Tensor
    privileged_obs_buf: Union[torch.Tensor, None] = None
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor
    extras: Dict[str, Any]

    def __init__(self, device: Union[torch.device, str] = "cuda"):
        """Initialize the vectorized environment.

        Args:
            device: Device to use for tensors. Defaults to "cuda".
        """
        self.device = device
        self.extras = {}

    @abstractmethod
    def get_observations(
        self,
    ) -> Union[TensorDict, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Return the current observations.

        Returns:
            Either a TensorDict containing observations or a tuple of
            (observations tensor, extras dict).

        Note:
            The observations can be structured using TensorDict to support
            multiple observation groups (e.g., "policy", "critic", "teacher").
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Union[TensorDict, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Reset all environment instances.

        Returns:
            Initial observations after reset. Same format as get_observations().
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Union[TensorDict, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        """Apply actions to the environment and step the simulation.

        Args:
            actions: Actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple containing:
                - observations: Observations from the environment.
                  Can be TensorDict or torch.Tensor.
                - rewards: Rewards. Shape: (num_envs,)
                - dones: Done flags (True when episode terminates). Shape: (num_envs,)
                - extras: Extra information dictionary containing:
                    - "time_outs": Boolean tensor for timeout-based terminations.
                      Shape: (num_envs,). True if episode ended due to time limit.
                    - "log": Dictionary of metrics for logging (keys should start with "/").

        Note:
            dones should be True for any episode termination (success, failure, timeout).
            Use extras["time_outs"] to distinguish timeout-based terminations.
        """
        raise NotImplementedError

    def reset_idx(
        self, env_ids: torch.Tensor
    ) -> Union[TensorDict, Tuple[torch.Tensor, Dict[str, Any]], None]:
        """Reset specific environments by index.

        This is useful for partial resets in GPU-accelerated environments where
        only some environments need to be reset (those with done=True).

        Args:
            env_ids: Indices of environments to reset. Shape: (num_reset_envs,)

        Returns:
            Observations for the reset environments, or None if observations
            are not immediately available (caller should call get_observations()).
        """
        # Default implementation: subclasses should override for efficiency
        return None

    @staticmethod
    def get_privileged_observations() -> Union[torch.Tensor, None]:
        """Return privileged observations (e.g., for critic).

        Returns:
            Privileged observations tensor or None if not supported.
            Shape: (num_envs, num_privileged_obs)
        """
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment (for checkpointing).

        Returns:
            Dictionary containing environment state.
        """
        return {
            "episode_length_buf": self.episode_length_buf.clone(),
            "extras": self.extras.copy(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the environment state (for resuming from checkpoint).

        Args:
            state: State dictionary from get_state().
        """
        if "episode_length_buf" in state:
            self.episode_length_buf = state["episode_length_buf"].to(self.device)
        if "extras" in state:
            self.extras.update(state["extras"])

    def close(self) -> None:
        """Close the environment and release resources."""
        pass

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Get observation space information.

        Returns:
            Dictionary containing observation space shape and dtype.
        """
        return {
            "shape": (self.num_obs,),
            "dtype": torch.float32,
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        """Get action space information.

        Returns:
            Dictionary containing action space shape, dtype, and bounds.
        """
        return {
            "shape": (self.num_actions,),
            "dtype": torch.float32,
            "low": -1.0,
            "high": 1.0,
        }

    def seed(self, seed: int) -> None:
        """Set random seed for the environment.

        Args:
            seed: Random seed value.
        """
        torch.manual_seed(seed)

    def render(self, mode: str = "rgb_array") -> Union[torch.Tensor, None]:
        """Render the environment.

        Args:
            mode: Rendering mode ("rgb_array", "human").

        Returns:
            Rendered image tensor if mode is "rgb_array", None otherwise.
        """
        return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class VecEnvWrapper(VecEnv):
    """Wrapper for vectorized environments.

    Provides a convenient way to modify behavior of a VecEnv without
    changing the underlying implementation.
    """

    def __init__(self, env: VecEnv):
        """Initialize the wrapper.

        Args:
            env: The vectorized environment to wrap.
        """
        self.env = env
        # Copy attributes from wrapped environment
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.num_obs = env.num_obs
        self.num_privileged_obs = getattr(env, "num_privileged_obs", 0)
        self.max_episode_length = env.max_episode_length
        self.device = env.device

    def get_observations(
        self,
    ) -> Union[TensorDict, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Return observations from wrapped environment."""
        return self.env.get_observations()

    def reset(self) -> Union[TensorDict, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Reset wrapped environment."""
        return self.env.reset()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Union[TensorDict, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        """Step wrapped environment."""
        return self.env.step(actions)

    def reset_idx(
        self, env_ids: torch.Tensor
    ) -> Union[TensorDict, Tuple[torch.Tensor, Dict[str, Any]], None]:
        """Reset specific environments in wrapped environment."""
        return self.env.reset_idx(env_ids)

    def close(self) -> None:
        """Close wrapped environment."""
        return self.env.close()

    @property
    def obs_buf(self) -> torch.Tensor:
        """Access observation buffer of wrapped environment."""
        return self.env.obs_buf

    @property
    def rew_buf(self) -> torch.Tensor:
        """Access reward buffer of wrapped environment."""
        return self.env.rew_buf

    @property
    def reset_buf(self) -> torch.Tensor:
        """Access reset buffer of wrapped environment."""
        return self.env.reset_buf

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Access episode length buffer of wrapped environment."""
        return self.env.episode_length_buf

    @property
    def extras(self) -> Dict[str, Any]:
        """Access extras of wrapped environment."""
        return self.env.extras

    @extras.setter
    def extras(self, value: Dict[str, Any]):
        """Set extras of wrapped environment."""
        self.env.extras = value


class DummyVecEnv(VecEnv):
    """Dummy vectorized environment for testing.

    This is a simple implementation that can be used for testing
    algorithms without a real environment.
    """

    def __init__(
        self,
        num_envs: int = 4096,
        num_obs: int = 48,
        num_actions: int = 12,
        device: Union[torch.device, str] = "cuda",
        max_episode_length: int = 1000,
    ):
        """Initialize dummy vectorized environment.

        Args:
            num_envs: Number of parallel environments.
            num_obs: Dimensionality of observations.
            num_actions: Dimensionality of actions.
            device: Device to use.
            max_episode_length: Maximum episode length.
        """
        super().__init__(device=device)
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.max_episode_length = max_episode_length

        # Initialize buffers
        self.obs_buf = torch.zeros(num_envs, num_obs, device=device)
        self.rew_buf = torch.zeros(num_envs, device=device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episode_length_buf = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )

    def get_observations(self) -> TensorDict:
        """Return current observations."""
        return TensorDict({"obs": self.obs_buf}, batch_size=self.num_envs)

    def reset(self) -> TensorDict:
        """Reset all environments."""
        self.obs_buf.zero_()
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()
        return self.get_observations()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the dummy environment.

        Simulates environment dynamics with random observations.
        """
        # Simple dummy dynamics
        self.obs_buf = torch.randn(self.num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.randn(self.num_envs, device=self.device)
        self.episode_length_buf += 1

        # Check for terminations
        time_outs = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = time_outs.clone()

        # Reset timed-out environments
        if time_outs.any():
            self.reset_idx(torch.where(time_outs)[0])

        extras = {
            "time_outs": time_outs,
            "log": {
                "/reward/mean": self.rew_buf.mean().item(),
                "/episode_length/mean": self.episode_length_buf.float().mean().item(),
            },
        }

        return self.get_observations(), self.rew_buf, self.reset_buf, extras

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specific environments."""
        self.obs_buf[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
