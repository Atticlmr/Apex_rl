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

"""Gymnasium environment wrapper for VecEnv interface.

This module provides wrappers to use Gymnasium environments with the
vectorized environment interface used by ApexRL algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from apexrl.envs.vecenv import VecEnv


class GymVecEnv(VecEnv):
    """Vectorized wrapper for Gymnasium environments.

    Wraps multiple Gymnasium environments to provide a VecEnv interface.
    Each environment runs independently (CPU-based).

    Example:
        >>> from apexrl.envs.gym_wrapper import GymVecEnv
        >>>
        >>> def make_env():
        ...     return gym.make("CartPole-v1")
        ...
        >>> env = GymVecEnv([make_env for _ in range(8)])
        >>> obs = env.reset()
        >>> actions = torch.randint(0, 2, (8,))
        >>> next_obs, rewards, dones, extras = env.step(actions)
    """

    def __init__(
        self,
        env_fns: List[callable],
        device: Union[torch.device, str] = "cpu",
    ):
        """Initialize GymVecEnv.

        Args:
            env_fns: List of callables that create gymnasium environments.
            device: Device for tensors.
        """
        # Create environments first
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        self._env = self.envs[0]

        # Get spaces from first environment
        self._obs_space = self._env.observation_space
        self._act_space = self._env.action_space

        # Infer dimensions BEFORE calling super().__init__
        if hasattr(self._obs_space, "shape"):
            self.num_obs = int(torch.prod(torch.tensor(self._obs_space.shape)))
        else:
            self.num_obs = 1

        if isinstance(self._act_space, gym.spaces.Discrete):
            self.num_actions = 1
        elif hasattr(self._act_space, "shape"):
            self.num_actions = (
                self._act_space.shape[0] if len(self._act_space.shape) > 0 else 1
            )
        else:
            self.num_actions = 1

        # Get max episode length if available
        self.max_episode_length = getattr(self._env.spec, "max_episode_steps", 1000)
        if self.max_episode_length is None:
            self.max_episode_length = 1000

        # Now call super().__init__ after setting num_obs
        super().__init__(device=device)
        self.device = torch.device(device)

        # Initialize buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Episode statistics
        self._ep_rewards = [0.0] * self.num_envs
        self._completed_episodes = []

        # Initial reset of all environments
        self.reset()

    def get_observations(self) -> torch.Tensor:
        """Return current observations."""
        return self.obs_buf

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert numpy observation to tensor."""
        if isinstance(obs, tuple):
            obs = obs[0]
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def reset(self) -> torch.Tensor:
        """Reset all environments."""
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=None)
            obs_list.append(self._obs_to_tensor(obs))
            self._ep_rewards[i] = 0.0
            self.episode_length_buf[i] = 0

        self.obs_buf = torch.stack(obs_list)
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self._completed_episodes.clear()

        return self.obs_buf

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specific environments."""
        env_ids = env_ids.cpu().numpy()
        for idx in env_ids:
            obs, _ = self.envs[idx].reset(seed=None)
            self.obs_buf[idx] = self._obs_to_tensor(obs)
            self._ep_rewards[idx] = 0.0
            self.episode_length_buf[idx] = 0

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step all environments.

        Args:
            actions: Actions to apply. Shape: (num_envs,) for discrete
                    or (num_envs, action_dim) for continuous.

        Returns:
            Tuple of (observations, rewards, dones, extras).
        """
        actions = actions.cpu().numpy()

        for i, env in enumerate(self.envs):
            # Handle discrete vs continuous actions
            if isinstance(self._act_space, gym.spaces.Discrete):
                action = int(actions[i])
            else:
                action = actions[i]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.obs_buf[i] = self._obs_to_tensor(obs)
            self.rew_buf[i] = float(reward)
            self.reset_buf[i] = done
            self.episode_length_buf[i] += 1
            self._ep_rewards[i] += float(reward)

            # Track completed episodes
            if done:
                self._completed_episodes.append(self._ep_rewards[i])
                self._ep_rewards[i] = 0.0

            # Auto-reset if done
            if done:
                obs, _ = env.reset(seed=None)
                self.obs_buf[i] = self._obs_to_tensor(obs)
                self.episode_length_buf[i] = 0

        # Check for timeouts
        time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        extras = {
            "time_outs": time_outs,
            "log": {},
        }

        return self.obs_buf, self.rew_buf, self.reset_buf, extras

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()


class GymVecEnvContinuous(VecEnv):
    """Vectorized wrapper for continuous action Gymnasium environments.

    Similar to GymVecEnv but specifically designed for continuous action spaces.
    Automatically handles action scaling to match environment bounds.
    """

    def __init__(
        self,
        env_fns: List[callable],
        device: Union[torch.device, str] = "cpu",
        clip_actions: bool = True,
    ):
        """Initialize GymVecEnvContinuous.

        Args:
            env_fns: List of callables that create gymnasium environments.
            device: Device for tensors.
            clip_actions: Whether to clip actions to action space bounds.
        """
        # Create environments first
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        self._env = self.envs[0]

        # Get spaces from first environment
        self._obs_space = self._env.observation_space
        self._act_space = self._env.action_space

        # Verify continuous action space
        assert isinstance(self._act_space, gym.spaces.Box), (
            f"GymVecEnvContinuous requires Box action space, got {type(self._act_space)}"
        )

        # Infer dimensions BEFORE calling super().__init__
        if hasattr(self._obs_space, "shape"):
            self.num_obs = int(torch.prod(torch.tensor(self._obs_space.shape)))
        else:
            self.num_obs = 1

        self.num_actions = (
            self._act_space.shape[0] if len(self._act_space.shape) > 0 else 1
        )

        # Get max episode length if available
        self.max_episode_length = getattr(self._env.spec, "max_episode_steps", 1000)
        if self.max_episode_length is None:
            self.max_episode_length = 1000

        # Now call super().__init__ after setting num_obs
        super().__init__(device=device)
        self.device = torch.device(device)
        self.clip_actions = clip_actions

        # Store action bounds
        self.action_low = torch.tensor(
            self._act_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.tensor(
            self._act_space.high, dtype=torch.float32, device=self.device
        )

        # Initialize buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Episode statistics
        self._ep_rewards = [0.0] * self.num_envs
        self._completed_episodes = []

        # Initial reset of all environments
        self.reset()

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions from [-1, 1] to environment bounds."""
        # Assuming input actions are in [-1, 1]
        scaled = (
            0.5 * (actions + 1.0) * (self.action_high - self.action_low)
            + self.action_low
        )
        if self.clip_actions:
            scaled = torch.clamp(scaled, self.action_low, self.action_high)
        return scaled

    def get_observations(self) -> torch.Tensor:
        """Return current observations."""
        return self.obs_buf

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert numpy observation to tensor."""
        if isinstance(obs, tuple):
            obs = obs[0]
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def reset(self) -> torch.Tensor:
        """Reset all environments."""
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=None)
            obs_list.append(self._obs_to_tensor(obs))
            self._ep_rewards[i] = 0.0
            self.episode_length_buf[i] = 0

        self.obs_buf = torch.stack(obs_list)
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self._completed_episodes.clear()

        return self.obs_buf

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specific environments."""
        env_ids = env_ids.cpu().numpy()
        for idx in env_ids:
            obs, _ = self.envs[idx].reset(seed=None)
            self.obs_buf[idx] = self._obs_to_tensor(obs)
            self._ep_rewards[idx] = 0.0
            self.episode_length_buf[idx] = 0

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step all environments.

        Args:
            actions: Actions to apply. Shape: (num_envs, num_actions).
                    Expected to be in [-1, 1] range.

        Returns:
            Tuple of (observations, rewards, dones, extras).
        """
        # Scale actions to environment bounds
        scaled_actions = self._scale_actions(actions)
        actions_np = scaled_actions.cpu().numpy()

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])
            done = terminated or truncated

            self.obs_buf[i] = self._obs_to_tensor(obs)
            self.rew_buf[i] = float(reward)
            self.reset_buf[i] = done
            self.episode_length_buf[i] += 1
            self._ep_rewards[i] += float(reward)

            # Track completed episodes
            if done:
                self._completed_episodes.append(self._ep_rewards[i])
                self._ep_rewards[i] = 0.0

            # Auto-reset if done
            if done:
                obs, _ = env.reset(seed=None)
                self.obs_buf[i] = self._obs_to_tensor(obs)
                self.episode_length_buf[i] = 0

        # Check for timeouts
        time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        extras = {
            "time_outs": time_outs,
            "log": {},
        }

        return self.obs_buf, self.rew_buf, self.reset_buf, extras

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
