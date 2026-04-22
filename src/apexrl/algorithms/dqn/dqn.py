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

"""DQN algorithm implementation."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from gymnasium import spaces

from apexrl.algorithms.dqn.config import DQNConfig
from apexrl.buffer.replay_buffer import ReplayBuffer
from apexrl.optimizers import get_optimizer
from apexrl.utils import (
    actor_space_from_observation_space,
    observation_to_tensor,
    space_to_spec,
    split_actor_critic_observations,
)


class DQN:
    """Deep Q-Network with target network and optional Double DQN targets."""

    def __init__(
        self,
        env: Any,
        cfg: DQNConfig | None = None,
        q_network_class: type | None = None,
        obs_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        q_network_cfg: dict[str, Any] | None = None,
        log_dir: str | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the DQN agent."""
        self.env = env
        self.cfg = cfg or DQNConfig()
        self.log_dir = log_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = None

        full_obs_space = obs_space or getattr(env, "observation_space_gym", None)
        self.action_space = action_space or getattr(env, "action_space_gym", None)
        if full_obs_space is None or self.action_space is None:
            raise ValueError("DQN requires obs_space and action_space")
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError(
                "DQN only supports Discrete action spaces, "
                f"got {type(self.action_space)}"
            )
        if q_network_class is None:
            raise ValueError("q_network_class is required for DQN")

        self.full_obs_space = full_obs_space
        self.obs_space = actor_space_from_observation_space(full_obs_space)
        self.num_envs = env.num_envs
        self.num_actions = self.action_space.n

        network_cfg = self._build_q_network_cfg(q_network_cfg)
        self.q_network = q_network_class(
            self.obs_space,
            self.action_space,
            network_cfg,
        ).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_q_network.eval()

        optimizer_cls = get_optimizer(self.cfg.optimizer)
        self.optimizer = optimizer_cls(
            self.q_network.parameters(),
            lr=self.cfg.learning_rate,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            obs_shape=space_to_spec(self.obs_space),
            action_shape=(),
            device=self.device,
            obs_dtype=torch.float32,
            action_dtype=torch.long,
        )

        self.iteration = 0
        self.total_timesteps = 0
        self.num_updates = 0

    def _build_q_network_cfg(
        self, q_network_cfg: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Build q-network configuration from DQN defaults and overrides."""
        cfg = {
            "hidden_dims": list(self.cfg.network_hidden_dims),
            "activation": self.cfg.activation,
            "layer_norm": self.cfg.layer_norm,
            "dueling": self.cfg.dueling,
        }
        if q_network_cfg:
            cfg.update(q_network_cfg)
        return cfg

    def _to_tensor_observation(self, obs: Any) -> Any:
        """Convert observations to tensors or TensorDicts on the agent device."""
        return observation_to_tensor(obs, device=self.device)

    def _actor_observation(self, obs: Any) -> Any:
        """Extract the actor-visible observation branch."""
        actor_obs, _ = split_actor_critic_observations(obs)
        return actor_obs

    def get_epsilon(self, total_timesteps: int) -> float:
        """Return the epsilon value for the given training step."""
        progress = min(
            float(total_timesteps) / float(self.cfg.epsilon_decay_steps),
            1.0,
        )
        return self.cfg.epsilon_start + progress * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def act(
        self,
        obs: Any,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """Select epsilon-greedy actions."""
        obs = self._actor_observation(self._to_tensor_observation(obs))
        epsilon = 0.0 if deterministic else (epsilon if epsilon is not None else 0.0)
        with torch.no_grad():
            return self.q_network.act(obs, epsilon=epsilon)

    def sample_random_actions(self) -> torch.Tensor:
        """Sample uniform random discrete actions."""
        return torch.randint(
            low=0,
            high=self.num_actions,
            size=(self.num_envs,),
            device=self.device,
        )

    def store_transition(
        self,
        observations: Any,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Any,
        dones: torch.Tensor,
    ) -> None:
        """Store a batch of transitions in replay."""
        self.replay_buffer.add(
            observations=self._actor_observation(
                self._to_tensor_observation(observations)
            ),
            actions=actions.to(self.device, dtype=torch.long),
            rewards=rewards.to(self.device, dtype=torch.float32),
            next_observations=self._actor_observation(
                self._to_tensor_observation(next_observations)
            ),
            dones=dones.to(self.device, dtype=torch.float32),
        )

    def _maybe_update_target_network(self) -> None:
        """Apply hard or soft target updates at the configured interval."""
        if self.num_updates % self.cfg.target_update_interval != 0:
            return
        tau = self.cfg.tau
        if tau >= 1.0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            return
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.lerp_(param.data, tau)

    def update(self) -> dict[str, float]:
        """Run one gradient update from a replay batch."""
        if len(self.replay_buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {}

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        observations = batch["observations"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        q_values = self.q_network(observations)
        chosen_q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                next_actions = self.q_network(next_observations).argmax(
                    dim=-1, keepdim=True
                )
                next_q = (
                    self.target_q_network(next_observations)
                    .gather(1, next_actions)
                    .squeeze(-1)
                )
            else:
                next_q = self.target_q_network(next_observations).max(dim=-1).values
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(chosen_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        self.num_updates += 1
        self._maybe_update_target_network()

        return {
            "train/q_loss": loss.item(),
            "train/mean_q": chosen_q.mean().item(),
            "train/td_target_mean": td_target.mean().item(),
            "train/grad_norm": float(grad_norm),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def save(self, path: str) -> None:
        """Save the DQN checkpoint."""
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "num_updates": self.num_updates,
            "config": self.cfg,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load the DQN checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("replay_buffer_state_dict"):
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])
        self.iteration = checkpoint.get("iteration", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.num_updates = checkpoint.get("num_updates", 0)

    def eval(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate greedy DQN policy on the current environment."""
        obs = self._to_tensor_observation(self.env.reset())
        episode_rewards = []
        current_rewards = torch.zeros(self.num_envs, device=self.device)
        episodes_completed = 0

        while episodes_completed < num_episodes:
            actions = self.act(obs, deterministic=True)
            next_obs, rewards, dones, _ = self.env.step(actions)
            obs = self._to_tensor_observation(next_obs)
            rewards = rewards.to(self.device, dtype=torch.float32)
            dones = dones.to(self.device).bool()

            current_rewards += rewards
            if dones.any():
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    if episodes_completed < num_episodes:
                        episode_rewards.append(current_rewards[idx].item())
                        episodes_completed += 1
                current_rewards = current_rewards * (~dones).float()

        rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float32)
        return {
            "eval/mean_reward": rewards_tensor.mean().item(),
            "eval/std_reward": rewards_tensor.std(unbiased=False).item(),
            "eval/min_reward": rewards_tensor.min().item(),
            "eval/max_reward": rewards_tensor.max().item(),
        }

    def learn(self, total_timesteps: int | None = None) -> dict[str, Any]:
        """Train through the canonical OffPolicyRunner entrypoint."""
        from apexrl.agent.off_policy_runner import OffPolicyRunner

        runner = OffPolicyRunner(
            agent=self,
            env=self.env,
            cfg=self.cfg,
            log_dir=None,
            save_dir=self.log_dir,
            device=self.device,
        )
        return runner.learn(total_timesteps=total_timesteps)
