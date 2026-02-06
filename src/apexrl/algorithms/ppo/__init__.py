"""PPO (Proximal Policy Optimization) algorithm module.

This module provides a flexible PPO implementation that supports:
- Custom Actor and Critic network architectures
- Multi-dimensional observations (images, vectors, dicts)
- Asymmetric actor-critic (different observations)
- GPU-accelerated vectorized environments

Example:
    >>> from apexrl.algorithms.ppo import PPO, PPOConfig
    >>> from apexrl.envs.vecenv import DummyVecEnv
    >>> from apexrl.models.mlp import MLPActor, MLPCritic
    >>> from gymnasium import spaces
    >>>
    >>> # Setup environment and spaces
    >>> env = DummyVecEnv(num_envs=4096, num_obs=48, num_actions=12)
    >>> obs_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(48,))
    >>> action_space = spaces.Box(low=-1, high=1, shape=(12,))
    >>>
    >>> # Create and train agent
    >>> cfg = PPOConfig(num_steps=24, learning_rate=3e-4)
    >>> agent = PPO(
    ...     env=env,
    ...     cfg=cfg,
    ...     actor_class=MLPActor,
    ...     critic_class=MLPCritic,
    ...     obs_space=obs_space,
    ...     action_space=action_space,
    ... )
    >>> agent.learn(total_timesteps=10_000_000)

Custom Network Example:
    >>> from apexrl.models.base import ContinuousActor, Critic
    >>>
    >>> class CustomActor(ContinuousActor):
    ...     def __init__(self, obs_space, action_space, cfg):
    ...         super().__init__(obs_space, action_space, cfg)
    ...         # Your custom architecture
    ...         self.encoder = nn.Sequential(...)
    ...         self.log_std = nn.Parameter(torch.zeros(self.action_dim))
    ...
    ...     def forward(self, obs):
    ...         return self.encoder(obs)
    ...
    ...     def get_action_dist(self, obs):
    ...         mean = self.forward(obs)
    ...         std = torch.exp(self.log_std)
    ...         return torch.distributions.Normal(mean, std)
"""

from apexrl.algorithms.ppo.config import PPOConfig, PPOStorageConfig
from apexrl.algorithms.ppo.ppo import PPO
from apexrl.algorithms.ppo.rollout_buffer import RolloutBuffer

__all__ = [
    "PPO",
    "PPOConfig",
    "PPOStorageConfig",
    "RolloutBuffer",
]
