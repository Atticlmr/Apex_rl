<p align="center">
  <img src="assets/logo-horizontal.svg" alt="ApexRL logo" width="520">
</p>

# Apex_rl

A reinforcement learning library focused on pragmatic, extensible training loops.

Documentation: https://apex-rl-doc.readthedocs.io/

## Installation

Clone and install from source:

```bash
git clone https://github.com/Atticlmr/Apex_rl.git
cd Apex_rl
pip install -e .
```

or with `uv`:

```bash
git clone https://github.com/Atticlmr/Apex_rl.git
cd Apex_rl
uv pip install -e .
```

Optional logging extras:

```bash
pip install -e ".[wandb]"
pip install -e ".[swanlab]"
```

or with `uv`:

```bash
uv pip install -e ".[wandb]"
uv pip install -e ".[swanlab]"
```

Core runtime dependencies:

- Python >= 3.10
- PyTorch >= 2.0
- Gymnasium >= 0.29
- TensorDict >= 0.6, < 0.12.2

## Status

| Algorithm | Status | Notes |
| --- | --- | --- |
| PPO | ✅ Available | `OnPolicyRunner`, discrete + continuous actions, asymmetric actor-critic |
| DQN | ✅ Available | `OffPolicyRunner`, Double DQN, Dueling DQN |
| SAC | ✅ Available | `OffPolicyRunner`, squashed Gaussian actor, twin critics |

## What Changed In Current Version

The current repository version supports structured observations end to end:

- `TensorDict` / nested dict observations
- multimodal actor observations such as image + vector
- privileged critic observations for asymmetric actor-critic
- the same observation structure through env wrappers, buffers, algorithms, and default MLP models

Recommended observation format:

```python
{
    "obs": {
        "image": image,
        "vector": vector,
    },
    "privileged_obs": {
        "state": state,
        "context": context,
    },
}
```

In this format:

- actor receives `obs`
- PPO asymmetric critic receives `privileged_obs`
- SAC stores actor and critic observation branches separately in replay

Current behavior:

- Env wrappers and buffers preserve the original dtype of each observation leaf, including raw `uint8` image branches and non-floating leaves in structured observations.
- Default MLP-based models still flatten structured observations into `float32` feature tensors internally, so existing vector-style training flows continue to work.
- Custom multimodal encoders receive the original per-leaf dtypes and can apply dtype-specific preprocessing explicitly inside the model.

## Quick Start

### PPO on a discrete Gymnasium task

```python
import gymnasium as gym
import torch

from apexrl.agent.on_policy_runner import OnPolicyRunner
from apexrl.algorithms.ppo import PPOConfig
from apexrl.envs.gym_wrapper import GymVecEnv
from apexrl.models import MLPDiscreteActor, MLPCritic


def make_env():
    return gym.make("CartPole-v1")


env = GymVecEnv([make_env for _ in range(8)], device="cpu")

cfg = PPOConfig(
    num_steps=128,
    num_epochs=4,
    minibatch_size=256,
    learning_rate=3e-4,
    learning_rate_schedule="constant",
    device="cpu",
    log_interval=1,
    save_interval=0,
)

runner = OnPolicyRunner(
    env=env,
    cfg=cfg,
    actor_class=MLPDiscreteActor,
    critic_class=MLPCritic,
    log_dir="./logs/cartpole_ppo",
    device=torch.device("cpu"),
)

runner.learn(total_timesteps=100_000)
runner.close()
```

## Logging Backends

The runner and algorithm configs support three logging backends:

- `tensorboard`
- `wandb`
- `swanlab`

`TensorBoard` works with the default install. `wandb` and `swanlab` require the
matching optional extras shown above.

Choose one backend per run:

```python
cfg = PPOConfig(
    logger_backend="wandb",
    logger_kwargs={
        "project": "apexrl",
        "entity": "your_team",
        "tags": ["ppo", "cartpole"],
    },
)
```

SwanLab example:

```python
cfg = PPOConfig(
    logger_backend="swanlab",
    logger_kwargs={
        "project": "apexrl",
        "experiment_description": "PPO CartPole run",
    },
)
```

The same `logger_backend` and `logger_kwargs` fields are available in
`PPOConfig`, `DQNConfig`, and `SACConfig`.

### PPO on a continuous Gymnasium task

```python
import gymnasium as gym
import torch

from apexrl.agent.on_policy_runner import OnPolicyRunner
from apexrl.algorithms.ppo import PPOConfig
from apexrl.envs.gym_wrapper import GymVecEnvContinuous
from apexrl.models import MLPActor, MLPCritic


def make_env():
    return gym.make("Pendulum-v1")


env = GymVecEnvContinuous([make_env for _ in range(8)], device="cpu")

runner = OnPolicyRunner(
    env=env,
    cfg=PPOConfig(device="cpu"),
    actor_class=MLPActor,
    critic_class=MLPCritic,
    log_dir="./logs/pendulum_ppo",
    device=torch.device("cpu"),
)

runner.learn(total_timesteps=100_000)
runner.close()
```

### DQN

```python
import gymnasium as gym
import torch

from apexrl.agent.off_policy_runner import OffPolicyRunner
from apexrl.algorithms.dqn import DQNConfig
from apexrl.envs.gym_wrapper import GymVecEnv
from apexrl.models import MLPQNetwork


env = GymVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)], device="cpu")

runner = OffPolicyRunner(
    env=env,
    cfg=DQNConfig(double_dqn=True, dueling=True),
    q_network_class=MLPQNetwork,
    device=torch.device("cpu"),
)

runner.learn(total_timesteps=50_000)
runner.close()
```

### SAC

```python
import gymnasium as gym
import torch

from apexrl.agent.off_policy_runner import OffPolicyRunner
from apexrl.algorithms.sac import SACConfig
from apexrl.envs.gym_wrapper import GymVecEnvContinuous


env = GymVecEnvContinuous(
    [lambda: gym.make("Pendulum-v1") for _ in range(2)],
    device="cpu",
)

runner = OffPolicyRunner(
    env=env,
    cfg=SACConfig(device="cpu"),
    algorithm="sac",
    device=torch.device("cpu"),
)

runner.learn(total_timesteps=100_000)
runner.close()
```

## Custom Multimodal Actor

If you want to define your own actor for image + vector input, subclass
`DiscreteActor` or `ContinuousActor` and process each branch explicitly.

```python
import torch
import torch.nn as nn

from apexrl.models.base import DiscreteActor


class MultiModalDiscreteActor(DiscreteActor):
    def __init__(self, obs_space, action_space, cfg=None):
        super().__init__(obs_space, action_space, cfg)

        image_shape = obs_space["image"].shape
        vector_dim = obs_space["vector"].shape[0]
        hidden_dim = (cfg or {}).get("hidden_dim", 256)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_shape[0], 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            image_dim = self.image_encoder(dummy).shape[-1]

        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(image_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(self, obs):
        image_feat = self.image_encoder(obs["image"])
        vector_feat = self.vector_encoder(obs["vector"])
        return self.head(torch.cat([image_feat, vector_feat], dim=-1))

    def get_action_dist(self, obs):
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)
```

Then plug it into the runner:

```python
runner = OnPolicyRunner(
    env=env,
    cfg=PPOConfig(use_asymmetric=True, device="cpu"),
    actor_class=MultiModalDiscreteActor,
    critic_class=MLPCritic,
    actor_cfg={"hidden_dim": 256},
)
```

## Smoke Benchmarks

Run the lightweight benchmark suite with:

```bash
/Users/air/workspace/abc/bin/python benchmarks/run_smoke_benchmarks.py --iterations 1 --num-envs 1
```

Current smoke tasks:

- `CartPole-v1` with PPO
- `CartPole-v1` with DQN
- `CartPole-v1` with Dueling DQN
- `Acrobot-v1` with DQN
- `Acrobot-v1` with Dueling DQN
- `Pendulum-v1` with PPO
- `MountainCarContinuous-v0` with PPO
- `Pendulum-v1` with SAC
- `MountainCarContinuous-v0` with SAC

## Roadmap

Planned algorithm work for upcoming versions:

- AMP
- Policy distillation

## License

Apache-2.0

## Citation

If you use this library in your research, please cite:

```bibtex
@software{li2025apexrl,
  author = {Li, Mingrui},
  title = {Apex\_rl: A Reinforcement Learning Library},
  url = {https://github.com/Atticlmr/Apex_rl},
  year = {2025}
}
```
