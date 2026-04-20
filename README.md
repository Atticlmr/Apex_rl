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

or use uv

```bash
git clone https://github.com/Atticlmr/Apex_rl.git
cd Apex_rl
uv pip install -e .
```

## Status

| Algorithm | Status      | Notes        |
| --------- | ----------- | ------------ |
| PPO       | ✅ Available | On-policy runner, continuous + discrete actions |
| DQN       | ✅ Available | Replay buffer, OffPolicyRunner, Double DQN, Dueling DQN |
| SAC       | 🚧 Planned  | Next Next release |

## Quick Start

### PPO

```python
import gymnasium as gym

from apexrl.agent.on_policy_runner import OnPolicyRunner
from apexrl.envs.gym_wrapper import GymVecEnv
from apexrl.models import MLPCritic, MLPDiscreteActor

env = GymVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)], device="cpu")

runner = OnPolicyRunner(
    env=env,
    algorithm="ppo",
    actor_class=MLPDiscreteActor,
    critic_class=MLPCritic,
)
runner.learn(total_timesteps=20_000)
```

### DQN / Dueling DQN

```python
import gymnasium as gym
import torch

from apexrl.agent.off_policy_runner import OffPolicyRunner
from apexrl.algorithms.dqn import DQNConfig
from apexrl.envs.gym_wrapper import GymVecEnv
from apexrl.models import MLPQNetwork

env = GymVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)], device="cpu")

cfg = DQNConfig(
    double_dqn=True,
    dueling=True,
    learning_starts=1_000,
)

runner = OffPolicyRunner(
    env=env,
    cfg=cfg,
    q_network_class=MLPQNetwork,
    device=torch.device("cpu"),
)
runner.learn(total_timesteps=50_000)
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

# License

Apache-2.0

# Citation

If you use this library in your research, please cite:
```
@software{li2025apexrl,
  author = {Li, Mingrui},
  title = {Apex\_rl: A Reinforcement Learning Library},
  url = {https://github.com/Atticlmr/Apex_rl},
  year = {2025}
}
```
