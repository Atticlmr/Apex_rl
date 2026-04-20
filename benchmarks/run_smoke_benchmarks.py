#!/usr/bin/env python3
"""Run lightweight smoke benchmarks across representative PPO and DQN tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from gymnasium import make


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apexrl.agent.on_policy_runner import OnPolicyRunner
from apexrl.agent.off_policy_runner import OffPolicyRunner
from apexrl.algorithms.dqn import DQNConfig
from apexrl.algorithms.ppo import PPO, PPOConfig
from apexrl.envs.gym_wrapper import GymVecEnv, GymVecEnvContinuous
from apexrl.models import MLPActor, MLPCritic, MLPDiscreteActor, MLPQNetwork


def run_cartpole(num_envs: int, iterations: int) -> dict:
    env = GymVecEnv([lambda: make("CartPole-v1") for _ in range(num_envs)], device="cpu")
    cfg = PPOConfig(
        num_steps=64,
        num_epochs=2,
        minibatch_size=64,
        learning_rate_schedule="constant",
        device="cpu",
    )
    runner = OnPolicyRunner(
        env=env,
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
    )
    try:
        return runner.learn(num_iterations=iterations)
    finally:
        runner.close()


def run_cartpole_dqn(num_envs: int, iterations: int) -> dict:
    env = GymVecEnv([lambda: make("CartPole-v1") for _ in range(num_envs)], device="cpu")
    cfg = DQNConfig(
        batch_size=32,
        buffer_size=2_048,
        learning_starts=32,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=8,
        log_interval=256,
        save_interval=0,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )
    try:
        return runner.learn(total_timesteps=iterations * 128 * num_envs)
    finally:
        runner.close()


def run_acrobot_dqn(num_envs: int, iterations: int) -> dict:
    env = GymVecEnv([lambda: make("Acrobot-v1") for _ in range(num_envs)], device="cpu")
    cfg = DQNConfig(
        batch_size=32,
        buffer_size=2_048,
        learning_starts=32,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=8,
        log_interval=256,
        save_interval=0,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )
    try:
        return runner.learn(total_timesteps=iterations * 128 * num_envs)
    finally:
        runner.close()


def run_cartpole_dueling_dqn(num_envs: int, iterations: int) -> dict:
    env = GymVecEnv([lambda: make("CartPole-v1") for _ in range(num_envs)], device="cpu")
    cfg = DQNConfig(
        dueling=True,
        batch_size=32,
        buffer_size=2_048,
        learning_starts=32,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=8,
        log_interval=256,
        save_interval=0,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )
    try:
        return runner.learn(total_timesteps=iterations * 128 * num_envs)
    finally:
        runner.close()


def run_acrobot_dueling_dqn(num_envs: int, iterations: int) -> dict:
    env = GymVecEnv([lambda: make("Acrobot-v1") for _ in range(num_envs)], device="cpu")
    cfg = DQNConfig(
        dueling=True,
        batch_size=32,
        buffer_size=2_048,
        learning_starts=32,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=8,
        log_interval=256,
        save_interval=0,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        q_network_class=MLPQNetwork,
        device=torch.device("cpu"),
    )
    try:
        return runner.learn(total_timesteps=iterations * 128 * num_envs)
    finally:
        runner.close()


def run_pendulum(num_envs: int, iterations: int) -> dict:
    env = GymVecEnvContinuous(
        [lambda: make("Pendulum-v1") for _ in range(num_envs)],
        device="cpu",
    )
    cfg = PPOConfig(
        num_steps=64,
        num_epochs=2,
        minibatch_size=64,
        learning_rate_schedule="constant",
        device="cpu",
    )
    agent = PPO(
        env=env,
        cfg=cfg,
        actor_class=MLPActor,
        critic_class=MLPCritic,
        device=torch.device("cpu"),
    )
    try:
        return agent.learn(total_timesteps=iterations * cfg.num_steps * num_envs)
    finally:
        env.close()


def run_mountain_car_continuous(num_envs: int, iterations: int) -> dict:
    env = GymVecEnvContinuous(
        [lambda: make("MountainCarContinuous-v0") for _ in range(num_envs)],
        device="cpu",
    )
    cfg = PPOConfig(
        num_steps=64,
        num_epochs=2,
        minibatch_size=64,
        learning_rate_schedule="constant",
        device="cpu",
    )
    agent = PPO(
        env=env,
        cfg=cfg,
        actor_class=MLPActor,
        critic_class=MLPCritic,
        device=torch.device("cpu"),
    )
    try:
        return agent.learn(total_timesteps=iterations * cfg.num_steps * num_envs)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=2)
    args = parser.parse_args()

    tasks = [
        ("CartPole-v1", run_cartpole),
        ("CartPole-v1 (DQN)", run_cartpole_dqn),
        ("CartPole-v1 (Dueling DQN)", run_cartpole_dueling_dqn),
        ("Acrobot-v1 (DQN)", run_acrobot_dqn),
        ("Acrobot-v1 (Dueling DQN)", run_acrobot_dueling_dqn),
        ("Pendulum-v1", run_pendulum),
        ("MountainCarContinuous-v0", run_mountain_car_continuous),
    ]

    for task_name, fn in tasks:
        print(f"\n=== {task_name} ===")
        result = fn(args.num_envs, args.iterations)
        print(
            f"final_iteration={result['final_iteration']} total_timesteps={result['total_timesteps']}"
        )


if __name__ == "__main__":
    main()
