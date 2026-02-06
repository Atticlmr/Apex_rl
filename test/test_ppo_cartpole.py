"""PPO test script using CartPole-v1 environment.

This script demonstrates how to use PPO with a discrete action space
environment (CartPole-v1) using the ApexRL framework.

Usage:
    python test/test_ppo_cartpole.py --train
    python test/test_ppo_cartpole.py --eval --checkpoint checkpoint.pt
    python test/test_ppo_cartpole.py --render --checkpoint checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import gymnasium as gym
import torch

from apexrl.algorithms.ppo import PPO, PPOConfig
from apexrl.envs.gym_wrapper import GymVecEnv
from apexrl.models.mlp import MLPCritic, MLPDiscreteActor


def make_cartpole_env():
    """Create a CartPole-v1 environment."""
    return gym.make("CartPole-v1")


def get_cartpole_spaces():
    """Get observation and action spaces for CartPole."""
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    env.close()
    return obs_space, action_space


def train(
    num_envs: int = 8,
    num_steps: int = 128,
    num_epochs: int = 4,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    log_dir: Optional[str] = None,
    device: str = "cpu",
):
    """Train PPO on CartPole-v1.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps per environment per rollout.
        num_epochs: Number of epochs to update policy for each rollout.
        total_timesteps: Total number of timesteps to train for.
        learning_rate: Learning rate for optimizer.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_range: Clipping parameter for surrogate loss.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy loss coefficient.
        log_dir: Directory for TensorBoard logs.
        device: Device to use ("cpu" or "cuda").
    """
    print("=" * 60)
    print("PPO Training on CartPole-v1")
    print("=" * 60)
    print(f"Num envs: {num_envs}")
    print(f"Num steps: {num_steps}")
    print(f"Num epochs: {num_epochs}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("=" * 60)

    # Create vectorized environment
    env_fns = [make_cartpole_env for _ in range(num_envs)]
    env = GymVecEnv(env_fns, device=device)

    # Get spaces
    obs_space, action_space = get_cartpole_spaces()

    # Configure PPO
    cfg = PPOConfig(
        num_steps=num_steps,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        learning_rate_schedule="constant",
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        normalize_advantages=True,
        log_interval=10,
        save_interval=50,
        device=device,
    )

    # Create agent
    agent = PPO(
        env=env,
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
        obs_space=obs_space,
        action_space=action_space,
        actor_cfg={"hidden_dims": [64, 64], "activation": "tanh"},
        critic_cfg={"hidden_dims": [64, 64], "activation": "tanh"},
        log_dir=log_dir,
        device=torch.device(device),
    )

    # Train
    print("\nStarting training...\n")
    agent.learn(total_timesteps=total_timesteps)

    # Save final model
    final_path = "cartpole_ppo_final.pt"
    agent.save(final_path)
    print(f"\nModel saved to {final_path}")

    # Evaluate
    print("\nEvaluating trained agent...")
    eval_stats = agent.eval(num_episodes=100)
    print(f"Mean reward: {eval_stats['eval/mean_reward']:.2f}")
    print(f"Std reward: {eval_stats['eval/std_reward']:.2f}")
    print(f"Min reward: {eval_stats['eval/min_reward']:.2f}")
    print(f"Max reward: {eval_stats['eval/max_reward']:.2f}")

    env.close()


def eval_agent(
    checkpoint_path: str,
    num_episodes: int = 100,
    num_envs: int = 8,
    device: str = "cpu",
):
    """Evaluate a trained PPO agent.

    Args:
        checkpoint_path: Path to model checkpoint.
        num_episodes: Number of episodes to evaluate.
        num_envs: Number of parallel environments.
        device: Device to use.
    """
    print("=" * 60)
    print("PPO Evaluation on CartPole-v1")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num episodes: {num_episodes}")
    print("=" * 60)

    # Create vectorized environment
    env_fns = [make_cartpole_env for _ in range(num_envs)]
    env = GymVecEnv(env_fns, device=device)

    # Get spaces
    obs_space, action_space = get_cartpole_spaces()

    # Configure PPO
    cfg = PPOConfig(device=device)

    # Create agent
    agent = PPO(
        env=env,
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
        obs_space=obs_space,
        action_space=action_space,
        actor_cfg={"hidden_dims": [64, 64], "activation": "tanh"},
        critic_cfg={"hidden_dims": [64, 64], "activation": "tanh"},
        device=torch.device(device),
    )

    # Load checkpoint
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Evaluate
    print("\nEvaluating...")
    eval_stats = agent.eval(num_episodes=num_episodes)
    print("\nResults:")
    print(f"Mean reward: {eval_stats['eval/mean_reward']:.2f}")
    print(f"Std reward: {eval_stats['eval/std_reward']:.2f}")
    print(f"Min reward: {eval_stats['eval/min_reward']:.2f}")
    print(f"Max reward: {eval_stats['eval/max_reward']:.2f}")

    env.close()


def render_agent(
    checkpoint_path: str,
    num_episodes: int = 5,
    delay: float = 0.05,
):
    """Render a trained PPO agent playing CartPole.

    Args:
        checkpoint_path: Path to model checkpoint.
        num_episodes: Number of episodes to render.
        delay: Delay between frames in seconds.
    """
    import time

    print("=" * 60)
    print("PPO Rendering on CartPole-v1")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num episodes: {num_episodes}")
    print("=" * 60)

    # Create render environment
    env = gym.make("CartPole-v1", render_mode="human")

    # Get spaces
    obs_space = env.observation_space
    action_space = env.action_space

    # Create a dummy VecEnv just to satisfy the interface
    # We'll use the agent's networks directly
    class DummyEnv:
        def __init__(self):
            self.num_envs = 1
            self.device = torch.device("cpu")

    # Configure PPO
    cfg = PPOConfig(device="cpu")

    # Create agent with dummy env
    agent = PPO(
        env=DummyEnv(),
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
        obs_space=obs_space,
        action_space=action_space,
        actor_cfg={"hidden_dims": [64, 64], "activation": "tanh"},
        critic_cfg={"hidden_dims": [64, 64], "activation": "tanh"},
        device=torch.device("cpu"),
    )

    # Load checkpoint
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Render episodes
    print("\nRendering episodes...")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Convert obs to tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Get action from policy
            with torch.no_grad():
                action, _ = agent.actor.act(obs_tensor, deterministic=True)

            # Step environment
            action_int = int(action.item())
            obs, reward, terminated, truncated, _ = env.step(action_int)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

            time.sleep(delay)

        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="PPO test script for CartPole-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new PPO agent",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a trained PPO agent",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a trained PPO agent",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cartpole_ppo_final.pt",
        help="Path to model checkpoint for eval/render",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="Number of steps per environment per rollout",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="Number of epochs to update policy for each rollout",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train for",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda parameter",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="Clipping parameter for surrogate loss",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function loss coefficient",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy loss coefficient",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between frames for rendering (seconds)",
    )

    args = parser.parse_args()

    # Default to train if no mode specified
    if not (args.train or args.eval or args.render):
        args.train = True

    if args.train:
        train(
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            num_epochs=args.num_epochs,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            log_dir=args.log_dir,
            device=args.device,
        )

    elif args.eval:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
        eval_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.num_episodes,
            num_envs=args.num_envs,
            device=args.device,
        )

    elif args.render:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
        render_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.num_episodes,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
