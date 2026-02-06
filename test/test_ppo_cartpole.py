"""PPO test script using CartPole-v1 environment.

This script demonstrates how to use PPO with a discrete action space
environment (CartPole-v1) using the ApexRL framework.

Usage:
    python test/test_ppo_cartpole.py --train
    python test/test_ppo_cartpole.py --train --plot  # 训练并实时绘制曲线
    python test/test_ppo_cartpole.py --eval --checkpoint checkpoint.pt
    python test/test_ppo_cartpole.py --render --checkpoint checkpoint.pt
    python test/test_ppo_cartpole.py --record --checkpoint checkpoint.pt --video-dir ./videos
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


class TrainingPlotter:
    """Real-time training curves plotter using matplotlib."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.timesteps = []

        if enabled:
            try:
                import matplotlib.pyplot as plt

                self.plt = plt
                self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
                self.fig.suptitle("PPO Training Progress")

                # Setup subplots
                self.ax_reward = self.axes[0, 0]
                self.ax_length = self.axes[0, 1]
                self.ax_policy_loss = self.axes[1, 0]
                self.ax_value_loss = self.axes[1, 1]

                self.ax_reward.set_title("Episode Reward")
                self.ax_reward.set_xlabel("Episode")
                self.ax_reward.set_ylabel("Reward")
                self.ax_reward.grid(True, alpha=0.3)

                self.ax_length.set_title("Episode Length")
                self.ax_length.set_xlabel("Episode")
                self.ax_length.set_ylabel("Steps")
                self.ax_length.grid(True, alpha=0.3)

                self.ax_policy_loss.set_title("Policy Loss")
                self.ax_policy_loss.set_xlabel("Iteration")
                self.ax_policy_loss.set_ylabel("Loss")
                self.ax_policy_loss.grid(True, alpha=0.3)

                self.ax_value_loss.set_title("Value Loss")
                self.ax_value_loss.set_xlabel("Iteration")
                self.ax_value_loss.set_ylabel("Loss")
                self.ax_value_loss.grid(True, alpha=0.3)

                self.fig.tight_layout()
                self.plt.ion()  # Interactive mode
                self.plt.show(block=False)

            except ImportError:
                print("Warning: matplotlib not installed, plotting disabled")
                self.enabled = False

    def update_episode(self, reward: float, length: float):
        """Update episode statistics."""
        if not self.enabled:
            return
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def update_loss(self, timestep: int, policy_loss: float, value_loss: float):
        """Update loss statistics."""
        if not self.enabled:
            return
        self.timesteps.append(timestep)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)

    def refresh(self):
        """Refresh the plot."""
        if not self.enabled:
            return

        # Update reward plot
        if self.episode_rewards:
            self.ax_reward.clear()
            self.ax_reward.plot(
                self.episode_rewards, "b-", alpha=0.6, label="Episode Reward"
            )
            # Add moving average
            window = min(10, len(self.episode_rewards))
            if window > 1:
                ma = [
                    sum(self.episode_rewards[max(0, i - window) : i + 1])
                    / min(window, i + 1)
                    for i in range(len(self.episode_rewards))
                ]
                self.ax_reward.plot(ma, "r-", linewidth=2, label=f"{window}-ep MA")
            self.ax_reward.set_title("Episode Reward")
            self.ax_reward.set_xlabel("Episode")
            self.ax_reward.set_ylabel("Reward")
            self.ax_reward.legend()
            self.ax_reward.grid(True, alpha=0.3)

        # Update length plot
        if self.episode_lengths:
            self.ax_length.clear()
            self.ax_length.plot(self.episode_lengths, "g-", alpha=0.6)
            self.ax_length.set_title("Episode Length")
            self.ax_length.set_xlabel("Episode")
            self.ax_length.set_ylabel("Steps")
            self.ax_length.grid(True, alpha=0.3)

        # Update policy loss plot
        if self.policy_losses:
            self.ax_policy_loss.clear()
            self.ax_policy_loss.plot(
                self.timesteps, self.policy_losses, "b-", alpha=0.7
            )
            self.ax_policy_loss.set_title("Policy Loss")
            self.ax_policy_loss.set_xlabel("Iteration")
            self.ax_policy_loss.set_ylabel("Loss")
            self.ax_policy_loss.grid(True, alpha=0.3)

        # Update value loss plot
        if self.value_losses:
            self.ax_value_loss.clear()
            self.ax_value_loss.plot(self.timesteps, self.value_losses, "r-", alpha=0.7)
            self.ax_value_loss.set_title("Value Loss")
            self.ax_value_loss.set_xlabel("Iteration")
            self.ax_value_loss.set_ylabel("Loss")
            self.ax_value_loss.grid(True, alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def save(self, path: str = "training_curves.png"):
        """Save the plot to file."""
        if not self.enabled:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {path}")

    def close(self):
        """Close the plot."""
        if self.enabled:
            self.plt.close(self.fig)


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
    plot: bool = False,
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
        plot: Whether to show real-time training curves.
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
    print(f"Real-time plot: {plot}")
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

    # Create plotter
    plotter = TrainingPlotter(enabled=plot)

    # Monkey-patch agent's learn method to capture episode data
    # original_learn = agent.learn

    def learn_with_plotting(total_timesteps: int):
        """Modified learn with plotting."""
        import time

        num_iterations = total_timesteps // (agent.cfg.num_steps * agent.num_envs)
        start_time = time.time()

        for iteration in range(num_iterations):
            agent.iteration = iteration

            # Collect rollout
            rollout_stats = agent.collect_rollout()

            # Update policy
            update_stats = agent.update()

            # Adjust learning rate
            agent.adjust_learning_rate(iteration, num_iterations)

            # Update plotter with losses
            plotter.update_loss(
                agent.total_timesteps,
                update_stats["train/policy_loss"],
                update_stats["train/value_loss"],
            )

            # Logging
            if agent.writer and iteration % agent.cfg.log_interval == 0:
                fps = (
                    agent.cfg.num_steps
                    * agent.num_envs
                    * agent.cfg.log_interval
                    / (time.time() - start_time)
                )
                start_time = time.time()

                agent.writer.add_scalar("time/fps", fps, agent.total_timesteps)
                agent.writer.add_scalar(
                    "time/iterations", iteration, agent.total_timesteps
                )

                for key, value in rollout_stats.items():
                    agent.writer.add_scalar(key, value, agent.total_timesteps)

                for key, value in update_stats.items():
                    agent.writer.add_scalar(key, value, agent.total_timesteps)

                # Log episode stats and update plotter
                if agent.episode_rewards:
                    mean_reward = sum(agent.episode_rewards) / len(
                        agent.episode_rewards
                    )
                    mean_length = sum(agent.episode_lengths) / len(
                        agent.episode_lengths
                    )
                    agent.writer.add_scalar(
                        "episode/mean_reward", mean_reward, agent.total_timesteps
                    )
                    agent.writer.add_scalar(
                        "episode/mean_length", mean_length, agent.total_timesteps
                    )

                    # Update plotter
                    for r, le in zip(agent.episode_rewards, agent.episode_lengths):
                        plotter.update_episode(r, le)

                    agent.episode_rewards.clear()
                    agent.episode_lengths.clear()

                    # Refresh plot
                    plotter.refresh()

                print(
                    f"Iteration {iteration}/{num_iterations} | "
                    f"Timesteps {agent.total_timesteps} | "
                    f"FPS {fps:.0f} | "
                    f"Policy Loss {update_stats['train/policy_loss']:.4f} | "
                    f"Value Loss {update_stats['train/value_loss']:.4f}"
                )

            # Save checkpoint
            if iteration % agent.cfg.save_interval == 0:
                agent.save(f"checkpoint_{iteration}.pt")

    # Train
    print("\nStarting training...\n")
    if plot:
        learn_with_plotting(total_timesteps=total_timesteps)
    else:
        agent.learn(total_timesteps=total_timesteps)

    # Save final plot
    if plot:
        plotter.save("training_curves.png")
        plotter.close()

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


def record_video(
    checkpoint_path: str,
    video_dir: str = "./videos",
    num_episodes: int = 5,
):
    """Record video of a trained PPO agent playing CartPole.

    Args:
        checkpoint_path: Path to model checkpoint.
        video_dir: Directory to save videos.
        num_episodes: Number of episodes to record.
    """
    from gymnasium.wrappers import RecordVideo

    print("=" * 60)
    print("PPO Video Recording on CartPole-v1")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Video dir: {video_dir}")
    print(f"Num episodes: {num_episodes}")
    print("=" * 60)

    # Create video directory
    os.makedirs(video_dir, exist_ok=True)

    # Create environment with video recording
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="cartpole_ppo",
    )

    # Get spaces
    obs_space = env.observation_space
    action_space = env.action_space

    # Create a dummy VecEnv just to satisfy the interface
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

    # Record episodes
    print("\nRecording episodes...")
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

        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    env.close()
    print(f"\nVideos saved to {video_dir}")


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
        help="Render a trained PPO agent (real-time display)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video of a trained PPO agent",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show real-time training curves during training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cartpole_ppo_final.pt",
        help="Path to model checkpoint for eval/render/record",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="Directory to save recorded videos",
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
        help="Number of episodes for evaluation/recording",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between frames for rendering (seconds)",
    )

    args = parser.parse_args()

    # Default to train if no mode specified
    if not (args.train or args.eval or args.render or args.record):
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
            plot=args.plot,
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

    elif args.record:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
        record_video(
            checkpoint_path=args.checkpoint,
            video_dir=args.video_dir,
            num_episodes=args.num_episodes,
        )


if __name__ == "__main__":
    main()
