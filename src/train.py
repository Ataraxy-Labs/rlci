"""
Training Pipeline for DQN Agent.

Trains the DQN agent over 2000 episodes, each simulating 100 commits.
Implements ε-greedy exploration with decay from 1.0 to 0.1.
Tracks convergence metrics and saves checkpoints.

Reference: Section III-C and IV-E of the paper.
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

from .environment import CICDEnvironment
from .agent import DQNAgent


def train(
    num_episodes: int = 2000,
    commits_per_episode: int = 100,
    beta: float = 5.0,
    seed: int = 42,
    save_path: Optional[str] = None,
    verbose: bool = True,
    log_interval: int = 100,
    device: Optional[str] = None,
) -> tuple[DQNAgent, dict]:
    """
    Train the DQN agent on the CI/CD simulation environment.

    Args:
        num_episodes: Number of training episodes (default: 2000)
        commits_per_episode: Commits per episode (default: 100)
        beta: Penalty weight for escaped defects (default: 5.0)
        seed: Random seed for reproducibility
        save_path: Path to save trained model
        verbose: Whether to print training progress
        log_interval: Episodes between progress logs
        device: Compute device

    Returns:
        Tuple of (trained agent, training history dict)
    """
    env = CICDEnvironment(
        commits_per_episode=commits_per_episode,
        beta=beta,
        seed=seed,
    )

    agent = DQNAgent(
        seed=seed,
        device=device,
    )

    history = {
        "episode_rewards": [],
        "episode_exec_times": [],
        "episode_dmr": [],        # Defect miss rate
        "episode_losses": [],
        "epsilon_values": [],
        "action_distributions": [],
    }

    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_losses = []

        for step in range(commits_per_episode):
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update network
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon(episode)

        # Update target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Record stats
        stats = env.get_episode_stats()
        history["episode_rewards"].append(episode_reward)
        history["episode_exec_times"].append(stats["total_execution_time"])
        history["episode_dmr"].append(stats["defect_miss_rate"])
        history["episode_losses"].append(
            np.mean(episode_losses) if episode_losses else 0.0
        )
        history["epsilon_values"].append(agent.epsilon)
        history["action_distributions"].append(stats["actions"].copy())

        agent.episode_rewards.append(episode_reward)

        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(history["episode_rewards"][-log_interval:])
            avg_dmr = np.mean(history["episode_dmr"][-log_interval:])
            avg_loss = np.mean(history["episode_losses"][-log_interval:])
            elapsed = time.time() - start_time
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg DMR: {avg_dmr:.3f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    total_time = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {total_time:.1f}s")

    history["total_training_time"] = total_time
    return agent, history


def check_convergence(history: dict, window: int = 100, threshold: float = 0.03) -> dict:
    """
    Check if training has converged.

    From Section V-E1: Agent policy converged after ~1500 episodes.
    Q-values stabilized and reward variance dropped below 3%.

    Args:
        history: Training history dict
        window: Rolling window size
        threshold: Variance threshold for convergence (3% from paper)

    Returns:
        Dict with convergence analysis
    """
    rewards = np.array(history["episode_rewards"])

    if len(rewards) < window:
        return {"converged": False, "message": "Not enough episodes"}

    # Check last window
    recent_rewards = rewards[-window:]
    mean_reward = np.mean(recent_rewards)
    std_reward = np.std(recent_rewards)

    # Coefficient of variation
    cv = abs(std_reward / mean_reward) if mean_reward != 0 else float("inf")

    converged = cv < threshold

    return {
        "converged": converged,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "coefficient_of_variation": float(cv),
        "threshold": threshold,
        "window": window,
    }


if __name__ == "__main__":
    agent, history = train(
        num_episodes=2000,
        commits_per_episode=100,
        beta=5.0,
        seed=42,
        save_path="models/dqn_agent.pt",
    )

    convergence = check_convergence(history)
    print(f"\nConvergence analysis: {convergence}")
