"""
Visualization module for training metrics and evaluation results.

Generates plots corresponding to figures in the paper:
- Figure 3: Pipeline time comparison
- Figure 4: Test overhead reduction
- Figure 5: Throughput improvement
- Training convergence curves
- β sensitivity analysis plots
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_training_curves(history: dict, save_dir: str = "plots"):
    """
    Plot training convergence curves.

    Shows episode rewards, loss, epsilon decay, and defect miss rate over training.
    Reference: Section V-E1 - Training convergence after ~1500 episodes.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards
    ax = axes[0, 0]
    rewards = history["episode_rewards"]
    ax.plot(rewards, alpha=0.3, color="blue")
    window = min(100, len(rewards))
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, color="blue", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward Convergence")
    ax.grid(True, alpha=0.3)

    # Training loss
    ax = axes[0, 1]
    losses = history["episode_losses"]
    ax.plot(losses, alpha=0.3, color="red")
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(losses)), smoothed, color="red", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # Epsilon decay
    ax = axes[1, 0]
    ax.plot(history["epsilon_values"], color="green", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate Decay")
    ax.grid(True, alpha=0.3)

    # Defect miss rate
    ax = axes[1, 1]
    dmr = history["episode_dmr"]
    ax.plot(dmr, alpha=0.3, color="orange")
    if len(dmr) >= window:
        smoothed = np.convolve(dmr, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(dmr)), smoothed, color="orange", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Defect Miss Rate")
    ax.set_title("Defect Miss Rate Over Training")
    ax.axhline(y=0.05, color="red", linestyle="--", label="5% threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(summary: dict, save_dir: str = "plots"):
    """
    Plot comparison between RL agent and baselines.

    Generates plots corresponding to:
    - Figure 3: Pipeline time comparison
    - Figure 4: Test overhead reduction
    - Figure 5: Throughput improvement
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    policies = list(summary.keys())
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    # Figure 3: Throughput comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    throughputs = [summary[p]["throughput"]["mean"] for p in policies]
    throughput_stds = [summary[p]["throughput"]["std"] for p in policies]
    bars = ax.bar(policies, throughputs, yerr=throughput_stds, color=colors,
                  capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Throughput (commits/min)")
    ax.set_title("Pipeline Throughput Comparison (Figure 5)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/throughput_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 4: Test time savings
    fig, ax = plt.subplots(figsize=(10, 6))
    tts = [summary[p]["test_time_savings"]["mean"] * 100 for p in policies]
    tts_stds = [summary[p]["test_time_savings"]["std"] * 100 for p in policies]
    bars = ax.bar(policies, tts, yerr=tts_stds, color=colors,
                  capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Test Time Savings (%)")
    ax.set_title("Test Overhead Reduction (Figure 4)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/test_time_savings.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Defect miss rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    dmr = [summary[p]["defect_miss_rate"]["mean"] * 100 for p in policies]
    dmr_stds = [summary[p]["defect_miss_rate"]["std"] * 100 for p in policies]
    bars = ax.bar(policies, dmr, yerr=dmr_stds, color=colors,
                  capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Defect Miss Rate (%)")
    ax.set_title("Defect Miss Rate Comparison")
    ax.axhline(y=5, color="red", linestyle="--", label="5% threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/defect_miss_rate.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_beta_sensitivity(beta_results: dict, save_dir: str = "plots"):
    """
    Plot β sensitivity study results (Section IV-C).

    Shows how throughput and DMR change across β ∈ {1, 3, 5, 10}.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    betas = sorted(beta_results.keys())
    throughputs = [beta_results[b]["throughput"] for b in betas]
    dmrs = [beta_results[b]["defect_miss_rate"] * 100 for b in betas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(betas, throughputs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax1.set_xlabel("β (Defect Penalty Weight)")
    ax1.set_ylabel("Throughput (commits/min)")
    ax1.set_title("Throughput vs β")
    ax1.grid(True, alpha=0.3)

    ax2.plot(betas, dmrs, "o-", color="#F44336", linewidth=2, markersize=8)
    ax2.set_xlabel("β (Defect Penalty Weight)")
    ax2.set_ylabel("Defect Miss Rate (%)")
    ax2.set_title("Defect Miss Rate vs β")
    ax2.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="5% threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/beta_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_action_distribution(history: dict, save_dir: str = "plots"):
    """Plot how action distribution evolves during training."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    distributions = history["action_distributions"]
    episodes = range(len(distributions))

    full = [d[0] for d in distributions]
    partial = [d[1] for d in distributions]
    skip = [d[2] for d in distributions]

    window = min(50, len(full))

    fig, ax = plt.subplots(figsize=(12, 6))

    for data, label, color in [
        (full, "Full Tests", "#F44336"),
        (partial, "Partial Tests", "#4CAF50"),
        (skip, "Skip Tests", "#2196F3"),
    ]:
        if len(data) >= window:
            smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(data)), smoothed, label=label,
                    color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Count per Episode")
    ax.set_title("Action Distribution Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/action_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
