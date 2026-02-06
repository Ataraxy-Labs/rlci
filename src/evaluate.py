"""
Evaluation and Metrics Module.

Implements all evaluation metrics from Section IV-B:
- Throughput (TP): Commits processed per unit time
- Defect Miss Rate (DMR): Percentage of undetected bugs
- Test Time Savings (TTS): Reduction in test execution time vs. baseline
- Sustainability Impact (SI): Compute savings in core-minutes

Also implements:
- β sensitivity study (Section IV-C): β ∈ {1, 3, 5, 10}
- Adversarial robustness evaluation (Section IV-D)
- 5 independent runs with mean ± std (Section IV-B)
"""

import numpy as np
from typing import Optional

from .environment import (
    CICDEnvironment,
    AdversarialCICDEnvironment,
    TEST_EXECUTION_TIME,
    ACTION_FULL_TEST,
)
from .agent import DQNAgent
from .baselines import (
    BaselinePolicy,
    StaticBaseline,
    HeuristicPolicy,
    SupervisedClassifier,
    train_supervised_baseline,
)
from .train import train


def evaluate_policy(
    env: CICDEnvironment,
    policy,
    num_episodes: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: CI/CD simulation environment
        policy: Agent or baseline policy with select_action method
        num_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        Dict with evaluation metrics
    """
    if seed is not None:
        env.rng = np.random.RandomState(seed)

    total_exec_time = 0.0
    total_bugs_introduced = 0
    total_bugs_escaped = 0
    total_commits = 0
    action_counts = {0: 0, 1: 0, 2: 0}
    episode_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(env.commits_per_episode):
            if isinstance(policy, DQNAgent):
                action = policy.select_action(state, training=False)
            else:
                action = policy.select_action(state)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            action_counts[action] += 1
            state = next_state

            if done:
                break

        stats = env.get_episode_stats()
        total_exec_time += stats["total_execution_time"]
        total_bugs_introduced += stats["bugs_introduced"]
        total_bugs_escaped += stats["bugs_escaped"]
        total_commits += stats["commits_processed"]
        episode_rewards.append(episode_reward)

    # Compute metrics
    # Throughput: commits per minute
    throughput = total_commits / total_exec_time if total_exec_time > 0 else total_commits

    # Defect Miss Rate
    dmr = total_bugs_escaped / total_bugs_introduced if total_bugs_introduced > 0 else 0.0

    # Test Time Savings vs static baseline (full tests always)
    static_time = total_commits * TEST_EXECUTION_TIME[ACTION_FULL_TEST]
    tts = (static_time - total_exec_time) / static_time if static_time > 0 else 0.0

    # Sustainability Impact: core-minutes saved
    sustainability_impact = static_time - total_exec_time

    return {
        "throughput": throughput,
        "defect_miss_rate": dmr,
        "test_time_savings": tts,
        "sustainability_impact": sustainability_impact,
        "total_execution_time": total_exec_time,
        "total_commits": total_commits,
        "total_bugs_introduced": total_bugs_introduced,
        "total_bugs_escaped": total_bugs_escaped,
        "mean_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
        "action_distribution": action_counts,
    }


def run_comparison(
    num_episodes: int = 2000,
    commits_per_episode: int = 100,
    beta: float = 20.0,
    num_runs: int = 5,
    training_episodes: int = 2000,
    verbose: bool = True,
    device: Optional[str] = None,
) -> dict:
    """
    Run full comparison: RL agent vs all baselines.

    All metrics reported as mean ± std over 5 independent runs (Section IV-B).

    Args:
        num_episodes: Evaluation episodes per run
        commits_per_episode: Commits per episode
        beta: Penalty weight
        num_runs: Number of independent runs (default: 5)
        training_episodes: Episodes for training the RL agent
        verbose: Print progress
        device: Compute device

    Returns:
        Dict with results for each policy across all runs
    """
    results = {
        "RL Agent (DQN)": [],
        "Static Baseline (SB)": [],
        "Heuristic Policy (HP)": [],
        "Supervised Classifier (SC)": [],
    }

    for run in range(num_runs):
        seed = run * 1000 + 42
        if verbose:
            print(f"\n{'='*60}")
            print(f"Run {run + 1}/{num_runs} (seed={seed})")
            print(f"{'='*60}")

        # Train RL agent
        if verbose:
            print("\nTraining RL agent...")
        agent, _ = train(
            num_episodes=training_episodes,
            commits_per_episode=commits_per_episode,
            beta=beta,
            seed=seed,
            verbose=verbose,
            device=device,
        )

        # Train supervised baseline
        sc = train_supervised_baseline(
            CICDEnvironment,
            num_episodes=200,
            commits_per_episode=commits_per_episode,
            seed=seed,
        )

        # Evaluate all policies
        policies = {
            "RL Agent (DQN)": agent,
            "Static Baseline (SB)": StaticBaseline(),
            "Heuristic Policy (HP)": HeuristicPolicy(),
            "Supervised Classifier (SC)": sc,
        }

        for name, policy in policies.items():
            env = CICDEnvironment(
                commits_per_episode=commits_per_episode,
                beta=beta,
                seed=seed + 1,
            )
            metrics = evaluate_policy(env, policy, num_episodes=num_episodes, seed=seed + 1)
            results[name].append(metrics)

            if verbose:
                print(
                    f"  {name}: TP={metrics['throughput']:.2f} | "
                    f"DMR={metrics['defect_miss_rate']:.3f} | "
                    f"TTS={metrics['test_time_savings']:.3f} | "
                    f"SI={metrics['sustainability_impact']:.0f}"
                )

    # Aggregate results: mean ± std
    summary = {}
    for name, run_results in results.items():
        metrics_keys = ["throughput", "defect_miss_rate", "test_time_savings",
                        "sustainability_impact", "mean_episode_reward"]
        summary[name] = {}
        for key in metrics_keys:
            values = [r[key] for r in run_results]
            summary[name][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return {"raw_results": results, "summary": summary}


def beta_sensitivity_study(
    beta_values: list[float] = [1.0, 3.0, 5.0, 10.0],
    num_episodes: int = 2000,
    commits_per_episode: int = 100,
    training_episodes: int = 2000,
    seed: int = 42,
    verbose: bool = True,
    device: Optional[str] = None,
) -> dict:
    """
    β sensitivity study (Section IV-C).

    Evaluates the RL agent trained with different β values
    to observe throughput and DMR changes.

    Args:
        beta_values: List of β values to test (default: {1, 3, 5, 10})
        num_episodes: Evaluation episodes
        commits_per_episode: Commits per episode
        training_episodes: Training episodes per β
        seed: Random seed
        verbose: Print progress
        device: Compute device

    Returns:
        Dict mapping β values to evaluation metrics
    """
    results = {}

    for beta in beta_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"β = {beta}")
            print(f"{'='*60}")

        # Train agent with this β
        agent, history = train(
            num_episodes=training_episodes,
            commits_per_episode=commits_per_episode,
            beta=beta,
            seed=seed,
            verbose=verbose,
            device=device,
        )

        # Evaluate
        env = CICDEnvironment(
            commits_per_episode=commits_per_episode,
            beta=beta,
            seed=seed + 1,
        )
        metrics = evaluate_policy(env, agent, num_episodes=num_episodes, seed=seed + 1)
        results[beta] = metrics

        if verbose:
            print(
                f"  β={beta}: TP={metrics['throughput']:.2f} | "
                f"DMR={metrics['defect_miss_rate']:.3f} | "
                f"TTS={metrics['test_time_savings']:.3f}"
            )

    return results


def adversarial_evaluation(
    agent: DQNAgent,
    num_episodes: int = 500,
    commits_per_episode: int = 100,
    beta: float = 20.0,
    seed: int = 42,
) -> dict:
    """
    Adversarial robustness evaluation (Section IV-D).

    Tests policy with adversarial commit sequences: low-diff commits
    followed by high-diff commits to confirm generalization.

    Args:
        agent: Trained DQN agent
        num_episodes: Number of adversarial episodes
        commits_per_episode: Commits per episode
        beta: Penalty weight
        seed: Random seed

    Returns:
        Dict with adversarial evaluation metrics
    """
    env = AdversarialCICDEnvironment(
        commits_per_episode=commits_per_episode,
        beta=beta,
        seed=seed,
    )

    metrics = evaluate_policy(env, agent, num_episodes=num_episodes, seed=seed)
    return metrics


if __name__ == "__main__":
    # Quick comparison
    results = run_comparison(
        num_episodes=100,
        training_episodes=500,
        num_runs=2,
        verbose=True,
    )

    print("\n\nSummary:")
    for name, metrics in results["summary"].items():
        print(f"\n{name}:")
        for key, val in metrics.items():
            print(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f}")
