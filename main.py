"""
Main entry point for the RL CI/CD Pipeline Optimizer.

Runs the full pipeline:
1. Train DQN agent (2000 episodes, 100 commits/episode)
2. Evaluate against baselines (5 independent runs)
3. β sensitivity study (β ∈ {1, 3, 5, 10})
4. Adversarial robustness evaluation
5. Generate visualization plots
"""

import argparse
import json
from pathlib import Path

from src.train import train, check_convergence
from src.evaluate import (
    run_comparison,
    beta_sensitivity_study,
    adversarial_evaluation,
)
from src.visualize import (
    plot_training_curves,
    plot_comparison,
    plot_beta_sensitivity,
    plot_action_distribution,
)


def main():
    parser = argparse.ArgumentParser(
        description="RL-based CI/CD Pipeline Optimizer"
    )
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Training episodes (default: 2000)")
    parser.add_argument("--commits", type=int, default=100,
                        help="Commits per episode (default: 100)")
    parser.add_argument("--beta", type=float, default=20.0,
                        help="Defect penalty weight (default: 20.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Independent evaluation runs (default: 5)")
    parser.add_argument("--eval-episodes", type=int, default=2000,
                        help="Evaluation episodes per run (default: 2000)")
    parser.add_argument("--model-path", type=str, default="models/dqn_agent.pt",
                        help="Model save path")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="Directory for plots")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (cpu/cuda/mps)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with reduced parameters for testing")
    args = parser.parse_args()

    if args.quick:
        args.episodes = 100
        args.commits = 50
        args.runs = 2
        args.eval_episodes = 50

    print("=" * 70)
    print("RL-based CI/CD Pipeline Optimizer")
    print("Reinforcement Learning for Dynamic Workflow Optimization")
    print("=" * 70)

    # 1. Train DQN Agent
    print("\n[1/5] Training DQN Agent...")
    print(f"  Episodes: {args.episodes}")
    print(f"  Commits/episode: {args.commits}")
    print(f"  β: {args.beta}")
    print(f"  Seed: {args.seed}")

    agent, history = train(
        num_episodes=args.episodes,
        commits_per_episode=args.commits,
        beta=args.beta,
        seed=args.seed,
        save_path=args.model_path,
        verbose=True,
        device=args.device,
    )

    # Check convergence
    convergence = check_convergence(history)
    print(f"\n  Convergence: {convergence}")

    # Plot training curves
    plot_training_curves(history, save_dir=args.plot_dir)
    plot_action_distribution(history, save_dir=args.plot_dir)
    print(f"  Training plots saved to {args.plot_dir}/")

    # 2. Run Comparison vs Baselines
    print(f"\n[2/5] Running comparison ({args.runs} runs)...")
    comparison = run_comparison(
        num_episodes=args.eval_episodes,
        commits_per_episode=args.commits,
        beta=args.beta,
        num_runs=args.runs,
        training_episodes=args.episodes,
        verbose=True,
        device=args.device,
    )

    print("\n  Summary (mean ± std):")
    for name, metrics in comparison["summary"].items():
        print(f"\n  {name}:")
        for key, val in metrics.items():
            print(f"    {key}: {val['mean']:.4f} ± {val['std']:.4f}")

    plot_comparison(comparison["summary"], save_dir=args.plot_dir)
    print(f"\n  Comparison plots saved to {args.plot_dir}/")

    # 3. β Sensitivity Study
    print("\n[3/5] Running β sensitivity study...")
    beta_results = beta_sensitivity_study(
        beta_values=[1.0, 3.0, 5.0, 10.0],
        num_episodes=args.eval_episodes,
        commits_per_episode=args.commits,
        training_episodes=args.episodes,
        seed=args.seed,
        verbose=True,
        device=args.device,
    )

    print("\n  β Sensitivity Results:")
    for beta, metrics in beta_results.items():
        print(
            f"  β={beta}: TP={metrics['throughput']:.2f} | "
            f"DMR={metrics['defect_miss_rate']:.3f} | "
            f"TTS={metrics['test_time_savings']:.3f}"
        )

    plot_beta_sensitivity(beta_results, save_dir=args.plot_dir)

    # 4. Adversarial Evaluation
    print("\n[4/5] Running adversarial evaluation...")
    adv_metrics = adversarial_evaluation(
        agent,
        num_episodes=min(500, args.eval_episodes),
        commits_per_episode=args.commits,
        beta=args.beta,
        seed=args.seed,
    )
    print(f"  Adversarial TP: {adv_metrics['throughput']:.2f}")
    print(f"  Adversarial DMR: {adv_metrics['defect_miss_rate']:.3f}")

    # 5. Save Results
    print("\n[5/5] Saving results...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (int, float, str, bool)):
            return obj
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return str(obj)

    all_results = {
        "training": {
            "convergence": convert(convergence),
            "final_epsilon": float(agent.epsilon),
            "total_episodes": args.episodes,
        },
        "comparison": convert(comparison["summary"]),
        "beta_sensitivity": {
            str(k): convert(v) for k, v in beta_results.items()
        },
        "adversarial": convert(adv_metrics),
    }

    with open(results_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"  Results saved to {results_dir}/results.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
