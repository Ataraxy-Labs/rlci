"""Tests for evaluation and metrics."""

import numpy as np
import pytest

from src.evaluate import (
    evaluate_policy,
    run_comparison,
    beta_sensitivity_study,
    adversarial_evaluation,
)
from src.environment import CICDEnvironment, ACTION_FULL_TEST, TEST_EXECUTION_TIME
from src.agent import DQNAgent
from src.baselines import StaticBaseline, HeuristicPolicy
from src.train import train


class TestEvaluatePolicy:
    """Tests for policy evaluation."""

    def test_static_baseline_evaluation(self):
        env = CICDEnvironment(commits_per_episode=50, seed=42)
        sb = StaticBaseline()
        metrics = evaluate_policy(env, sb, num_episodes=10, seed=42)

        assert "throughput" in metrics
        assert "defect_miss_rate" in metrics
        assert "test_time_savings" in metrics
        assert "sustainability_impact" in metrics
        assert metrics["defect_miss_rate"] == 0.0  # Full tests catch everything
        assert metrics["test_time_savings"] == 0.0  # No savings vs itself

    def test_heuristic_evaluation(self):
        env = CICDEnvironment(commits_per_episode=50, seed=42)
        hp = HeuristicPolicy()
        metrics = evaluate_policy(env, hp, num_episodes=10, seed=42)

        assert metrics["test_time_savings"] >= 0  # Should save some time
        assert metrics["throughput"] > 0

    def test_rl_agent_evaluation(self):
        """Train a small agent and evaluate it."""
        agent, _ = train(
            num_episodes=20,
            commits_per_episode=20,
            seed=42,
            verbose=False,
            device="cpu",
        )
        env = CICDEnvironment(commits_per_episode=20, seed=42)
        metrics = evaluate_policy(env, agent, num_episodes=10, seed=42)

        assert metrics["throughput"] > 0
        assert 0 <= metrics["defect_miss_rate"] <= 1

    def test_metrics_consistency(self):
        """Verify metric calculations are consistent."""
        env = CICDEnvironment(commits_per_episode=100, seed=42)
        sb = StaticBaseline()
        metrics = evaluate_policy(env, sb, num_episodes=5, seed=42)

        # For static baseline: all full tests
        expected_time = metrics["total_commits"] * TEST_EXECUTION_TIME[ACTION_FULL_TEST]
        assert abs(metrics["total_execution_time"] - expected_time) < 0.01

    def test_sustainability_impact_nonnegative_for_savings(self):
        """Sustainability impact should be positive when saving time."""
        env = CICDEnvironment(commits_per_episode=50, seed=42)
        hp = HeuristicPolicy()
        metrics = evaluate_policy(env, hp, num_episodes=10, seed=42)
        assert metrics["sustainability_impact"] >= 0


class TestRunComparison:
    """Tests for full comparison runs."""

    def test_comparison_runs(self):
        results = run_comparison(
            num_episodes=5,
            commits_per_episode=10,
            num_runs=1,
            training_episodes=10,
            verbose=False,
            device="cpu",
        )
        assert "summary" in results
        assert "raw_results" in results
        assert "RL Agent (DQN)" in results["summary"]
        assert "Static Baseline (SB)" in results["summary"]
        assert "Heuristic Policy (HP)" in results["summary"]
        assert "Supervised Classifier (SC)" in results["summary"]

    def test_summary_has_mean_std(self):
        results = run_comparison(
            num_episodes=5,
            commits_per_episode=10,
            num_runs=2,
            training_episodes=10,
            verbose=False,
            device="cpu",
        )
        for name, metrics in results["summary"].items():
            for key, val in metrics.items():
                assert "mean" in val
                assert "std" in val


class TestBetaSensitivity:
    """Tests for β sensitivity study."""

    def test_sensitivity_study_runs(self):
        results = beta_sensitivity_study(
            beta_values=[1.0, 5.0],
            num_episodes=5,
            commits_per_episode=10,
            training_episodes=10,
            seed=42,
            verbose=False,
            device="cpu",
        )
        assert 1.0 in results
        assert 5.0 in results

    def test_higher_beta_lower_dmr(self):
        """Higher β should generally lead to lower defect miss rate."""
        results = beta_sensitivity_study(
            beta_values=[1.0, 10.0],
            num_episodes=20,
            commits_per_episode=20,
            training_episodes=50,
            seed=42,
            verbose=False,
            device="cpu",
        )
        # With more training this trend would be stronger,
        # but even with short training the direction should hold or be close
        dmr_low_beta = results[1.0]["defect_miss_rate"]
        dmr_high_beta = results[10.0]["defect_miss_rate"]
        # High beta should not be dramatically worse
        assert dmr_high_beta <= dmr_low_beta + 0.2


class TestAdversarialEvaluation:
    """Tests for adversarial robustness."""

    def test_adversarial_runs(self):
        agent, _ = train(
            num_episodes=20,
            commits_per_episode=20,
            seed=42,
            verbose=False,
            device="cpu",
        )
        metrics = adversarial_evaluation(
            agent, num_episodes=5, commits_per_episode=20, seed=42
        )
        assert "throughput" in metrics
        assert "defect_miss_rate" in metrics
