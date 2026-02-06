"""Tests for training pipeline."""

import numpy as np
import pytest
import tempfile
import os

from src.train import train, check_convergence


class TestTraining:
    """Tests for the DQN training pipeline."""

    def test_short_training_runs(self):
        """Verify training completes without errors."""
        agent, history = train(
            num_episodes=10,
            commits_per_episode=20,
            beta=5.0,
            seed=42,
            verbose=False,
            device="cpu",
        )
        assert agent is not None
        assert len(history["episode_rewards"]) == 10

    def test_training_produces_history(self):
        _, history = train(
            num_episodes=20,
            commits_per_episode=20,
            seed=42,
            verbose=False,
            device="cpu",
        )
        assert "episode_rewards" in history
        assert "episode_exec_times" in history
        assert "episode_dmr" in history
        assert "episode_losses" in history
        assert "epsilon_values" in history
        assert "action_distributions" in history
        assert "total_training_time" in history

    def test_epsilon_decays_during_training(self):
        agent, history = train(
            num_episodes=50,
            commits_per_episode=20,
            seed=42,
            verbose=False,
            device="cpu",
        )
        # Epsilon should decrease
        assert history["epsilon_values"][-1] < history["epsilon_values"][0]

    def test_model_saving(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent, _ = train(
                num_episodes=5,
                commits_per_episode=10,
                seed=42,
                save_path=path,
                verbose=False,
                device="cpu",
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_different_beta_values(self):
        """Training should work with different β values."""
        for beta in [1.0, 5.0, 10.0]:
            agent, history = train(
                num_episodes=5,
                commits_per_episode=10,
                beta=beta,
                seed=42,
                verbose=False,
                device="cpu",
            )
            assert len(history["episode_rewards"]) == 5

    def test_reproducibility(self):
        """Same seed should produce same results."""
        _, h1 = train(num_episodes=10, commits_per_episode=20, seed=42, verbose=False, device="cpu")
        _, h2 = train(num_episodes=10, commits_per_episode=20, seed=42, verbose=False, device="cpu")
        np.testing.assert_array_almost_equal(
            h1["episode_rewards"], h2["episode_rewards"], decimal=4
        )


class TestConvergenceCheck:
    """Tests for convergence checking."""

    def test_converged_history(self):
        """Stable rewards should indicate convergence."""
        history = {
            "episode_rewards": [-50.0] * 200,  # Very stable
        }
        result = check_convergence(history, window=100, threshold=0.03)
        assert result["converged"]

    def test_not_converged_short(self):
        history = {"episode_rewards": [-50.0] * 10}
        result = check_convergence(history, window=100)
        assert not result["converged"]

    def test_not_converged_volatile(self):
        """Volatile rewards should not indicate convergence."""
        np.random.seed(42)
        rewards = list(np.random.uniform(-100, 0, 200))
        history = {"episode_rewards": rewards}
        result = check_convergence(history, window=100, threshold=0.01)
        assert not result["converged"]
