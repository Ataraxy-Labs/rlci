"""Tests for baseline strategies."""

import numpy as np
import pytest

from src.baselines import (
    StaticBaseline,
    HeuristicPolicy,
    SupervisedClassifier,
    train_supervised_baseline,
)
from src.environment import (
    CICDEnvironment,
    ACTION_FULL_TEST,
    ACTION_PARTIAL_TEST,
    ACTION_SKIP_TEST,
    STATE_DIM,
)


class TestStaticBaseline:
    """Tests for Static Baseline (SB): Always runs full test suite."""

    def test_always_full_test(self):
        sb = StaticBaseline()
        for _ in range(100):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            assert sb.select_action(state) == ACTION_FULL_TEST

    def test_name(self):
        sb = StaticBaseline()
        assert "Static" in sb.name()


class TestHeuristicPolicy:
    """Tests for Heuristic Policy (HP): Partial tests if diff < 20 LOC."""

    def test_small_diff_partial(self):
        hp = HeuristicPolicy(diff_threshold=0.1)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = 0.05  # Small diff
        assert hp.select_action(state) == ACTION_PARTIAL_TEST

    def test_large_diff_full(self):
        hp = HeuristicPolicy(diff_threshold=0.1)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = 0.5  # Large diff
        assert hp.select_action(state) == ACTION_FULL_TEST

    def test_threshold_boundary(self):
        hp = HeuristicPolicy(diff_threshold=0.1)
        # At threshold: should run full tests
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = 0.1
        assert hp.select_action(state) == ACTION_FULL_TEST

    def test_name(self):
        hp = HeuristicPolicy()
        assert "Heuristic" in hp.name()


class TestSupervisedClassifier:
    """Tests for Supervised Classifier (SC): Logistic regression."""

    def test_untrained_default(self):
        sc = SupervisedClassifier()
        state = np.random.randn(STATE_DIM).astype(np.float32)
        risk = sc.predict_risk(state)
        assert risk == 0.5  # Default when untrained

    def test_training(self):
        sc = SupervisedClassifier(seed=42)
        # Generate training data
        states = np.random.randn(200, STATE_DIM).astype(np.float32)
        labels = (np.random.random(200) > 0.5).astype(int)
        sc.train(states, labels)
        assert sc.is_trained

    def test_prediction_after_training(self):
        sc = SupervisedClassifier(seed=42)
        states = np.random.randn(200, STATE_DIM).astype(np.float32)
        labels = (np.random.random(200) > 0.5).astype(int)
        sc.train(states, labels)

        # Should return valid probability
        risk = sc.predict_risk(states[0])
        assert 0 <= risk <= 1

    def test_action_selection_thresholds(self):
        sc = SupervisedClassifier(low_threshold=0.1, high_threshold=0.3, seed=42)
        states = np.random.randn(500, STATE_DIM).astype(np.float32)
        labels = (np.random.random(500) > 0.5).astype(int)
        sc.train(states, labels)

        # Run many predictions and check all actions are possible
        actions = set()
        for i in range(500):
            action = sc.select_action(states[i])
            actions.add(action)
        # Should use at least 2 different actions
        assert len(actions) >= 2

    def test_name(self):
        sc = SupervisedClassifier()
        assert "Supervised" in sc.name()


class TestTrainSupervisedBaseline:
    """Tests for supervised baseline training function."""

    def test_trains_successfully(self):
        sc = train_supervised_baseline(
            CICDEnvironment,
            num_episodes=10,
            commits_per_episode=50,
            seed=42,
        )
        assert sc.is_trained

    def test_produces_valid_predictions(self):
        sc = train_supervised_baseline(
            CICDEnvironment,
            num_episodes=10,
            commits_per_episode=50,
            seed=42,
        )
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = sc.select_action(state)
        assert action in [ACTION_FULL_TEST, ACTION_PARTIAL_TEST, ACTION_SKIP_TEST]
