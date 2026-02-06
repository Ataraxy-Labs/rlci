"""
Baseline Strategies for CI/CD Pipeline Optimization.

Three baselines from Section III-D:
1. Static Baseline (SB): Always runs full test suite
2. Heuristic Policy (HP): Partial tests if diff < 20 LOC; otherwise full tests
3. Supervised Classifier (SC): Logistic regression predicts bug risk
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Optional

from .environment import (
    ACTION_FULL_TEST,
    ACTION_PARTIAL_TEST,
    ACTION_SKIP_TEST,
    STATE_DIM,
    NUM_ACTIONS,
)


class BaselinePolicy:
    """Base class for baseline policies."""

    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class StaticBaseline(BaselinePolicy):
    """
    Static Baseline (SB): Always runs the full test suite.

    This is the most conservative strategy - guarantees 100% detection
    but at maximum execution time cost.
    """

    def select_action(self, state: np.ndarray) -> int:
        return ACTION_FULL_TEST

    def name(self) -> str:
        return "Static Baseline (SB)"


class HeuristicPolicy(BaselinePolicy):
    """
    Heuristic Policy (HP): Partial tests if diff < 20 LOC; otherwise full tests.

    Uses a simple rule based on diff size. The diff_size in the state vector
    is normalized, so we use a threshold corresponding to ~20 LOC.
    Assuming max diff is ~200 LOC, threshold = 20/200 = 0.1.
    """

    def __init__(self, diff_threshold: float = 0.1):
        """
        Args:
            diff_threshold: Normalized diff size threshold (default: 0.1 ~ 20 LOC)
        """
        self.diff_threshold = diff_threshold

    def select_action(self, state: np.ndarray) -> int:
        diff_size = state[0]  # First element is diff_size
        if diff_size < self.diff_threshold:
            return ACTION_PARTIAL_TEST
        return ACTION_FULL_TEST

    def name(self) -> str:
        return "Heuristic Policy (HP)"


class SupervisedClassifier(BaselinePolicy):
    """
    Supervised Classifier (SC): Logistic regression predicts bug risk.

    Trains a logistic regression model on historical commit data to predict
    bug probability. If predicted risk is below threshold, skips or reduces tests.

    Decision logic:
    - risk < low_threshold: skip tests
    - risk < high_threshold: partial tests
    - risk >= high_threshold: full tests
    """

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Args:
            low_threshold: Below this, skip tests
            high_threshold: Above this, run full tests
            seed: Random seed for reproducibility
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.model = LogisticRegression(random_state=seed, max_iter=1000)
        self.is_trained = False

    def train(self, states: np.ndarray, labels: np.ndarray):
        """
        Train the logistic regression classifier.

        Args:
            states: Array of state vectors (N, 10)
            labels: Array of binary bug labels (N,)
        """
        self.model.fit(states, labels)
        self.is_trained = True

    def predict_risk(self, state: np.ndarray) -> float:
        """Predict bug probability for a commit state."""
        if not self.is_trained:
            return 0.5  # Default to medium risk
        state_2d = state.reshape(1, -1)
        return self.model.predict_proba(state_2d)[0, 1]

    def select_action(self, state: np.ndarray) -> int:
        risk = self.predict_risk(state)
        if risk < self.low_threshold:
            return ACTION_SKIP_TEST
        elif risk < self.high_threshold:
            return ACTION_PARTIAL_TEST
        return ACTION_FULL_TEST

    def name(self) -> str:
        return "Supervised Classifier (SC)"


def train_supervised_baseline(
    env_class,
    num_episodes: int = 200,
    commits_per_episode: int = 100,
    seed: int = 42,
) -> SupervisedClassifier:
    """
    Train the supervised classifier baseline on simulated data.

    Generates training data by running the environment and collecting
    commit states with their ground-truth bug labels.

    Args:
        env_class: Environment class to use for data generation
        num_episodes: Number of episodes for training data
        commits_per_episode: Commits per episode
        seed: Random seed

    Returns:
        Trained SupervisedClassifier
    """
    env = env_class(commits_per_episode=commits_per_episode, seed=seed)
    states_list = []
    labels_list = []

    for _ in range(num_episodes):
        env.reset()
        for commit in env.commits:
            states_list.append(commit.to_state_vector())
            labels_list.append(int(commit.has_bug))

    states = np.array(states_list)
    labels = np.array(labels_list)

    classifier = SupervisedClassifier(seed=seed)
    classifier.train(states, labels)
    return classifier
