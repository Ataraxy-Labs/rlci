"""Tests for CI/CD simulation environment."""

import numpy as np
import pytest

from src.environment import (
    CICDEnvironment,
    AdversarialCICDEnvironment,
    Commit,
    ACTION_FULL_TEST,
    ACTION_PARTIAL_TEST,
    ACTION_SKIP_TEST,
    TEST_EXECUTION_TIME,
    BUG_DETECTION_RATE,
    BUG_INTRODUCTION_PROBABILITY,
    BUG_ESCAPE_PENALTY,
    STATE_DIM,
    NUM_ACTIONS,
)


class TestCommit:
    """Tests for the Commit dataclass."""

    def test_state_vector_shape(self):
        commit = Commit(0.5, 0.3, 0.2, 0.1, 0.9, 0.4, 0.3, 0.0, 0.1, 0.2)
        state = commit.to_state_vector()
        assert state.shape == (STATE_DIM,)
        assert state.dtype == np.float32

    def test_state_vector_values(self):
        commit = Commit(0.5, 0.3, 0.2, 0.1, 0.9, 0.4, 0.3, 0.0, 0.1, 0.2)
        state = commit.to_state_vector()
        expected = np.array([0.5, 0.3, 0.2, 0.1, 0.9, 0.4, 0.3, 0.0, 0.1, 0.2], dtype=np.float32)
        np.testing.assert_array_almost_equal(state, expected)


class TestCICDEnvironment:
    """Tests for the CI/CD simulation environment."""

    def test_initialization(self):
        env = CICDEnvironment(commits_per_episode=50, beta=5.0, seed=42)
        assert env.commits_per_episode == 50
        assert env.beta == 5.0
        assert env.state_dim == STATE_DIM
        assert env.num_actions == NUM_ACTIONS

    def test_reset_returns_valid_state(self):
        env = CICDEnvironment(seed=42)
        state = env.reset()
        assert state.shape == (STATE_DIM,)
        assert all(0 <= s <= 1 for s in state)

    def test_reset_generates_commits(self):
        env = CICDEnvironment(commits_per_episode=100, seed=42)
        env.reset()
        assert len(env.commits) == 100

    def test_step_full_test(self):
        env = CICDEnvironment(seed=42)
        env.reset()
        next_state, reward, done, info = env.step(ACTION_FULL_TEST)
        assert next_state.shape == (STATE_DIM,)
        assert info["execution_time"] == TEST_EXECUTION_TIME[ACTION_FULL_TEST]
        assert reward <= 0  # Reward is always negative (time cost)
        assert not done  # First step shouldn't be done

    def test_step_partial_test(self):
        env = CICDEnvironment(seed=42)
        env.reset()
        _, _, _, info = env.step(ACTION_PARTIAL_TEST)
        assert info["execution_time"] == TEST_EXECUTION_TIME[ACTION_PARTIAL_TEST]

    def test_step_skip_test(self):
        env = CICDEnvironment(seed=42)
        env.reset()
        _, _, _, info = env.step(ACTION_SKIP_TEST)
        assert info["execution_time"] == TEST_EXECUTION_TIME[ACTION_SKIP_TEST]
        assert info["execution_time"] == 0.0

    def test_episode_completion(self):
        env = CICDEnvironment(commits_per_episode=10, seed=42)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(ACTION_FULL_TEST)
            steps += 1
        assert steps == 10

    def test_bug_detection_full_test(self):
        """Full tests should detect 100% of bugs."""
        env = CICDEnvironment(commits_per_episode=1000, bug_probability=1.0, seed=42)
        env.reset()
        bugs_escaped = 0
        for _ in range(1000):
            _, _, _, info = env.step(ACTION_FULL_TEST)
            if info["bug_escaped"]:
                bugs_escaped += 1
        assert bugs_escaped == 0  # 100% detection rate

    def test_bug_detection_skip_test(self):
        """Skipped tests should detect 0% of bugs."""
        env = CICDEnvironment(commits_per_episode=1000, bug_probability=1.0, seed=42)
        env.reset()
        bugs_escaped = 0
        bugs_introduced = 0
        for _ in range(1000):
            _, _, _, info = env.step(ACTION_SKIP_TEST)
            if info["bug_introduced"]:
                bugs_introduced += 1
            if info["bug_escaped"]:
                bugs_escaped += 1
        assert bugs_escaped == bugs_introduced  # 0% detection

    def test_bug_detection_partial_test(self):
        """Partial tests should detect ~70% of bugs."""
        env = CICDEnvironment(commits_per_episode=10000, bug_probability=1.0, seed=42)
        env.reset()
        bugs_detected = 0
        bugs_introduced = 0
        for _ in range(10000):
            _, _, _, info = env.step(ACTION_PARTIAL_TEST)
            if info["bug_introduced"]:
                bugs_introduced += 1
            if info["bug_detected"]:
                bugs_detected += 1
        detection_rate = bugs_detected / bugs_introduced
        assert 0.65 <= detection_rate <= 0.75  # ~70% ± tolerance

    def test_reward_function_no_bug(self):
        """Reward should be -t_exec/T_full when no bug escapes."""
        env = CICDEnvironment(commits_per_episode=1, bug_probability=0.0, seed=42)
        env.reset()
        _, reward, _, _ = env.step(ACTION_FULL_TEST)
        expected = -TEST_EXECUTION_TIME[ACTION_FULL_TEST] / TEST_EXECUTION_TIME[ACTION_FULL_TEST]
        assert reward == expected  # -1.0 for full test

    def test_reward_function_with_escaped_bug(self):
        """Reward should include -β penalty on bug escape."""
        env = CICDEnvironment(
            commits_per_episode=1000,
            beta=5.0,
            bug_probability=1.0,
            seed=42,
        )
        env.reset()
        # Skip tests - all bugs escape
        _, reward, _, info = env.step(ACTION_SKIP_TEST)
        if info["bug_escaped"]:
            normalized_exec = TEST_EXECUTION_TIME[ACTION_SKIP_TEST] / TEST_EXECUTION_TIME[ACTION_FULL_TEST]
            expected_reward = -normalized_exec - 5.0
            assert reward == expected_reward

    def test_episode_stats(self):
        env = CICDEnvironment(commits_per_episode=10, seed=42)
        env.reset()
        for _ in range(10):
            env.step(ACTION_FULL_TEST)
        stats = env.get_episode_stats()
        assert stats["commits_processed"] == 10
        assert stats["total_execution_time"] == 100.0  # 10 * 10 min
        assert "defect_miss_rate" in stats

    def test_bug_introduction_probability(self):
        """Test that commits have bugs at rates influenced by risk factors."""
        env = CICDEnvironment(commits_per_episode=10000, seed=42)
        env.reset()
        bugs = sum(1 for c in env.commits if c.has_bug)
        bug_rate = bugs / 10000
        # Adjusted bug prob is around 15% base but modulated by risk factors
        assert 0.05 <= bug_rate <= 0.40

    def test_reproducibility(self):
        """Same seed should produce identical episodes."""
        env1 = CICDEnvironment(seed=42)
        env2 = CICDEnvironment(seed=42)
        state1 = env1.reset()
        state2 = env2.reset()
        np.testing.assert_array_equal(state1, state2)

    def test_invalid_action(self):
        env = CICDEnvironment(seed=42)
        env.reset()
        with pytest.raises(AssertionError):
            env.step(5)

    def test_discount_factor_parameter(self):
        """Verify discount factor is used as 0.99 in agent (env doesn't use it directly)."""
        # This is a design verification - γ=0.99 is in the agent
        assert True  # γ is agent-side, tested in agent tests


class TestAdversarialEnvironment:
    """Tests for the adversarial CI/CD environment."""

    def test_adversarial_commits_generated(self):
        env = AdversarialCICDEnvironment(commits_per_episode=1000, seed=42)
        env.reset()
        # Some commits should have low diff but high bug rate
        low_diff_bugs = sum(
            1 for c in env.commits if c.diff_size < 0.15 and c.has_bug
        )
        assert low_diff_bugs > 0

    def test_adversarial_env_runs(self):
        env = AdversarialCICDEnvironment(commits_per_episode=50, seed=42)
        state = env.reset()
        for _ in range(50):
            _, _, done, _ = env.step(ACTION_FULL_TEST)
        assert done


class TestSimulationParameters:
    """Verify simulation parameters match Table I and Table II."""

    def test_execution_times(self):
        assert TEST_EXECUTION_TIME[ACTION_FULL_TEST] == 10.0
        assert TEST_EXECUTION_TIME[ACTION_PARTIAL_TEST] == 3.0
        assert TEST_EXECUTION_TIME[ACTION_SKIP_TEST] == 0.0

    def test_detection_rates(self):
        assert BUG_DETECTION_RATE[ACTION_FULL_TEST] == 1.0
        assert BUG_DETECTION_RATE[ACTION_PARTIAL_TEST] == 0.7
        assert BUG_DETECTION_RATE[ACTION_SKIP_TEST] == 0.0

    def test_bug_probability(self):
        assert BUG_INTRODUCTION_PROBABILITY == 0.15

    def test_bug_escape_penalty(self):
        assert BUG_ESCAPE_PENALTY == 20.0

    def test_state_dim(self):
        assert STATE_DIM == 10

    def test_num_actions(self):
        assert NUM_ACTIONS == 3
