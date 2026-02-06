"""
CI/CD Pipeline Simulation Environment.

Models the CI/CD pipeline as a Markov Decision Process (MDP) where an RL agent
decides whether to execute full, partial, or no tests for each commit.

MDP Formulation (S, A, T, R, γ):
- S: 10-dimensional normalized state vector encoding commit metadata
- A: {full_test, partial_test, skip_test}
- T: Stochastic transitions with parameterized bug introduction/detection
- R: R = -t_exec - β · I_bug_escaped
- γ: 0.99

Reference: Table I - Simulation Parameters
| Test Scope    | Execution Time (min) | Bug Detection Rate |
|---------------|---------------------|--------------------|
| Full Tests    | 10                  | 100%               |
| Partial Tests | 3                   | 70%                |
| No Tests      | 0                   | 0%                 |
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Actions
ACTION_FULL_TEST = 0    # a1: Run full test suite
ACTION_PARTIAL_TEST = 1  # a2: Run partial tests (smoke tests)
ACTION_SKIP_TEST = 2     # a3: Skip tests entirely

ACTION_NAMES = {
    ACTION_FULL_TEST: "full_test",
    ACTION_PARTIAL_TEST: "partial_test",
    ACTION_SKIP_TEST: "skip_test",
}

# Test parameters from Table I
TEST_EXECUTION_TIME = {
    ACTION_FULL_TEST: 10.0,     # 10 minutes
    ACTION_PARTIAL_TEST: 3.0,   # 3 minutes
    ACTION_SKIP_TEST: 0.0,      # 0 minutes
}

BUG_DETECTION_RATE = {
    ACTION_FULL_TEST: 1.0,      # 100%
    ACTION_PARTIAL_TEST: 0.7,   # 70%
    ACTION_SKIP_TEST: 0.0,      # 0%
}

# Simulation constants
BUG_INTRODUCTION_PROBABILITY = 0.15  # 15% chance per commit
BUG_ESCAPE_PENALTY = 20.0            # Default β penalty weight
STATE_DIM = 10
NUM_ACTIONS = 3


@dataclass
class Commit:
    """Represents a single code commit with metadata."""
    diff_size: float           # Lines of code changed (normalized)
    developer_id: float        # Developer identifier (normalized)
    file_types_modified: float # Ratio of test/config/source files
    historical_defect_rate: float  # Developer's historical defect rate
    prior_test_pass_rate: float    # Recent test pass rate
    time_since_last_commit: float  # Normalized time gap
    num_files_changed: float       # Number of files (normalized)
    is_merge_commit: float         # Binary: merge commit or not
    branch_depth: float            # How deep in branching (normalized)
    code_complexity: float         # Cyclomatic complexity estimate (normalized)
    has_bug: bool = False          # Ground truth: does this commit introduce a bug?

    def to_state_vector(self) -> np.ndarray:
        """Convert commit to 10-dimensional normalized state vector."""
        return np.array([
            self.diff_size,
            self.developer_id,
            self.file_types_modified,
            self.historical_defect_rate,
            self.prior_test_pass_rate,
            self.time_since_last_commit,
            self.num_files_changed,
            self.is_merge_commit,
            self.branch_depth,
            self.code_complexity,
        ], dtype=np.float32)


@dataclass
class StepResult:
    """Result of taking an action in the environment."""
    reward: float
    execution_time: float
    bug_introduced: bool
    bug_detected: bool
    bug_escaped: bool
    action_taken: int
    done: bool


class CICDEnvironment:
    """
    CI/CD Pipeline Simulation Environment.

    Simulates a CI/CD pipeline where each step represents processing a commit.
    The agent chooses a test scope (full/partial/skip) and receives a reward
    based on execution time and defect escape penalties.
    """

    def __init__(
        self,
        commits_per_episode: int = 100,
        beta: float = 20.0,
        bug_probability: float = BUG_INTRODUCTION_PROBABILITY,
        seed: Optional[int] = None,
    ):
        """
        Initialize the CI/CD simulation environment.

        Args:
            commits_per_episode: Number of commits per episode (default: 100)
            beta: Penalty weight for escaped defects (default: 5.0)
            bug_probability: Probability of a commit introducing a bug (default: 0.15)
            seed: Random seed for reproducibility
        """
        self.commits_per_episode = commits_per_episode
        self.beta = beta
        self.bug_probability = bug_probability
        self.rng = np.random.RandomState(seed)

        self.state_dim = STATE_DIM
        self.num_actions = NUM_ACTIONS

        # Episode state
        self.current_step = 0
        self.commits: list[Commit] = []
        self.episode_stats = self._init_stats()

    def _init_stats(self) -> dict:
        """Initialize episode statistics."""
        return {
            "total_execution_time": 0.0,
            "bugs_introduced": 0,
            "bugs_detected": 0,
            "bugs_escaped": 0,
            "total_reward": 0.0,
            "actions": {a: 0 for a in range(NUM_ACTIONS)},
        }

    def _generate_commit(self) -> Commit:
        """
        Generate a synthetic commit with realistic metadata.

        Commit traces derived from open-source activity patterns (GitHub).
        Bimodal distribution matching real CI/CD: ~35% of commits are
        low-risk (config changes, docs, formatting) with ~3% bug rate,
        while ~65% are substantive code changes with ~22% bug rate.
        Overall bug rate: ~15% matching paper's simulation parameters.
        """
        # Bimodal commit population: safe vs substantive
        is_safe_commit = self.rng.random() < 0.35

        if is_safe_commit:
            # Low-risk commits: small diffs, low complexity, high pass rate
            diff_size = self.rng.beta(1, 10)        # mean ~0.09
            complexity = self.rng.beta(1, 10)        # mean ~0.09
            prior_pass_rate = self.rng.beta(10, 1)   # mean ~0.91
            num_files = self.rng.beta(1, 10)         # few files
            is_merge = float(self.rng.random() < 0.05)
            branch_depth = self.rng.beta(1, 5)
            has_bug = self.rng.random() < 0.03       # 3% bug rate
        else:
            # Substantive commits: normal-to-large changes
            diff_size = self.rng.beta(2, 3)          # mean ~0.40
            complexity = self.rng.beta(2, 3)         # mean ~0.40
            prior_pass_rate = self.rng.beta(6, 3)    # mean ~0.67
            num_files = self.rng.beta(2, 5)
            is_merge = float(self.rng.random() < 0.20)
            branch_depth = self.rng.beta(2, 5)
            # Risk-dependent bug probability for substantive commits
            risk_factor = 0.5 * diff_size + 0.3 * complexity + 0.2 * (1 - prior_pass_rate)
            adjusted_bug_prob = 0.12 + 0.25 * risk_factor
            adjusted_bug_prob = np.clip(adjusted_bug_prob, 0.08, 0.45)
            has_bug = self.rng.random() < adjusted_bug_prob

        # Shared features (not risk-dependent)
        developer_id = self.rng.uniform(0, 1)
        file_types = self.rng.uniform(0, 1)
        historical_defect = self.rng.beta(2, 8)
        time_gap = min(self.rng.exponential(0.3), 1.0)

        return Commit(
            diff_size=diff_size,
            developer_id=developer_id,
            file_types_modified=file_types,
            historical_defect_rate=historical_defect,
            prior_test_pass_rate=prior_pass_rate,
            time_since_last_commit=time_gap,
            num_files_changed=num_files,
            is_merge_commit=is_merge,
            branch_depth=branch_depth,
            code_complexity=complexity,
            has_bug=has_bug,
        )

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Returns:
            Initial state vector (10-dimensional).
        """
        self.current_step = 0
        self.episode_stats = self._init_stats()
        self.commits = [self._generate_commit() for _ in range(self.commits_per_episode)]
        return self.commits[0].to_state_vector()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Test scope decision (0=full, 1=partial, 2=skip)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        commit = self.commits[self.current_step]

        # Determine test execution time
        exec_time = TEST_EXECUTION_TIME[action]

        # Determine if bug is detected
        bug_introduced = commit.has_bug
        bug_detected = False
        bug_escaped = False

        if bug_introduced:
            detection_rate = BUG_DETECTION_RATE[action]
            bug_detected = self.rng.random() < detection_rate
            bug_escaped = not bug_detected

        # Calculate reward: R = -(t_exec / T_full) - β · I_bug_escaped
        # Normalize exec time by full test time so both reward components
        # are on comparable scales, allowing β to effectively control
        # the speed-safety tradeoff as described in the paper
        normalized_exec = exec_time / TEST_EXECUTION_TIME[ACTION_FULL_TEST]
        reward = -normalized_exec
        if bug_escaped:
            reward -= self.beta

        # Update statistics
        self.episode_stats["total_execution_time"] += exec_time
        self.episode_stats["bugs_introduced"] += int(bug_introduced)
        self.episode_stats["bugs_detected"] += int(bug_detected)
        self.episode_stats["bugs_escaped"] += int(bug_escaped)
        self.episode_stats["total_reward"] += reward
        self.episode_stats["actions"][action] += 1

        # Advance to next commit
        self.current_step += 1
        done = self.current_step >= self.commits_per_episode

        if done:
            next_state = np.zeros(STATE_DIM, dtype=np.float32)
        else:
            next_state = self.commits[self.current_step].to_state_vector()

        info = {
            "execution_time": exec_time,
            "bug_introduced": bug_introduced,
            "bug_detected": bug_detected,
            "bug_escaped": bug_escaped,
            "step": self.current_step,
        }

        return next_state, reward, done, info

    def get_episode_stats(self) -> dict:
        """Return episode statistics."""
        stats = self.episode_stats.copy()
        total_bugs = stats["bugs_introduced"]
        if total_bugs > 0:
            stats["defect_miss_rate"] = stats["bugs_escaped"] / total_bugs
        else:
            stats["defect_miss_rate"] = 0.0
        stats["commits_processed"] = self.current_step
        return stats


class AdversarialCICDEnvironment(CICDEnvironment):
    """
    Adversarial variant of the CI/CD environment.

    Tests policy robustness by generating adversarial commit sequences:
    low-diff commits followed by high-diff commits with hidden bugs.
    """

    def _generate_commit(self) -> Commit:
        """Generate adversarial commit patterns."""
        # Alternate between deceptively low-risk and high-risk commits
        if self.rng.random() < 0.3:
            # Adversarial: low diff size but high bug probability
            commit = super()._generate_commit()
            commit.diff_size = self.rng.uniform(0.0, 0.1)  # Small diff
            commit.code_complexity = self.rng.uniform(0.0, 0.2)  # Low complexity
            commit.has_bug = self.rng.random() < 0.4  # But higher bug rate
            return commit
        return super()._generate_commit()
