"""Tests for DQN agent."""

import numpy as np
import torch
import pytest
import tempfile
import os

from src.agent import DQNAgent, QNetwork, ReplayBuffer
from src.environment import STATE_DIM, NUM_ACTIONS


class TestQNetwork:
    """Tests for the Q-Network architecture."""

    def test_architecture(self):
        """Verify 3-layer feedforward with correct dimensions."""
        net = QNetwork(STATE_DIM, NUM_ACTIONS)
        layers = list(net.network.children())
        # Should have: Linear, ReLU, Linear, ReLU, Linear = 5 layers
        assert len(layers) == 5
        assert isinstance(layers[0], torch.nn.Linear)
        assert isinstance(layers[1], torch.nn.ReLU)
        assert isinstance(layers[2], torch.nn.Linear)
        assert isinstance(layers[3], torch.nn.ReLU)
        assert isinstance(layers[4], torch.nn.Linear)

    def test_input_output_dimensions(self):
        """Input: 10-dim state, Output: 3-dim Q-values."""
        net = QNetwork(STATE_DIM, NUM_ACTIONS)
        x = torch.randn(1, STATE_DIM)
        out = net(x)
        assert out.shape == (1, NUM_ACTIONS)

    def test_batch_forward(self):
        net = QNetwork(STATE_DIM, NUM_ACTIONS)
        x = torch.randn(32, STATE_DIM)
        out = net(x)
        assert out.shape == (32, NUM_ACTIONS)


class TestReplayBuffer:
    """Tests for the experience replay buffer."""

    def test_capacity(self):
        """Buffer capacity should be 10,000 (Section IV-E)."""
        buffer = ReplayBuffer(capacity=10000)
        assert buffer.buffer.maxlen == 10000

    def test_push_and_sample(self):
        buffer = ReplayBuffer(capacity=100)
        for i in range(50):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            next_state = np.random.randn(STATE_DIM).astype(np.float32)
            buffer.push(state, i % 3, -5.0, next_state, False)

        assert len(buffer) == 50
        states, actions, rewards, next_states, dones = buffer.sample(16)
        assert states.shape == (16, STATE_DIM)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, STATE_DIM)
        assert dones.shape == (16,)

    def test_overflow(self):
        """Buffer should discard oldest when full."""
        buffer = ReplayBuffer(capacity=10)
        for i in range(20):
            buffer.push(np.zeros(STATE_DIM), 0, 0.0, np.zeros(STATE_DIM), False)
        assert len(buffer) == 10


class TestDQNAgent:
    """Tests for the DQN agent."""

    def test_initialization(self):
        agent = DQNAgent(seed=42, device="cpu")
        assert agent.state_dim == STATE_DIM
        assert agent.num_actions == NUM_ACTIONS
        assert agent.gamma == 0.99  # Paper: γ = 0.99
        assert agent.epsilon == 1.0  # Start at 1.0
        assert agent.batch_size == 64  # Paper: minibatch 64

    def test_select_action_returns_valid(self):
        agent = DQNAgent(seed=42, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = agent.select_action(state, training=False)
        assert 0 <= action < NUM_ACTIONS

    def test_epsilon_greedy_exploration(self):
        """With ε=1.0, all actions should be random."""
        agent = DQNAgent(seed=42, device="cpu")
        agent.epsilon = 1.0
        actions = set()
        for _ in range(1000):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            action = agent.select_action(state, training=True)
            actions.add(action)
        assert len(actions) == NUM_ACTIONS  # All actions explored

    def test_greedy_action_selection(self):
        """With training=False, should always select max Q-value action."""
        agent = DQNAgent(seed=42, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)
        # Should be deterministic
        actions = [agent.select_action(state, training=False) for _ in range(10)]
        assert len(set(actions)) == 1

    def test_epsilon_decay(self):
        """ε should decay from 1.0 to 0.1 over training."""
        agent = DQNAgent(
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_episodes=1500,
            device="cpu",
        )
        agent.decay_epsilon(0)
        assert agent.epsilon == 1.0

        agent.decay_epsilon(750)
        assert 0.4 < agent.epsilon < 0.7

        agent.decay_epsilon(1500)
        assert abs(agent.epsilon - 0.1) < 0.01

        agent.decay_epsilon(3000)
        assert abs(agent.epsilon - 0.1) < 0.01

    def test_store_transition(self):
        agent = DQNAgent(seed=42, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        agent.store_transition(state, 0, -10.0, next_state, False)
        assert len(agent.replay_buffer) == 1

    def test_update_requires_sufficient_buffer(self):
        agent = DQNAgent(seed=42, device="cpu")
        loss = agent.update()
        assert loss is None  # Buffer too small

    def test_update_with_data(self):
        agent = DQNAgent(seed=42, device="cpu", batch_size=16)
        for _ in range(32):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            next_state = np.random.randn(STATE_DIM).astype(np.float32)
            agent.store_transition(state, np.random.randint(3), -5.0, next_state, False)

        loss = agent.update()
        assert loss is not None
        assert loss > 0

    def test_target_network_update(self):
        agent = DQNAgent(seed=42, device="cpu")
        # Modify policy net
        state = np.random.randn(STATE_DIM).astype(np.float32)
        for _ in range(100):
            agent.store_transition(state, 0, -1.0, state, False)
        agent.update()

        # Target and policy should differ
        policy_params = list(agent.policy_net.parameters())
        target_params = list(agent.target_net.parameters())

        # Update target
        agent.update_target_network()

        # Now they should match
        for pp, tp in zip(policy_params, target_params):
            torch.testing.assert_close(pp, tp)

    def test_save_and_load(self):
        agent = DQNAgent(seed=42, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action1 = agent.select_action(state, training=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            new_agent = DQNAgent(seed=99, device="cpu")
            new_agent.load(path)
            action2 = new_agent.select_action(state, training=False)
            assert action1 == action2
        finally:
            os.unlink(path)

    def test_get_q_values(self):
        agent = DQNAgent(seed=42, device="cpu")
        state = np.random.randn(STATE_DIM).astype(np.float32)
        q_values = agent.get_q_values(state)
        assert q_values.shape == (NUM_ACTIONS,)

    def test_learning_rate(self):
        """Verify Adam learning rate is 1e-4."""
        agent = DQNAgent(learning_rate=1e-4, device="cpu")
        for param_group in agent.optimizer.param_groups:
            assert param_group["lr"] == 1e-4
