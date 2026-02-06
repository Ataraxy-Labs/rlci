"""
Deep Q-Network (DQN) Agent for CI/CD Pipeline Optimization.

Architecture:
- 3-layer feedforward neural network with ReLU activations
- 10-dimensional normalized state vector input
- 3-dimensional output (Q-values for each action)
- ε-greedy exploration: decay from 1.0 to 0.1
- Replay buffer: 10,000 transitions, minibatch size 64
- Optimizer: Adam with learning rate 1e-4
- Discount factor γ = 0.99
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional
from pathlib import Path

from .environment import STATE_DIM, NUM_ACTIONS


class QNetwork(nn.Module):
    """
    3-layer feedforward neural network for Q-value estimation.

    Architecture from paper:
    - Input: 10-dimensional state vector
    - Hidden layers with ReLU activations
    - Output: Q-values for 3 actions
    """

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Capacity: 10,000 transitions (from Section IV-E).
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for CI/CD pipeline optimization.

    Implements the DQN algorithm with experience replay and ε-greedy exploration.
    The agent learns to select optimal test scopes (full/partial/skip) for each
    commit based on the commit's metadata state vector.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_episodes: int = 1500,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim: Dimension of state vector (default: 10)
            num_actions: Number of possible actions (default: 3)
            learning_rate: Adam learning rate (default: 1e-4)
            gamma: Discount factor (default: 0.99)
            epsilon_start: Initial exploration rate (default: 1.0)
            epsilon_end: Final exploration rate (default: 0.1)
            epsilon_decay_episodes: Episodes over which to decay ε (default: 1500)
            replay_buffer_size: Replay buffer capacity (default: 10000)
            batch_size: Training minibatch size (default: 64)
            target_update_freq: Episodes between target network updates
            seed: Random seed for reproducibility
            device: Compute device ('cpu', 'cuda', or 'mps')
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        # Networks
        self.policy_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Training stats
        self.training_losses = []
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        During training, uses ε-greedy exploration.
        During evaluation, always selects the greedy action.

        Args:
            state: Current state vector
            training: Whether we're in training mode

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Perform one step of DQN training.

        Samples a minibatch from the replay buffer and updates the policy network
        using the standard DQN loss:
            L = E[(r + γ max_a' Q_target(s', a') - Q(s, a))²]

        Returns:
            Training loss value, or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # MSE loss
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def decay_epsilon(self, episode: int):
        """
        Decay exploration rate linearly from ε_start to ε_end.

        ε-greedy exploration decays from 1.0 to 0.1 across training episodes.
        """
        fraction = min(1.0, episode / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Save model weights and training state."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_losses": self.training_losses,
            "episode_rewards": self.episode_rewards,
        }, path)

    def load(self, path: str):
        """Load model weights and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_losses = checkpoint["training_losses"]
        self.episode_rewards = checkpoint["episode_rewards"]
