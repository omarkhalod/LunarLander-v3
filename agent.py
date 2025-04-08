import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from network import Network
from replay_memory import ReplayMemory

class Agent():
    """DQN Agent that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, learning_rate=0.0005, 
                 replay_buffer_size=int(1e5), interpolation_parameter=2e-3):
        """Initialize an Agent object."""
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.interpolation_parameter = interpolation_parameter

        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.local_qnetwork.parameters(), lr=learning_rate)

        self.memory = ReplayMemory(replay_buffer_size)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done, minibatch_size=128, discount_factor=0.99):
        """Save experience in replay memory, and use random sample to learn."""

        self.memory.push((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilion=0.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.local_qnetwork.eval()

        with torch.inference_mode():
            action_values = self.local_qnetwork(state)

        self.local_qnetwork.train()

        # Epsilon-Greedy Action Selection
        if random.random() > epsilion:
            return np.argmax(action_values.cpu().data.numpy())  # Exploit
        else:
            return random.choice(np.arange(self.action_size))   # Explore

    def learn(self, experiences, discount_factor):
        """Update value parameters using given batch of experience tuples."""
        state, next_state, action, reward, done = experiences

        self.target_qnetwork.eval()

        # Compute Q targets for current states
        next_q_targets = self.target_qnetwork(
            next_state).detach().max(1)[0].unsqueeze(1)

        # calculate target Q values using Bellman equation
        q_targets = reward + (discount_factor * next_q_targets * (1 - done))
        q_expected = self.local_qnetwork(state).gather(1, action)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_qnetwork,
                         self.target_qnetwork, self.interpolation_parameter)

    def soft_update(self, local_model, target_modal, interpolation_parameter):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_modal.parameters(), local_model.parameters()):
            target_param.data.copy_(
                interpolation_parameter * local_param.data +
                (1.0 - interpolation_parameter) * target_param.data
            )
