import random
import numpy as np
import torch
from collections import deque


class ReplayMemory(object):
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, capacity):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, event):
        """Add a new experience to memory."""
        self.memory.append(event)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        # Extract and convert experiences to tensors
        states = torch.from_numpy(np.vstack(
            [e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(
            np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones
