import random
import numpy as np
import torch
from sum_tree import SumTree

class PriorityReplayMemory(object):
    def __init__(self, capacity, epsilon, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.Pmemory = SumTree(capacity)
        
    def push(self, event):
        # When adding a new experience, use maximum priority
        # This ensures new experiences are sampled at least once
        max_priority = 1.0 if self.Pmemory.tree[0] == 0 else self.Pmemory.total()
        
        # Convert to priority using alpha
        priority = (max_priority + self.epsilon) ** self.alpha
        
        # Add to memory
        self.Pmemory.add(priority, event)
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities"""
        batch = []
        indices = []
        priorities = []

        # Calculate segment size for sampling
        segment = self.Pmemory.total() / batch_size

        # Increment beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Sample from each segment
        for i in range(batch_size):
            # Calculate segment bounds
            a = segment * i
            b = segment * (i + 1)

            # Sample uniformly from segment
            s = random.uniform(a, b)

            # Retrieve experience and its priority
            idx, priority, data = self.Pmemory.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        # P(i) = p_i^α / sum(p_k^α)
        sampling_probabilities = np.array(priorities) / self.Pmemory.total()

        # Importance sampling weight: (1/N * 1/P(i))^β
        is_weights = np.power(self.Pmemory.n_entries * sampling_probabilities, -self.beta)

        # Normalize weights to have max weight = 1
        is_weights /= is_weights.max()



         # Extract and convert experiences to tensors
        states = torch.from_numpy(np.vstack(
            [e[0] for e in batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in batch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e[2] for e in batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e[3] for e in batch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None]).astype(
            np.uint8)).float().to(self.device)

        batch = states, next_states, actions, rewards, dones
        
        return batch, indices, is_weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            # Convert TD error to priority
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        
            # Update priority in tree
            self.Pmemory.update(idx, priority)
        