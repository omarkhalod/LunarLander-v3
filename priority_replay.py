import random
import numpy as np
import torch
from sum_tree import SumTree


class PriorityReplayMemory:
    def __init__(self, capacity, epsilon=0.01, alpha=0.4, beta_start=0.4, beta_frames=10000):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def push(self, transition):
        p = (np.max(self.tree.tree[-self.tree.capacity:]
                    ) + self.epsilon) ** self.alpha
        if p == 0:
            p = (1.0 + self.epsilon) ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta_start + self.frame *
                        (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            s = min(s, self.tree.total() - 1e-5)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        # compute IS weights
        probs = np.array(priorities) / self.tree.total()
        N = self.tree.n_entries
        weights = (N * probs) ** (-self.beta)
        weights /= weights.max()
        # convert to torch
        is_weights = torch.tensor(
            weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        # extract transitions
        states = torch.tensor(
            np.vstack([t[0] for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            np.vstack([t[1] for t in batch]), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(
            np.vstack([t[2] for t in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            np.vstack([t[3] for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.vstack([t[4] for t in batch]).astype(
            np.uint8), dtype=torch.float32, device=self.device)
        return (states, actions, rewards, next_states, dones), idxs, is_weights

    def update_priorities(self, idxs, errors):
        for idx, err in zip(idxs, errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
