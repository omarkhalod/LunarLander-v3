import random
import torch
import numpy as np
from priority_replay import PriorityReplayMemory  # adjust import
from linear_replay import ReplayMemory        # your linear replay


class MixedReplayMemory:
    """
    Mixes PriorityReplayMemory with uniform ReplayMemory.
      - capacity: total buffer size
      - eta: fraction of each batch drawn from prioritized replay (0.0–1.0)
      - **per_kwargs: passed through to PriorityReplayMemory
      - **unif_kwargs: passed through to ReplayMemory (just capacity)
    """

    def __init__(self, capacity, eta=0.8, **per_kwargs):
        self.capacity = capacity
        self.eta = eta

        # Prioritized buffer
        self.per = PriorityReplayMemory(capacity, **per_kwargs)
        # Uniform buffer
        self.unif = ReplayMemory(capacity)

        # device for tensors
        self.device = self.per.device

    def push(self, experience):
        # Add to both buffers
        self.per.push(experience)
        self.unif.push(experience)

    def sample(self, batch_size):
        # number of PER vs. uniform samples
        k_per = int(self.eta * batch_size)
        k_unif = batch_size - k_per

        w_u = torch.ones((k_unif, 1), device=self.device)

        # 1) PER sample
        if (self.eta == 1):
            return self.per.sample(batch_size)

        # 2) Uniform sample
        # ReplayMemory.sample returns (states, next_states, actions, rewards, dones)
        if (self.eta == 0):
            return self.unif.sample(batch_size), [], w_u

        # 3) Mixed sample

        # 1) PER sample: returns (states, actions, rewards, next_states, dones)

        (s_p, a_p, r_p, ns_p, d_p), idxs_p, w_p = self.per.sample(k_per)

        # 2) Uniform sample: returns (states, next_states, actions, rewards, dones)
        s_u, a_u, r_u, ns_u, d_u = self.unif.sample(k_unif)

        # importance weights = 1 for uniform
        w_u = torch.ones((k_unif, 1), device=self.device)

        # 3) Concatenate
        states = torch.cat([s_p,      s_u],      dim=0)
        next_states = torch.cat([ns_p,     ns_u],     dim=0)
        actions = torch.cat([a_p,      a_u],      dim=0)
        rewards = torch.cat([r_p,      r_u],      dim=0)
        dones = torch.cat([d_p,      d_u],      dim=0)
        is_weights = torch.cat([w_p,      w_u],      dim=0)

        # 4) Shuffle
        perm = torch.randperm(batch_size, device=self.device)
        batch = (
            states[perm],
            actions[perm],
            rewards[perm],
            next_states[perm],
            dones[perm]
        )
        # reorder weights—but PER idxs and errs refer only to the first k_per before shuffling
        is_weights = is_weights[perm]

        return batch, idxs_p, is_weights

    def update_priorities(self, idxs, errors):
        # only update the PER buffer
        self.per.update_priorities(idxs, errors)
