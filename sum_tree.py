import numpy as np


class SumTree:
    def __init__(self, capacity):
        # Initialize the tree with zeros
        # Tree size is 2*capacity-1 (internal nodes + leaf nodes)
        self.tree = np.zeros(2 * capacity - 1)

        # Store data separately
        self.data = np.zeros(capacity, dtype=object)

        # Current position for writing
        self.write = 0

        # Maximum capacity of leaf nodes
        self.capacity = capacity

        # Current number of elements
        self.n_entries = 0

    def _propagate(self, idx, change):
        # Update parent nodes from a leaf change
        parent = (idx - 1) // 2
        self.tree[parent] += change

        # Continue propagating up if not at root
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        # Find sample on leaf node
        left = 2 * idx + 1
        right = left + 1

        # If we're at a leaf, return it
        if left >= len(self.tree):
            return idx

        # Traverse left or right based on the value s
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        # Total priority sum (root node)
        return self.tree[0]

    def add(self, priority, data):
        # Add a new entry to the data structure

        self.data[self.write] = data

        # self.capacity - 1 = first leaf node from the left
        # + self.write = leaf node index

        idx = self.write + self.capacity - 1

        # handles the back propgation of the tree and adding thhe new priority
        self.update(idx, priority)

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0
            self.n_entries = self.capacity
        else:
            self.n_entries += 1

    def update(self, idx, priority):
        if idx < self.capacity - 1 or idx >= 2 * self.capacity - 1:
            raise IndexError("Invalid tree index")
        # change is the difference between the new and old priority
        # if there is no old node the equatiopn is priority - 0
        change = priority - self.tree[idx]

        # Update the leaf node
        self.tree[idx] = priority

        # Propagate the change up to the root
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)

        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]
