from Visualization import plot_scores, show_video_of_model
import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
from priority_replay import PriorityReplayMemory


class Network(nn.Module):
    """Deep Q-Network architecture definition."""

    def __init__(self, state_size, action_size, seed=42):
        # Calling the init (constructor) of the super class for nn.module
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Forward pass through the network."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


# Setting up the environment
env = gym.make("LunarLander-v3")
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("State shape: ", state_shape)
print("State size: ", state_size)
print("number of actions: ", action_size)


# Hyperparameters
learning_rate = 6e-4
minibatch_size = 128
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 2e-3

# seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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

        return states, next_states, actions, rewards, dones


class Agent():
    """DQN Agent that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size):
        """Initialize an Agent object."""
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork.load_state_dict(
            self.local_qnetwork.state_dict())

        self.optimizer = optim.Adam(
            self.local_qnetwork.parameters(), lr=learning_rate)

        self.memory = ReplayMemory(replay_buffer_size)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample to learn."""

        self.memory.push((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.local_qnetwork.eval()

        with torch.inference_mode():
            action_values = self.local_qnetwork(state)

        self.local_qnetwork.train()

        # Epsilon-Greedy Action Selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())  # Exploit
        else:
            return random.choice(np.arange(self.action_size))   # Explore

    def learn(self, experiences, discount_factor):
        """Update value parameters using given batch of experience tuples."""
        state, next_state, action, reward, done = experiences

        self.target_qnetwork.eval()

        # Compute Q targets for current states
        # 1. Forward propagation to get predicted values (returns matrix of shape [batch_size, action_size])
        # 2. .detach() ensures we don't update target network weights
        # 3. max(1)[0] gets maximum Q-value for each batch sample
        # 4. unsqueeze(1) reshapes from [batch_size] to [batch_size, 1]
        next_q_targets = self.target_qnetwork(
            next_state).detach().max(1)[0].unsqueeze(1)

        # calculate target Q values using Bellman equation
        q_targets = reward + (discount_factor * next_q_targets * (1 - done))
        q_expected = self.local_qnetwork(state).gather(1, action)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(
        #     self.local_qnetwork.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.soft_update(self.local_qnetwork,
                         self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_model, target_modal, interpolation_parameter):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_modal.parameters(), local_model.parameters()):
            target_param.data.copy_(
                interpolation_parameter * local_param.data +
                (1.0 - interpolation_parameter) * target_param.data
            )


class AgentPR:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_qnetwork = Network(state_dim, action_dim).to(self.device)
        self.target_qnetwork = Network(state_dim, action_dim).to(self.device)
        self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=5e-4)
        self.memory = PriorityReplayMemory(100000, alpha=0.8, epsilon=0.01)
        self.gamma = 0.99
        self.t_step = 0

    def act(self, state, eps=0.9):
        if random.random() < eps:
            return random.randrange(self.local_qnetwork.fc3.out_features)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            q = self.local_qnetwork(state)
        self.local_qnetwork.train()
        return q.argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and self.memory.tree.n_entries > 128:
            experiences = self.memory.sample(128)
            self.learn(experiences)

    def learn(self, experiences):
        (states, actions, rewards, next_states,
         dones), idxs, is_weights = experiences
        # compute target
        next_q = self.target_qnetwork(next_states).detach().max(1)[
            0].unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (1 - dones)
        # current Q
        curr_q = self.local_qnetwork(states).gather(1, actions)
        # compute loss
        td_errors = F.mse_loss(curr_q, target_q, reduction='none')
        loss = (is_weights * td_errors).mean()
        # optimize
        self.optimizer.zero_grad()

        loss.backward()
        # to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(
        #     self.local_qnetwork.parameters(), max_norm=1.0)
        self.optimizer.step()

        # update priorities
        errors = td_errors.detach().cpu().squeeze().tolist()
        self.memory.update_priorities(idxs, errors)
        # soft-update target
        for t_param, l_param in zip(self.target_qnetwork.parameters(), self.local_qnetwork.parameters()):
            t_param.data.copy_(2e-3*l_param.data + (1-2e-3)*t_param.data)


def trainging_loop(agent, number_of_episodes, max_number_of_timesteps_per_episode, epsilon_ending, epsilon_decay):
    eps = 1.0
    scores_window = deque(maxlen=100)

    for episode in range(1, number_of_episodes + 1):
        state, _ = env.reset(seed=SEED)
        done = False
        score = 0
        t = 0
        while not done and t < max_number_of_timesteps_per_episode:
            action = agent.act(state, eps)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            score += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            t += 1

        scores_window.append(score)
        eps = max(epsilon_ending, epsilon_decay * eps)
        print(
            f"\rEpisode {episode:3d}	Average(100): {np.mean(scores_window):.2f}", end="")
        if np.mean(scores_window) >= 200.0 and episode >= 100:
            print(f"\nSolved in {episode - 100} episodes!")
            os.makedirs('Models', exist_ok=True)
            torch.save(agent.local_qnetwork.state_dict(),
                       f'Models/checkpoint{episode - 100}.pth')
            break


# Training parameters
number_of_episodes = 2000
max_number_of_timesteps_per_episode = 1000
epsilon_starting = 1.0
epsilon_ending = 0.01
epsilon_decay = 0.995

agent = Agent(state_size, action_size)

eps = epsilon_starting


trainging_loop(agent, number_of_episodes, max_number_of_timesteps_per_episode,
               epsilon_ending, epsilon_decay)

agentPR = AgentPR(state_size, action_size)

trainging_loop(agentPR, number_of_episodes, max_number_of_timesteps_per_episode,
               epsilon_ending, epsilon_decay)

show_video_of_model(agentPR, 'LunarLander-v3', "video_prioritized")

show_video_of_model(agent, 'LunarLander-v3', "video_standard")
