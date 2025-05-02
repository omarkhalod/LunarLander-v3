from Visualization import plot_comparison, show_video_of_model
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
from linear_replay import ReplayMemory
from MixedReplay import MixedReplayMemory


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


# === TUNED HYPERPARAMETER SECTION ===
LEARNING_RATE = 1e-3               # increased LR for faster convergence
MINIBATCH_SIZE = 128               # smaller batch for more frequent updates
DISCOUNT_FACTOR = 0.995            # slightly higher gamma for longer-term rewards
REPLAY_BUFFER_SIZE = int(1e5)      # larger buffer to increase coverage
INTERPOLATION_PARAMETER = 5e-3     # faster target-network blending

ETA = 0.9                          # 90% PER, 10% uniform
PER_EPSILON = 1e-4                 # very small floor to avoid zero priority
PER_ALPHA = 0.7                    # stronger prioritization
PER_BETA_START = 0.5               # start IS correction higher
PER_BETA_FRAMES = 50000            # anneal beta over more frames

EPSILON_STARTING = 1.0             # initial exploration rate
EPSILON_ENDING = 0.01             # minimum exploration rate
EPSILON_DECAY = 0.99              # slower decay for more exploration

NUMBER_OF_EPISODES = 3000          # more episodes for deeper learning
MAX_NUMBER_OF_TIMESTEPS_PER_EPISODE = 1000
UPDATE_EVERY = 4                   # learn at every step

SEED = 42                          # reproducibility

# set seeds and initialize environment
env = gym.make("LunarLander-v3")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
env.reset(seed=SEED)


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
            self.local_qnetwork.parameters(), lr=LEARNING_RATE)

        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample to learn."""

        self.memory.push((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > MINIBATCH_SIZE:
                experiences = self.memory.sample(MINIBATCH_SIZE)
                self.learn(experiences, DISCOUNT_FACTOR)

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
                         self.target_qnetwork, INTERPOLATION_PARAMETER)

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
        self.optimizer = optim.Adam(
            self.local_qnetwork.parameters(), LEARNING_RATE)
        self.memory = PriorityReplayMemory(
            REPLAY_BUFFER_SIZE, alpha=0.8, epsilon=0.01)
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
        target_q = rewards + DISCOUNT_FACTOR * next_q * (1 - dones)
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
        if (ETA > 0):
            self.memory.update_priorities(idxs, errors)
        # soft-update target network
        self.soft_update(self.local_qnetwork,
                         self.target_qnetwork, INTERPOLATION_PARAMETER)

    def soft_update(self, local_model, target_modal, interpolation_parameter):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_modal.parameters(), local_model.parameters()):
            target_param.data.copy_(
                interpolation_parameter * local_param.data +
                (1.0 - interpolation_parameter) * target_param.data
            )


class AgentMR:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_qnetwork = Network(state_dim, action_dim).to(self.device)
        self.target_qnetwork = Network(state_dim, action_dim).to(self.device)
        self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())
        self.optimizer = optim.Adam(
            self.local_qnetwork.parameters(), lr=LEARNING_RATE)
        self.memory = MixedReplayMemory(
            REPLAY_BUFFER_SIZE,
            eta=ETA,
            epsilon=PER_EPSILON,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES
        )
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
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and self.memory.per.tree.n_entries > MINIBATCH_SIZE:
            experiences = self.memory.sample(MINIBATCH_SIZE)
            self.learn(experiences)

    def learn(self, experiences):
        (states, actions, rewards, next_states,
         dones), idxs, is_weights = experiences
        # compute targets
        next_q = self.target_qnetwork(next_states).detach().max(1)[
            0].unsqueeze(1)
        q_targets = rewards + self.gamma * next_q * (1 - dones)
        # expected
        q_expected = self.local_qnetwork(states).gather(1, actions)
        # loss with IS weights
        td_errors = (q_expected - q_targets).detach().cpu().squeeze().numpy()
        loss = (is_weights * F.mse_loss(q_expected,
                q_targets, reduction='none')).mean()
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.local_qnetwork.parameters(), max_norm=1.0)
        self.optimizer.step()
        # update priorities
        self.memory.update_priorities(idxs, td_errors)
        # soft-update target
        self.soft_update(self.local_qnetwork,
                         self.target_qnetwork, INTERPOLATION_PARAMETER)

    def soft_update(self, local_model, target_modal, interpolation_parameter):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_modal.parameters(), local_model.parameters()):
            target_param.data.copy_(
                interpolation_parameter * local_param.data +
                (1.0 - interpolation_parameter) * target_param.data
            )


def trainging_loop(agent):
    eps = EPSILON_STARTING
    replay_type = 'PR' if ETA == 1 else 'LR' if ETA == 0 else 'MR'
    scores_avr = np.array([-200.])
    scores_window = deque(maxlen=100)
    for ep in range(1, NUMBER_OF_EPISODES + 1):
        state, _ = env.reset()
        done, score, t = False, 0, 0
        while not done and t < MAX_NUMBER_OF_TIMESTEPS_PER_EPISODE:
            action = agent.act(state, eps)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            t += 1
        scores_window.append(score)
        eps = max(EPSILON_ENDING, EPSILON_DECAY * eps)
        print(
            f"\rEpisode {ep:4d} Avg(100): {np.mean(scores_window):.2f}", end="")
        if (ep % 10 == 0):
            scores_avr = np.append(scores_avr, np.mean(scores_window))
        if np.mean(scores_window) >= 200.0 and ep >= 100:
            print(f"\nSolved in {ep-100} episodes!")
            scores_avr = np.append(scores_avr, np.mean(scores_window))
            os.makedirs('Models', exist_ok=True)
            torch.save(agent.local_qnetwork.state_dict(),
                       f'Models/checkpoint{ep-100}_{replay_type}.pth')
            break

    return scores_avr


if __name__ == '__main__':
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # priority
    ETA = 1
    agent = AgentMR(state_size, action_size)
    score_pr = trainging_loop(agent)
    show_video_of_model(agent, "LunarLander-v3", "video_priority")

    # linear
    ETA = 0
    agent = AgentMR(state_size, action_size)
    score_lr = trainging_loop(agent)
    show_video_of_model(agent, "LunarLander-v3", "video_linear")

    # mixed
    ETA = 0.6
    agent = AgentMR(state_size, action_size)
    score_mr = trainging_loop(agent)

    show_video_of_model(agent, "LunarLander-v3", "video_mixed")

    plot_comparison(score_lr, score_pr, score_mr)
