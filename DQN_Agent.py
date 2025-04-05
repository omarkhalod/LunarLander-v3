import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import io
import base64
import imageio
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import torch.nn.functional as F
from collections import deque

print(torch.cuda.is_available())


class Network(nn.Module):

    def __init__(self, state_size, action_size, seed=42):
        # Calling the init (constructor) of the super class for nn.module
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
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
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3


# Experience replay buffer
class ReplayMemory(object):

    def __init__(self, capacity):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
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

    def __init__(self, state_size, action_size):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilion=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        # Epsilon-Greedy Action Selection
        if random.random() > epsilion:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        state, next_state, action, reward, done = experiences
        self.target_qnetwork.eval()
        # first we forward propgation to get the predicted values we get back a matix (batch_size, action_size)
        # next we make sure that we dont update the weights by the .eval() and .detach()
        # After that we get the maximum Q-value for each row and the [0] extracts only the max value
        # After getting the max the array is (batch_size,) but we want it (batch_size,1) so we use the unsqueeze(1)
        next_q_targets = self.target_qnetwork(
            next_state).detach().max(1)[0].unsqueeze(1)
        q_targets = reward + (discount_factor * next_q_targets * (1 - done))
        q_expected = self.local_qnetwork(state).gather(1, action)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork,
                         self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_model, target_modal, interpolation_parameter):
        for target_param, local_param in zip(target_modal.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (
                1.0 - interpolation_parameter) * target_param.data)

# Create the environment and agent


agent = Agent(state_size, action_size)

# Training the agent


number_of_episodes = 2000
max_number_of_timesteps_per_episode = 1000
epsilion_starting = 1.0
epsilion_ending = 0.01
epsilion_decay = 0.995
epsilion = epsilion_starting
score_for_100_episodes = deque(maxlen=minibatch_size)
score_for_all_episodes = []

for episode in range(1, number_of_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(max_number_of_timesteps_per_episode):
        action = agent.act(state, epsilion)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    score_for_100_episodes.append(score)
    epsilion = max(epsilion_ending, epsilion_decay * epsilion)
    print('\rEpisode: {}\t Average Score: {:.2f}'.format(
        episode, np.mean(score_for_100_episodes)), end="")
    if (episode % 100 == 0):
        score_for_all_episodes.append(np.mean(score_for_100_episodes))
        print('\rEpisode: {}\t Average Score: {:.2f}'.format(
            episode, np.mean(score_for_100_episodes)))
        print(score_for_all_episodes)
    if (np.mean(score_for_100_episodes) > 200.0):
        print('\nEnviroment solved in {:d} Episodes!\tAverage Score: {:.2f}'.format(
            episode - 100, np.mean(score_for_100_episodes)))
        score_for_all_episodes.append(np.mean(score_for_100_episodes))
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

# visualizing the agent


def plot_scores(scores):
    episodes = np.arange(100, (100 * len(score_for_all_episodes)) + 100, 100)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, score_for_all_episodes, label="Average Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Performance")
    plt.legend()
    plt.show()


plot_scores(score_for_all_episodes)


def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)


show_video_of_model(agent, 'LunarLander-v3')


def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


show_video()
