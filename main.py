import gymnasium as gym
import numpy as np
import torch
import os
from collections import deque
from agent import Agent
from visualization import plot_scores, show_video_of_model, show_video

# Setting up the environment
env = gym.make("LunarLander-v3")
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("State shape: ", state_shape)
print("State size: ", state_size)
print("number of actions: ", action_size)

# Hyperparameters
learning_rate = 0.0005
minibatch_size = 128
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 2e-3

# Initialize agent
agent = Agent(state_size, action_size, learning_rate, replay_buffer_size, interpolation_parameter)

# Training parameters
number_of_episodes = 2000
max_number_of_timesteps_per_episode = 1000
epsilion_starting = 1.0
epsilion_ending = 0.01
epsilion_decay = 0.995
epsilion = epsilion_starting
score_for_100_episodes = deque(maxlen=minibatch_size)
score_for_all_episodes = []

# Training loop
for episode in range(1, number_of_episodes + 1):
    state, _ = env.reset()
    score = 0

    for t in range(max_number_of_timesteps_per_episode):
        action = agent.act(state, epsilion)
        next_state, reward, done, _, _ = env.step(action)

        agent.step(state, action, reward, next_state, done, minibatch_size, discount_factor)

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
        os.makedirs('Models', exist_ok=True)
        torch.save(agent.local_qnetwork.state_dict(), 'Models/checkpoint.pth')
        break

# Plot training performance
plot_scores(score_for_all_episodes)

# Generate video of trained agent
show_video_of_model(agent, 'LunarLander-v3')

# Display the video
show_video()
