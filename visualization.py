import matplotlib.pyplot as plt
import gymnasium as gym
import os
import imageio
import glob
import io
import base64
from IPython.display import HTML, display
import numpy as np

def plot_scores(scores):
    """Plot the training scores over episodes."""
    final_episode = len(scores) * 100

    episodes = []
    plotted_scores = []

    for i, score in enumerate(scores):
        episode_num = (i + 1) * 100
        episodes.append(episode_num)
        plotted_scores.append(score)

    plt.figure(figsize=(12, 6))

    plt.plot(episodes, plotted_scores, 'o-', color='blue', markersize=8,
             linewidth=2, label="Average Score (per 100 episodes)")

    plt.axhline(y=200, color='green', linestyle='--',
                label="Solving Threshold (200)")

    plt.xlim(0, final_episode + 10)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Score", fontsize=12)
    plt.title("Training Performance - Deep Q-Learning on LunarLander", fontsize=14)

    solving_episode = episodes[-1]
    solving_score = plotted_scores[-1]
    plt.annotate(f"Solved at episode {solving_episode}\nScore: {solving_score:.2f}",
                 xy=(solving_episode, solving_score),
                 xytext=(solving_episode-100, solving_score+20),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def show_video_of_model(agent, env_name):
    """Generate a video of the trained agent performance."""
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item() if hasattr(action, 'item') else action)

    env.close()
    os.makedirs('Video', exist_ok=True)
    imageio.mimsave('Video/video.mp4', frames, fps=30)

def show_video():
    """Display the generated MP4 video in the notebook."""
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
