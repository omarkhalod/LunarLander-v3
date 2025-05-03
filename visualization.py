import matplotlib.pyplot as plt
import gymnasium as gym
import os
import imageio
import glob
import io
import base64
from IPython.display import HTML, display


def __init__(self):
    self.frames = []


def plot_comparison(scores_lr, scores_pr, scores_mr):
    """
    Plot training performance for three agents:
      - scores_lr: list of average scores per episode for Linear Replay
      - scores_pr: list of average scores per episode for Prioritized Replay
      - scores_mr: list of average scores per episode for Mixed Replay
    Handles different lengths by plotting each over its own episode range.
    """
    # Compute episode indices for each score list
    eps_lr = list(range(1, len(scores_lr) + 1))
    eps_pr = list(range(1, len(scores_pr) + 1))
    eps_mr = list(range(1, len(scores_mr) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(eps_lr, scores_lr, label='Linear Replay')
    plt.plot(eps_pr, scores_pr, label='Prioritized Replay')
    plt.plot(eps_mr, scores_mr, label='Mixed Replay')

    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100-episode window)')
    plt.title('DQN Replay Buffer Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs('Graphs', exist_ok=True)
    plt.savefig('Graphs/comparison.png')
    plt.show()


def plot_comparison2(scores_lr, scores_mr, name):
    """
    Plot training performance for three agents:
      - scores_lr: list of average scores per episode for Linear Replay
      - scores_pr: list of average scores per episode for Prioritized Replay
      - scores_mr: list of average scores per episode for Mixed Replay
    Handles different lengths by plotting each over its own episode range.
    """
    # Compute episode indices for each score list
    eps_lr = list(range(1, len(scores_lr) + 1))
    eps_mr = list(range(1, len(scores_mr) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(eps_lr, scores_lr, label='Linear Replay')
    plt.plot(eps_mr, scores_mr, label='Mixed Replay')

    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100-episode window)')
    plt.title('DQN Replay Buffer Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs('Graphs', exist_ok=True)
    plt.savefig(f'Graphs/comparison{name}.png')
    # plt.show()


def show_video_of_model(agent, env_name, name="video_standard", max_steps=1000):

    # 1) Create env in RGB_ARRAY mode
    env = gym.make(env_name, render_mode='rgb_array')

    # 2) Reset and prepare
    state, _ = env.reset(seed=123)
    done = False
    frames = []

    # 3) Switch agent to eval if using PyTorch modules
    try:
        agent.local_qnetwork.eval()
    except AttributeError:
        pass

    # 4) Roll out with eps=0 (fully greedy)
    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        # force greedy: eps=0
        action = agent.act(state, eps=0.0)

        # step
        state, _, term, trunc, _ = env.step(
            action.item() if hasattr(action, 'item') else action
        )
        done = term or trunc
        if done:
            break

    env.close()

    # 5) Save video
    os.makedirs('Video', exist_ok=True)
    imageio.mimsave(f'Video/{name}.mp4', frames, fps=30, macro_block_size=1)


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
