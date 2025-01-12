import os.path

import cv2
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import torch.nn as nn
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


IMAGE_SIZE = (63, 112)

def preprocess(image: np.ndarray, background: int = 200, threshold: int = 254):
    image[image[:, :, 0] == background] = [255, 255, 255]
    grayscale = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    binary_image = (grayscale < threshold).astype(np.uint8)

    return cv2.resize(binary_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


class DQNetwork(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(DQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self._calculate_fc_input(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

# Evaluation Function
def fully_evaluate_model(env, model_path, num_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    input_shape = IMAGE_SIZE
    np.expand_dims(input_shape, axis=0)
    num_actions = env.action_space.n
    q_network = DQNetwork(input_shape, num_actions).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    q_network.eval()  # Set the model to evaluation mode

    scores = []
    best_score = -np.inf
    best_episode = None


    for episode in range(num_episodes):
        env.reset()
        state = preprocess(env.render())
        state = np.expand_dims(state, axis=0)
        total_reward = 0
        done = False

        frames = []

        print(f"Episode {episode + 1}:")
        score = 0

        while not done:
            # Select the best action based on the Q-network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

            # Perform the action
            next_state, reward, done, truncated, score = env.step(action)

            frame = env.render()

            frames.append(frame)

            next_state = preprocess(frame)
            next_state = np.expand_dims(next_state, axis=0)

            state = next_state
            total_reward += reward

        if score['score'] > best_score:
            best_score = score['score']
            best_episode = frames

        print(f"Total Reward: {total_reward}, Score:{score['score']}")
        scores.append(score['score'])

    env.close()

    return scores, best_score, best_episode


# Environment setup
env = gym.make("FlappyBird-v0", render_mode="rgb_array", background=None, use_lidar=False)

# Path to the saved model
model_path = "flappybird_dql.pth"

# Evaluate the model
scores, best_score, best_episode = fully_evaluate_model(env, model_path, num_episodes=100)

clip = ImageSequenceClip(best_episode, fps=30)
clip.write_videofile(os.path.join('.', 'video', 'agent.mp4'), codec="libx264")

#print scores in file
with open('./scores.csv', 'w') as f:
    for score in scores:
        f.write(f'{score},')

print(np.mean(np.array(scores)))
print(best_score)