import random
from collections import deque
import flappy_bird_gymnasium
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
IMAGE_SIZE = (63, 112)  # Resize images to 84x84
LEARNING_RATE = 1e-4
GAMMA = 0.99
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPSILON_START = 0.8
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 1000

# Environment setup
env = gym.make("FlappyBird-v0", use_lidar = False, render_mode='rgb_array', background=None)


def preprocess(image: np.ndarray, background: int = 200, threshold: int = 254):
    image[image[:, :, 0] == background] = [255, 255, 255]
    grayscale = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    binary_image = (grayscale > threshold).astype(np.uint8)

    return cv2.resize(binary_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)


# Neural Network Model
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

    def _calculate_fc_input(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.conv(dummy_input)
            return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Training Loop
def train_dql(env):
    # Initialize networks
    input_shape = (1, *IMAGE_SIZE)  # Single grayscale channel
    num_actions = env.action_space.n
    q_network = DQNetwork(input_shape, num_actions).to(device)
    target_network = DQNetwork(input_shape, num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPSILON_START
    total_steps = 0

    for episode in range(NUM_EPISODES):
        env.reset()
        state = preprocess(env.render())
        state = np.expand_dims(state, axis=0)  # Add channel dimension (1, 84, 84)
        total_reward = 0
        done = False

        score = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = torch.argmax(q_values).item()  # Exploit

            # Perform action in the environment
            next_state, reward, done, _, score = env.step(action)
            next_state = preprocess(env.render())
            next_state = np.expand_dims(next_state, axis=0)

            # if episode >= 590:
            #     print(f"....{reward}")

            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the network
            if replay_buffer.size() > BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                # Compute Q-values and targets
                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = target_network(next_states).max(1)[0]
                    targets = rewards + (1 - dones) * GAMMA * max_next_q_values

                # Compute loss and update Q-network
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_steps += 1

            # Update target network periodically
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_network.load_state_dict(q_network.state_dict())

        # Decay epsilon
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Score: {score['score']}")

    env.close()
    return q_network


# Device setup (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the agent
trained_q_network = train_dql(env)

# Save the trained model
torch.save(trained_q_network.state_dict(), "flappybird_dql.pth")