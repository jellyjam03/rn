import cv2
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import torch.nn as nn

IMAGE_SIZE = (63, 112)

def preprocess(image: np.ndarray, background: int = 200, threshold: int = 254):
    image[image[:, :, 0] == background] = [255, 255, 255]
    grayscale = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    binary_image = (grayscale < threshold).astype(np.uint8)

    return cv2.resize(binary_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


# Neural Network Model (same as used during training)
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


# Evaluation Function
def evaluate_model(env, model_path, num_episodes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    input_shape = IMAGE_SIZE
    np.expand_dims(input_shape, axis=0)
    num_actions = env.action_space.n
    q_network = DQNetwork(input_shape, num_actions).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    q_network.eval()  # Set the model to evaluation mode

    for episode in range(num_episodes):
        env.reset()
        state = preprocess(env.render())
        state = np.expand_dims(state, axis=0)
        total_reward = 0
        done = False

        print(f"Episode {episode + 1}:")

        score = 0
        while not done:
            # Select the best action based on the Q-network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

            # Perform the action
            next_state, reward, done, _, score = env.step(action)
            next_state = preprocess(env.render())
            next_state = np.expand_dims(next_state, axis=0)

            state = next_state
            total_reward += reward

        print(f"Total Reward: {total_reward}, Score:{score['score']}")

    env.close()


# Environment setup
env = gym.make("FlappyBird-v0", render_mode="human", background=None, use_lidar=False)

# Path to the saved model
model_path = "flappybird_dql_z.pth"

# Evaluate the model
evaluate_model(env, model_path, num_episodes=10)