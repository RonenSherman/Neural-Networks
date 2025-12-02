import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# NN
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

def forward(self, x):
        return torch.softmax(self.layers(x), dim=-1)


# Setup
env = gym.make("CartPole-v1", render_mode="human")
Network = Network()
optimizer = optim.Adam(Network.parameters(), lr=1e-2)

def choose_action(obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    probs = Network(obs)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


# Training Loop
for episode in range(200):
    obs, info = env.reset()
    done = False
    log_probs = []
    rewards = []

    while not done:
        action, log_prob = choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        done = terminated or truncated  # stops when pole falls over

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}  |  Reward = {sum(rewards)}")

env.close()
