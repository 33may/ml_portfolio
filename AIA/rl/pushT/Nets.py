import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) * 512

        return x

class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class OUNoise:
    """Ornstein-Uhlenbeck process with controlled decay"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, initial_sigma=50, final_sigma=5, decay_steps=9000):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.decay_steps = decay_steps
        self.sigma = initial_sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.step_count = 0

    def sample(self):
        self.step_count += 1
        dx = self.theta * (self.mu - self.state)
        dx += self.current_sigma() * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

    def current_sigma(self):
        # Linear decay
        progress = min(self.step_count / self.decay_steps, 1.0)
        return self.initial_sigma + (self.final_sigma - self.initial_sigma) * progress