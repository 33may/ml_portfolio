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
        x = self.fc3(x)

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