import os
import pickle
import random

import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = x0

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class ReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.state_buf = np.zeros((self.max_size, obs_dim))
        self.next_state_buf = np.zeros((self.max_size, obs_dim))
        self.action_buf = np.zeros((self.max_size, action_dim))
        self.reward_buf = np.zeros(self.max_size)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.ptr % self.max_size
        self.state_buf[idx] = state
        self.next_state_buf[idx] = next_state
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.done_buf[idx] = 1 - done
        self.ptr += 1

    def sample_buffer(self, batch_size):
        max_idx = min(self.ptr, self.max_size)
        batch_idxs = np.random.choice(max_idx, batch_size)

        return (
            self.state_buf[batch_idxs],
            self.action_buf[batch_idxs],
            self.reward_buf[batch_idxs],
            self.next_state_buf[batch_idxs],
            self.done_buf[batch_idxs],
        )


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = 400
        self.fc2_dims = 300
        self.n_actions = n_actions

        self.state_encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.n_actions, self.fc2_dims),
            nn.ReLU(),
        )


        self.n_actions = n_actions

        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'mps')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.state_encoder(state)

        action_value = self.action_encoder(action)

        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = 400
        self.fc2_dims = 300
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions),
            nn.Tanh()
        )


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'mps')

        self.to(self.device)

    def forward(self, state):

        x = self.net(state)

        return x


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, noise, gamma=0.99,
                 n_actions=2, max_size=10000, batch_size=256,
                 expert_data=None, expert_ratio=0.25):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, n_actions)
        self.critic = CriticNetwork(beta, input_dims, n_actions)

        self.target_actor = ActorNetwork(alpha, input_dims, n_actions)
        self.target_critic = CriticNetwork(beta, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=noise, x0=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

        self.expert_data = expert_data
        self.expert_ratio = expert_ratio

    def choose_action(self, observation, eval=False):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)

        noise = self.noise()

        mu_prime = mu + T.tensor(noise, dtype=T.float).to(self.actor.device) if not eval else mu

        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_mixed_batch(self):
        exp_bs = int(self.expert_ratio * self.batch_size)
        live_bs = self.batch_size - exp_bs

        expert_batch = random.choices(self.expert_data, k=exp_bs) if len(self.expert_data) > 0 else []
        if expert_batch:
            exp_s, exp_a, exp_r, exp_d, exp_s2 = zip(*expert_batch)
            exp_s = np.array(exp_s, dtype=np.float32)
            exp_a = np.array(exp_a, dtype=np.float32)
            exp_r = np.array(exp_r, dtype=np.float32)
            exp_s2 = np.array(exp_s2, dtype=np.float32)
            exp_d = np.array(exp_d, dtype=np.float32)
        else:
            exp_s = exp_a = exp_r = exp_s2 = exp_d = np.zeros((0,))

        ls, la, lrw, ls2, ld = self.memory.sample_buffer(live_bs)

        states = np.vstack([exp_s, ls])
        actions = np.vstack([exp_a, la])
        rewards = np.concatenate([exp_r, lrw])
        states_ = np.vstack([exp_s2, ls2])
        dones = np.concatenate([exp_d, ld])

        perm = np.random.permutation(self.batch_size)
        return (
            states[perm],
            actions[perm],
            rewards[perm],
            states_[perm],
            dones[perm],
        )

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, done = \
            self.memory.sample_buffer(self.batch_size)

        dev = self.critic.device
        states = T.tensor(states, dtype=T.float32, device=dev)
        actions = T.tensor(actions, dtype=T.float32, device=dev)
        rewards = T.tensor(rewards, dtype=T.float32, device=dev).view(-1, 1)
        new_states = T.tensor(new_states, dtype=T.float32, device=dev)
        done = T.tensor(done, dtype=T.float32, device=dev).view(-1, 1)

        with T.no_grad():
            next_actions = self.target_actor(new_states)
            q_next = self.target_critic(new_states, next_actions)
            q_target = rewards + self.gamma * q_next * done

        self.critic.train()
        self.critic.optimizer.zero_grad()
        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, q_target)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.train()
        self.actor.optimizer.zero_grad()
        mu = self.actor(states)
        actor_loss = -self.critic(states, mu).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else tau

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
