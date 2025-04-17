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
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions):
        super().__init__()

        self.state_net = nn.Sequential(
            nn.Linear(input_dims, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 128),
            nn.LayerNorm(128)
        )

        self.action_net = nn.Sequential(
            nn.Linear(n_actions, 128),
            nn.ReLU()
        )

        self.q_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, state, action):
        state_out = self.state_net(state)
        action_out = self.action_net(action)
        combined = state_out + action_out
        q_value = self.q_net(combined)
        return q_value

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dims, 400),
                nn.LayerNorm(400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.LayerNorm(300),
                nn.ReLU(),
                nn.Linear(300, n_actions),
                nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'mps')

        self.to(self.device)

    def forward(self, state):
        raw_output = self.net(state)
        scaled = (raw_output + 1) * 256
        return scaled


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, noise, gamma=0.99,
                 n_actions=2, max_size=1000000, batch_size=256,
                 expert_data=None, expert_ratio=0.25):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, n_actions)
        self.critic = CriticNetwork(beta, input_dims, n_actions)

        self.target_actor = ActorNetwork(alpha, input_dims, n_actions)
        self.target_critic = CriticNetwork(beta, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=noise)

        self.update_network_parameters(tau=1)

        self.expert_data = expert_data or []
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
        # number of expert and live samples
        exp_bs = int(self.expert_ratio * self.batch_size)
        live_bs = self.batch_size - exp_bs

        # sample expert (with replacement if needed)
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

        # sample live from replay buffer
        ls, la, lrw, ls2, ld = self.memory.sample_buffer(live_bs)

        # concatenate
        states = np.vstack([exp_s, ls])
        actions = np.vstack([exp_a, la])
        rewards = np.concatenate([exp_r, lrw])
        states_ = np.vstack([exp_s2, ls2])
        dones = np.concatenate([exp_d, ld])

        # shuffle
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

        # 1) sample mixed batch
        s, a, r, s2, d = self.sample_mixed_batch()

        # 2) to torch
        device = self.critic.device
        states   = torch.tensor(s,  dtype=torch.float32).to(device)
        actions  = torch.tensor(a,  dtype=torch.float32).to(device)
        rewards  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1).to(device)
        states_  = torch.tensor(s2, dtype=torch.float32).to(device)
        dones    = torch.tensor(d,  dtype=torch.float32).unsqueeze(1).to(device)

        # 3) target Q
        with torch.no_grad():
            a2       = self.target_actor(states_)
            q_next   = self.target_critic(states_, a2)
            q_target = rewards + self.gamma * q_next * dones

        # 4) critic update
        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, q_target)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # 5) actor update
        self.actor.optimizer.zero_grad()
        mu = self.actor(states)
        actor_loss = -self.critic(states, mu).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # 6) soft update
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
