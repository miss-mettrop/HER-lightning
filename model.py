from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import optim

HID_SIZE = 64


class Actor(nn.Module):
    def __init__(self, obs_size, goal_size, act_size, action_bounds, offset):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            # state + goal
            nn.Linear(obs_size + goal_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()
        )

        self.action_bounds = nn.Parameter(action_bounds, requires_grad=False)
        self.offset = nn.Parameter(offset, requires_grad=False)

    def forward(self, state, goal):
        return (self.net(torch.cat([state, goal], dim=1)) * self.action_bounds) + self.offset


class Critic(nn.Module):
    def __init__(self, obs_size, goal_size, act_size, H):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size + goal_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
            nn.Sigmoid()
        )

        self.H = nn.Parameter(torch.tensor(H, dtype=torch.float32, requires_grad=False), requires_grad=False)

    def forward(self, state, goal, action):
        return -self.net(torch.cat([state, goal, action], dim=1)) * self.H


class Agent():
    def __init__(self, net, action_clips, random_eps, noise_eps):
        self.net = net
        self.random_eps = random_eps
        self.noise_eps = noise_eps
        self.action_clips = action_clips

    def __call__(self, states, goals):
        mu_v = self.net(states, goals)
        actions = mu_v.data.detach().cpu().numpy()

        for i in range(len(actions)):
            if np.random.random() < self.random_eps:
                actions[i] = np.random.uniform(self.action_clips[0], self.action_clips[1])

        action_distribution_mean = (self.action_clips[0] + self.action_clips[1]) / 2
        action_deviation = self.action_clips[1] - action_distribution_mean
        action_standard_deviation = action_deviation * 68 / 100
        actions += self.noise_eps * np.random.normal(action_distribution_mean, action_standard_deviation)
        actions = np.clip(actions, self.action_clips[0], self.action_clips[1])
        return actions

    def test(self, states, goals):
        mu_v = self.net(states, goals)
        actions = mu_v.data.detach().cpu().numpy()
        actions = np.clip(actions, self.action_clips[0], self.action_clips[1])
        return actions


class DDPG(nn.Module):
    def __init__(self, params, obs_size, goal_size, act_size, action_clips, action_bounds, action_offset):
        super().__init__()
        self.actor = Actor(obs_size, goal_size, act_size, action_bounds, action_offset)
        self.critic = Critic(obs_size, goal_size, act_size, params.H)
        self.agent = Agent(self.actor, action_clips, params.random_eps, params.noise_eps)
        self.tgt_act_net = deepcopy(self.actor)
        self.tgt_crt_net = deepcopy(self.critic)
        self.act_opt = optim.Adam(self.actor.parameters(), lr=params.lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=params.lr_critic)

    def alpha_sync(self, alpha):
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.actor.state_dict()
        tgt_state = self.tgt_act_net.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.tgt_act_net.load_state_dict(tgt_state)

        state = self.critic.state_dict()
        tgt_state = self.tgt_crt_net.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.tgt_crt_net.load_state_dict(tgt_state)