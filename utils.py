import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit
from recordtype import recordtype

from FetchBulletEnv import FetchBulletEnv

Experience = recordtype('Experience', 'state action next_state reward done goal', default=None)


def make_env(params, worker_id=-1, render=False):
    if params.env_name == 'FetchBulletEnv':
        if render:
            env = FetchBulletEnv(seed=params.seed + worker_id, render_mode='GUI')
        else:
            env = FetchBulletEnv(seed=params.seed + worker_id)
        env = TimeLimit(env, max_episode_steps=params.max_timesteps)
    else:
        env = gym.make(params.env_name)
        env._max_episode_steps = params.max_timesteps
    env.seed(params.seed + worker_id)
    return env


def get_env_boundaries():
    action_bounds_np = np.array([1., 1., 1., .05])
    action_offset_np = np.array([0., 0., 0., 0.])
    action_offset = torch.FloatTensor(action_offset_np.reshape(1, -1))
    action_clip_low = np.array([-1.0 * action_bounds_np])
    action_clip_high = np.array([action_bounds_np])
    action_bounds = torch.FloatTensor(action_bounds_np.reshape(1, -1))

    state_bounds_np = np.array([0.075, 0.0865, 0.075, .02])
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1))
    state_offset_np = np.array([-0.025, 0.1135, -0.605, .02])
    state_offset = torch.FloatTensor(state_offset_np.reshape(1, -1))
    state_clip_low = np.array([-0.1, 0.027, -0.68, 0.])
    state_clip_high = np.array([0.05, 0.2, -0.53, 0.04])

    return (action_offset, action_bounds, action_clip_low, action_clip_high), (
    state_offset, state_bounds, state_clip_low, state_clip_high)
