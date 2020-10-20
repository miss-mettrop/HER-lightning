import gym
from gym.wrappers import TimeLimit
from recordtype import recordtype

from FetchBulletEnv import FetchBulletEnv

Experience = recordtype('Experience', 'state action next_state reward done goal', default=None)


def make_env(params, worker_id=-1, render=False):
    if params['ENV_NAME'] == 'FetchBulletEnv':
        if render:
            env = FetchBulletEnv(seed=params['SEED'] + worker_id, render_mode='GUI')
        else:
            env = FetchBulletEnv(seed=params['SEED'] + worker_id)
        env = TimeLimit(env, max_episode_steps=params['EP_LENGTH'])
    else:
        env = gym.make(params['ENV_NAME'])
        env._max_episode_steps = params['EP_LENGTH']
    env.seed(params['SEED'] + worker_id)
    return env


def normalize_actions(actions, action_bounds):
    # first get them between 0 and 1
    normalized_actions =  (actions - action_bounds[0]) / (action_bounds[1] - action_bounds[0])
    # then change it to -1 to 1
    return normalized_actions * 2.0 - 1