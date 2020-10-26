import os

import numpy as np
import torch
import torch.multiprocessing as mp

from utils import make_env, Experience


def spawn_processes(params, replay_buffer, model, state_normalizer, goal_normalizer, log_func):
    # limit the number of threads started by OpenMP
    os.environ['OMP_NUM_THREADS'] = "1"

    data_proc_list = []
    for proc_idx in range(params.np):
        p_args = (proc_idx, params, replay_buffer, model, state_normalizer, goal_normalizer, log_func)
        data_proc = mp.Process(target=process_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)

    return data_proc_list


def process_func(proc_idx, params, replay_buffer, model, state_normalizer, goal_normalizer, log_func):
    env = make_env(params, proc_idx)
    w = Worker(params, env, replay_buffer, model, state_normalizer, goal_normalizer, log_func)
    print(f"Spawning worker with id: {proc_idx}")
    w.loop()


class Worker:
    def __init__(self, params, env, replay_buffer, model, state_normalizer, goal_normalizer, log_func):
        self.params = params
        self.env = env
        self.replay_buffer = replay_buffer
        self.model = model
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer
        self.log_func = log_func

    def loop(self):
        device = next(self.model.actor.parameters()).device
        done = False
        obs = self.env.reset()
        goal = torch.from_numpy(obs['desired_goal']).float().unsqueeze(0).to(device)
        norm_goal = self.goal_normalizer.normalize(goal)
        episode_transitions = []

        state_shape = self.env.observation_space['observation'].sample().shape[0]
        new_states = np.zeros((self.params.max_timesteps, state_shape), dtype=np.float32)
        idx = 0

        while True:
            if done:
                self.state_normalizer.update(new_states)
                self.state_normalizer.recompute_stats()
                idx = 0

                self.create_her_transition(episode_transitions)
                episode_transitions = []
                obs = self.env.reset()
                goal = torch.from_numpy(obs['desired_goal']).float().unsqueeze(0).to(device)

                self.goal_normalizer.update(goal)
                self.goal_normalizer.recompute_stats()

                norm_goal = self.goal_normalizer.normalize(goal)

            new_states[idx] = obs['observation']
            idx += 1

            state = torch.from_numpy(obs['observation']).float().unsqueeze(0).to(device)
            norm_state = self.state_normalizer.normalize(state)

            with torch.no_grad():
                action = self.model.agent(norm_state, norm_goal)[0]

            new_obs, reward, done, _ = self.env.step(action)

            episode_transitions.append((obs, action, reward, new_obs, done))

            obs = new_obs

    def create_her_transition(self, episode_transitions):
        episode_obs = np.array(episode_transitions)
        for idx, (obs, action, reward, new_obs, done) in enumerate(episode_obs):
            exp = Experience(state=obs['observation'], action=action, next_state=new_obs['observation'], reward=reward,
                             done=done, goal=obs['desired_goal'])
            self.replay_buffer.append(exp)

            # using future-k method
            if (episode_obs.shape[0] - idx - 1) > 0:
                future_offset = np.unique(np.random.choice(range(episode_obs.shape[0] - idx - 1), self.params.replay_k))
                future_idx = future_offset + idx + 1
                future_idx = future_idx.astype(int)

                for future_o in episode_obs[future_idx][:, 0]:
                    new_reward = self.env.compute_reward(achieved_goal=new_obs['achieved_goal'], desired_goal=future_o['achieved_goal'], info=None)
                    new_exp = Experience(state=obs['observation'], action=action, next_state=new_obs['observation'],
                                         reward=new_reward, done=False, goal=future_o['achieved_goal'])
                    self.replay_buffer.append(new_exp)