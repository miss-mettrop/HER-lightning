import numpy as np
import torch

from utils import Experience


class Worker:
    def __init__(self, id, params, env, replay_buffer, model, state_normalizer, goal_normalizer):
        self.id = id
        self.params = params
        self.env = env
        self.replay_buffer = replay_buffer
        self.model = model
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer

    def play_episode(self):
        device = next(self.model.actor.parameters()).device
        obs = self.env.reset()
        goal = torch.from_numpy(obs['desired_goal']).float().unsqueeze(0).to(device)
        norm_goal = self.goal_normalizer.normalize(goal)
        episode_transitions = []
        episode_reward = 0

        state_shape = self.env.observation_space['observation'].sample().shape[0]
        goal_shape = self.env.observation_space['achieved_goal'].sample().shape[0]
        new_states = np.zeros((self.params.max_timesteps, state_shape), dtype=np.float32)
        new_goals = np.zeros((self.params.max_timesteps, goal_shape), dtype=np.float32)
        idx = 0

        while True:
            new_states[idx] = obs['observation']
            new_goals[idx] = obs['achieved_goal']
            idx += 1

            state = torch.from_numpy(obs['observation']).float().unsqueeze(0).to(device)
            norm_state = self.state_normalizer.normalize(state)

            with torch.no_grad():
                action = self.model.agent(norm_state, norm_goal)[0]

            new_obs, reward, done, _ = self.env.step(action)
            episode_reward += reward

            episode_transitions.append((obs, action, reward, new_obs, done))

            obs = new_obs

            if done:
                self.state_normalizer.update(new_states)
                self.state_normalizer.recompute_stats()
                self.goal_normalizer.update(new_goals)

                self.create_her_transition(episode_transitions)
                obs = self.env.reset()
                env_goals = torch.tensor([obs['desired_goal']] * self.params.max_timesteps).to(device)

                self.goal_normalizer.update(env_goals)
                self.goal_normalizer.recompute_stats()

                return episode_reward

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