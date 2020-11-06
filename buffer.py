import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data.dataset import IterableDataset

from worker import LOW_STATE_IDX


class SharedReplayBuffer:
    def __init__(self, buffer_size: int, state_shape, action_shape, goal_shape):
        self.count = torch.tensor([0], dtype=torch.int64)
        self.capacity = buffer_size
        self.pos = torch.tensor([0], dtype=torch.int64)

        self.states = torch.zeros((buffer_size, state_shape), dtype=torch.float32)
        self.next_states = torch.zeros((buffer_size, state_shape), dtype=torch.float32)
        self.goals = torch.zeros((buffer_size, goal_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_shape), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.int8)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool)

        self.count.share_memory_()
        self.pos.share_memory_()
        self.states.share_memory_()
        self.actions.share_memory_()
        self.next_states.share_memory_()
        self.rewards.share_memory_()
        self.dones.share_memory_()
        self.goals.share_memory_()

        self.lock = mp.Lock()

    def __len__(self):
        return self.count

    def sample(self, batch_size):
        if self.count[0] <= batch_size:
            with self.lock:
                nr = self.count[0]
                return [self.states[:nr], self.actions[:nr], self.next_states[:nr], self.rewards[:nr], self.dones[:nr], self.goals[:nr]]

        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(min(self.count[0], self.capacity), batch_size, replace=True)
        with self.lock:
            return (self.states[keys], self.actions[keys], self.next_states[keys], self.rewards[keys], self.dones[keys],
                    self.goals[keys])

    def append(self, sample):
        assert type(sample).__name__ == 'Experience'

        with self.lock:
            if self.count[0] < self.capacity:
                self.count[0] += 1

            pos = self.pos[0]
            self.states[pos] = torch.tensor(sample.state, dtype=torch.float32)
            self.actions[pos] = torch.tensor(sample.action, dtype=torch.float32)
            self.next_states[pos] = torch.tensor(sample.next_state, dtype=torch.float32)
            self.rewards[pos] = torch.tensor(sample.reward, dtype=torch.int8)
            self.dones[pos] = torch.tensor(sample.done, dtype=torch.bool)
            self.goals[pos] = torch.tensor(sample.goal, dtype=torch.float32)

            self.pos[0] = (self.pos[0] + 1) % self.capacity

    def empty(self):
        with self.lock:
            self.count[0] = 0
            self.pos[0] = 0


class RLDataset(IterableDataset):
    def __init__(self, buffers, batch_size, n_batches, replay_initial):
        self.buffers = buffers
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.replay_initial = replay_initial

    def __iter__(self):
        while len(self.buffers[-1]) < self.replay_initial:
            time.sleep(0.3)

        for i in range(self.n_batches):
            for j in range(len(self.buffers)):
                yield self.buffers[j].sample(self.batch_size)


class TestDataset(IterableDataset):
    def __init__(self, hparams, test_env, high_model, low_model, high_state_normalizer, low_state_normalizer,
                 env_goal_normalizer):
        self.hparams = hparams
        self.test_env = test_env
        self.high_model = high_model
        self.low_model = low_model
        self.high_state_normalizer = high_state_normalizer
        self.low_state_normalizer = low_state_normalizer
        self.env_goal_normalizer = env_goal_normalizer

    @torch.no_grad()
    def __iter__(self):
        total_reward = 0.0
        accuracy = 0.0
        device = next(self.high_model.actor.parameters()).device

        for _ in range(self.hparams.n_test_rollouts):
            goal_reached_this_round = False
            obs = self.test_env.reset()
            goal = torch.from_numpy(obs['desired_goal']).float().unsqueeze(0).to(device)
            norm_goal = self.env_goal_normalizer.normalize(goal)

            while True:
                state = torch.from_numpy(obs['observation']).float().unsqueeze(0).to(device)
                norm_state = self.high_state_normalizer.normalize(state)

                norm_target_np = self.high_model.agent(norm_state, norm_goal)[0]
                norm_target = torch.from_numpy(norm_target_np).float().unsqueeze(0).to(device)
                target_np = self.low_state_normalizer.denormalize(norm_target).detach().cpu().numpy()[0]
                self.test_env.update_markers(target_np)

                for i in range(self.hparams.H):
                    low_state = torch.from_numpy(obs['observation'][LOW_STATE_IDX]).float().unsqueeze(0).to(device)
                    norm_low_state = self.low_state_normalizer.normalize(low_state)

                    action = self.low_model.agent(norm_low_state, norm_target)[0]
                    new_obs, reward, done, info = self.test_env.step(action)

                    total_reward += reward

                    obs = new_obs

                    if done or info['is_success']:
                        break

                if info['is_success']:
                    if not goal_reached_this_round:
                        accuracy += 1
                        goal_reached_this_round = True

                if done:
                    break

        tqdm_dict = {
            'test_mean_reward': total_reward / self.hparams.n_test_rollouts,
            'test_accuracy': accuracy / self.hparams.n_test_rollouts
        }
        yield tqdm_dict
