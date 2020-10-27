import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data.dataset import IterableDataset

from utils import make_env
from worker import Worker


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
    def __init__(self, params, buffer, model, state_normalizer, goal_normalizer):
        self.params = params
        self.buffer = buffer

        env = make_env(params)
        self.worker = Worker(0, params, env, self.buffer, model, state_normalizer, goal_normalizer)
        self.env_generated = False

        self.fill_buffer()

    def fill_buffer(self):
        for i in range(self.params.replay_initial // self.params.max_timesteps):
            self.worker.play_episode()

    def __iter__(self):
        if self.env_generated is False:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                proc_idx = worker_info.id
                self.worker.id = proc_idx
                self.worker.env = make_env(self.params, worker_info.seed)
            else:
                self.worker.env = make_env(self.params)
            self.env_generated = True

        for c in range(self.params.n_cycles):
            ep_rewards = []
            for i in range(self.params.ep_per_cycle // self.params.num_workers):
                ep_reward = self.worker.play_episode()
                ep_rewards.append(ep_reward)

            for i in range(self.params.n_batches // self.params.num_workers):
                yield (self.buffer.sample(self.params.batch_size), ep_rewards)

class TestDataset(IterableDataset):
    def __init__(self, hparams, test_env, model, state_normalizer, goal_normalizer):
        self.hparams = hparams
        self.test_env = test_env
        self.model = model
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer

    @torch.no_grad()
    def __iter__(self):
        total_reward = 0.0
        accuracy = 0.0
        device = next(self.model.actor.parameters()).device

        for _ in range(self.hparams.n_test_rollouts):
            goal_reached_this_round = False
            obs = self.test_env.reset()
            goal = torch.from_numpy(obs['desired_goal']).float().unsqueeze(0).to(device)

            while True:
                state = torch.from_numpy(obs['observation']).float().unsqueeze(0).to(device)
                norm_state = self.state_normalizer.normalize(state)
                norm_goal = self.goal_normalizer.normalize(goal)

                action = self.model.agent.test(norm_state, norm_goal)[0]

                new_obs, reward, done, info = self.test_env.step(action)

                total_reward += reward

                if info['is_success']:
                    if not goal_reached_this_round:
                        accuracy += 1
                        goal_reached_this_round = True

                if done:
                    break

                obs = new_obs

        tqdm_dict = {
            'test_mean_reward': total_reward / self.hparams.n_test_rollouts,
            'accuracy': accuracy / self.hparams.n_test_rollouts
        }
        yield tqdm_dict