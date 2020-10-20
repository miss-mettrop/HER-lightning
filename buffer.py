import torch
from torch.utils.data.dataset import IterableDataset
import torch.multiprocessing as mp
import numpy as np
import time

class SharedReplayBuffer:
    def __init__(self, buffer_size:int, state_shape, action_shape):
        self.count = torch.tensor([0], dtype=torch.int64)
        self.capacity = buffer_size
        self.pos = torch.tensor([0], dtype=torch.int64)

        self.states = torch.zeros((buffer_size, state_shape), dtype=torch.float32)
        self.next_states = torch.zeros((buffer_size, state_shape), dtype=torch.float32)
        self.goals = torch.zeros((buffer_size, state_shape), dtype=torch.float32)
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
                return [self.states, self.actions, self.next_states, self.rewards, self.dones, self.goals]

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
    def __init__(self, buffer, batch_size, n_batches, replay_initial):
        self.buffer = buffer
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.replay_initial = replay_initial

    def __iter__(self):
        while len(self.buffer) < self.replay_initial:
            time.sleep(0.3)

        for i in range(self.n_batches):
            yield self.buffer.sample(self.batch_size)
