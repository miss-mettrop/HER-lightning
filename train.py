import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
import gym
import numpy as np
import torch
from tqdm import tqdm

from arguments import get_args
from buffer import RLDataset, SharedReplayBuffer
from model import Shared_DDPG, DDPG
from utils import normalize_actions


class HER(pl.LightningModule):
    def __init__(self, hparams):
        super(HER, self).__init__()

        self.hparams = hparams

        self.model = DDPG()

        self.shared_model = Shared_DDPG()

        # TODO: extract this from the env
        state_shape = None
        action_shape = None
        self.action_bounds = []

        self.replay_buffer = SharedReplayBuffer(self.hparams.buffer_size, state_shape, action_shape)

    def collate_fn(self, batch):
        return collate.default_convert(batch)

    def __dataloader(self) -> DataLoader:
        dataset = RLDataset(self.replay_buffer, self.hparams.batch_size, self.hparams.n_batches,
                            self.hparams.replay_initial)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=1,
            pin_memory=True
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def configure_optimizers(self):
        return [self.model.crt_opt, self.model.act_opt], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        states_v, actions_v, next_states_v, rewards_v, dones_mask, goals_v = batch

        if optimizer_idx == 0:
            # train critic
            self.model.crt_opt.zero_grad()
            q_v = self.model.critic(states_v, goals_v, actions_v)
            with torch.no_grad():
                next_act_v = self.model.tgt_act_net.target_model(
                    next_states_v, goals_v)
                q_next_v = self.model.tgt_crt_net.target_model(
                    next_states_v, goals_v, next_act_v)
                q_next_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_next_v * self.gamma
                # clip the q value
                clip_return = 1 / (1 - self.hparams.gamma)
                q_ref_v = torch.clamp(q_ref_v, -clip_return, 0)
            critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
            tqdm_dict = {
                'critic_loss': critic_loss_v
            }
            self.log_dict(tqdm_dict)

            return critic_loss_v

        elif optimizer_idx == 1:
            # train actor
            self.model.act_opt.zero_grad()
            cur_actions_v = self.model.actor(states_v, goals_v)
            actor_loss_v = -self.model.critic(states_v, goals_v, cur_actions_v)
            actor_loss_v = actor_loss_v.mean() + (cur_actions_v / normalize_actions(cur_actions_v, self.action_bounds)).pow(2).mean()
            tqdm_dict = {
                'actor_loss': actor_loss_v
            }
            self.log_dict(tqdm_dict)

            return actor_loss_v


if __name__ == '__main__':
    hparams = get_args()

    if hparams.debug:
        hparams.logger = None
        hparams.profiler = SimpleProfiler()
        hparams.num_workers = 0
    else:
        hparams.logger = WandbLogger(project=hparams.project)

    seed_everything(hparams.seed)
    her = HER(hparams)
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(her)
