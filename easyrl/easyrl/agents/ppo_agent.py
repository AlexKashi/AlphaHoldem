from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from easyrl.agents.base_agent import BaseAgent
from easyrl.configs import cfg
from easyrl.utils.common import linear_decay_percent
from easyrl.utils.rl_logger import logger
from easyrl.utils.torch_util import action_entropy
from easyrl.utils.torch_util import action_from_dist
from easyrl.utils.torch_util import action_log_prob
from easyrl.utils.torch_util import clip_grad
from easyrl.utils.torch_util import load_ckpt_data
from easyrl.utils.torch_util import load_state_dict
from easyrl.utils.torch_util import move_to
from easyrl.utils.torch_util import save_model
from easyrl.utils.torch_util import torch_float
from easyrl.utils.torch_util import torch_to_np


@dataclass
class PPOAgent(BaseAgent):
    actor: nn.Module
    critic: nn.Module
    same_body: float = False

    def __post_init__(self):
        move_to([self.actor, self.critic],
                device=cfg.alg.device)
        if cfg.alg.vf_loss_type == 'mse':
            self.val_loss_criterion = nn.MSELoss().to(cfg.alg.device)
        elif cfg.alg.vf_loss_type == 'smoothl1':
            self.val_loss_criterion = nn.SmoothL1Loss().to(cfg.alg.device)
        else:
            raise TypeError(f'Unknown value loss type: {cfg.alg.vf_loss_type}!')
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        # keep unique elements only. The following code works for python >=3.7
        # for earlier version of python, u need to use OrderedDict
        self.all_params = dict.fromkeys(all_params).keys()
        if (cfg.alg.linear_decay_lr or cfg.alg.linear_decay_clip_range) and \
                cfg.alg.max_steps > cfg.alg.max_decay_steps:
            logger.warning('max_steps should not be greater than max_decay_steps.')
            cfg.alg.max_decay_steps = int(cfg.alg.max_steps * 1.5)
            logger.warning(f'Resetting max_decay_steps to {cfg.alg.max_decay_steps}!')
        total_epochs = int(np.ceil(cfg.alg.max_decay_steps / (cfg.alg.num_envs *
                                                              cfg.alg.episode_steps)))
        if cfg.alg.linear_decay_clip_range:
            self.clip_range_decay_rate = cfg.alg.clip_range / float(total_epochs)

        p_lr_lambda = partial(linear_decay_percent,
                              total_epochs=total_epochs)
        optim_args = dict(
            lr=cfg.alg.policy_lr,
            weight_decay=cfg.alg.weight_decay
        )
        if not cfg.alg.sgd:
            optim_args['amsgrad'] = cfg.alg.use_amsgrad
            optim_func = optim.Adam
        else:
            optim_args['nesterov'] = True if cfg.alg.momentum > 0 else False
            optim_args['momentum'] = cfg.alg.momentum
            optim_func = optim.SGD
        if self.same_body:
            optim_args['params'] = self.all_params
        else:
            optim_args['params'] = [{'params': self.actor.parameters(),
                                     'lr': cfg.alg.policy_lr},
                                    {'params': self.critic.parameters(),
                                     'lr': cfg.alg.value_lr}]

        self.optimizer = optim_func(**optim_args)

        if self.same_body:
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                         lr_lambda=[p_lr_lambda])
        else:
            v_lr_lambda = partial(linear_decay_percent,
                                  total_epochs=total_epochs)
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                         lr_lambda=[p_lr_lambda, v_lr_lambda])

    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        self.eval_mode()
        t_ob = torch_float(ob, device=cfg.alg.device)
        act_dist, val = self.get_act_val(t_ob)
        action = action_from_dist(act_dist,
                                  sample=sample)
        log_prob = action_log_prob(action, act_dist)
        entropy = action_entropy(act_dist, log_prob)
        action_info = dict(
            log_prob=torch_to_np(log_prob),
            entropy=torch_to_np(entropy),
            val=torch_to_np(val)
        )
        return torch_to_np(action), action_info

    def get_act_val(self, ob, *args, **kwargs):
        ob = torch_float(ob, device=cfg.alg.device)
        act_dist, body_out = self.actor(ob)
        if self.same_body:
            val, body_out = self.critic(body_x=body_out)
        else:
            val, body_out = self.critic(x=ob)
        val = val.squeeze(-1)
        return act_dist, val

    @torch.no_grad()
    def get_val(self, ob, *args, **kwargs):
        self.eval_mode()
        ob = torch_float(ob, device=cfg.alg.device)
        val, body_out = self.critic(x=ob)
        val = val.squeeze(-1)
        return val

    def optimize(self, data, *args, **kwargs):
        pre_res = self.optim_preprocess(data)
        processed_data = pre_res
        processed_data['entropy'] = torch.mean(processed_data['entropy'])
        loss_res = self.cal_loss(**processed_data)
        loss, pg_loss, vf_loss, ratio = loss_res
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = clip_grad(self.all_params, cfg.alg.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            approx_kl = 0.5 * torch.mean(torch.pow(processed_data['old_log_prob'] -
                                                   processed_data['log_prob'], 2))
            clip_frac = np.mean(np.abs(torch_to_np(ratio) - 1.0) > cfg.alg.clip_range)
        optim_info = dict(
            pg_loss=pg_loss.item(),
            vf_loss=vf_loss.item(),
            total_loss=loss.item(),
            entropy=processed_data['entropy'].item(),
            approx_kl=approx_kl.item(),
            clip_frac=clip_frac
        )
        optim_info['grad_norm'] = grad_norm
        return optim_info

    def optim_preprocess(self, data):
        self.train_mode()
        for key, val in data.items():
            data[key] = torch_float(val, device=cfg.alg.device)
        ob = data['ob']
        action = data['action']
        ret = data['ret']
        adv = data['adv']
        old_log_prob = data['log_prob']
        old_val = data['val']

        act_dist, val = self.get_act_val(ob)
        log_prob = action_log_prob(action, act_dist)
        entropy = action_entropy(act_dist, log_prob)
        if not all([x.ndim == 1 for x in [val, entropy, log_prob]]):
            raise ValueError('val, entropy, log_prob should be 1-dim!')
        processed_data = dict(
            val=val,
            old_val=old_val,
            ret=ret,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            adv=adv,
            entropy=entropy
        )
        return processed_data

    def cal_loss(self, val, old_val, ret, log_prob, old_log_prob, adv, entropy):
        vf_loss = self.cal_val_loss(val=val, old_val=old_val, ret=ret)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = adv * ratio
        surr2 = adv * torch.clamp(ratio,
                                  1 - cfg.alg.clip_range,
                                  1 + cfg.alg.clip_range)
        pg_loss = -torch.mean(torch.min(surr1, surr2))

        loss = pg_loss - entropy * cfg.alg.ent_coef + \
               vf_loss * cfg.alg.vf_coef
        return loss, pg_loss, vf_loss, ratio

    def cal_val_loss(self, val, old_val, ret):
        if cfg.alg.clip_vf_loss:
            clipped_val = old_val + torch.clamp(val - old_val,
                                                -cfg.alg.clip_range,
                                                cfg.alg.clip_range)
            vf_loss1 = torch.pow(val - ret, 2)
            vf_loss2 = torch.pow(clipped_val - ret, 2)
            vf_loss = 0.5 * torch.mean(torch.max(vf_loss1,
                                                 vf_loss2))
        else:
            # val = torch.squeeze(val)
            vf_loss = 0.5 * self.val_loss_criterion(val, ret)
        return vf_loss

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def decay_lr(self):
        self.lr_scheduler.step()

    def get_lr(self):
        cur_lr = self.lr_scheduler.get_lr()
        lrs = {'policy_lr': cur_lr[0]}
        if len(cur_lr) > 1:
            lrs['value_lr'] = cur_lr[1]
        return lrs

    def decay_clip_range(self):
        cfg.alg.clip_range -= self.clip_range_decay_rate

    def save_model(self, is_best=False, step=None):
        self.save_env(cfg.alg.model_dir)
        data_to_save = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }

        if cfg.alg.linear_decay_clip_range:
            data_to_save['clip_range'] = cfg.alg.clip_range
            data_to_save['clip_range_decay_rate'] = self.clip_range_decay_rate
        save_model(data_to_save, cfg.alg, is_best=is_best, step=step)

    def load_model(self, step=None, pretrain_model=None):
        self.load_env(cfg.alg.model_dir)
        ckpt_data = load_ckpt_data(cfg.alg, step=step,
                                   pretrain_model=pretrain_model)
        load_state_dict(self.actor,
                        ckpt_data['actor_state_dict'])
        load_state_dict(self.critic,
                        ckpt_data['critic_state_dict'])
        if pretrain_model is not None:
            return
        self.optimizer.load_state_dict(ckpt_data['optim_state_dict'])
        self.lr_scheduler.load_state_dict(ckpt_data['lr_scheduler_state_dict'])
        if cfg.alg.linear_decay_clip_range:
            self.clip_range_decay_rate = ckpt_data['clip_range_decay_rate']
            cfg.alg.clip_range = ckpt_data['clip_range']
        return ckpt_data['step']

    def print_param_grad_status(self):
        logger.info('Requires Grad?')
        logger.info('================== Actor ================== ')
        for name, param in self.actor.named_parameters():
            print(f'{name}: {param.requires_grad}')
        logger.info('================== Critic ================== ')
        for name, param in self.critic.named_parameters():
            print(f'{name}: {param.requires_grad}')
