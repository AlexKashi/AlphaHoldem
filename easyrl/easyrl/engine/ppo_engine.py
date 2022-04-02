import time
from itertools import chain
from itertools import count

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from easyrl.configs import cfg
from easyrl.engine.basic_engine import BasicEngine
from easyrl.utils.common import get_list_stats
from easyrl.utils.common import save_traj
from easyrl.utils.gae import cal_gae
from easyrl.utils.torch_util import EpisodeDataset


class PPOEngine(BasicEngine):
    def train(self):
        for iter_t in count():
            if iter_t % cfg.alg.eval_interval == 0:
                print(cfg.alg.test_num)
                det_log_info, _ = self.eval(eval_num=cfg.alg.test_num, sample=False, smooth=True)
                sto_log_info, _ = self.eval(eval_num=cfg.alg.test_num,
                                            sample=True, smooth=False)

                det_log_info = {f'det/{k}': v for k, v in det_log_info.items()}
                sto_log_info = {f'sto/{k}': v for k, v in sto_log_info.items()}
                eval_log_info = {**det_log_info, **sto_log_info}
                self.agent.save_model(is_best=self._eval_is_best,
                                      step=self.cur_step)
            else:
                eval_log_info = None
            traj, rollout_time = self.rollout_once(sample=True,
                                                   get_last_val=True,
                                                   time_steps=cfg.alg.episode_steps)

            train_log_info = self.train_once(traj)
            if iter_t % cfg.alg.log_interval == 0:
                train_log_info['train/rollout_time'] = rollout_time
                if eval_log_info is not None:
                    train_log_info.update(eval_log_info)
                if cfg.alg.linear_decay_lr:
                    train_log_info.update(self.agent.get_lr())
                if cfg.alg.linear_decay_clip_range:
                    train_log_info.update(dict(clip_range=cfg.alg.clip_range))
                scalar_log = {'scalar': train_log_info}
                self.tf_logger.save_dict(scalar_log, step=self.cur_step)
            if self.cur_step > cfg.alg.max_steps:
                break
            if cfg.alg.linear_decay_lr:
                self.agent.decay_lr()
            if cfg.alg.linear_decay_clip_range:
                self.agent.decay_clip_range()

    @torch.no_grad()
    def eval(self, render=False, save_eval_traj=False, eval_num=1,
             sleep_time=0, sample=True, smooth=True, no_tqdm=None):
        time_steps = []
        rets = []
        lst_step_infos = []
        successes = []
        if no_tqdm:
            disable_tqdm = bool(no_tqdm)
        else:
            disable_tqdm = not cfg.alg.test
        for idx in tqdm(range(eval_num), disable=disable_tqdm):
            traj, _ = self.rollout_once(time_steps=cfg.alg.episode_steps,
                                        return_on_done=True,
                                        sample=cfg.alg.sample_action and sample,
                                        render=render,
                                        sleep_time=sleep_time,
                                        render_image=save_eval_traj,
                                        evaluation=True)

        
            tsps = traj.steps_til_done.copy().tolist()
            rewards = traj.raw_rewards
            infos = traj.infos
            for ej in range(traj.num_envs):
                ret = np.sum(rewards[:tsps[ej], ej])
                rets.append(ret)
                lst_step_infos.append(infos[tsps[ej] - 1][ej])
            time_steps.extend(tsps)
            if save_eval_traj:
                save_traj(traj, cfg.alg.eval_dir)
            if 'success' in infos[0][0]:
                successes.extend([infos[tsps[ej] - 1][ej]['success'] for ej in range(rewards.shape[1])])

        raw_traj_info = {'return': rets,
                         'episode_length': time_steps,
                         'lst_step_info': lst_step_infos}
        log_info = dict()
        for key, val in raw_traj_info.items():
            if 'info' in key:
                continue
            val_stats = get_list_stats(val)
            for sk, sv in val_stats.items():
                log_info['eval/' + key + '/' + sk] = sv
        if len(successes) > 0:
            log_info['eval/success'] = np.mean(successes)
        if smooth:
            if self.smooth_eval_return is None:
                self.smooth_eval_return = log_info['eval/return/mean']
            else:
                self.smooth_eval_return = self.smooth_eval_return * self.smooth_tau
                self.smooth_eval_return += (1 - self.smooth_tau) * log_info['eval/return/mean']
            log_info['eval/smooth_return/mean'] = self.smooth_eval_return
            if self.smooth_eval_return > self._best_eval_ret:
                self._eval_is_best = True
                self._best_eval_ret = self.smooth_eval_return
            else:
                self._eval_is_best = False
        return log_info, raw_traj_info

    def rollout_once(self, *args, **kwargs):
        t0 = time.perf_counter()
        self.agent.eval_mode()
        traj = self.runner(**kwargs)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        return traj, elapsed_time

    def train_once(self, traj):
        self.optim_stime = time.perf_counter()
        self.cur_step += traj.total_steps
        rollout_dataloader = self.traj_preprocess(traj)
        optim_infos = []
        for oe in range(cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                optim_info = self.agent.optimize(batch_data)
                optim_infos.append(optim_info)
        return self.get_train_log(optim_infos, traj)

    def traj_preprocess(self, traj):
        action_infos = traj.action_infos
        vals = np.array([ainfo['val'] for ainfo in action_infos])
        log_prob = np.array([ainfo['log_prob'] for ainfo in action_infos])
        adv = self.cal_advantages(traj)
        ret = adv + vals
        if cfg.alg.normalize_adv:
            adv = adv.astype(np.float64)
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        data = dict(
            ob=traj.obs,
            action=traj.actions,
            ret=ret,
            adv=adv,
            log_prob=log_prob,
            val=vals
        )
        rollout_dataset = EpisodeDataset(**data)
        rollout_dataloader = DataLoader(rollout_dataset,
                                        batch_size=cfg.alg.batch_size,
                                        shuffle=True)
        return rollout_dataloader

    def cal_advantages(self, traj):
        rewards = traj.rewards
        action_infos = traj.action_infos
        vals = np.array([ainfo['val'] for ainfo in action_infos])
        last_val = traj.extra_data['last_val']
        adv = cal_gae(gamma=cfg.alg.rew_discount,
                      lam=cfg.alg.gae_lambda,
                      rewards=rewards,
                      value_estimates=vals,
                      last_value=last_val,
                      dones=traj.dones)
        return adv

