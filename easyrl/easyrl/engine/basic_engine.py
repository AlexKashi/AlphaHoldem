from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from easyrl.configs import cfg
from easyrl.utils.rl_logger import TensorboardLogger
from easyrl.utils.common import get_list_stats

@dataclass
class BasicEngine:
    agent: Any
    runner: Any

    def __post_init__(self):
        self.cur_step = 0
        self._best_eval_ret = -np.inf
        self._eval_is_best = False
        if cfg.alg.test or cfg.alg.resume:
            self.cur_step = self.agent.load_model(step=cfg.alg.resume_step)
        else:
            if cfg.alg.pretrain_model is not None:
                self.agent.load_model(pretrain_model=cfg.alg.pretrain_model)
            cfg.alg.create_model_log_dir()
        self.train_ep_return = deque(maxlen=100)
        self.smooth_eval_return = None
        self.smooth_tau = cfg.alg.smooth_eval_tau
        self.optim_stime = None
        if not cfg.alg.test:
            self.tf_logger = TensorboardLogger(log_dir=cfg.alg.log_dir)

    def train(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError

    def get_train_log(self, optim_infos, traj=None):
        log_info = dict()
        vector_keys = set()
        scalar_keys = set()
        for oinf in optim_infos:
            for key in oinf.keys():
                if 'vec_' in key:
                    vector_keys.add(key)
                else:
                    scalar_keys.add(key)

        for key in scalar_keys:
            log_info[key] = np.mean([inf[key] for inf in optim_infos if key in inf])

        for key in vector_keys:
            k_stats = get_list_stats([inf[key] for inf in optim_infos if key in inf])
            for sk, sv in k_stats.items():
                log_info[f'{key}/' + sk] = sv
        if traj is not None:
            actions_stats = get_list_stats(traj.actions)
            for sk, sv in actions_stats.items():
                log_info['rollout_action/' + sk] = sv
            log_info['rollout_steps_per_iter'] = traj.total_steps

            ep_returns_stats = get_list_stats(self.runner.train_ep_return)
            for sk, sv in ep_returns_stats.items():
                log_info['episode_return/' + sk] = sv

            if len(self.runner.train_success) > 0:
                log_info['episode_success'] = np.mean(self.runner.train_success)

        train_log_info = dict()
        for key, val in log_info.items():
            train_log_info['train/' + key] = val
        return train_log_info