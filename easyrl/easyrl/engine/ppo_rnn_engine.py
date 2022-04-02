import numpy as np
from torch.utils.data import DataLoader

from easyrl.configs import cfg
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.utils.torch_util import DictDataset


class PPORNNEngine(PPOEngine):
    def __init__(self, agent, runner):
        super().__init__(agent=agent,
                         runner=runner)

    def traj_preprocess(self, traj):
        action_infos = traj.action_infos
        vals = np.array([ainfo['val'] for ainfo in action_infos])
        log_prob = np.array([ainfo['log_prob'] for ainfo in action_infos])
        hidden_state = action_infos[0]['in_hidden_state']
        if hidden_state is not None:
            hidden_state = hidden_state.swapaxes(0, 1)
        else:
            hidden_state_shape = self.runner.hidden_state_shape
            hidden_state = np.zeros((vals.shape[1], hidden_state_shape[0], hidden_state_shape[2]))
        adv = self.cal_advantages(traj)
        ret = adv + vals
        if cfg.alg.normalize_adv:
            adv = adv.astype(np.float64)
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        # TxN --> NxT
        data = dict(
            ob=traj.obs.swapaxes(0, 1),
            action=traj.actions.swapaxes(0, 1),
            ret=ret.swapaxes(0, 1),
            adv=adv.swapaxes(0, 1),
            log_prob=log_prob.swapaxes(0, 1),
            val=vals.swapaxes(0, 1),
            done=traj.step_extras.swapaxes(0, 1), # we use the mask here instead of true_done
            hidden_state=hidden_state,
        )
        rollout_dataset = DictDataset(**data)
        rollout_dataloader = DataLoader(rollout_dataset,
                                        batch_size=cfg.alg.batch_size,
                                        shuffle=True)
        return rollout_dataloader
