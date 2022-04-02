import numpy as np

from easyrl.envs.vec_env import VecEnvWrapper
from easyrl.utils.common import RunningMeanStd


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, training=True, ob=True, ret=True, clipob=10.,
                 cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            for idx, inf in enumerate(infos):
                inf['raw_reward'] = rews[idx]
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon),
                           -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def get_states(self):
        data = dict(
            ob_rms=self.ob_rms,
            ret_rms=self.ret_rms,
            clipob=self.clipob,
            cliprew=self.cliprew,
            gamma=self.gamma,
            epsilon=self.epsilon
        )
        return data

    def set_states(self, data):
        assert isinstance(data, dict)
        keys = ['ob_rms', 'ret_rms', 'clipob',
                'cliprew', 'gamma', 'epsilon']
        for key in keys:
            if key in data:
                setattr(self, key, data[key])
            else:
                print(f'Warning: {key} does not exist in data.')
