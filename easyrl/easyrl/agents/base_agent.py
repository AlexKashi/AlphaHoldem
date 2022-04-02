from dataclasses import dataclass
import gym
from easyrl.envs.vec_normalize import VecNormalize
from easyrl.utils.gym_util import save_vec_normalized_env
from easyrl.utils.gym_util import load_vec_normalized_env

@dataclass
class BaseAgent:
    env: gym.Env

    def get_action(self, ob, sample=True, **kwargs):
        raise NotImplementedError

    def optimize(self, data, **kwargs):
        raise NotImplementedError

    def save_env(self, save_dir):
        if isinstance(self.env, VecNormalize):
            save_vec_normalized_env(self.env, save_dir)

    def load_env(self, save_dir):
        if isinstance(self.env, VecNormalize):
            load_vec_normalized_env(self.env, save_dir)
