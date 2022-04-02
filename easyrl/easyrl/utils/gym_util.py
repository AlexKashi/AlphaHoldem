from copy import deepcopy

import gym
import numpy as np
from easyrl.envs.dummy_vec_env import DummyVecEnv
from easyrl.envs.shmem_vec_env import ShmemVecEnv
from easyrl.envs.subproc_vec_env import SubprocVecEnv
from easyrl.envs.timeout import NoTimeOutEnv
from easyrl.envs.vec_normalize import VecNormalize
from easyrl.utils.common import load_from_pickle
from easyrl.utils.common import pathlib_file
from easyrl.utils.common import save_to_pickle
from easyrl.utils.rl_logger import logger
from gym.spaces import Box
from gym.spaces import Dict
from gym.spaces import Discrete
from gym.spaces import MultiBinary
from gym.spaces import MultiDiscrete
from gym.spaces import Tuple
from gym.wrappers.time_limit import TimeLimit


def num_space_dim(space):
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return int(space.n)
    elif isinstance(space, Tuple):
        return int(sum([num_space_dim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([num_space_dim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError

def make_vec_env(env_id=None, num_envs=1, seed=1, env_func=None, no_timeout=False,
                 env_kwargs=None, distributed=False,
                 extra_wrapper=None, wrapper_kwargs=None):
    logger.info(f'Creating {num_envs} environments.')
    if env_kwargs is None:
        env_kwargs = {}
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    if distributed:
        import horovod.torch as hvd
        seed_offset = hvd.rank() * 100000
        seed += seed_offset

    def make_env(env_id, rank, seed, no_timeout, env_kwargs, extra_wrapper, wrapper_kwargs):
        def _thunk():
            if env_func is not None:
                env = env_func(**env_kwargs)
            else:
                env = gym.make(env_id, **env_kwargs)
            if no_timeout:
                env = NoTimeOutEnv(env)
            if extra_wrapper is not None:
                env = extra_wrapper(env, **wrapper_kwargs)
            env.seed(seed + rank)
            return env
        return _thunk

    envs = [make_env(env_id,
                     idx,
                     seed,
                     no_timeout,
                     env_kwargs,
                     extra_wrapper,
                     wrapper_kwargs
                     ) for idx in range(num_envs)]
    if num_envs > 1:
        envs = ShmemVecEnv(envs, context='spawn')
    else:
        envs = DummyVecEnv(envs)
    return envs


def get_render_images(env):
    try:
        img = env.get_images()
    except AttributeError:
        try:
            img = env.render('rgb_array')
        except AttributeError:
            raise AttributeError('Cannot get rendered images.')
    return deepcopy(img)


def is_time_limit_env(env):
    if not (isinstance(env, TimeLimit)):
        if not hasattr(env, 'env'):
            return False
        else:
            return is_time_limit_env(env.env)
    return True


def save_vec_normalized_env(env, save_dir):
    save_dir = pathlib_file(save_dir)
    save_file = save_dir.joinpath('vecnorm_env.pkl')
    assert isinstance(env, VecNormalize)
    data = env.get_states()
    save_to_pickle(data, save_file)


def load_vec_normalized_env(env, save_dir):
    save_dir = pathlib_file(save_dir)
    save_file = save_dir.joinpath('vecnorm_env.pkl')
    assert isinstance(env, VecNormalize)
    data = load_from_pickle(save_file)
    env.set_states(data)


def get_true_done(done, info):
    return done and not info.get('TimeLimit.truncated', False)
