from dataclasses import dataclass
from typing import Any

from easyrl.configs.ppo_config import PPOConfig
from easyrl.configs.sac_config import SACConfig
from easyrl.utils.rl_logger import logger


@dataclass
class CFG:
    alg: Any = None


cfg = CFG()


def set_config(alg=None, config_func=None):
    global cfg
    if config_func is not None:
        cfg.alg = config_func()
        logger.info(f'Alogrithm type:{config_func.__name__}')
        return
    if alg == 'ppo':
        cfg.alg = PPOConfig()
    elif alg == 'sac':
        cfg.alg = SACConfig()
    else:
        raise ValueError(f'Unimplemented algorithm: {alg}')
    logger.info(f'Alogrithm type:{type(cfg.alg)}')
