import torch
import torch.nn as nn

from easyrl.agents.sac_agent import SACAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.engine.sac_engine import SACEngine
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.replays.circular_buffer import CyclicBuffer
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env


def main():
    torch.set_num_threads(1)
    set_config('sac')
    cfg_from_cmd(cfg.alg)
    if cfg.alg.resume or cfg.alg.test:
        if cfg.alg.test:
            skip_params = [
                'test_num',
                'num_envs',
                'sample_action',
            ]
        else:
            skip_params = []
        cfg.alg.restore_cfg(skip_params=skip_params)
    if cfg.alg.env_name is None:
        cfg.alg.env_name = 'HalfCheetah-v3'
    if not cfg.alg.test:
        cfg.alg.test_num = 10
    set_random_seed(cfg.alg.seed)
    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs,
                       seed=cfg.alg.seed)
    # env = SingleEnvWrapper(gym.make(cfg.alg.env_name))
    eval_env = make_vec_env(cfg.alg.env_name,
                            cfg.alg.num_envs,
                            seed=cfg.alg.seed)
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    actor_body = MLP(input_size=ob_size,
                     hidden_sizes=[256],
                     output_size=256,
                     hidden_act=nn.ReLU,
                     output_act=nn.ReLU)
    q1_body = MLP(input_size=ob_size + act_size,
                  hidden_sizes=[256],
                  output_size=256,
                  hidden_act=nn.ReLU,
                  output_act=nn.ReLU)
    q2_body = MLP(input_size=ob_size + act_size,
                  hidden_sizes=[256],
                  output_size=256,
                  hidden_act=nn.ReLU,
                  output_act=nn.ReLU)
    actor = DiagGaussianPolicy(actor_body, action_dim=act_size,
                               tanh_on_dist=True,
                               std_cond_in=True,
                               clamp_log_std=True)
    q1 = ValueNet(q1_body)
    q2 = ValueNet(q2_body)
    memory = CyclicBuffer(capacity=cfg.alg.replay_size)
    agent = SACAgent(actor=actor, q1=q1, q2=q2, env=env, memory=memory)
    runner = EpisodicRunner(agent=agent, env=env, eval_env=eval_env)

    engine = SACEngine(agent=agent,
                       runner=runner)
    if not cfg.alg.test:
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(render=cfg.alg.render,
                                               save_eval_traj=cfg.alg.save_test_traj,
                                               eval_num=cfg.alg.test_num,
                                               sleep_time=0.04)
        import pprint
        pprint.pprint(stat_info)
    env.close()


if __name__ == '__main__':
    main()
