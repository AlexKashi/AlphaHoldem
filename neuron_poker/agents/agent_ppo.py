"""manual keypress agent"""
#from rl.memory import SequentialMemory
import time

from agents.agent_keras_rl_dqn import TrumpPolicy, memory_limit, window_length
from gym_env import env


#added from easyrl
import gym
from pathlib import Path
import torch
from torch import nn
from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env
from easyrl.envs.dummy_vec_env import DummyVecEnv
from easyrl.utils.common import load_from_json


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.model = None


        set_config('ppo')
        if(name is not None):
            self.load(name)
        else:
            now = time.strftime("%Y-%m-%d-%H:%M:%S")
            cfg.alg.env_name = f"PPO-{now}"
            cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
        episode_steps = 128
        itters = 10000
        print(f"Training for {episode_steps * itters} Steps!")
        cfg.alg.num_envs = 1
        cfg.alg.episode_steps = episode_steps
        cfg.alg.log_interval = 2
        cfg.alg.eval_interval = 2
        cfg.alg.max_steps = episode_steps * itters
        cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = cfg.alg.device
        print(cfg.alg.device)



     #   self.env = make_vec_env(name,1)



    def load(self, path = None):
        """Load a model"""
        cfg.alg.resume = True
        if path is None:
            print("Path is empty, using default!")
            path = "data/PPO/default/seed_0"
        print(f"Loading: {path}")
        cfg.alg.restore_cfg(skip_params=[], path = Path(path))


    def initiate_agent(self, env):
        self.env = env

        def wrapper():
            return self.env
        self.envwrapped = DummyVecEnv([wrapper])

        nb_obs = self.env.observation_space[0]
        nb_actions = self.env.action_space.n

        nlayers = 3
        nnodes = 512


     #   assert False
        actor_body = MLP(input_size=nb_obs,
                         hidden_sizes=[nnodes] * nlayers,
                         output_size=nnodes,
                         hidden_act=nn.Tanh,
                         output_act=nn.Tanh).to(self.device)


        critic_body = MLP(input_size=nb_obs,
                         hidden_sizes=[nnodes] * nlayers,
                         output_size=nnodes,
                         hidden_act=nn.Tanh,
                         output_act=nn.Tanh).to(self.device)

 
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.actor = CategoricalPolicy(actor_body,
                                     in_features=nnodes,
                                     action_dim=nb_actions)
        else:
            raise TypeError(f'Unknown action space type: {env.action_space}')

           
        self.critic = ValueNet(critic_body, in_features=nnodes)
        self.agent = PPOAgent(actor=self.actor, critic=self.critic, env=self.envwrapped)


        # memory = SequentialMemory(limit=memory_limit, window_length=window_length)  # pylint: disable=unused-variable
        # policy = TrumpPolicy()  # pylint: disable=unused-variable


    def train(self, env_name):
        """Train a model"""
        # initiate training loop


        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        # tensorboard = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
        #                           write_images=False)

        # self.dqn.fit(self.env, nb_max_start_steps=nb_max_start_steps, nb_steps=nb_steps, visualize=False, verbose=2,
        #              start_step_policy=self.start_step_policy, callbacks=[tensorboard])

        # env = make_vec_env("neuron_poker-v0",
        #                    1)




        runner = EpisodicRunner(agent=self.agent, env=self.envwrapped)
        engine = PPOEngine(agent=self.agent,
                           runner=runner)
        engine.train()
        # assert False


        # # Save the architecture
        # dqn_json = self.model.to_json()
        # with open("dqn_{}_json.json".format(env_name), "w") as json_file:
        #     json.dump(dqn_json, json_file)

        # # After training is done, we save the final weights.
        # self.dqn.save_weights('dqn_{}_weights.h5'.format(env_name), overwrite=True)

        # # Finally, evaluate our algorithm for 5 episodes.
        # self.dqn.test(self.env, nb_episodes=5, visualize=False)





    def play(self, nb_episodes=5, render=True):
        """Let the agent play"""

        set_config('ppo')
        cfg.alg.num_envs = 1
        cfg.alg.episode_steps = 2048
        cfg.alg.log_interval = 2
        cfg.alg.eval_interval = 2
        cfg.alg.max_steps = cfg.alg.episode_steps * 100
        cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.alg.test = True

        skip_params = ['test_num', "num_envs", "sample_action"]

        # path = "data/PPO/default/seed_0"
        path = self.name        

       # if path is none:
       #     path = "data/PPO-2022-05-02-16:55:32/default/seed_0"
 
        cfg.alg.restore_cfg(skip_params=skip_params, path = Path(path))

        print(cfg.alg.resume, cfg.alg.test)
       # assert False

        runner = EpisodicRunner(agent=self.agent, env=self.envwrapped)
        engine = PPOEngine(agent=self.agent,
                           runner=runner)
   
        stat_info, raw_traj_info = engine.eval(render=render,
                                               save_eval_traj=False,
                                               eval_num=nb_episodes,
                                               sleep_time=0.04)

        import pprint
        pprint.pprint(stat_info)
        return stat_info, raw_traj_info

    def read_tf_log(log_dir, scalar='train/episode_return/mean'):
        log_dir = Path(log_dir)
        log_files = list(log_dir.glob(f'**/events.*'))
        if len(log_files) < 1:
            return None
        log_file = log_files[0]

        event_acc = EventAccumulator(log_file.as_posix())
        event_acc.Reload()
        tags = event_acc.Tags()

        scalar_return = event_acc.Scalars(scalar)
        returns = [x.value for x in scalar_return]
        steps = [x.step for x in scalar_return]

        return steps, returns

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use,unused-argument
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = (observation, info)  # not using the observation for random decision
        
        action = self.agent.get_action(observation, sample = True)[0]
        print(action)
        return action
