"""manual keypress agent"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from rl.memory import SequentialMemory
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
        cfg.alg.num_envs = 1
        cfg.alg.episode_steps = 1024
        cfg.alg.log_interval = 1
        cfg.alg.eval_interval = 20
        
        cfg.alg.max_steps = 100



     #   self.env = make_vec_env(name,1)

    def initiate_agent(self, env):
        self.env = env

        def wrapper():
            return self.env
        self.envwrapped = DummyVecEnv([wrapper])

        nb_obs = self.env.observation_space[0]
        nb_actions = self.env.action_space.n
        cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.alg.env_name = "name"
        cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
        cfg.alg.save_dir += '/' + "name"
        # self.model = Sequential()
        # self.model.add(Dense(512, activation='relu', input_shape=env.observation_space))  # pylint: disable=no-member
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(512, activation='relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(512, activation='relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(nb_actions, activation='linear'))
        assert False
        actor_body = MLP(input_size=nb_obs,
                         hidden_sizes=[64, 64],
                         output_size=64,
                         hidden_act=nn.Tanh,
                         output_act=nn.Tanh)

        critic_body = MLP(input_size=nb_obs,
                         hidden_sizes=[64, 64],
                         output_size=64,
                         hidden_act=nn.Tanh,
                         output_act=nn.Tanh)
        assert False

        if isinstance(env.action_space, gym.spaces.Discrete):
            act_size = env.action_space.n
            self.actor = CategoricalPolicy(actor_body,
                                     in_features=64,
                                     action_dim=act_size)
        elif isinstance(env.action_space, gym.spaces.Box):
            act_size = env.action_space.shape[0]
            self.actor = DiagGaussianPolicy(actor_body,
                                       in_features=64,
                                       action_dim=act_size,
                                       tanh_on_dist=cfg.alg.tanh_on_dist,
                                       std_cond_in=cfg.alg.std_cond_in)
        else:
            raise TypeError(f'Unknown action space type: {env.action_space}')

           
        self.critic = ValueNet(critic_body, in_features=64)
        self.agent = PPOAgent(actor=self.actor, critic=self.critic, env=self.envwrapped)
        # # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # # even the metrics!
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





    def action(self, action_space, observation, info):  # pylint: disable=no-self-use,unused-argument
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = (observation, info)  # not using the observation for random decision
        assert False
        action = self.agent.get_action(observation)[0].tolist()
        return None
        return action
