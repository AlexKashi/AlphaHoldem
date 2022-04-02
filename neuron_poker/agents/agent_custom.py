"""manual keypress agent"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from rl.memory import SequentialMemory
import time

from agents.agent_keras_rl_dqn import TrumpPolicy, memory_limit, window_length
from gym_env import env


#added from easyrl
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

    def __init__(self, name='Custom_Q1'):
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
        
        cfg.alg.max_steps = 100000



     #   self.env = make_vec_env(name,1)

    def initiate_agent(self, env):
        self.env = env 
        nb_obs = env.observation_space[0]
        nb_actions = env.action_space.n
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
        print(env.observation_space)
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
        self.actor = CategoricalPolicy(actor_body,
                                 in_features=64,
                                 action_dim=nb_actions)            
        self.critic = ValueNet(critic_body, in_features=64)


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


        def wrapper():
            return self.env
        env = DummyVecEnv([wrapper])

        self.agent = PPOAgent(actor=self.actor, critic=self.critic, env=env)

        runner = EpisodicRunner(agent=self.agent, env=env)
        engine = PPOEngine(agent=self.agent,
                           runner=runner)
        engine.train()



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
        action = None

        self.initiate_agent(4)

        # decide if explore or explot

        # forward

        # save to memory

        # backward
        # decide what to use for training
        # update model
        # save weights

        return action
