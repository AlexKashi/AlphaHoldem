import logging

import gym
import numpy as np
import pandas as pd
from docopt import docopt

from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger




from agents.agent_ppo import Player as PPOPlayer
from agents.agent_consider_equity import Player as EquityPlayer
from agents.agent_keras_rl_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
env_name = 'neuron_poker-v0'




plot = False
render = False
num_episodes = 30
stack = 500
env = gym.make('neuron_poker-v0', initial_stacks= stack, funds_plot=plot, render=render)

np.random.seed(54)
env.seed(54)



agent1 = PPOPlayer(name="data/PPO-2022-05-03-00:30:38/default/seed_0")
dqn = DQNPlayer(name="output/dqn_dqn1_20220503-022055_dqn1",load_model = True,  env=env, enable_double_dqn = True, enable_dueling_network = True)
agent2 = PPOPlayer(name="data/PPO-2022-05-03-00:30:20/default/seed_0")

env.add_player(dqn)
# env.add_player(agent1)
env.add_player(PlayerShell(name='ppo', stack_size=stack))

env.reset()

#ppoAgent = PPOPlayer(name="data/PPO-2022-05-02-16:55:32/default/seed_0")
dqn.initiate_agent(env)
agent1.initiate_agent(env)
agent2.initiate_agent(env)

agent2.play(nb_episodes=num_episodes, render=render)


