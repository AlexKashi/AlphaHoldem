import logging

import gym
import numpy as np
import pandas as pd
from docopt import docopt
import matplotlib.pyplot as plt
from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


import torch
from easyrl.configs import cfg
from easyrl.configs import set_config
from agents.agent_ppo import Player as PPOPlayer
from agents.agent_consider_equity import Player as EquityPlayer
from agents.agent_keras_rl_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
env_name = 'neuron_poker-v0'


set_config('ppo')
cfg.alg.num_envs = 1
cfg.alg.episode_steps = 2048
cfg.alg.log_interval = 2
cfg.alg.eval_interval = 2
cfg.alg.max_steps = cfg.alg.episode_steps * 100
cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.alg.test = True




plot = False
render = False
random = True
DQN = True
num_episodes = 10
stack = 500
env = gym.make('neuron_poker-v0', initial_stacks= stack, funds_plot=plot, render=render)

np.random.seed(54)
env.seed(54)

equityLogFilesPPO = ["data/PPO-2022-05-02-16:53:32", "data/PPO-2022-05-03-00:29:59", "data/PPO-2022-05-03-00:30:14"]
randomLogsPPO = ["data/PPO-2022-05-02-16:55:32", "data/PPO-2022-05-03-00:30:20", "data/PPO-2022-05-03-00:30:38"]

equityLogFilesPPO.extend(["data/PPO-2022-05-04-04:10:32", "data/PPO-2022-05-04-04:10:42", "data/PPO-2022-05-04-04:10:47"])
randomLogsPPO.extend(["data/PPO-2022-05-04-04:11:01", "data/PPO-2022-05-04-04:11:04", "data/PPO-2022-05-04-04:11:08"])

agentsList = equityLogFilesPPO + randomLogsPPO

res = []
for file in agentsList:
    res.append(file + "/default/seed_0")
agentsList = res
print(agentsList)



randomLogsDQN = ["Graph-random/20220504-040921_dqn1", "Graph-random/20220504-040928_dqn1", "Graph-random/20220504-040935_dqn1", "Graph/20220504-170004_dqn1", "Graph/20220504-170007_dqn1"]
randomLogsDoubleDQN = ["Graph-random/20220504-040942_dqn1", "Graph-random/20220504-040947_dqn1", "Graph-random/20220504-040952_dqn1", "Graph/20220504-170010_dqn1", "Graph/20220504-170012_dqn1", "Graph/20220504-170014_dqn1"]
randomLogsDuelingDQN = ["Graph-random/20220504-040955_dqn1", "Graph-random/20220504-040958_dqn1", "Graph-random/20220504-041000_dqn1", "Graph/20220504-170017_dqn1" , "Graph/20220504-170022_dqn1", "Graph/20220504-170027_dqn1"]
randomLogsDoubleDuelingDQN = ["Graph/20220504-170121_dqn1", "Graph/20220504-170136_dqn1", "Graph/20220504-170151_dqn1"]
equityLogsDQN = ["20220503-005836_dqn1-normal", "20220503-005943_dqn1-normal", "20220503-005951_dqn1-normal", "20220503-113448_output-normal", "20220503-113543_output-normal"]
equityLogsDoubleDQN = ["20220503-010131_dqn1-double", "20220503-010149_dqn1-double", "20220503-010156_dqn1-double", "20220503-113615_output-double", "20220503-113638_output-double", "20220503-113654_output-double"]
eqityLogsDuelingDQN = ["20220503-010210_dqn1-dueling", "20220503-010225_dqn1-dueling", "20220503-022055_dqn1-dueling"]

import glob


agentsList = list("_".join(f.split("_")[:-1]) for f in glob.glob("output/*.h5"))
print(agentsList)
#assert False
# "dqn_dqn1_20220504-170151_dqn1",
# agentsList = randomLogsDQN + randomLogsDoubleDQN + randomLogsDuelingDQN + randomLogsDoubleDuelingDQN + equityLogsDQN + equityLogsDoubleDQN  + eqityLogsDuelingDQN






# for random in [True, False]:
#     bestAgent = None
#     bestReward = -1e9

#     for agentFile in agentsList:
#         env = gym.make('neuron_poker-v0', initial_stacks= stack, funds_plot=plot, render=render)


#         np.random.seed(123)
#         env.seed(123)

#         if random:
#             print("Adding Random Agent")
#             env.add_player(RandomPlayer())
#         else:
#             print("Adding Equity Agent")
#             env.add_player(EquityPlayer(name='equity/50/70', min_call_equity=.5, min_bet_equity=.7))

#         if(DQN):

#             dqn = DQNPlayer(name = agentFile, load_model=True, env=env, enable_double_dqn = True, enable_dueling_network = True)
#             env.add_player(dqn)
#             # env.add_player(PlayerShell(name='keras-rl', stack_size=stack))
#             env.reset()
#             dqn.initiate_agent(env)
#       #      dqn.initiate_agent()
#             # dqn.play(nb_episodes=num_episodes, render=render)
#         else:
#             env.add_player(PlayerShell(name='ppo', stack_size=stack))
#             agent = PPOPlayer(name=agentFile)
#             env.reset()
#             agent.initiate_agent(env)
#             results = agent.play(nb_episodes=num_episodes, render=render)

#         curReward = results["eval/return/mean"]
#         if(curReward > bestReward):
#             print(f"Update Best VS Random:{random}: {bestReward} to {curReward} {agentFile}")
#             bestReward = curReward
#             bestAgent = agentFile
#         else:
#             print(f"Skip Update VS Random:{random}: {bestReward} to {curReward} {agentFile}")
#     print(f"@@@@@ Best Agent VS Random:{random} {bestAgent}: {bestReward}")

bestAgentVsEquityPPO = "data/PPO-2022-05-02-16:53:32/default/seed_0" # ~230
bestAgentVsRandomPPO = "data/PPO-2022-05-04-04:11:08/default/seed_0" # 627
plt.rcParams.update({'font.size': 50})

def runPPO(fileName, random = False, num_episodes = 50):
    env = gym.make('neuron_poker-v0', initial_stacks= stack, funds_plot=False, render=render)

    np.random.seed(1)
    env.seed(1)

    if random:
        print("Adding Random Agent")
        env.add_player(RandomPlayer())
    else:
        print("Adding Equity Agent")
        env.add_player(EquityPlayer(name='equity/50/70', min_call_equity=.5, min_bet_equity=.7))

    env.add_player(PlayerShell(name='ppo', stack_size=stack))
    agent = PPOPlayer(name=fileName)
    env.reset()
    agent.initiate_agent(env)
    results = agent.play(nb_episodes=num_episodes, render=render)

    print("WHAT THE FUCK")
    print(results[0])
    print("FOFOF")
    print(results[1])
    print(results[1]["return"])

    wins = sum([ret > 0 for ret in results[1]["return"]])
    losses = num_episodes - wins
    plt.figure()
    plt.bar([0, 1], [wins,losses])
    agentName = "Random" if random else "Equity"
    plt.title(f"Best agent Vs {agentName} Agent")

    bars = ["Wins", "Losses"]
    plt.xticks([0,1], bars)
   # plt.show()
# runPPO(bestAgentVsEquityPPO, False)
# runPPO(bestAgentVsRandomPPO, True)

plt.plot()
runPPO(bestAgentVsEquityPPO, False)
runPPO(bestAgentVsRandomPPO, True)

plt.show()





print(bestAgent)
agent1 = PPOPlayer(name=bestAgentVsEquityPPO)
env.add_player(EquityPlayer(name='equity/50/70', min_call_equity=.5, min_bet_equity=.7))




# dqn = DQNPlayer(name="output/dqn_dqn1_20220503-022055_dqn1",load_model = True,  env=env, enable_double_dqn = True, enable_dueling_network = True)
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


