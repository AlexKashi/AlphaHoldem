import gym

import torch
from tqdm import tqdm
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
import math
import glob
import io
import base64
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


def read_tf_log(log_dir, scalar='train/episode_return/mean'):
    log_dir = Path(log_dir)
    log_files = sorted(list(log_dir.glob(f'**/events.*')))
    if len(log_files) < 1:
        return None
    steps = [0]
    returns = []
    print(log_files)
    for log_file in log_files:
 #   log_file = log_files[-1]
        try:
            event_acc = EventAccumulator(log_file.as_posix())
            event_acc.Reload()
            tags = event_acc.Tags()
            print(tags)
            scalar_return = event_acc.Scalars(scalar)
            returns.extend([x.value for x in scalar_return])
            s_0 = steps[-1] - scalar_return[0].step
            steps.extend([x.step + s_0  for x in scalar_return])
        except:
            pass

    return steps[1:], returns


# 'scalars': ['train/vf_loss',\
# 'train/entropy',\
# 'train/clip_frac',\
# 'train/grad_norm', \
# 'train/total_loss',\
# 'train/approx_kl',\
# 'train/pg_loss', \
# 'train/rollout_action/min',\
# 'train/rollout_action/max',\
# 'train/rollout_action/mean',\
# 'train/rollout_action/median',\
# 'train/rollout_steps_per_iter', \
# 'train/episode_return/min',\
# 'train/episode_return/max',\
# 'train/episode_return/mean',\
# 'train/episode_return/median',\
# 'train/rollout_time', \
# 'det/eval/return/min',\
# 'det/eval/return/max',\
# 'det/eval/return/mean', \
# 'det/eval/return/median',\
# 'det/eval/episode_length/min', \
# 'det/eval/episode_length/max',\
# 'det/eval/episode_length/mean',\
# 'det/eval/episode_length/median',\
# 'det/eval/smooth_return/mean',
# 'sto/eval/return/min', 
# 'sto/eval/return/max',
# 'sto/eval/return/mean',
# 'sto/eval/return/median',
# 'sto/eval/episode_length/min', 
# 'sto/eval/episode_length/max',
# 'sto/eval/episode_length/mean',
# 'sto/eval/episode_length/median']


def read_dqn_logs(logDir):
    steps, returns = read_tf_log(logDir, scalar = "epoch_episode_reward")
    rets = [0]
    alpha = 0.99
    for ret in returns:
        rets.append(rets[-1] * alpha + (1 - alpha) * ret)
    returns = rets[1:]

    return steps, returns
    # plt.plot(steps, returns)
    # plt.title("DQN Average Reward")
    # plt.xlabel("Steps")
    # plt.ylabel("Episode Return (Smoothed)")


def plot(logs, x_key, y_key, legend_key, **kwargs):
    nums = len(logs[legend_key].unique())
    palette = sns.color_palette("hls", nums)
    if 'palette' not in kwargs:
        kwargs['palette'] = palette
    sns.lineplot(x=x_key, y=y_key, data=logs, hue=legend_key, **kwargs)




y1 = [0, 2, 2]
y2 = [0, 2, 3, 4]
# x1 = [1, 2, 4]    


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def makePlot(logFiles, title = "Something", scalar = "train/episode_return/mean", cutoff = None, label = None):
    logs = []
    if(label is None):
        plt.figure()
    for logFile in logFiles:
        if(scalar is "train/episode_return/mean"):
            steps, returns = read_tf_log(logFile)
        else:
            steps, returns = read_dqn_logs(logFile)
        logs.append(returns)


    sns.set_theme()
    plt.rcParams.update({'font.size': 50})
    y, error = tolerant_mean(logs)
    if(cutoff is not None):
        y = y[:cutoff]
        error = error[:cutoff]

    if label is None:
        plt.plot(np.arange(len(y))+1, y, color='blue')
    else:
        plt.plot(np.arange(len(y))+1, y,  label = label)
    lower_limit_mean  = y+error
    higher_limit_mean = y-error

    plt.fill_between(list(range(1,len(y)+1)), higher_limit_mean, lower_limit_mean, alpha=0.2)
    plt.title(title, fontsize = 50)
    plt.xticks(fontsize = 50)
    plt.yticks(fontsize = 50)
    plt.xlabel("Iteration", fontsize = 50)
    plt.ylabel("Episode Reward (mean)", fontsize = 50) 



# plt.show()


# logs = pd.DataFrame({'smooth_reward':y1, 'smooth_reward':y2, 'episode':x1, "episode":x2})
# print(logs)

# ax = sns.lineplot(x=logs["episode"], y="smooth_reward", data=logs, estimator = 'mean', label = "PPO")
# plt.show()

# logs = pd.DataFrame({'smooth_reward':y1, 'smooth_reward':y2, 'episode':x1, "episode":x2})

# def plotPPOReturn(logDir):
#     steps, returns = read_tf_log(logDir, scalar = "train/episode_return/mean")
#     plt.title("PPO Average Reward")
#     plt.xlabel("Steps")
#     plt.ylabel("Episode Return (mean)")
#     plt.plot(steps, returns)
#     plt.show()


def main():
    logDir = "data/ppo-equity-run-1/"
    logDir = "data/PPO-2022-05-02-16:53:32/" #equity
    logDir = "data/PPO-2022-05-02-16:55:32/" #random

    def makePPO():
        equityLogFilesPPO = ["data/PPO-2022-05-02-16:53:32", "data/PPO-2022-05-03-00:29:59", "data/PPO-2022-05-03-00:30:14"]
        randomLogsPPO = ["data/PPO-2022-05-02-16:55:32", "data/PPO-2022-05-03-00:30:20", "data/PPO-2022-05-03-00:30:38"]

        equityLogFilesPPO.extend(["data/PPO-2022-05-04-04:10:32", "data/PPO-2022-05-04-04:10:42", "data/PPO-2022-05-04-04:10:47"])
        randomLogsPPO.extend(["data/PPO-2022-05-04-04:11:01", "data/PPO-2022-05-04-04:11:04", "data/PPO-2022-05-04-04:11:08"])

        makePlot(equityLogFilesPPO, title = "PPO Training vs Equity Agent")
        makePlot(randomLogsPPO, title = "PPO Training vs Random Agent", cutoff = 630) 
        plt.show()


    randomLogsDQN = ["Graph-random/20220504-040921_dqn1", "Graph-random/20220504-040928_dqn1", "Graph-random/20220504-040935_dqn1", "Graph/20220504-170004_dqn1", "Graph/20220504-170007_dqn1"]
    randomLogsDoubleDQN = ["Graph-random/20220504-040942_dqn1", "Graph-random/20220504-040947_dqn1", "Graph-random/20220504-040952_dqn1", "Graph/20220504-170010_dqn1", "Graph/20220504-170012_dqn1", "Graph/20220504-170014_dqn1"]
    randomLogsDuelingDQN = ["Graph-random/20220504-040955_dqn1", "Graph-random/20220504-040958_dqn1", "Graph-random/20220504-041000_dqn1", "Graph/20220504-170017_dqn1" , "Graph/20220504-170022_dqn1", "Graph/20220504-170027_dqn1"]
    randomLogsDoubleDuelingDQN = ["Graph/20220504-170121_dqn1", "Graph/20220504-170136_dqn1", "Graph/20220504-170151_dqn1"]

    # makePlot(randomLogsDQN,scalar ="epoch_episode_reward",  title = "DQN Training vs Random Agent")#, cutoff = 240)
    # makePlot(randomLogsDoubleDQN,scalar ="epoch_episode_reward",  title = "Double DQN Training vs Random Agent", cutoff = 1450)
    # makePlot(randomLogsDuelingDQN,scalar ="epoch_episode_reward",  title = "Dueling DQN Training vs Random Agent",cutoff = 1450)
    # makePlot(randomLogsDoubleDuelingDQN,scalar ="epoch_episode_reward",  title = "Double Dueling DQN Training vs Random Agent", cutoff = 1250)#, cutoff = 440)

    # plt.figure()
    # makePlot(randomLogsDQN,scalar ="epoch_episode_reward",  title = "DQN Training vs Random Agent", label = "DQN")#, cutoff = 240)
    # makePlot(randomLogsDoubleDQN,scalar ="epoch_episode_reward",  title = "Double DQN Training vs Random Agent", cutoff = 1450, label = "Double DQN")
    # makePlot(randomLogsDuelingDQN,scalar ="epoch_episode_reward",  title = "Dueling DQN Training vs Random Agent",cutoff = 1450, label = "Dueling DQN")
    # makePlot(randomLogsDoubleDuelingDQN,scalar ="epoch_episode_reward",  title = "Double Dueling DQN Training vs Random Agent", cutoff = 1250, label = "Double Dueling DQN")#, cutoff = 440)
    # plt.legend(fontsize = 30)
    # plt.title("DQN vs Random Agent", fontsize = 50)
    # plt.show()
    # exit(0)
    equityLogsDQN = ["20220503-005836_dqn1-normal", "20220503-005943_dqn1-normal", "20220503-005951_dqn1-normal", "20220503-113448_output-normal", "20220503-113543_output-normal"]
    equityLogsDoubleDQN = ["20220503-010131_dqn1-double", "20220503-010149_dqn1-double", "20220503-010156_dqn1-double", "20220503-113615_output-double", "20220503-113638_output-double", "20220503-113654_output-double"]
    eqityLogsDuelingDQN = ["20220503-010210_dqn1-dueling", "20220503-010225_dqn1-dueling", "20220503-022055_dqn1-dueling"]



    for i in range(len(equityLogsDQN)):
        equityLogsDQN[i] = "Graph-equity/" + equityLogsDQN[i]
    for i in range(len(equityLogsDoubleDQN)):
        equityLogsDoubleDQN[i] = "Graph-equity/" + equityLogsDoubleDQN[i]
    for i in range(len(eqityLogsDuelingDQN)):
        eqityLogsDuelingDQN[i] = "Graph-equity/" + eqityLogsDuelingDQN[i]

    makePlot(equityLogsDQN,scalar ="epoch_episode_reward",  title = "DQN Training vs Equity Agent")
    makePlot(equityLogsDoubleDQN,scalar ="epoch_episode_reward",  title = "Double DQN Training vs Equity Agent", cutoff = 2050)
    makePlot(eqityLogsDuelingDQN,scalar ="epoch_episode_reward",  title = "Dueling DQN Training vs Equity Agent")
#    plt.show()
    plt.figure()
    makePlot(equityLogsDQN,scalar ="epoch_episode_reward",  title = "DQN Training vs Equity Agent", label = "DQN")
    makePlot(equityLogsDoubleDQN,scalar ="epoch_episode_reward",  title = "Double DQN Training vs Equity Agent", cutoff = 2050, label = "Double DQN")
    makePlot(eqityLogsDuelingDQN,scalar ="epoch_episode_reward",  title = "Dueling DQN Training vs Equity Agent", label = "Dueling DQN")
    plt.legend(fontsize = 30)
    plt.title("DQN vs Equity Agent", fontsize = 50)
    plt.show()

 #   plt.show()
    # equityLogsPPO = []
    # for logFile in equityLogFilesPPO:
    #     steps, returns = read_tf_log(logFile)
    #     equityLogsPPO.append(returns)
    # plt.show()

    # equityLogsPPO = []
    # for logFile in equityLogFilesPPO:
    #     steps, returns = read_tf_log(logFile)
    #     equityLogsPPO.append(returns)
    # makePlot(randomLogsPPO, title = "PPO Training vs Equity Agent")
    # plt.show()



    #new data
    logDir = "Graph-random/20220504-040921_dqn1"
    # logDir = "data/PPO-2022-05-03-00:30:20"
    # logDir = "data/PPO-2022-05-03-00:30:38"
  #  logDir = "data/PPO"
   # steps, returns = read_tf_log(logDir)


  #  plotDQNReturn(logDir)
  #  plt.show()
    # plotPPOReturn("data/PPO-2022-05-02-16:53:32/")

if __name__ == '__main__':
    main()

