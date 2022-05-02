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
def read_tf_log(log_dir, scalar='train/episode_return/mean'):
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob(f'**/events.*'))
    if len(log_files) < 1:
        return None
    log_file = log_files[0]

    event_acc = EventAccumulator(log_file.as_posix())
    event_acc.Reload()

    tags = event_acc.Tags()
  #  print(tags)
#    assert False
    scalar_return = event_acc.Scalars(scalar)
    returns = [x.value for x in scalar_return]
    steps = [x.step for x in scalar_return]

    return steps, returns


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

def main():
    logDir = "data/ppo-equity-run-1/"
  #  logDir = "data/PPO"
   # steps, returns = read_tf_log(logDir)
   # steps, returns = read_tf_log(logDir, scalar = "train/rollout_steps_per_iter")
   # #determanistic
    steps, returns = read_tf_log(logDir, scalar = "det/eval/smooth_return/mean")
    steps, returns = read_tf_log(logDir, scalar = "det/eval/episode_length/mean")
   # steps, returns = read_tf_log(logDir, scalar = "train/rollout_steps_per_iter")

    plt.title("PPO Average Reward")
    plt.xlabel("Steps")
    plt.ylabel("Episode Return (mean)")
    plt.plot(steps, returns)
    plt.show()

if __name__ == '__main__':
    main()