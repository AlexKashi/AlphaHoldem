import numpy as np
import torch

from easyrl.utils.torch_util import torch_float


def cal_gae(gamma, lam, rewards, value_estimates, last_value, dones):
    advs = np.zeros_like(rewards)
    last_gae_lam = 0
    if len(value_estimates.shape) > 1:
        last_value = last_value.reshape(1, -1)
    value_estimates = np.concatenate((value_estimates,
                                      last_value),
                                     axis=0)
    for t in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * value_estimates[t + 1] * non_terminal - value_estimates[t]
        last_gae_lam = delta + gamma * lam * non_terminal * last_gae_lam
        advs[t] = last_gae_lam.copy()
    return advs


def cal_gae_torch(gamma, lam, rewards, value_estimates, last_value, dones):
    device = value_estimates.device
    rewards = torch_float(rewards, device)
    value_estimates = torch_float(value_estimates, device)
    last_value = torch_float(last_value, device)
    if len(value_estimates.shape) > 1:
        last_value = last_value.view(1, -1)
    dones = torch_float(dones, device)
    advs = torch.zeros_like(rewards).to(device)
    last_gae_lam = 0
    value_estimates = torch.cat((value_estimates,
                                 last_value),
                                dim=0)
    for t in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * value_estimates[t + 1] * non_terminal - value_estimates[t]
        last_gae_lam = delta + gamma * lam * non_terminal * last_gae_lam
        advs[t] = last_gae_lam.clone()
    return advs
