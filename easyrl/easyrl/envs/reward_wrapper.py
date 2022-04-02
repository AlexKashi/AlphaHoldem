from easyrl.envs.vec_env import VecEnvWrapper


class RewardScaler(VecEnvWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """

    def __init__(self, venv, scale=0.01, observation_space=None, action_space=None):
        super(RewardScaler, self).__init__(venv,
                                           observation_space=observation_space,
                                           action_space=action_space)
        self.scale = scale

    def step(self, action):
        observation, reward, done, info = super().step(action)
        for idx, inf in enumerate(info):
            inf['raw_reward'] = reward[idx]
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        return reward * self.scale

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


class RewardMinMaxNorm(VecEnvWrapper):
    def __init__(self, venv, min_rew, max_rew, observation_space=None, action_space=None):
        super(RewardMinMaxNorm, self).__init__(venv,
                                               observation_space=observation_space,
                                               action_space=action_space)
        self.min_rew = min_rew
        self.max_rew = max_rew

    def step(self, action):
        observation, reward, done, info = super().step(action)
        for idx, inf in enumerate(info):
            inf['raw_reward'] = reward[idx]
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        reward = (reward - self.min_rew) / (self.max_rew - self.min_rew)
        return reward

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()
