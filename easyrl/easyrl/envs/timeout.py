import gym


class NoTimeOutEnv(gym.Wrapper):
    def __init__(self, env):
        super(NoTimeOutEnv, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # set done to True only if it's not done because of timeout
        done = done and not info.get('TimeLimit.truncated', False)
        return observation, reward, done, info
