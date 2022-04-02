import time
from copy import deepcopy

import torch

from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory
from easyrl.utils.gym_util import get_render_images
from easyrl.utils.gym_util import is_time_limit_env
from easyrl.utils.common import list_to_numpy

class StepRunner(BasicRunner):
    # Simulate the environment for T steps,
    # and in the next call, the environment will continue
    # from where it's left in the previous call.
    # only single env (no parallel envs) is supported for now.
    # we also assume the environment is wrapped by TimeLimit
    # from https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
    def __init__(self, agent, env, eval_env=None):
        super().__init__(agent=agent,
                         env=env,
                         eval_env=eval_env)
        self.obs = None

    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False, render_image=False,
                 sleep_time=0, reset_first=False,
                 env_reset_kwargs=None, agent_reset_kwargs=None,
                 action_kwargs=None, random_action=False):
        traj = Trajectory()
        if env_reset_kwargs is None:
            env_reset_kwargs = {}
        if agent_reset_kwargs is None:
            agent_reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        action_kwargs['eval'] = evaluation
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        if self.obs is None or reset_first or evaluation:
            self.reset(env=env,
                       env_reset_kwargs=env_reset_kwargs,
                       agent_reset_kwargs=agent_reset_kwargs)
        ob = self.obs
        ob = deepcopy(ob)
        for t in range(time_steps):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = get_render_images(env)
            if random_action:
                action = env.action_space.sample()
                if len(action.shape) == 1:
                    # the first dim is num_envs
                    action = list_to_numpy(action, expand_dims=0)
                action_info = dict()
            else:
                action, action_info = self.agent.get_action(ob,
                                                            sample=sample,
                                                            **action_kwargs)
            next_ob, reward, done, info = env.step(action)
            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)
            true_done = deepcopy(done)
            for iidx, inf in enumerate(info):
                true_done[iidx] = true_done[iidx] and not inf.get('TimeLimit.truncated',
                                                                  False)
            sd = StepData(ob=ob,
                          action=action,
                          action_info=action_info,
                          next_ob=next_ob,
                          reward=reward,
                          done=true_done,
                          info=info)
            ob = next_ob
            traj.add(sd)
            if return_on_done and done:
                break
            if done:
                ob = self.reset(env, env_reset_kwargs, agent_reset_kwargs)
        self.obs = None if evaluation else deepcopy(ob)
        return traj

    def reset(self, env=None, *args, **kwargs):
        if env is None:
            env = self.train_env
        self.obs = env.reset(*args, **kwargs)
        return ob
