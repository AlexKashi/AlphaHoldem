import time
from copy import deepcopy
from collections import deque
import numpy as np
import torch
from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory
from easyrl.utils.torch_util import torch_to_np


class RNNRunner(BasicRunner):
    def __init__(self, *args, **kwargs):
        super(RNNRunner, self).__init__(*args, **kwargs)
        self.hidden_states = None
        self.hidden_state_shape = None


    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False,
                 render_image=False,
                 sleep_time=0, reset_first=False,
                 reset_kwargs=None, action_kwargs=None,
                 get_last_val=False):
        traj = Trajectory()
        if reset_kwargs is None:
            reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        # In RL^2, we should always reset in the begining of a rollout
        if self.obs is None or reset_first or evaluation:
            self.reset(**reset_kwargs)
        ob = self.obs
        hidden_state = self.hidden_states
        # this is critical for some environments depending
        # on the returned ob data. use deepcopy() to avoid
        # adding the same ob to the traj

        # only add deepcopy() when a new ob is generated
        # so that traj[t].next_ob is still the same instance as traj[t+1].ob
        ob = deepcopy(ob)
        if return_on_done:
            all_dones = np.zeros(env.num_envs, dtype=bool)
        else:
            all_dones = None
        done = None
        for t in range(time_steps):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = deepcopy(env.get_images())

            action, action_info, hidden_state = self.agent.get_action(ob,
                                                                      sample=sample,
                                                                      hidden_state=hidden_state,
                                                                      **action_kwargs)
            if self.hidden_state_shape is None:
                self.hidden_state_shape = hidden_state.shape
            next_ob, reward, done, info = env.step(action)

            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)

            true_next_ob, true_done, all_dones = self.get_true_done_next_ob(next_ob,
                                                                            done,
                                                                            reward,
                                                                            info,
                                                                            all_dones,
                                                                            skip_record=evaluation)

            sd = StepData(ob=ob,
                          action=action,
                          action_info=action_info,
                          next_ob=true_next_ob,
                          reward=reward,
                          done=true_done,
                          info=info,
                          extra=done,  # this is a flag that can tell whether the environment
                          # is reset or not so that we know whether we need to
                          # reset the hidden state or not. We save it in "extra"
                          )
            ob = next_ob
            traj.add(sd)
            if return_on_done and np.all(all_dones):
                break

            # the order of next few lines matter, do not exchange
            if get_last_val and not evaluation and t == time_steps - 1:
                last_val, _ = self.agent.get_val(traj[-1].next_ob,
                                                 hidden_state=hidden_state)
                traj.add_extra('last_val', torch_to_np(last_val))
            hidden_state = self.check_hidden_state(hidden_state, done=done)
        self.obs = ob if not evaluation else None
        self.hidden_states = hidden_state.detach() if not evaluation else None
        return traj

    def reset(self, env=None, *args, **kwargs):
        super().reset(env, *args, **kwargs)
        self.hidden_states = None

    def get_hidden_state_shape(self):
        obs = self.train_env.reset()
        done = None
        action, action_info, hidden_state = self.agent.get_action(ob,
                                                                  sample=True,
                                                                  hidden_state=None,
                                                                  prev_done=done)
        self.hidden_state_shape = hidden_state.shape
        return self.hidden_state_shape

    def check_hidden_state(self, hidden_state, done=None):
        if done is not None:
            # if the last step is the end of an episode,
            # then reset hidden state
            done_idx = np.argwhere(done).flatten()
            if done_idx.size > 0:
                ld, b, hz = hidden_state.shape
                hidden_state[:, done_idx] = torch.zeros(ld, done_idx.size, hz,
                                                        device=hidden_state.device)
        return hidden_state
