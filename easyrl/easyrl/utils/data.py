from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List

import numpy as np


@dataclass
class StepData:
    ob: Any = None
    state: Any = None
    action: Any = None
    # store action infomation such as log probability, entropy
    action_info: Any = None
    next_ob: Any = None
    next_state: Any = None
    reward: Any = None
    done: Any = None
    info: Any = None
    extra: Any = None

    def __post_init__(self):
        """
        If the ob is a dict containing keys: ob and state
        then store them into ob and state separately
        """
        if isinstance(self.ob, dict):
            self.ob, self.state = self.dict_ob_state(self.ob)
        if isinstance(self.next_ob, dict):
            self.next_ob, self.next_state = self.dict_ob_state(self.next_ob)

    def dict_ob_state(self, ob):
        keys = ['ob', 'state']
        for key in keys:
            if key not in ob:
                raise ValueError('ob must have `ob` and `state` '
                                 'as keys if it is a dict!')
        state = ob['state']
        ob = ob['ob']
        return ob, state


@dataclass
class Trajectory:
    extra_data: Dict = field(default_factory=dict)
    traj_data: List[StepData] = field(default_factory=list)

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, item):
        return self.traj_data[item]

    def add(self, step_data=None, **kwargs):
        if step_data is not None:
            if not isinstance(step_data, StepData):
                raise TypeError('step_data should be an '
                                'instance of StepData!')
        else:
            step_data = StepData(**kwargs)
        self.traj_data.append(step_data)

    def add_extra(self, key, value):
        self.extra_data[key] = value

    @property
    def obs(self):
        return np.array([step_data.ob for step_data in self.traj_data])

    @property
    def states(self):
        return np.array([step_data.state for step_data in self.traj_data])

    @property
    def actions(self):
        return np.array([step_data.action for step_data in self.traj_data])

    @property
    def action_infos(self):
        return [step_data.action_info for step_data in self.traj_data]

    @property
    def next_obs(self):
        return np.array([step_data.next_ob for step_data in self.traj_data])

    @property
    def next_states(self):
        return np.array([step_data.next_state for step_data in self.traj_data])

    @property
    def rewards(self):
        return np.array([step_data.reward for step_data in self.traj_data])

    @property
    def step_extras(self):
        return np.array([step_data.extra for step_data in self.traj_data])

    @property
    def raw_rewards(self):
        if len(self.traj_data) > 0 and 'raw_reward' in self.traj_data[0].info[0]:
            raw_rews = []
            for step_data in self.traj_data:
                info = step_data.info
                step_raw_reward = [x['raw_reward'] for x in info]
                raw_rews.append(step_raw_reward)
            raw_rews = np.array(raw_rews)
        else:
            raw_rews = self.rewards
        return raw_rews

    @property
    def dones(self):
        return np.array([step_data.done for step_data in self.traj_data])

    @property
    def infos(self):
        return [step_data.info for step_data in self.traj_data]

    @property
    def total_steps(self):
        return len(self.traj_data[0].action) * len(self.traj_data)

    @property
    def num_envs(self):
        return self.dones.shape[1]

    @property
    def steps_til_done(self):
        steps = []
        dones = self.dones
        for i in range(dones.shape[1]):
            di = dones[:, i]
            if not np.any(di):
                steps.append(len(di))
            else:
                steps.append(np.argmax(dones[:, i]) + 1)
        return np.array(steps)

    @property
    def episode_returns(self):
        """
        return the undiscounted return in each episode (between any two dones)

        Returns:
            list: a list of length-num_envs,
            each element in this list is a list of episodic return values

        """
        all_epr = []
        dones = self.dones
        rewards = self.raw_rewards
        for i in range(dones.shape[1]):
            epr = []
            di = dones[:, i]
            if not np.any(di):
                epr.append(np.sum(rewards[:, i]))
            else:
                done_idx = np.where(di)[0]
                t = 0
                for idx in done_idx:
                    epr.append(np.sum(rewards[t: idx + 1, i]))
                    t = idx + 1
            all_epr.append(epr)
        return all_epr

    @property
    def episode_steps(self):
        """
        return the number of steps in each episode

        Returns:
            list: a list of length-num_envs,
            each element in this list is a list of # of steps in an episode

        """
        all_epl = []
        dones = self.dones
        for i in range(dones.shape[1]):
            epl = []
            di = dones[:, i]

            if not np.any(di):
                epl.append(di.shape[0])
            else:
                di = np.insert(di, 0, 1)
                done_idx = np.where(di)[0]
                leng = np.diff(done_idx)
                epl.extend(leng.tolist())
            all_epl.append(epl)
        return all_epl

    def pop(self):
        """
        Remove and return the last element from the trajectory
        """
        step_data = self.traj_data.pop(-1)
        return step_data

    def popleft(self):
        """
        Remove and return the first element from the trajectory
        """
        step_data = self.traj_data.pop(0)
        return step_data
