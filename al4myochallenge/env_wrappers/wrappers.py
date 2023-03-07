from abc import ABC, abstractmethod

import gym
import numpy as np
import scipy.stats
from tonic import logger

import al4myochallenge.utils as utils


class AbstractWrapper(gym.Wrapper, ABC):
    def merge_args(self, args):
        if args is not None:
            for k, v in args.items():
                setattr(self, k, v)

    def apply_args(self):
        pass


class GymWrapper(AbstractWrapper):
    def __init__(self, *args, **kwargs):
        self._step = {
            # "step_myo": self._step_die_myochallenge,
            "step_die": self._step_die,
            "step_baoding": self._step_baoding,
            "step_baoding_test": self._step_baoding_test,
        }
        if "Baoding" not in str(type(args[0])):
            self.step_fn = "step_die"
        else:
            self.step_fn = "step_baoding"
        self.effort_coeff = 0
        self.indic_active = 0
        super().__init__(*args, **kwargs)
        self.old_state = None
        self.starting_steps = 0

    def merge_args(self, *args, **kwargs):
        super().merge_args(*args, **kwargs)
        self.effort_coeff = np.abs(self.effort_coeff)

    def step(self, action):
        self.starting_steps += 1
        obs, reward, done, info = self.unwrapped.step(action)
        info["episode_number"] = self.episode_number
        return self._step[self.step_fn](obs, reward, done, info)

    def reset(self, *args, **kwargs):
        if "schedule" in self.step_fn:
            scale = 1 - np.clip(self.starting_steps / 1e3, 0, 1)
            T = 100
            sampled_time_period = np.random.uniform(
                self.goal_time_period[0] + scale * T,
                self.goal_time_period[1] + scale * T,
            )
            kwargs["time_period"] = sampled_time_period
        self.active_duration = np.random.zipf(2, size=(1,))[0]
        self.indic_active = 0
        if hasattr(self, "old_qpos"):
            self.old_qpos[:] = 0
        self.episode_number = np.random.randint(0, 10000000)
        if self.old_state is not None:
            if "Reorient" not in str(type(self.unwrapped)):
                kwargs["reset_pose"] = self.old_state
                if hasattr(self, "old_vel"):
                    kwargs["reset_vel"] = self.old_vel
            self.old_state = None
        obs = super().reset(*args, **kwargs)
        if "Reorient" in str(type(self.unwrapped)):
            utils.relabel_and_get_trifinger_reward(obs, self.goal_obj_offset)
        return obs.copy()

    def _step_die(self, obs, reward, done, info):
        """
        Replace rewards and state with keypoint representation from nvidia trifinger
        paper.
        Give an additional reward for perfectly fixating the cube.
        """
        reward, _ = utils.relabel_and_get_trifinger_reward(
            obs, self.goal_obj_offset
        )
        if self.rwd_dict["solved"]:
            reward += 0.1
        reward += self.effort_coeff * self.rwd_dict["act_square"]
        return obs.copy(), reward.cpu().numpy()[0], done, info

    def _step_baoding_test(self, obs, reward, done, info):
        return obs, reward, done, info

    def _step_baoding(self, obs, reward, done, info):
        self.old_state = self.sim.data.qpos[:].copy()
        self.old_vel = self.sim.data.qvel[:].copy()
        return self._sub_step_baoding(obs, reward, done, info)

    def _sub_step_baoding(self, obs, reward, done, info):
        reward = -1
        if self.rwd_dict["solved_tight"]:
            reward += 10
        if self.rwd_dict["done"]:
            reward = -200
            self.old_state = None
            self.old_vel = None
        if self.rwd_dict["solved"]:
            self.indic_active += 1
        else:
            self.indic_active = 0
        if self.indic_active >= self.active_duration:
            reward += 10
            done = True
            self.indic_active = 0
        reward += self.effort_coeff * self.rwd_dict["act_series"]
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        kwargs["mode"] = "window"
        self.unwrapped.sim.render(*args, **kwargs)

    @property
    def _max_episode_steps(self):
        """
        This is on purpose, use episode length 200 for balls and cube, even though cube evaluation will be with
        150
        """
        return 200


class ExceptionWrapper(GymWrapper):
    """
    Catches MuJoCo related exception thrown mostly by instabilities in the simulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        if not np.any(np.isnan(observation)):
            self.last_observation = observation.copy()
        else:
            return self.reset(**kwargs)
        return observation

    def step(self, action):
        try:
            observation, reward, done, info = super().step(action)
            if np.any(np.isnan(observation)):
                raise Exception()
        except Exception as e:
            logger.log(f"NaN detected, resetting environment! Exception: {e}")

            observation = self.last_observation
            reward = 0
            done = 1
            self.reset()
        return observation, reward, done, info
