import os

import numpy as np
from gym.envs.registration import register

curr_dir = os.path.dirname(os.path.abspath(__file__))

register(
    id="myoChallengeBaodingP2_al4muscles-v1",
    entry_point="al4myochallenge.myo_env.baoding_v1:BaodingEnvV1",
    max_episode_steps=200,
    kwargs={
        "model_path": curr_dir + "/myo_hand_baoding.mjb",
        "normalize_act": True,
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        # Randomization in physical properties of the baoding balls
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (
            0.030,
            0.300,
        ),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (
            0.2,
            0.001,
            0.00002,
        ),  # nominal: 1.0, 0.005, 0.0001
        # 'obj_size_range': (0.022, 0.022),       # Object size range. Nominal 0.022
        # 'obj_mass_range': (0.043, 0.043),       # Object weight range. Nominal 43 gms
        # 'obj_friction_change': (0.0, 0.0, 0.0), # nominal: 1.0, 0.005, 0.0001
        "task_choice": "fixed",
    },
)


register(
    id="myoChallengeBaodingP2_al4muscles_eval-v1",
    entry_point="al4myochallenge.myo_env.baoding_v1:BaodingEnvV1",
    max_episode_steps=200,
    kwargs={
        "model_path": curr_dir + "/myo_hand_baoding.mjb",
        "normalize_act": True,
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        # Randomization in physical properties of the baoding balls
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (
            0.030,
            0.300,
        ),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (
            0.2,
            0.001,
            0.00002,
        ),  # nominal: 1.0, 0.005, 0.0001
        # 'obj_size_range': (0.022, 0.022),       # Object size range. Nominal 0.022
        # 'obj_mass_range': (0.043, 0.043),       # Object weight range. Nominal 43 gms
        # 'obj_friction_change': (0.0, 0.0, 0.0), # nominal: 1.0, 0.005, 0.0001
        "task_choice": "random",
    },
)
