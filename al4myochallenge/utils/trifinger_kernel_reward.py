import gym
import myosuite
import torch
from myosuite.utils.quat_math import *
from scipy.spatial.transform import Rotation as R

from al4myochallenge.utils.keypoint_test import gen_keypoints

object_dist_weight = 2000
dt = 0.002


@torch.jit.script
def _lgsk_kernel(
    x: torch.Tensor, scale: float = 50.0, eps: float = 2
) -> torch.Tensor:
    """Defines logistic kernel function to bound input to [-0.25, 0)
    Ref: https://arxiv.org/abs/1901.08652 (page 15)
    Args:
        x: Input tensor.
        scale: Scaling of the kernel function (controls how wide the 'bell' shape is')
        eps: Controls how 'tall' the 'bell' shape is.
    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    # return 1.0 / (scaled.exp())
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())


def _get_obj_info(observations, goal_offset):
    obj_pos = observations[45:48]
    obj_rot = observations[54:57]
    goal_pos = observations[48:51] - goal_offset
    goal_rot = observations[57:60]

    if len(obj_pos.shape) == 1:
        obj_rot = torch.as_tensor(
            euler2quat(obj_rot), dtype=torch.float32, device="cpu"
        )[None, :]
        goal_rot = torch.as_tensor(
            euler2quat(goal_rot), dtype=torch.float32, device="cpu"
        )[None, :]
        obj_pos = torch.as_tensor(obj_pos, dtype=torch.float32, device="cpu")[
            None, :
        ]
        goal_pos = torch.as_tensor(
            goal_pos, dtype=torch.float32, device="cpu"
        )[None, :]
    else:
        obj_rot = torch.as_tensor(
            euler2quat(obj_rot), dtype=torch.float32, device="cpu"
        )
        goal_rot = torch.as_tensor(
            euler2quat(goal_rot), dtype=torch.float32, device="cpu"
        )
        obj_pos = torch.as_tensor(obj_pos, dtype=torch.float32, device="cpu")
        goal_pos = torch.as_tensor(goal_pos, dtype=torch.float32, device="cpu")
    return obj_pos, obj_rot, goal_pos, goal_rot


def _replace_state(observations, keypoints_object, keypoints_goal):
    obj_key = keypoints_object.flatten(1, 2).cpu().numpy()
    goal_key = keypoints_goal.flatten(1, 2).cpu().numpy()
    # with activity
    observations[45:-39] = 0.0

    observations[45:69] = obj_key
    observations[69:93] = goal_key
    observations[93:117] = goal_key - obj_key


def _compute_rew_from_key(keypoints_object, keypoints_goal):
    keypoints_object = torch.as_tensor(keypoints_object)
    keypoints_goal = torch.as_tensor(keypoints_goal)
    delta = keypoints_goal - keypoints_object
    dist_l2 = torch.norm(delta, p=2, dim=-1)
    keypoints_kernel_sum = _lgsk_kernel(dist_l2, scale=200, eps=2.0).mean(
        dim=-1
    )
    dist_l2 = np.linalg.norm(delta.cpu().numpy())
    return object_dist_weight * dt * keypoints_kernel_sum, dist_l2


def relabel_and_get_trifinger_reward(observations, goal_offset):
    obj = _get_obj_info(observations, goal_offset)
    keypoints_object = gen_keypoints(obj[0], obj[1])
    keypoints_goal = gen_keypoints(obj[2], obj[3])
    pose_reward, dist_l2 = _compute_rew_from_key(
        keypoints_object, keypoints_goal
    )
    _replace_state(observations, keypoints_object, keypoints_goal)
    return pose_reward, dist_l2


if __name__ == "__main__":
    pass
