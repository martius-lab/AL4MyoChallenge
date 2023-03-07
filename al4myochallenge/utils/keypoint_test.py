from typing import Deque, Dict, Tuple, Union

import numpy as np
import torch

"""
Credit for keypoint generation goes to:
Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning
https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
Also see file: nvidia_license.txt
"""


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def local_to_world_space(
    pos_offset_local: torch.Tensor,
    pos_global: torch.Tensor,
    rot_global: torch.Tensor,
):
    """Convert a point from the local frame to the global frame
    Args:
        pos_offset_local: Point in local frame. Shape: [N, 3]
        pose_global: The spatial pose of this point. Shape: [N, 7]
    Returns:
        Position in the global frame. Shape: [N, 3]
    """
    quat_pos_local = torch.cat(
        [
            pos_offset_local,
            torch.zeros(
                pos_offset_local.shape[0],
                1,
                dtype=torch.float32,
                # device=pos_offset_local.device,
                device="cpu",
            ),
        ],
        dim=-1,
    )
    quat_global = rot_global
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(
        quat_global, quat_mul(quat_pos_local, quat_global_conj)
    )[:, 0:3]

    result_pos_gloal = pos_offset_global + pos_global

    return result_pos_gloal


@torch.jit.script
def gen_keypoints(
    pos: torch.Tensor,
    rot: torch.Tensor,
    num_keypoints: int = 8,
    size: Tuple[float, float, float] = (0.026, 0.026, 0.026),
):
    num_envs = pos.shape[0]

    keypoints_buf = torch.ones(
        # num_envs, num_keypoints, 3, dtype=torch.float32, device=pos.device
        num_envs,
        num_keypoints,
        3,
        dtype=torch.float32,
        device="cpu",
    )

    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = (
            [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        )
        corner = (
            # torch.tensor(corner_loc, dtype=torch.float32, device=pos.device)
            torch.tensor(corner_loc, dtype=torch.float32, device="cpu")
            * keypoints_buf[:, i, :]
        )
        keypoints_buf[:, i, :] = local_to_world_space(corner, pos, rot)
    return keypoints_buf


if __name__ == "__main__":
    num_envs = 1
    num_keypoints = 8
    size = [0.1, 0.1, 0.1]

    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32)
    pose = torch.rand(size=(num_envs, 7))
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = (
            [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        )
        corner = (
            torch.tensor(corner_loc, dtype=torch.float32)
            * keypoints_buf[:, i, :]
        )
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)

    print(keypoints_buf)
