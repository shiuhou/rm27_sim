"""Explicit NWU→NED world and FLU→FRD body basis transformations."""

import mujoco
import numpy as np

BASIS = np.diag([1.0, -1.0, -1.0])
GRAVITY = np.array([0.0, 0.0, -9.81])


def to_ap(vector):
    return BASIS @ np.asarray(vector, dtype=float)


def attitude_to_ap(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError("expected finite nonzero WXYZ quaternion")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, q / np.linalg.norm(q))
    transformed = BASIS @ rotation.reshape(3, 3) @ BASIS.T
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, transformed.ravel())
    return result


def specific_force(rotation_body_to_world, acceleration_world, gravity=GRAVITY):
    return to_ap(
        np.asarray(rotation_body_to_world).T
        @ (np.asarray(acceleration_world) - gravity)
    )
