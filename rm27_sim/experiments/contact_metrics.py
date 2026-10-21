"""Read-only contact measurements; no referee damage or structural model."""

import mujoco
import numpy as np


def armor_contacts(model, data, armor_id, outward_normal):
    normal = np.asarray(outward_normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError('Expected finite nonzero armor normal')
    normal = normal / np.linalg.norm(normal)
    records = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if armor_id not in contact.geom:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        # Contact-frame axes are rows, normal from geom[0] toward geom[1].
        on_second = contact.frame.reshape(3, 3).T @ wrench[:3]
        on_armor = on_second if contact.geom[1] == armor_id else -on_second
        records.append({
            'geoms': [model.geom(int(g)).name for g in contact.geom],
            'position_world_m': contact.pos.tolist(),
            'force_on_armor_world_N': on_armor.tolist(),
            'compressive_normal_force_N': max(0., -float(normal @ on_armor)),
            'signed_normal_force_N': -float(normal @ on_armor),
            'distance_m': float(contact.dist),
        })
    return records


def point_approach_speed(com_velocity, angular_velocity_world, point_from_com_world,
                         plate_velocity_world, outward_normal):
    """Positive toward the surface; all inputs in the same world frame."""
    vectors = np.asarray([com_velocity, angular_velocity_world, point_from_com_world,
                          plate_velocity_world, outward_normal], dtype=float)
    if vectors.shape != (5, 3) or not np.isfinite(vectors).all() or np.linalg.norm(vectors[4]) == 0:
        raise ValueError('Expected five finite world vectors and nonzero normal')
    v, omega, r, plate, n = vectors
    return -float((v + np.cross(omega, r) - plate) @ (n / np.linalg.norm(n)))
