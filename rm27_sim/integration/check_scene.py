"""Unpowered four-vehicle settling check: python -m rm27.integration.check_scene scene.xml."""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.nq != 28 or model.nu != 16:
        raise ValueError("Expected four free vehicles and sixteen motors")
    initial = data.qpos.copy()
    initial_contacts = data.ncon
    if initial_contacts:
        raise ValueError("Initial vehicle penetration/contact requires review")
    maximum_force = 0.0
    maximum_penetration = 0.0
    for step in range(2500):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError(f"Nonfinite state at step {step}")
        if np.max(np.abs(data.qvel)) > 1:
            raise ValueError("Unexpected settling velocity")
        for i in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            maximum_force = max(maximum_force, float(np.linalg.norm(force[:3])))
            maximum_penetration = max(maximum_penetration, -float(data.contact[i].dist))
    mujoco.mj_forward(model, data)
    slots = []
    for i in range(4):
        slot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"slot_{i+1}")
        if slot < 0:
            raise ValueError("Missing slot")
        p = data.qpos[i * 7 : i * 7 + 3]
        drift = float(np.linalg.norm(p[:2] - initial[i * 7 : i * 7 + 2]))
        height = float(p[2] - data.site_xpos[slot, 2])
        bottom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"drone_{i+1}_bottom")
        expected_height = -float(model.site_pos[bottom, 2]) if bottom >= 0 else 0.02
        if drift > 0.001 or abs(height - expected_height) > 0.002:
            raise ValueError("Vehicle did not settle in its assigned slot")
        slots.append(
            {
                "drone": i + 1,
                "position_m": p.tolist(),
                "xy_drift_m": drift,
                "height_above_slot_m": height,
                "expected_height_above_slot_m": expected_height,
            }
        )
    if (
        maximum_force > 20
        or maximum_penetration > 0.005
        or np.max(np.abs(data.qvel)) > 0.01
    ):
        raise ValueError("Unstable contact response")
    result = {
        "passed": True,
        "test": "unpowered settling, not flight",
        "steps": 2500,
        "physics_seconds": data.time,
        "initial_contacts": initial_contacts,
        "final_contacts": data.ncon,
        "maximum_contact_force_N": maximum_force,
        "maximum_penetration_m": maximum_penetration,
        "max_final_abs_qvel": float(np.max(np.abs(data.qvel))),
        "vehicles": slots,
    }
    args.scene.with_name("settling.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
