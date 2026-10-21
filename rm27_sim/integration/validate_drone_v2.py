from rm27_sim.paths import resolve_asset
"""Headless Ver2 physics checks and a truth-feedback takeoff/hover/return/landing.

This validates the model interface, not PX4/ArduPilot or real-aircraft performance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def run(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = {}
    model = mujoco.MjModel.from_xml_path(str(package / "dynamic/scene.xml"))
    uid = model.body("rm27_drone").id
    mass = model.body_mass[uid]
    inertia = model.body_inertia[uid]
    info = json.loads((package / "model_info.json").read_text())
    checks["six_dof_four_motors_150g"] = bool(model.nq == 7 and model.nv == 6 and model.nu == 4 and abs(mass - .15) < 1e-12)
    checks["reviewed_parts_only"] = bool(len(info["parts"]) == 30 and not any("jiangbao3" in p["name"] for p in info["parts"]))
    visual = mujoco.MjModel.from_xml_path(str(package / "visual/model.xml"))
    checks["visual_only_entry"] = bool(visual.nq == visual.nu == 0 and not np.any(visual.geom_contype))
    checks["portable_relative_assets"] = all(not Path(asset.get("file")).is_absolute() for kind in ["dynamic", "visual"]
        for asset in ET.parse(package / kind / "model.xml").findall("asset/*"))
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "relocated"
        for d in ["assets", "visual", "dynamic"]:
            shutil.copytree(package / d, target / d)
        relocated = mujoco.MjModel.from_xml_path(str(target / "dynamic/scene.xml"))
        checks["relocated_bundle_compiles"] = bool(relocated.nu == 4)

    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, .5]
    data.act[:] = data.ctrl[:] = mass * 9.81 / 4
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    mujoco.mj_step(model, data, nstep=1000)
    hover_drift = float(np.linalg.norm(data.qpos[:3] - initial[:3]))
    checks["balanced_thrust_hover"] = bool(hover_drift < 1e-6 and np.linalg.norm(data.qvel) < 1e-6)
    rotor_positions = np.array([model.site_pos[model.site(f"rm27_drone_rotor_{i}").id] for i in range(4)])
    torque_columns = np.c_[rotor_positions[:, 1], -rotor_positions[:, 0], model.actuator_gear[:, 5]].T
    mixing = np.vstack([np.ones(4), torque_columns])
    checks["independent_roll_pitch_yaw_control"] = bool(np.linalg.matrix_rank(mixing) == 4)
    pulse_errors = []
    for i in range(4):
        d = mujoco.MjData(model)
        d.qpos[:3] = [0, 0, .5]
        d.act[:] = d.ctrl[:] = mass * 9.81 / 4
        d.act[i] += .1
        d.ctrl[i] += .1
        mujoco.mj_forward(model, d)
        expected = torque_columns[:, i] * .1 / inertia
        pulse_errors.append(float(np.max(np.abs(d.qacc[3:] - expected))))
    checks["motor_wrench_signs"] = max(pulse_errors) < 1e-7

    # A ray through a rotor opening must not hit a convex hull filling the cage.
    d = mujoco.MjData(model)
    d.qpos[:3] = [0, 0, .5]
    mujoco.mj_forward(model, d)
    groups = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    rays = []
    for x, y, _ in rotor_positions:
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, d, np.array([x + .011, y + .004, .6]),
                                np.array([0., 0., -1.]), groups, 1, -1, hit)
        rays.append(float(distance))
    checks["rotor_openings_not_filled"] = all(v < 0 for v in rays)

    # Passive contact and rest; no controller can conceal missing ground collision.
    d = mujoco.MjData(model)
    mujoco.mj_step(model, d, nstep=1500)
    mujoco.mj_forward(model, d)
    rest_height = float(d.qpos[2])
    checks["passive_ground_contact"] = bool(d.ncon > 0 and abs(rest_height - .0225) < .002 and np.linalg.norm(d.qvel) < .01)

    def smooth(t, begin, duration, start, end):
        u = np.clip((t - begin) / duration, 0, 1)
        s = 10*u**3 - 15*u**4 + 6*u**5
        v = (30*u**2 - 60*u**3 + 30*u**4) / duration
        return start + (end-start)*s, (end-start)*v

    # Simple full-state model exercise; commands are individual thrusts in newtons.
    d = mujoco.MjData(model)
    d.qpos[:3] = [0, 0, .0235]
    records, errors, saturations = [], [], 0
    for step in range(6500):
        t = float(d.time)
        reference = np.array([0., 0., .0235])
        velocity = np.zeros(3)
        if 1 <= t < 3:
            reference[2], velocity[2] = smooth(t, 1, 2, .0235, .30)
        elif 3 <= t < 9:
            reference[2] = .30
            if 5 <= t < 7:
                reference[0], velocity[0] = smooth(t, 5, 2, 0., .15)
            elif t >= 7:
                reference[0], velocity[0] = smooth(t, 7, 2, .15, 0.)
        elif 9 <= t < 11:
            reference[2], velocity[2] = smooth(t, 9, 2, .30, .0235)
        mujoco.mj_forward(model, d)
        if 1 <= t < 11:
            acceleration = np.array([8., 8., 25.])*(reference-d.qpos[:3]) + np.array([5., 5., 10.])*(velocity-d.qvel[:3]) + [0, 0, 9.81]
            zdes = acceleration / np.linalg.norm(acceleration)
            ydes = np.cross(zdes, [1., 0., 0.]); ydes /= np.linalg.norm(ydes)
            desired = np.column_stack([np.cross(ydes, zdes), ydes, zdes])
            rotation = d.xmat[uid].reshape(3, 3)
            skew = (desired.T @ rotation - rotation.T @ desired) / 2
            error = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            omega = d.qvel[3:]
            torque = inertia*(-100*error - 18*omega) + np.cross(omega, inertia*omega)
            total = mass * np.dot(acceleration, rotation[:, 2])
            control = np.linalg.solve(mixing, np.r_[total, torque])
            saturations += int(np.any((control < 0) | (control > model.actuator_ctrlrange[:, 1])))
            d.ctrl[:] = np.clip(control, 0, model.actuator_ctrlrange[:, 1])
            errors.append(float(np.linalg.norm(reference-d.qpos[:3])))
        else:
            d.ctrl[:] = 0
        if step % 5 == 0:
            records.append(np.r_[t, d.qpos, d.qvel, d.act, d.ctrl, reference])
        mujoco.mj_step(model, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise ValueError("Non-finite model exercise")
    mujoco.mj_forward(model, d)
    trace = np.array(records)
    np.savez_compressed(output / "trajectory.npz", samples=trace,
                        columns=np.array(["t"] + [f"qpos_{i}" for i in range(7)] + [f"qvel_{i}" for i in range(6)] +
                                         [f"act_{i}" for i in range(4)] + [f"ctrl_{i}" for i in range(4)] + ["target_x", "target_y", "target_z"]))
    checks["truth_feedback_takeoff_hover_translate_return_land"] = bool(max(errors) < .04 and
        np.max(trace[:, 3]) > .29 and np.max(trace[:, 1]) > .14 and
        np.linalg.norm(d.qpos[:2]) < .01 and abs(d.qpos[2]-.0225) < .002 and np.linalg.norm(d.qvel) < .01 and d.ncon > 0)
    summary = dict(passed=all(checks.values()), mode="headless MuJoCo truth-feedback model exercise; no SITL", checks=checks,
                   mass_kg=float(mass), hover_drift_m=hover_drift, passive_rest_height_m=rest_height,
                   maximum_tracking_error_m=max(errors), maximum_motor_wrench_error=max(pulse_errors),
                   rotor_opening_ray_distances_m=rays, saturation_steps=saturations,
                   final_position_m=d.qpos[:3].tolist(), final_speed_m_s=float(np.linalg.norm(d.qvel[:3])),
                   duration_s=float(d.time), package=str(package),
                   file_hashes={str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(package.rglob("*")) if p.is_file() and p.suffix in (".xml", ".obj", ".json") and "legacy_v1" not in p.parts},
                   limitations="Basic model interface and rigid contact checks only. No autopilot, field mission, real sensor, material or propulsion calibration.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "file_hashes"}, indent=2))
    if not summary["passed"]:
        raise RuntimeError("Ver2 model checks failed; see summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=resolve_asset("components/rm_mjcf/rm27_drone_mjcf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.package, args.output)
