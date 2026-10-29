"""Plant, explicit uncertainty draws, and estimate/action boundary.

World/body +Z up. R maps body to world; omega is body-frame rad/s.
Only this plant adapter and the evaluator read MuJoCo hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from rm27_contact_physics.drone import add as add_drone, import_mjcf


def unit(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-10:
        raise ValueError("Expected finite nonzero vector")
    return v / np.linalg.norm(v)


def frame(normal):
    z = unit(normal)
    heading = [1., 0., 0.] if abs(z[0]) < .99 else [0., 1., 0.]
    y = unit(np.cross(z, heading))
    return np.column_stack((np.cross(y, z), y, z))


def target_frame(target):
    """A plane normal alone cannot specify a rectangular armor footprint."""
    R = frame(target["normal"])
    angle = target.get("in_plane_rotation_deg", 0.)
    return R if angle == 0 else R @ Rotation.from_euler("z", angle, degrees=True).as_matrix()


def normal_alignment(source_normal, target_normal):
    """Return the shortest rotation that maps one plane normal to another.

    A plane's in-plane rectangle heading is intentionally excluded.  The
    marker is mounted at a fixed world offset in the base asset; when a test
    perturbs the target normal, only that normal-induced pose change should be
    applied to the marker offset.
    """
    source = unit(source_normal)
    target = unit(target_normal)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1., 1.))
    if sine < 1e-10:
        if cosine > 0.:
            return np.eye(3)
        # Antipodal normals have infinitely many shortest axes. Pick a stable
        # axis orthogonal to the source so the result remains deterministic.
        axis_hint = np.array([1., 0., 0.])
        if abs(source[0]) > .9:
            axis_hint = np.array([0., 1., 0.])
        axis = unit(np.cross(source, axis_hint))
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / sine
    return Rotation.from_rotvec(axis * np.arctan2(sine, cosine)).as_matrix()


def marker_center(target, marker_config):
    """Return the configured guidance-lamp center for a target pose.

    The lamp is a geometric aid for the flight-control visibility objective. Its
    local offset is taken from ``base_target.json`` and carried with the armor
    frame when a scenario perturbs the target pose. This is not a vision
    measurement or a claim that the lamp pose is hardware-calibrated.
    """
    base = marker_config
    base_pose = dict(center_m=base["armor_surface_center_m"],
                     normal=base["armor_normal"],
                     in_plane_rotation_deg=base.get("in_plane_rotation_deg", 0.))
    base_offset = (np.asarray(base["marker_center_m"], float)
                   - np.asarray(base["armor_surface_center_m"], float))
    alignment = normal_alignment(base["armor_normal"], target["normal"])
    return np.asarray(target["center_m"], float) + alignment @ base_offset


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    dt, control = cfg["physics_dt_s"], cfg["control_dt_s"]
    if dt <= 0 or control < dt or not np.isclose(control / dt, round(control / dt)):
        raise ValueError("Control period must be an integer multiple of positive physics dt")
    if cfg["horizon_s"] <= 0 or cfg["contact"]["radius_m"] <= 0:
        raise ValueError("Invalid horizon/contact radius")
    unit(cfg["target"]["normal"])
    return cfg


@dataclass(frozen=True)
class VehicleEstimate:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    omega_body: np.ndarray
    timestamp: float
    source: str = "MUJOCO_TRUTH_TEACHER"


@dataclass(frozen=True)
class TargetState:
    position: np.ndarray
    normal: np.ndarray
    covariance: np.ndarray
    timestamp: float
    confidence: float = 1.0
    source: str = "SYNTHETIC_STATIC_ESTIMATE"

    def valid(self, now, max_age=0.25):
        return (0 <= now - self.timestamp <= max_age and self.confidence > 0
                and np.isfinite(self.position).all() and np.isfinite(self.normal).all()
                and np.linalg.norm(self.normal) > 1e-8)


def scenario(cfg, seed=None, scale=1.0):
    """Draw physical perturbations once; no trial draws depend on method/action."""
    c = copy.deepcopy(cfg)
    c["draw"] = dict(seed=seed, scale=scale, start_dp=[0., 0., 0.],
                     start_dv=[0., 0., 0.], start_rotvec=[0., 0., 0.],
                     thrust_scale=1., mass_scale=1., motor_tau_scale=1.)
    if seed is None:
        return c
    rng = np.random.default_rng(seed)
    u, d = cfg["uncertainty"], c["draw"]
    def uniform(key):
        lo, hi = u[key]
        return float(1 + scale * (rng.uniform(lo, hi) - 1))
    d["start_dp"] = (rng.normal(size=3) * u["start_position_std_m"] * scale).tolist()
    d["start_dv"] = (rng.normal(size=3) * u["start_velocity_std_m_s"] * scale).tolist()
    d["start_rotvec"] = (rng.normal(size=3) * np.deg2rad(u["start_attitude_std_deg"]) * scale).tolist()
    c["target"]["center_m"] = (np.array(c["target"]["center_m"]) + rng.normal(size=3)
                                 * u["target_position_std_m"] * scale).tolist()
    n = Rotation.from_rotvec(rng.normal(size=3) * np.deg2rad(u["target_angle_std_deg"]) * scale).apply(c["target"]["normal"])
    c["target"]["normal"] = unit(n).tolist()
    for key in ("thrust_scale", "mass_scale", "motor_tau_scale"):
        d[key] = uniform(key + "_range")
    for key, source in (("friction", "friction_range"), ("solref_damping_ratio", "contact_damping_range")):
        lo, hi = u[source]
        nominal = cfg["target"][key]
        c["target"][key] = float(nominal + scale * (rng.uniform(lo, hi) - nominal))
    c["target"]["solref_s"] *= uniform("contact_time_scale_range")
    c["contact"]["offset_body_m"] = (np.array(c["contact"]["offset_body_m"]) + rng.normal(size=3)
                                    * u["contact_offset_std_m"] * scale).tolist()
    return c


def build_model(cfg, nominal_cfg, *, vehicle_config):
    root = ET.Element("mujoco", model="rm27_aggressive_contact_research")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=str(cfg["physics_dt_s"]), integrator="implicitfast",
                  gravity="0 0 -9.81", iterations="60", tolerance="1e-10")
    default = ET.SubElement(root, "default")
    t = cfg["target"]
    ET.SubElement(default, "geom", friction=f'{t["friction"]} .005 .0001',
                  solref=f'{t["solref_s"]} {t["solref_damping_ratio"]}', solimp=".95 .99 .001")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")
    ET.SubElement(world, "geom", name="ground", type="plane", size="4 4 .05", rgba=".18 .2 .23 1")
    if "landing_pad" in cfg:
        # Optional local test fixture: the central recess clears the assumed
        # protruding bumper while the CAD feet rest on the surrounding surface.
        # This is not the competition dock. Absent configuration preserves the
        # existing contact/RL scene byte for byte.
        pad = cfg["landing_pad"]
        cx, cy = pad["center_xy_m"]
        half, hole = pad["half_size_m"], pad["recess_half_size_m"]
        height, depth = pad["height_m"], pad["recess_depth_m"]
        if not np.isfinite([cx, cy, half, hole, height, depth]).all() or not (0 < hole < half and 0 < depth < height):
            raise ValueError("Invalid landing pad dimensions")
        pieces = [("floor", [cx, cy, (height-depth)/2], [half, half, (height-depth)/2])]
        for sign in (-1, 1):
            pieces.append((f"x_{sign}", [cx+sign*(half+hole)/2, cy, height/2],
                           [(half-hole)/2, half, height/2]))
            pieces.append((f"y_{sign}", [cx, cy+sign*(half+hole)/2, height/2],
                           [hole, (half-hole)/2, height/2]))
        for name, position, size in pieces:
            ET.SubElement(world, "geom", name="landing_pad_"+name, type="box",
                          pos=" ".join(map(str, position)), size=" ".join(map(str, size)),
                          friction="0.6 .005 .0001", solref=".004 1", rgba=".25 .55 .3 1")
    R = target_frame(t)
    xyzw = Rotation.from_matrix(R).as_quat()
    plate = ET.SubElement(world, "body", name="target", pos=" ".join(map(str, t["center_m"])),
                          quat=" ".join(map(str, xyzw[[3, 0, 1, 2]])))
    x, y, thick = np.array(t["dimensions_m"]) / 2
    def box(name, center, size, rgba):
        ET.SubElement(plate, "geom", name=name, type="box", pos=" ".join(map(str, center)),
                      size=" ".join(map(str, size)), rgba=rgba)
    box("armor", [0, 0, -thick], [x, y, thick], ".1 .35 .95 1")
    ox, oy = t["rim_outer_halfsize_m"]
    for sign in (-1, 1):
        box(f"rim_x_{sign}", [sign*(ox+x)/2, 0, -thick], [(ox-x)/2, oy, thick], ".5 .3 .3 1")
        box(f"rim_y_{sign}", [0, sign*(oy+y)/2, -thick], [x, (oy-y)/2, thick], ".5 .3 .3 1")
    vcfg = copy.deepcopy(vehicle_config)
    # Preserve nominal actuator/inertia reference; uncertain physics is hidden from controller.
    initial = np.array(nominal_cfg["target"]["center_m"]) + nominal_cfg["start_offset_m"]
    # A vehicle config with ``mjcf_path`` imports the runtime aircraft MJCF instead of the generated proxy.
    info = (import_mjcf(root, "uav", initial, vcfg) if "mjcf_path" in vcfg
            else add_drone(root, "uav", initial, [1, 0, 0, 0], vcfg))
    info["rotor_arm_m"] = vcfg["rotor_center_offset_m"]
    info["yaw_torque_per_thrust_m"] = vcfg.get("dynamics_override", {}).get(
        "yaw_torque_per_thrust_m", vcfg["engineering_estimates"]["yaw_torque_per_thrust_m"])
    body = world.find("body[@name='uav']")
    if cfg["contact"].get("bumper", True):
        ET.SubElement(body, "geom", name="bumper", type="sphere", size=str(cfg["contact"]["radius_m"]),
                      pos=" ".join(map(str, cfg["contact"]["offset_body_m"])), rgba=".1 .9 .3 1")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    uid = model.body("uav").id
    model.body_mass[uid] *= cfg["draw"]["mass_scale"]
    model.body_inertia[uid] *= cfg["draw"]["mass_scale"]
    model.actuator_gear[:] *= cfg["draw"]["thrust_scale"]
    model.actuator_dynprm[:, 0] *= cfg["draw"]["motor_tau_scale"]
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    data.qpos[:3] += cfg["draw"]["start_dp"]
    q = Rotation.from_rotvec(cfg["draw"]["start_rotvec"]).as_quat()
    data.qpos[3:7] = q[[3, 0, 1, 2]]
    data.qvel[:3] = cfg["draw"]["start_dv"]
    data.act[:] = vcfg["mass_kg"] * 9.81 / 4
    mujoco.mj_forward(model, data)
    return model, data, info, xml


def truth_estimate(model, data):
    return VehicleEstimate(data.qpos[:3].copy(), data.qvel[:3].copy(),
                           data.xmat[model.body("uav").id].reshape(3, 3).copy(),
                           data.qvel[3:6].copy(), float(data.time))


def advance_physics_interval(model, data, before, *, consume_interval, saturated=False):
    """Advance once, consume the interval solve, then refresh the endpoint.

    ``implicitfast`` leaves the completed interval's contact solve in ``data``.
    The task-owned consumer must read it before ``mj_forward`` replaces it with
    an endpoint snapshot. This function has no task, controller or reward logic.
    """
    mujoco.mj_step(model, data)
    consume_interval(model, data, before.timestamp, saturated)
    mujoco.mj_forward(model, data)
    return truth_estimate(model, data)
