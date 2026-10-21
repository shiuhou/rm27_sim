"""Provisional 120 mm, 180 g geometry; not the frozen G1 flight plant."""

import itertools
import xml.etree.ElementTree as ET
import numpy as np


def add(root, name, position, quaternion, cfg):
    if cfg.get("model") == "cad_v2":
        from .drone_v2 import add as add_v2
        return add_v2(root, name, position, quaternion, cfg)
    width = cfg["outer_width_m"]
    a = cfg["rotor_center_offset_m"]
    r = cfg["propeller_diameter_m"] / 2
    guard_r = 0.002
    rail = width / 2 - guard_r
    if a + r > rail - guard_r or 2 * a < 2 * r:
        raise ValueError("Propellers overlap each other or the guard envelope")
    estimates = cfg["engineering_estimates"]
    mass = cfg["mass_kg"]
    motor_mass = estimates["motor_point_mass_kg"]
    central = mass - 4 * motor_mass
    if central <= 0:
        raise ValueError("Invalid mass allocation")
    inertia = (
        central
        / 12
        * np.array([0.03**2 + 0.02**2, 0.04**2 + 0.02**2, 0.04**2 + 0.03**2])
    )
    inertia += motor_mass * np.array([4 * a * a, 4 * a * a, 8 * a * a])
    override = cfg.get('dynamics_override', {})
    if override:
        inertia = np.asarray(override['inertia_kg_m2'], float)
    body = ET.SubElement(
        root.find("worldbody"),
        "body",
        name=name,
        pos=" ".join(map(str, position)),
        quat=" ".join(map(str, quaternion)),
    )
    ET.SubElement(body, "freejoint", name=f"{name}_free")
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=str(mass),
        diaginertia=" ".join(map(str, inertia)),
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_fuselage",
        type="box",
        size=".02 .015 .01",
        rgba=".2 .25 .3 1",
    )
    # Top/bottom rails and corner posts form a replaceable collision proxy.
    corners = list(itertools.product((-rail, rail), repeat=2))
    edges = []
    for z in (-0.018, 0.018):
        for axis in (0, 1):
            for side in (-rail, rail):
                start, end = [side, side, z], [side, side, z]
                start[axis], end[axis] = -rail, rail
                edges.append((start, end))
    edges += [([x, y, -0.018], [x, y, 0.018]) for x, y in corners]
    for i, (start, end) in enumerate(edges):
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_guard_{i}",
            type="capsule",
            size=str(guard_r),
            fromto=" ".join(map(str, start + end)),
            rgba=".8 .65 .15 1",
        )
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(root, "actuator")
    maximum = estimates["thrust_to_weight"] * mass * 9.81 / 4
    maximum = override.get('max_rotor_thrust_N', maximum)
    tau = override.get('motor_time_constant_s', estimates['motor_time_constant_s'])
    yaw_torque = override.get('yaw_torque_per_thrust_m', estimates['yaw_torque_per_thrust_m'])
    for i, (x, y) in enumerate(((a, a), (a, -a), (-a, -a), (-a, a))):
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_arm_{i}",
            type="capsule",
            size=".002",
            fromto=f"0 0 0 {x} {y} 0",
            rgba=".2 .2 .2 1",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_prop_{i}",
            type="cylinder",
            size=f"{r} .001",
            pos=f"{x} {y} .008",
            contype="0",
            conaffinity="0",
            rgba=".2 .5 .8 .8",
        )
        site = f"{name}_rotor_{i}"
        ET.SubElement(body, "site", name=site, pos=f"{x} {y} .008", size=".001")
        ET.SubElement(
            actuators,
            "general",
            name=f"{name}_motor_{i}",
            site=site,
            gear=f"0 0 1 0 0 {(-1)**i * yaw_torque}",
            ctrlrange=f"0 {maximum}",
            dyntype="filterexact",
            dynprm=str(tau),
        )
    return {
        "name": name,
        "mass_kg": mass,
        "inertia_kg_m2": inertia.tolist(),
        "max_rotor_thrust_N_assumed": maximum,
        "flight_verified": False,
    }
