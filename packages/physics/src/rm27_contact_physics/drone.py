"""Provisional 120 mm, 180 g geometry; not the frozen G1 flight plant."""

import itertools
import xml.etree.ElementTree as ET
import numpy as np


def add(root, name, position, quaternion, cfg):
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
        group="5",
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
            group="5",
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
            group="5",
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
            group="5",
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


def import_mjcf(root, name, position, cfg):
    """Insert the free body of a runtime aircraft MJCF (e.g. rm_mjcf/rm27_drone_mjcf/dynamic).

    Only collision proxies, sites and actuators are copied; visual meshes and sensors are
    dropped so no asset files are needed. Names are re-prefixed from the source body name
    to ``name``. Mass, inertia and actuator values come from ``dynamics_override`` when a
    calibration profile is applied, and must otherwise be present in the source file.
    """
    source = ET.parse(cfg["mjcf_path"]).getroot()
    bodies = [b for b in source.find("worldbody").findall("body") if b.find("freejoint") is not None]
    if len(bodies) != 1:
        raise ValueError("Expected exactly one free body in the aircraft MJCF")
    src = bodies[0]
    prefix = src.get("name")
    rename = lambda value: name + value[len(prefix):] if value and value.startswith(prefix) else value
    override = cfg.get("dynamics_override", {})
    body = ET.SubElement(root.find("worldbody"), "body", name=name,
                         pos=" ".join(map(str, position)), quat="1 0 0 0")
    ET.SubElement(body, "freejoint", name=f"{name}_free")
    inertial = src.find("inertial")
    mass = float(override.get("mass_kg", inertial.get("mass")))
    inertia = np.asarray(override.get("inertia_kg_m2", [float(v) for v in inertial.get("diaginertia").split()]), float)
    if np.linalg.norm(np.asarray([float(v) for v in inertial.get("pos", "0 0 0").split()])) > 0:
        raise ValueError("The aircraft MJCF must place the COM at the body origin")
    ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass), diaginertia=" ".join(map(str, inertia)))
    for child in src:
        if child.tag == "site" or (child.tag == "geom" and child.get("type") != "mesh"
                                   and child.get("contype", "1") != "0"):
            copy = ET.SubElement(body, child.tag, dict(child.attrib))
            copy.set("name", rename(child.get("name")))
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(root, "actuator")
    motors = source.find("actuator").findall("general")
    for i, motor in enumerate(motors):
        attrib = dict(motor.attrib)
        attrib.update(name=f"{name}_motor_{i}", site=rename(motor.get("site")))
        gear = [float(v) for v in attrib["gear"].split()]
        if "yaw_torque_per_thrust_m" in override:
            gear[5] = np.sign(gear[5]) * override["yaw_torque_per_thrust_m"]
        attrib["gear"] = " ".join(map(str, gear))
        if "max_rotor_thrust_N" in override:
            attrib["ctrlrange"] = f"0 {override['max_rotor_thrust_N']}"
        if "motor_time_constant_s" in override:
            attrib["dynprm"] = str(override["motor_time_constant_s"])
        ET.SubElement(actuators, "general", attrib)
    return {
        "name": name,
        "mass_kg": mass,
        "inertia_kg_m2": inertia.tolist(),
        "max_rotor_thrust_N_assumed": float(attrib["ctrlrange"].split()[1]),
        "flight_verified": False,
        "source_mjcf": str(cfg["mjcf_path"]),
    }
