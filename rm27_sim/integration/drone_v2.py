"""Build and compose the reviewed Ver2 CAD with provisional MuJoCo dynamics.

Runtime files belong in components/rm_mjcf; CAD conversion and evidence stay in this repo.
Run ``python -m rm27.integration.drone_v2 --help`` for the reproducible exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from rm27_sim.paths import ROOT, resolve_asset
CONFIG = ROOT / "configs/drone_v2.json"
COLORS = {
    "frame": [.32, .39, .48, 1], "supports": [.76, .78, .81, 1],
    "motor": [.2, .63, .86, 1], "fc": [.12, .63, .36, 1],
    "battery": [.96, .59, .14, 1], "top": [.52, .36, .72, 1],
    "camera": [.86, .28, .29, 1], "core_board": [.10, .72, .65, 1],
}


def vec(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def group(name):
    if "jiangbao" in name:
        return "frame"
    if "hashira" in name or "betweenboard" in name:
        return "supports"
    return {"dmotor": "motor", "2627飛控": "fc", "26dronebattery": "battery",
            "26dronetoplayer4": "top", "core_board_30x65x5": "core_board"}.get(name, "camera")


def read_parts(source, cfg):
    """Retain XCAF occurrence transforms and apply only the two approved edits."""
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.collections import Sequence_TDF_Label
    from OCP.TDF import TDF_Label
    from OCP.TDataStd import TDataStd_Name
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != cfg["source_sha256"]:
        raise ValueError("Ver2 STEP changed; re-audit before exporting")
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError("Cannot read Ver2 STEP")
    doc = TDocStd_Document(TCollection_ExtendedString("Ver2"))
    if not reader.Transfer(doc):
        raise ValueError("Cannot transfer Ver2 STEP")
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = Sequence_TDF_Label()
    st.GetFreeShapes(roots)
    if roots.Length() != 1:
        raise ValueError("Expected one aircraft assembly")

    def bounds(shape):
        b = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, b, False, False)
        return np.array([b.CornerMin().Coord(), b.CornerMax().Coord()])

    parts, removed = [], []

    def walk(label, parent=TopLoc_Location()):
        ref = TDF_Label()
        definition = ref if st.GetReferredShape_s(label, ref) else label
        attr = TDataStd_Name()
        if not definition.FindAttribute(TDataStd_Name.GetID_s(), attr):
            raise ValueError("Unnamed CAD part")
        name = attr.Get().ToExtString()
        loc = parent.Multiplied(st.GetLocation_s(label))
        children = Sequence_TDF_Label()
        st.GetComponents_s(definition, children)
        if children.Length():
            for i in range(1, children.Length() + 1):
                walk(children.Value(i), loc)
        elif name in cfg["removed_parts"]:
            removed.append(name)
        else:
            shape = st.GetShape_s(definition).Moved(loc)
            if not BRepCheck_Analyzer(shape).IsValid():
                raise ValueError(f"Invalid CAD part: {name}")
            if TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID).More():
                raise ValueError(f"Unexpected free surfaces: {name}")
            parts.append((name, shape, bounds(shape)))

    walk(roots.Value(1))
    if sorted(removed) != sorted(cfg["removed_parts"]) or len(parts) != 29:
        raise ValueError("Unexpected Ver2 assembly inventory")
    battery = next(b for n, _, b in parts if n == "26dronebattery")
    top = next(b for n, _, b in parts if n == "26dronetoplayer4")
    w, length, height = cfg["core_board_size_mm"]
    center = battery.mean(axis=0)
    point = [center[0] - w / 2, top[1, 1], center[2] - length / 2]
    block = BRepPrimAPI_MakeBox(gp_Pnt(*point), w, height, length).Shape()
    parts.append(("core_board_30x65x5", block, bounds(block)))
    return parts


def build(output, config=CONFIG):
    """Create a new self-contained visual/dynamic bundle; never overwrite a run."""
    import trimesh
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    output = Path(output).resolve()
    cfg = json.loads(Path(config).read_text())
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    parts = read_parts(resolve_asset(cfg["source_cad"]), cfg)
    boxes = np.array([b for _, _, b in parts])
    low, high = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
    basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    origin = (low + high) / 2
    origin[1] = low[1] + cfg["com_height_above_bottom_m_assumed"] * 1000
    output.mkdir(parents=True)
    assets = output / "assets"
    assets.mkdir()
    grouped = {k: [] for k in COLORS}
    inventory = []
    for i, (name, shape, b) in enumerate(parts):
        BRepMesh_IncrementalMesh(shape, .05, False, .15, False).Perform()
        path = assets / f"part_{i:02d}.stl"
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(shape, str(path)):
            raise ValueError(f"Cannot tessellate {name}")
        mesh = trimesh.load(path, force="mesh", process=False)
        path.unlink()  # Only this builder's intermediate STL; final runtime uses OBJ.
        mesh.vertices = (mesh.vertices - origin) @ basis.T / 1000
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"Invalid mesh {name}")
        grouped[group(name)].append(mesh)
        bb = (b - origin) @ basis.T / 1000
        inventory.append(dict(name=name, group=group(name), bounds_m=bb.tolist(),
                              bounds_cad_mm=b.tolist(), triangles=len(mesh.faces)))
    for key, meshes in grouped.items():
        trimesh.util.concatenate(meshes).export(assets / f"{key}.obj")
    dims = (high - low) @ basis.T / 1000
    np.testing.assert_allclose(dims, [.114, .114, .045], atol=2e-6)
    dimensions = np.array([cfg["outer_width_m"], cfg["outer_width_m"], cfg["outer_height_m"]])
    inertia = cfg["mass_kg"] / 12 * (np.sum(dimensions**2) - dimensions**2)

    def document(dynamic):
        root = ET.Element("mujoco", model=cfg["name"])
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
        ET.SubElement(root, "option", timestep=".002", integrator="implicitfast")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="1000", offheight="800")
        ET.SubElement(visual, "headlight", ambient=".5 .5 .5", diffuse=".6 .6 .6")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        body = ET.SubElement(world, "body", name="rm27_drone", pos="0 0 .1" if dynamic else "0 0 0")
        if dynamic:
            ET.SubElement(body, "freejoint", name="rm27_drone_free")
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(cfg["mass_kg"]), diaginertia=vec(inertia))
        for key in grouped:
            ET.SubElement(asset, "mesh", name=f"rm27_drone_visual_{key}", file=f"../assets/{key}.obj", maxhullvert="4")
            ET.SubElement(body, "geom", name=f"rm27_drone_visual_{key}", type="mesh", mesh=f"rm27_drone_visual_{key}",
                          rgba=vec(COLORS[key]), contype="0", conaffinity="0", group="2", mass="0")
        ET.SubElement(body, "site", name="rm27_drone_imu", pos="0 0 0", size=".001", rgba="0 0 0 0")
        ET.SubElement(body, "site", name="rm27_drone_bottom", pos=vec([0, 0, -cfg["com_height_above_bottom_m_assumed"]]), size=".001", rgba="0 0 0 0")
        if dynamic:
            add_collision(body, inventory, cfg)
            actuators = ET.SubElement(root, "actuator")
            a = cfg["rotor_center_offset_m"]
            rotors = [p for p in inventory if p["name"] == "dmotor"]
            rotor_z = float(np.array(rotors[0]["bounds_m"])[:, 2].mean())
            for i, (x, y) in enumerate(((a, a), (a, -a), (-a, -a), (-a, a))):
                site = f"rm27_drone_rotor_{i}"
                ET.SubElement(body, "site", name=site, pos=vec([x, y, rotor_z]), size=".001", rgba="0 0 0 0")
                e = cfg["engineering_estimates"]
                ET.SubElement(actuators, "general", name=f"rm27_drone_motor_{i}", site=site,
                              gear=vec([0, 0, 1, 0, 0, (-1)**i * e["yaw_torque_per_thrust_m"]]),
                              ctrlrange=vec([0, e["max_rotor_thrust_N"]]), dyntype="filterexact", dynprm=str(e["motor_time_constant_s"]))
            sensor = ET.SubElement(root, "sensor")
            ET.SubElement(sensor, "gyro", name="rm27_drone_gyro", site="rm27_drone_imu")
            ET.SubElement(sensor, "accelerometer", name="rm27_drone_accel", site="rm27_drone_imu")
            ET.SubElement(sensor, "framequat", name="rm27_drone_attitude_truth", objtype="body", objname="rm27_drone")
        ET.SubElement(root, "statistic", center="0 0 .1" if dynamic else "0 0 0", extent=".25")
        return root

    for kind in ["visual", "dynamic"]:
        (output / kind).mkdir()
        tree = document(kind == "dynamic")
        ET.indent(tree)
        ET.ElementTree(tree).write(output / kind / "model.xml", encoding="unicode")
    scene = ET.fromstring('<mujoco model="Ver2_ground_scene"><include file="model.xml"/><worldbody>'
                          '<light pos="1 -1 2" dir="-1 1 -2"/>'
                          '<geom name="ground" type="plane" size="2 2 .1" rgba=".22 .25 .29 1" friction=".6 .005 .0001"/>'
                          '</worldbody></mujoco>')
    ET.indent(scene)
    ET.ElementTree(scene).write(output / "dynamic/scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "dynamic/scene.xml"))
    assert model.nq == 7 and model.nv == 6 and model.nu == 4
    metadata = dict(schema_version=1, source_cad=cfg["source_cad"], source_sha256=cfg["source_sha256"],
                    removed_parts=cfg["removed_parts"], added_core_board_mm=cfg["core_board_size_mm"],
                    cad_to_body=basis.tolist(), body_origin_cad_mm=origin.tolist(), dimensions_m=dims.tolist(),
                    mass_kg=cfg["mass_kg"], mass_evidence=cfg["mass_evidence"], inertia_kg_m2_assumed=inertia.tolist(),
                    configuration=cfg, parts=inventory,
                    collision_geoms=int(np.count_nonzero((model.geom_contype != 0) & (model.geom_bodyid == model.body("rm27_drone").id))),
                    limitations=cfg["limitations"], generator="rm27_sim/integration/drone_v2.py")
    (output / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def add_collision(body, parts, cfg):
    """Explicit rigid proxies; ring openings stay open, visual meshes never collide."""
    e = cfg["engineering_estimates"]
    common = dict(group="3", rgba=".8 .2 .8 .4", mass="0", friction=vec(e["contact_friction"]),
                  solref=vec(e["contact_solref"]), contype="1", conaffinity="1")
    def geom(name, **attrs):
        return ET.SubElement(body, "geom", name=f"rm27_drone_collision_{name}", **common, **attrs)
    a, r, tube = cfg["rotor_center_offset_m"], e["ring_center_radius_m"], e["ring_tube_radius_m"]
    centers = [(a, a), (a, -a), (-a, -a), (-a, a)]
    for layer, name in [("upper", "26dronejiangbao5.0"), ("lower", "26dronejiangbao5-2")]:
        bb = np.array(next(p["bounds_m"] for p in parts if p["name"] == name))
        z = bb[:, 2].mean()
        for i, (x, y) in enumerate(centers):
            angles = np.linspace(0, 2 * np.pi, e["ring_segments"] + 1)
            points = np.c_[x + r * np.cos(angles), y + r * np.sin(angles), np.full(len(angles), z)]
            for j in range(len(angles) - 1):
                geom(f"{layer}_ring_{i}_{j}", type="capsule", size=str(tube), fromto=vec(np.r_[points[j], points[j+1]]))
            if layer == "upper":
                for j, theta in enumerate(np.pi / 4 + np.arange(4) * np.pi / 2):
                    end = [x + (r - tube) * np.cos(theta), y + (r - tube) * np.sin(theta), z]
                    geom(f"motor_strut_{i}_{j}", type="capsule", size=".0015", fromto=vec([x, y, z] + end))
    for i, p in enumerate(parts):
        name = p["name"]
        bb = np.array(p["bounds_m"])
        center, half = bb.mean(axis=0), (bb[1] - bb[0]) / 2
        if "jiangbao" in name or name == "26dronetoplayer4":
            continue
        if name == "dmotor":
            # Only central motor can collide; swept propeller disks are visual-only.
            center[2] = bb[1, 2] - .005
            geom(f"motor_{i}", type="cylinder", pos=vec(center), size=".0075 .005")
        elif name == "2627飛控":
            # Use the 37 mm local square, rotated 45 degrees, not its 52 mm AABB.
            geom(f"fc_{i}", type="box", pos=vec(center), size=".0185 .0185 .002",
                 quat=vec([np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)]))
        elif "hashira" in name:
            geom(f"post_{i}", type="cylinder", pos=vec(center), size=vec([.0026, half[2]]))
        else:
            geom(f"{p['group']}_{i}", type="box", pos=vec(center), size=vec(half))
    # The purple top plate is open: a ring and three support spokes instead of a solid box.
    bb = np.array(next(p["bounds_m"] for p in parts if p["name"] == "26dronetoplayer4"))
    z = bb[:, 2].mean()
    for i, theta in enumerate(np.arange(8) * np.pi / 4):
        start = [.018 * np.cos(theta), .018 * np.sin(theta), z]
        end = [.018 * np.cos(theta + np.pi/4), .018 * np.sin(theta + np.pi/4), z]
        geom(f"top_ring_{i}", type="capsule", size=".001", fromto=vec(start + end))
    for i, p in enumerate(p for p in parts if p["name"] == "26drone12mmhashira"):
        endpoint = np.array(p["bounds_m"]).mean(axis=0)
        endpoint[2] = z
        start = endpoint.copy()
        start[:2] *= .018 / np.linalg.norm(start[:2])
        geom(f"top_spoke_{i}", type="capsule", size=".001", fromto=vec(np.r_[start, endpoint]))


def add(root, name, position, quaternion, cfg):
    """Compose packaged Ver2 assets with the existing root four-motor interface."""
    package = resolve_asset(cfg["runtime_package"])
    source = package / "dynamic/model.xml"
    template = ET.parse(source).getroot()
    info = json.loads((package / "model_info.json").read_text())
    if info["configuration"] != {k: v for k, v in cfg.items() if k not in ("dynamics_override", "calibration")}:
        # Calibration changes mass; compare the remaining reviewed geometry/configuration.
        reviewed = dict(info["configuration"])
        supplied = {k: v for k, v in cfg.items() if k not in ("dynamics_override", "calibration")}
        if "dynamics_override" in cfg:
            reviewed["mass_kg"] = supplied["mass_kg"]
        if reviewed != supplied:
            raise ValueError("Ver2 config differs from runtime bundle; rebuild it")
    def section(tag):
        element = root.find(tag)
        return element if element is not None else ET.SubElement(root, tag)
    for asset in template.findall("asset/*"):
        asset.set("file", str((source.parent / asset.get("file")).resolve()))
    for element in template.iter():
        for key, value in list(element.attrib.items()):
            if value.startswith("rm27_drone"):
                element.set(key, name + value[len("rm27_drone"):])
    body = template.find("worldbody/body")
    body.set("pos", vec(position))
    body.set("quat", vec(quaternion))
    override = cfg.get("dynamics_override", {})
    if override:
        inertial = body.find("inertial")
        inertial.set("mass", str(override["mass_kg"]))
        inertial.set("diaginertia", vec(override["inertia_kg_m2"]))
        for i, actuator in enumerate(template.findall("actuator/*")):
            actuator.set("ctrlrange", vec([0, override["max_rotor_thrust_N"]]))
            actuator.set("dynprm", str(override["motor_time_constant_s"]))
            actuator.set("gear", vec([0, 0, 1, 0, 0, (-1)**i * override["yaw_torque_per_thrust_m"]]))
    section("worldbody").append(body)
    for tag in ("asset", "actuator", "sensor"):
        section(tag).extend(list(template.find(tag)))
    return dict(name=name, mass_kg=cfg["mass_kg"],
                inertia_kg_m2=override.get("inertia_kg_m2", info["inertia_kg_m2_assumed"]),
                max_rotor_thrust_N_assumed=override.get("max_rotor_thrust_N", cfg["engineering_estimates"]["max_rotor_thrust_N"]),
                flight_verified=False, model=cfg["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = build(args.output, args.config)
    print(json.dumps({k: result[k] for k in ["dimensions_m", "mass_kg", "collision_geoms"]}, indent=2))
