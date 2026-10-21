"""Audit the supplied rectilinear dock STEP and build an exact box partition.

CAD: millimetres, +Y up. Dock: metres, +Z up; (x,y,z)=(X,-Z,Y)/1000.
The partition is accepted only after CAD bounds, volume and occupancy checks.
"""

from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import xml.etree.ElementTree as ET


def boxes():
    # Disjoint partition of base, perimeter and cross divider, in dock metres.
    return [
        ("bottom", (0, 0, 0.08), (0.15, 0.15, 0.08)),
        ("west", (-0.1475, 0, 0.18), (0.0025, 0.15, 0.02)),
        ("east", (0.1475, 0, 0.18), (0.0025, 0.15, 0.02)),
        ("south", (0, -0.1475, 0.18), (0.145, 0.0025, 0.02)),
        ("north", (0, 0.1475, 0.18), (0.145, 0.0025, 0.02)),
        ("divider_x", (0, 0, 0.18), (0.0025, 0.145, 0.02)),
        ("divider_y_w", (-0.07375, 0, 0.18), (0.07125, 0.0025, 0.02)),
        ("divider_y_e", (0.07375, 0, 0.18), (0.07125, 0.0025, 0.02)),
    ]


def slot_centers():
    return {
        f"slot_{i+1}": (x, y, 0.16)
        for i, (x, y) in enumerate(itertools.product((-0.07375, 0.07375), repeat=2))
    }


def export(source, output):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    import trimesh

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    text = source.read_bytes()
    if b"SI_UNIT ( .MILLI., .METRE. )" not in text:
        raise ValueError("Expected explicit millimetre STEP units")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError("STEP import failed")
    reader.TransferRoots()
    shape = reader.OneShape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("CAD solid is invalid")
    bound = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bound)
    low, high = bound.CornerMin().Coord(), bound.CornerMax().Coord()
    if not np.allclose([low, high], [[-150, 0, -150], [150, 200, 150]], atol=1e-5):
        raise ValueError(f"Dock geometry changed; review partition: {low}, {high}")
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    proxy_volume = sum(8 * np.prod(size) for _, _, size in boxes())
    cad_volume = props.Mass() * 1e-9
    if not np.isclose(cad_volume, proxy_volume, rtol=1e-8):
        raise ValueError("CAD/proxy solid volumes differ")
    classifier = BRepClass3d_SolidClassifier(shape)
    rng = np.random.default_rng(270912)
    points = rng.uniform([-0.16, -0.16, -0.01], [0.16, 0.16, 0.21], (6000, 3))
    failures = 0
    for p in points:
        classifier.Perform(gp_Pnt(p[0] * 1000, p[2] * 1000, -p[1] * 1000), 1e-7)
        cad_inside = classifier.State() in (TopAbs_IN, TopAbs_ON)
        proxy_inside = any(
            np.all(np.abs(p - center) <= size) for _, center, size in boxes()
        )
        failures += cad_inside != proxy_inside
    if failures:
        raise ValueError(f"CAD/proxy occupancy mismatches: {failures}")
    output.mkdir(parents=True)
    BRepMesh_IncrementalMesh(shape, 0.05, False, 0.1, False).Perform()
    if not StlAPI_Writer().Write(shape, str(output / "dock_cad_mm.stl")):
        raise RuntimeError("CAD tessellation export failed")
    mesh = trimesh.load(output / "dock_cad_mm.stl", force="mesh")
    original = mesh.vertices.copy()
    mesh.vertices = original[:, [0, 2, 1]] * np.array([0.001, -0.001, 0.001])
    mesh.export(output / "dock_visual.stl")
    report = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cad_units": "mm",
        "cad_bounds_mm": [low, high],
        "cad_to_dock": "(X,-Z,Y)/1000",
        "dimensions_m": [0.3, 0.3, 0.2],
        "slot_clear_dimensions_m": [0.1425, 0.1425, 0.04],
        "slot_centers_m": slot_centers(),
        "collision_boxes": len(boxes()),
        "cad_volume_m3": cad_volume,
        "collision_volume_m3": float(proxy_volume),
        "occupancy_samples": len(points),
        "occupancy_mismatches": int(failures),
        "visual_triangles": len(mesh.faces),
        "limitations": "Static rigid dock; no doors, charging or retention mechanism.",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    return report


def add_to_world(root, mesh_path, position, quaternion):
    asset = root.find("asset")
    ET.SubElement(
        asset, "mesh", name="rm27_dock_visual", file=str(Path(mesh_path).resolve())
    )
    body = ET.SubElement(
        root.find("worldbody"),
        "body",
        name="rm27_dock",
        pos=" ".join(map(str, position)),
        quat=" ".join(map(str, quaternion)),
    )
    ET.SubElement(
        body,
        "geom",
        name="rm27_dock_visual",
        type="mesh",
        mesh="rm27_dock_visual",
        contype="0",
        conaffinity="0",
        group="2",
        rgba="0.3 0.36 0.42 1",
    )
    for name, center, size in boxes():
        ET.SubElement(
            body,
            "geom",
            name=f"rm27_dock_collision_{name}",
            type="box",
            pos=" ".join(map(str, center)),
            size=" ".join(map(str, size)),
            contype="1",
            conaffinity="1",
            group="3",
            rgba="0 .8 .4 0",
            friction=".8 .01 .001",
            solref=".01 1",
            solimp=".95 .99 .001",
        )
    ET.SubElement(body, "site", name="dock_frame", size=".002", rgba="1 1 0 1")
    for name, p in slot_centers().items():
        ET.SubElement(
            body,
            "site",
            name=name,
            pos=" ".join(map(str, p)),
            type="box",
            size=".055 .055 .0005",
            rgba=".1 .8 .3 .6",
        )
    return body
