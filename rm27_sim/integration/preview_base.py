"""Export the complete base STEP as a static, visual-only CAD preview."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def render(scene):
    """Save two oblique views per CAD group without assuming an up direction."""
    import mujoco
    from PIL import Image, ImageDraw

    scene = Path(scene).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 600
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in ("negative_x", "positive_x"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"Not a base preview: missing body {name}")
    canvas = Image.new("RGB", (1600, 1260), "white")
    draw = ImageDraw.Draw(canvas)
    with mujoco.Renderer(model, height=600, width=800) as renderer:
        for i, (name, azimuth) in enumerate(
            (
                ("negative_x", 45),
                ("negative_x", 225),
                ("positive_x", 45),
                ("positive_x", 225),
            )
        ):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.xpos[body_id]
            camera.distance = 3.8
            camera.azimuth = azimuth
            camera.elevation = -25
            renderer.update_scene(data, camera=camera)
            x, y = (i % 2) * 800, (i // 2) * 630
            canvas.paste(Image.fromarray(renderer.render()), (x, y + 30))
            draw.text(
                (x + 10, y + 8), f"{name}, CAD axes, azimuth {azimuth}", fill="black"
            )
    canvas.save(scene.parent / "views.png")
    print(scene.parent / "views.png")


def export(source, output):
    import numpy as np
    import trimesh
    import mujoco
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    reader = STEPControl_Reader()
    print("Reading full STEP assembly…", flush=True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError("STEP import failed")
    shape = reader.OneShape()
    print("Tessellating all surfaces, including faces outside solids…", flush=True)
    BRepMesh_IncrementalMesh(shape, 0.5, False, 0.25, False).Perform()
    output.mkdir(parents=True)
    if not StlAPI_Writer().Write(shape, str(output / "assembly_mm.stl")):
        raise RuntimeError("Full assembly STL export failed")
    mesh = trimesh.load(output / "assembly_mm.stl", force="mesh", process=False)
    mesh.vertices *= 0.001
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
        raise ValueError("Invalid tessellation")
    triangle_x = mesh.triangles[:, :, 0]
    if np.any((triangle_x.min(axis=1) < 0) & (triangle_x.max(axis=1) > 0)):
        raise ValueError("A triangle crosses the proposed X=0 preview partition")
    root = ET.Element("mujoco", model="base_CAD_visual_preview")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    groups = []
    for name, mask in (
        ("negative_x", triangle_x.mean(axis=1) < 0),
        ("positive_x", triangle_x.mean(axis=1) >= 0),
    ):
        part = mesh.submesh([np.flatnonzero(mask)], append=True, repair=False)
        center = part.bounds.mean(axis=0)
        original_bounds = part.bounds.copy()
        part.vertices -= center
        path = output / f"{name}.obj"
        part.export(path)
        ET.SubElement(asset, "mesh", name=name, file=str(path))
        body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, center)))
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            contype="0",
            conaffinity="0",
            rgba="0.65 0.68 0.72 1",
        )
        groups.append(
            {
                "name": name,
                "triangles": len(part.faces),
                "bounds_cad_axes_m": original_bounds.tolist(),
                "center_cad_axes_m": center.tolist(),
            }
        )
    ET.SubElement(world, "light", pos="0 -5 8", dir="0 0 -1", directional="true")
    ET.indent(root)
    ET.ElementTree(root).write(output / "scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    if model.nq or np.any(model.geom_contype) or np.any(model.geom_conaffinity):
        raise AssertionError("Preview must be static and visual-only")
    report = {
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "triangles": len(mesh.faces),
        "groups": groups,
        "units": "metres; imported CAD axes preserved",
        "linear_tessellation_tolerance_mm": 0.5,
        "collision_geoms": 0,
        "limitations": "No color recovery, semantic labels, flight integration or validated upright orientation. CAD exporter can skip faces with null triangulation; inspect its warnings. Surface completeness is not established.",
    }
    (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render source as an existing scene.xml",
    )
    args = parser.parse_args()
    if args.render_only:
        render(args.source)
    else:
        if args.output is None:
            parser.error("Export requires a new output directory")
        export(args.source, args.output)
