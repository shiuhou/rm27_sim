"""Render a generated scene: MUJOCO_GL=egl python -m rm27_sim.simulation.render <scene.xml>."""

import argparse
from pathlib import Path
import mujoco
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    m = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    identifier = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rm27_dock")
    if identifier < 0:
        raise ValueError("No rm27_dock body")
    center = d.xpos[identifier].copy()
    center[2] += 0.15
    r = mujoco.Renderer(m, height=700, width=1000)
    canvas = Image.new("RGB", (2000, 1460), "white")
    draw = ImageDraw.Draw(canvas)
    views = [
        ("Dock and four parked proxies", center, 0.95, 135, -40),
        ("Slot layout", center, 0.85, 90, -89.9),
        ("H platform placement", center, 4.5, 215, -25),
        ("Whole field", (0, 0, 0.7), 34, 225, -40),
    ]
    for i, (name, look, distance, azimuth, elevation) in enumerate(views):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = look
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        r.update_scene(d, camera=camera, scene_option=options)
        im = Image.fromarray(r.render())
        im.save(args.scene.parent / f"view_{i}.png")
        x = i % 2 * 1000
        y = i // 2 * 730
        canvas.paste(im, (x, y + 30))
        draw.text((x + 10, y + 8), name, fill="black")
    canvas.save(args.scene.parent / "views.png")
    r.close()
    print(args.scene.parent / "views.png")


if __name__ == "__main__":
    main()
