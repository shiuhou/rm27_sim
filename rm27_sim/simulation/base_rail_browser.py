"""Display an external moving-base field with the upstream mjviser viewer.

Adapted from CosmosMount/rmuc2027_mjarena ffd1eb7 (MIT); see
.agents/docs/mjarena_LICENSE.txt. Target control uses our shared RailMotion.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from rmuc2027_mjarena.scene import Attachment, Simulation, load_scene
from .base_rail_control import RailMotion


def apply_target_input(rail, keys, pressed):
    """Apply browser inputs in the simulation thread; never step physics."""
    axis = int('ArrowRight' in keys) - int('ArrowLeft' in keys)
    old_axis = rail.axis
    rail.axis = float(axis)
    if old_axis and not axis:
        rail.stop()
    if 'KeyT' in pressed:
        rail.random_goal()
    elif 'Home' in pressed:
        rail.set_goal(0)
    elif 'Space' in pressed:
        rail.stop()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scene", type=Path, required=True, help="External complete moving-field model.xml")
  parser.add_argument("--model", type=Path, help="Optional external MJCF to attach")
  parser.add_argument("--prefix", default="object/", help="Prefix for attached model names")
  parser.add_argument("--position", type=float, nargs=3, default=(0, 0, 0), metavar=("X", "Y", "Z"))
  parser.add_argument("--headless", action="store_true", help="Run without a viewer")
  parser.add_argument("--steps", type=int, help="Number of simulation steps (headless default: 1000)")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--frame-rate-hz", type=float, default=30)
  parser.add_argument("--fpv-camera", help="Name of an MJCF camera for the browser FPV view")
  parser.add_argument("--first-person", action="store_true", help="Start in the FPV view (requires --fpv-camera)")
  parser.add_argument("--target-speed", type=float, default=0.15, help="Base target command speed in m/s")
  parser.add_argument("--target-seed", type=int, default=42)
  parser.add_argument("--target-random", action="store_true", help="Move to one seeded random position, then hold")
  args = parser.parse_args()
  if args.steps is not None and args.steps <= 0:
    parser.error("--steps must be positive")
  if not 0 <= args.port <= 65535:
    parser.error("--port must be between 0 and 65535")
  if args.first_person and not args.fpv_camera:
    parser.error("--first-person requires --fpv-camera")
  if not math.isfinite(args.target_speed) or args.target_speed <= 0:
    parser.error("--target-speed must be finite and positive")
  arena = args.scene
  attachments = [Attachment(args.model, args.prefix, tuple(args.position))] if args.model else []
  sim = Simulation(load_scene(arena, attachments))
  import mujoco
  has_target = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "blue_base_slide") >= 0
  if args.target_random and not has_target:
    parser.error("--target-random requires a scene with blue_base_slide")
  rail = RailMotion(sim.model, sim.data, speed=args.target_speed, seed=args.target_seed) if has_target else None
  if rail and args.target_random:
    rail.random_goal()
  if args.headless:
    for _ in range(args.steps or 1000):
      if rail:
        rail.tick()
      sim.step()
    print(f"steps={args.steps or 1000} time={sim.data.time:.6f}s")
    if rail:
      print(f"target={rail.goal:.6f} position={sim.data.qpos[rail.qadr]:.6f} reached={rail.reached()}")
    return
  from rmuc2027_mjarena.input import WebServer
  from rmuc2027_mjarena.viewer import BrowserViewer

  with BrowserViewer(sim.model, port=0, frame_rate_hz=args.frame_rate_hz,
                     fpv_camera=args.fpv_camera, first_person=args.first_person) as viewer:
    keys = {"KeyP", "KeyR"}
    if rail:
      keys.update({"ArrowLeft", "ArrowRight", "KeyT", "Home", "Space"})
    if args.fpv_camera:
      keys.add("Tab")
    web = WebServer(args.host, args.port, viewer.server.get_port(), keys=frozenset(keys))
    try:
      print(f"Viewer: http://{args.host}:{web.port} (P pause, R reset" +
            (", arrows target, T random, Home center, Space stop" if rail else "") +
            (", Tab FPV" if args.fpv_camera else "") + ")", flush=True)
      paused = False
      steps = 0
      deadline = time.monotonic()
      while args.steps is None or steps < args.steps:
        web.set_first_person(viewer.first_person_enabled, viewer.fpv_fovy_rad)
        state = web.poll()
        if "KeyP" in state.pressed:
          paused = not paused
        if "KeyR" in state.pressed:
          sim.reset()
          if rail:
            rail.reset(seed=args.target_seed)
            if args.target_random:
              rail.random_goal()
        if "Tab" in state.pressed:
          viewer.toggle_first_person()
        if rail:
          apply_target_input(rail, state.keys, state.pressed)
        if not paused:
          if rail:
            rail.tick()
          sim.step()
          steps += 1
        viewer.draw(sim.data)
        deadline += sim.model.opt.timestep if not paused else 0.01
        delay = deadline - time.monotonic()
        if delay > 0:
          time.sleep(delay)
        else:
          deadline = time.monotonic()
    except KeyboardInterrupt:
      pass
    finally:
      web.close()


if __name__ == "__main__":
  main()
