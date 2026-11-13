"""Run the base carriage with a separate Pygame keyboard controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import mujoco
import numpy as np

from rm27_sim.simulation.base_rail_control import RailMotion


def load_scene(scene):
    scene = Path(scene).resolve()
    manifest_path = scene.parent/'assembly_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if (scene.parent/'model.mjb').exists() and manifest.get('mujoco_version') == mujoco.__version__:
        if (scene.name == 'model.xml' and manifest.get('runtime_hashes') and
                all((scene.parent/name).is_file() and
                    hashlib.sha256((scene.parent/name).read_bytes()).hexdigest() == digest
                    for name, digest in manifest['runtime_hashes'].items())):
            return mujoco.MjModel.from_binary_path(str(scene.parent/'model.mjb')), manifest
    return mujoco.MjModel.from_xml_path(str(scene)), manifest


class RailLink:
    """Dedicated local keyboard channel; it does not change flight IPC."""
    def __init__(self, motion):
        self.motion = motion
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 0))
        self.socket.setblocking(False)
        self.port = self.socket.getsockname()[1]
        self.peer = None
        self.sequence = -1
        self.received = None
        self.last_telemetry = 0.
        self.closed = False

    def poll(self, now):
        for _ in range(64):
            try:
                raw, address = self.socket.recvfrom(1024)
            except BlockingIOError:
                break
            if address[0] != '127.0.0.1' or (self.peer is not None and address != self.peer):
                continue
            try:
                msg = json.loads(raw)
                seq, axis = msg['seq'], msg['axis']
                action = msg.get('action')
                if (type(seq) is not int or seq <= self.sequence or
                        type(axis) not in (int, float) or not np.isfinite(axis) or abs(axis) > 1 or
                        action not in (None, 'random', 'center', 'stop', 'quit')):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            self.peer = address
            self.sequence, self.received = seq, now
            old_axis = self.motion.axis
            self.motion.axis = float(axis)
            if axis == 0 and old_axis:
                self.motion.set_goal(self.motion.command)
            if action == 'random': self.motion.random_goal()
            elif action == 'center': self.motion.set_goal(0)
            elif action == 'stop': self.motion.stop()
            elif action == 'quit': self.closed = True
        if self.received is not None and now-self.received > .3:
            self.motion.stop()
        if self.peer and now-self.last_telemetry > 1/20:
            self.socket.sendto(json.dumps({'position': float(self.motion.data.qpos[self.motion.qadr]),
                'target': self.motion.goal, 'limits': self.motion.limits.tolist()}).encode(), self.peer)
            self.last_telemetry = now

    def close(self):
        self.socket.close()


def controller(port):
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((760, 230))
    pygame.display.set_caption('Base rail control | focus this window for arrow keys')
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0)); sock.setblocking(False)
    state = None; seq = 0; running = True; armed = False; last_status = time.monotonic()
    try:
        while running:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    action = 'quit'; running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: action = 'quit'; running = False
                    elif armed and event.key == pygame.K_r: action = 'random'
                    elif armed and event.key == pygame.K_HOME: action = 'center'
                    elif armed and event.key == pygame.K_SPACE: action = 'stop'
            keys = pygame.key.get_pressed()
            focused = pygame.key.get_focused() and pygame.display.get_active()
            movement = bool(keys[pygame.K_LEFT] or keys[pygame.K_RIGHT])
            if not focused:
                armed = False
                if action != 'quit': action = 'stop'
            elif not movement:
                armed = True
            axis = int(keys[pygame.K_RIGHT])-int(keys[pygame.K_LEFT]) if armed and focused else 0
            sock.sendto(json.dumps({'seq': seq, 'axis': axis, 'action': action}).encode(), ('127.0.0.1', port))
            seq += 1
            while True:
                try:
                    raw, _ = sock.recvfrom(1024)
                    state = json.loads(raw); last_status = time.monotonic()
                except BlockingIOError:
                    break
            screen.fill((22, 26, 32))
            lines = ['BLUE BASE / UPPER CARRIAGE', 'Hold LEFT / RIGHT: slide     R: random target',
                     'HOME: center     SPACE: stop     ESC: close',
                     'Keyboard ready' if focused and armed else 'Click here; release arrow keys to enable']
            if state:
                lines.append(f"Position {state['position']*100:+.1f} cm   Target {state['target']*100:+.1f} cm")
                low, high = state['limits']
                pygame.draw.line(screen, (100,110,125), (25,205), (735,205), 5)
                for value, color in [(state['target'], (255,170,60)), (state['position'], (70,210,140))]:
                    x = int(25+710*(np.clip(value,low,high)-low)/(high-low))
                    pygame.draw.circle(screen, color, (x,205), 7)
            else:
                lines.append('Connecting to the simulation...')
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, (225,232,240)), (22, 16+i*32))
            pygame.display.flip(); clock.tick(30)
            if time.monotonic()-last_status > 5:
                running = False
    finally:
        sock.sendto(json.dumps({'seq': seq+1,'axis':0,'action':'quit'}).encode(), ('127.0.0.1',port))
        sock.close(); pygame.quit()


def camera_setup(camera):
    camera.lookat[:] = [11.60, 0, 1.24]
    camera.distance = 1.5
    camera.azimuth = 0
    camera.elevation = -16


def view(scene, seed=0, *, speed=None):
    import mujoco.viewer
    model, manifest = load_scene(scene)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    motion = RailMotion(model, data, seed=seed, speed=(speed if speed is not None else
                       manifest.get('source_config', {}).get('dynamics_estimate', {}).get('speed_m_s', .15)))
    link = RailLink(motion)
    process = None
    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                camera_setup(viewer.cam)
                viewer.opt.geomgroup[3] = 0
            process = subprocess.Popen([sys.executable, '-m', 'rm27_sim.simulation.base_rail', '--controller', str(link.port)])
            print('Use the Base rail control window: LEFT / RIGHT, R=random, HOME=center.', flush=True)
            deadline = time.monotonic(); last_render = deadline
            while viewer.is_running() and not link.closed and process.poll() is None:
                now = time.monotonic(); link.poll(now)
                motion.tick(); mujoco.mj_step(model, data)
                if now-last_render >= 1/30:
                    viewer.sync(state_only=True); last_render = now
                deadline += model.opt.timestep
                now = time.monotonic()
                if deadline > now: time.sleep(min(deadline-now, model.opt.timestep))
                elif now-deadline > .1: deadline = now
    finally:
        link.close()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=3)


def exercise(scene, *, seed=0, random_targets=5, video=None, speed=None):
    if random_targets < 0:
        raise ValueError('Random target count must be nonnegative')
    model, manifest = load_scene(scene)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    motion = RailMotion(model, data, seed=seed, speed=(speed if speed is not None else
                       manifest.get('source_config', {}).get('dynamics_estimate', {}).get('speed_m_s', .15)))
    renderer = writer = None
    if video:
        import cv2
        renderer = mujoco.Renderer(model, 720, 1000)
        camera = mujoco.MjvCamera(); camera_setup(camera)
        option = mujoco.MjvOption(); option.geomgroup[3] = 0
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1000,720))
        if not writer.isOpened(): raise RuntimeError('Cannot open video output')
    targets = [float(motion.limits[0]), float(motion.limits[1]), 0.]
    targets += [None] * random_targets
    results = []; maximum_limit_excess = 0.; next_frame = 0.
    try:
        for goal in targets:
            goal = motion.random_goal() if goal is None else motion.set_goal(goal)
            start = data.time
            timeout = abs(goal-motion.command)/motion.speed + 4.
            while data.time-start < timeout:
                motion.tick(); mujoco.mj_step(model, data)
                q = float(data.qpos[motion.qadr])
                maximum_limit_excess = max(maximum_limit_excess, motion.limits[0]-q, q-motion.limits[1])
                if writer and data.time >= next_frame:
                    renderer.update_scene(data, camera, option)
                    frame = renderer.render()
                    import cv2
                    cv2.putText(frame, f'Rail: {q*100:+.1f} cm  target: {goal*100:+.1f} cm',
                                (20,40), cv2.FONT_HERSHEY_SIMPLEX, .8, (240,240,240), 2)
                    writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)); next_frame += 1/30
                if motion.reached(): break
            results.append({'target_m': float(goal), 'position_m': float(data.qpos[motion.qadr]),
                            'elapsed_sim_s': float(data.time-start), 'reached': bool(motion.reached())})
    finally:
        if writer: writer.release()
        if renderer: renderer.close()
    return {'mujoco_version': mujoco.__version__, 'seed':seed, 'range_m':motion.limits.tolist(),
            'speed_m_s': motion.speed,
            'maximum_limit_excess_m': float(maximum_limit_excess), 'targets':results,
            'passed': all(r['reached'] for r in results) and maximum_limit_excess < .0005}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path('artifacts/base_rail/model.xml'))
    parser.add_argument('--controller', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--speed', type=float, help='Command speed in m/s; defaults to the scene setting (0.15)')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--random-targets', type=int, default=5)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.speed is not None and (not np.isfinite(args.speed) or args.speed <= 0):
        parser.error('--speed must be finite and greater than zero (m/s)')
    if args.controller is not None: controller(args.controller)
    elif args.headless or args.video:
        result = exercise(args.scene, seed=args.seed, random_targets=args.random_targets, video=args.video,
                          speed=args.speed)
        if args.report: args.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        if not result['passed']: raise SystemExit(1)
    else: view(args.scene, args.seed, speed=args.speed)


if __name__ == '__main__':
    main()
