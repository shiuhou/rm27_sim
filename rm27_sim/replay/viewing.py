"""MuJoCo live viewer and recorded-state playback; never write the control plant."""
import copy
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
from .recording import load_model
from .fpv import DEFAULT_MOUNT, configure_camera, load_mount


class LiveView:
    def __init__(self, model, data, speed=1., camera='follow', fovy=None, camera_config=DEFAULT_MOUNT):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Positive finite playback speed required")
        import mujoco.viewer
        self.model = copy.deepcopy(model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, data)
        self.speed, self.paused, self.overview = speed, False, False
        if camera not in ('follow','overview','fpv'):
            raise ValueError('Unknown viewing camera')
        self.camera_mode=camera
        self.overview=camera=='overview'
        self.fpv_id=None
        self.fovy=fovy
        self.camera_config=camera_config
        if camera=='fpv':
            self.fpv_id=configure_camera(self.model,load_mount(camera_config),fovy)
            mujoco.mj_camlight(self.model,self.data)
        self.wall_start, self.sim_start, self.last_draw = time.monotonic(), float(data.time), -np.inf
        self.label = "Live simulation"
        self.body = self.model.body("uav").id
        threads = set(threading.enumerate())
        self.handle = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key,
                                                   show_left_ui=False, show_right_ui=False)
        self.threads = set(threading.enumerate())-threads
        with self.handle.lock():
            self.handle.cam.azimuth = 120
            self.handle.cam.elevation = -22
            self.handle.cam.distance = 18 if self.overview else 1.3
            self.handle.cam.lookat[:] = data.qpos[:3]
            self.handle.opt.geomgroup[3] = 0
            self._set_camera()

    def _set_camera(self):
        """Called with the display lock held; control model/data remain untouched."""
        if self.camera_mode=='fpv':
            if self.fpv_id is None:
                self.fpv_id=configure_camera(self.model,load_mount(self.camera_config),self.fovy)
            mujoco.mj_camlight(self.model,self.data)
            self.handle.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
            self.handle.cam.fixedcamid=self.fpv_id
            self.handle.opt.sitegroup[:]=0
        else:
            self.handle.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            self.handle.cam.lookat[:]=[-5.4,.5,1.5] if self.overview else self.data.qpos[:3]

    def key(self, key):
        if key == 32:  # space
            self.paused = not self.paused
        elif key in (ord('V'),ord('v'),ord('1'),ord('2'),ord('6')):
            if key==ord('6'):
                self.camera_mode='fpv'
            elif key==ord('1'):
                self.camera_mode='follow'
            elif key==ord('2'):
                self.camera_mode='overview'
            else:
                self.camera_mode='follow' if self.overview else 'overview'
            self.overview=self.camera_mode=='overview'
            with self.handle.lock():
                self.handle.cam.distance = 18 if self.overview else 1.3
                if self.camera_mode!='fpv' and self.fpv_id is not None:
                    self.handle.opt.sitegroup[:]=1
                self._set_camera()

    def update(self, data, phase, *, force=False):
        if not self.handle.is_running():
            return False
        while self.paused and self.handle.is_running():
            before = time.monotonic()
            self.handle.sync(state_only=True)
            time.sleep(.02)
            self.wall_start += time.monotonic()-before
        deadline = self.wall_start+(data.time-self.sim_start)/self.speed
        while time.monotonic() < deadline and self.handle.is_running():
            time.sleep(max(0., min(.01, deadline-time.monotonic())))
        now = time.monotonic()
        if not force and now-self.last_draw < 1/45:
            return self.handle.is_running()
        with self.handle.lock():
            mujoco.mj_copyData(self.data, self.model, data)
            self._set_camera()
        self.handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               "RM27 contact research\nPhase\nTime / speed\nControls",
                               f"{self.label}\n{phase}\n{data.time:.2f} s / {self.speed:g}x\nSpace: pause   1: follow   2: overview   6: first person   V: overview/follow"))
        self.handle.sync(state_only=True)
        self.last_draw = now
        return self.handle.is_running()

    def close(self):
        self.handle.close()
        for thread in self.threads:
            thread.join(timeout=5)


def view_episode(source, speed=.5, loop=False, camera='follow', fovy=None, camera_config=DEFAULT_MOUNT):
    source = Path(source)
    archive = np.load(source/"trajectory.npz")
    trace = archive["trace"]
    phases = archive["phases"] if "phases" in archive else np.array(["RECORDED"]*len(trace))
    model = load_model(source)
    data = mujoco.MjData(model)
    def restore(row):
        data.time = row[0]
        data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:] = row[1:8], row[8:14], row[14:18], row[18:22]
        mujoco.mj_forward(model, data)
    restore(trace[0])
    view = LiveView(model, data, speed, camera, fovy, camera_config)
    view.label = "Recorded-state playback (not a new flight)"
    try:
        while view.handle.is_running():
            view.wall_start, view.sim_start = time.monotonic(), trace[0, 0]
            for row, phase in zip(trace, phases):
                restore(row)
                if not view.update(data, str(phase)):
                    return
            if not loop:
                view.paused = True
                view.update(data, "FINISHED — close window", force=True)
                break
    finally:
        view.close()


def render_episode(source, output, whole=False):
    """Headless image QA and compact animation of the recorded physical mission."""
    from PIL import Image, ImageDraw
    from .io import write_json
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(source)
    data = mujoco.MjData(model)
    trace = np.load(source/"trajectory.npz")["trace"]
    summary = json.loads((source/"summary.json").read_text())
    contact = summary.get("terminal", summary).get("first_contact")
    renderer = mujoco.Renderer(model, height=480, width=800)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    def frame_at(t, overview=False):
        row = trace[np.argmin(abs(trace[:, 0]-t))]
        data.time = row[0]
        data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:] = row[1:8], row[8:14], row[14:18], row[18:22]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [-5.4, .5, 1.5] if overview else data.qpos[:3]
        cam.distance, cam.azimuth, cam.elevation = (18, 90, -45) if overview else (1.15, 115, -22)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        image = Image.fromarray(renderer.render())
        ImageDraw.Draw(image).text((15, 15), f"RM27 recorded state | t={row[0]:.2f}s | {'overview' if overview else 'follow'}", fill="white")
        return image
    try:
        frame_at(0, True).save(output/"arena.png")
        if contact:
            for label, t in [("approach", contact["time_s"]-.35), ("contact", contact["time_s"]),
                             ("recovery", contact["time_s"]+1.)]:
                frame_at(t).save(output/f"{label}.png")
            times = np.linspace(max(trace[0, 0], contact["time_s"]-1.3), min(trace[-1, 0], contact["time_s"]+2.), 100)
            frames = [frame_at(t) for t in times]
            frames[0].save(output/"contact.gif", save_all=True, append_images=frames[1:], duration=66, loop=0)
        frame_at(trace[-1, 0]).save(output/("finish.png" if whole else "return.png"))
        if whole:
            times = np.linspace(trace[0, 0], trace[-1, 0], max(2, int(np.ceil((trace[-1, 0]-trace[0, 0])*25))))
            frames = [frame_at(t) for t in times]
            frames[0].save(output/'maneuver.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
    finally:
        renderer.close()
    result = dict(passed=True, scope="headless render of recorded states; not live-view or a new flight",
                  source=str(source), images=[p.name for p in output.iterdir()])
    write_json(output/"summary.json", result)
    return result
