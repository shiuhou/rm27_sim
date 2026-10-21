"""Observe the flight state without allowing viewer controls to alter physics."""

import copy
import threading
import time

import mujoco
import numpy as np


class FlightView:
    def __init__(self, plant):
        import mujoco.viewer

        self.plant = plant
        # A render snapshot, never stepped. UI perturbations and physics settings
        # affect only this copy, never the authoritative ArduPilot plant.
        self.model = copy.deepcopy(plant.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, plant.data)
        existing_threads = set(threading.enumerate())
        self.handle = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )
        self.render_threads = set(threading.enumerate()) - existing_threads
        with self.handle.lock():
            self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.handle.cam.fixedcamid = -1
            self.handle.cam.lookat[:] = plant.origin + [0, 0, 0.25]
            self.handle.cam.distance = 1.5
            self.handle.cam.azimuth = 45
            self.handle.cam.elevation = -30
            self.handle.opt.geomgroup[3] = 0
        self.last_update = -float("inf")
        self.updates = 0

    def is_running(self):
        return self.handle.is_running()

    def update(self, phase, telemetry, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_update < 1 / 30:
            return
        local = self.plant.qpos[:3] - self.plant.origin
        with self.handle.lock():
            mujoco.mj_copyData(self.data, self.model, self.plant.data)
            if phase == 'ROUTE':
                self.handle.cam.lookat[:] = self.plant.qpos[:3]
        self.handle.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "RM27 | drone 1\nPhase\nPhysics time\nHeight above slot\nXY distance from slot\nMode / armed",
                f"Live simulation\n{phase}\n{self.plant.data.time:.2f} s\n"
                f"{local[2]:.3f} m\n{np.linalg.norm(local[:2]) * 1000:.1f} mm\n"
                f"{telemetry.get('mode') or 'Waiting'} / {telemetry.get('armed', False)}",
            )
        )
        self.handle.sync(state_only=True)
        self.last_update = time.monotonic()
        self.updates += 1

    def close(self):
        self.handle.close()
        # launch_passive owns a daemon GUI thread. Let it finish OpenGL teardown
        # before Python exits; otherwise shutdown can race native rendering.
        for thread in self.render_threads:
            thread.join(timeout=5)
