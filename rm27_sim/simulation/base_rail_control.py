"""Portable moving-base control; copy this file into another MuJoCo project.

Only NumPy is imported. Call tick() immediately before each mj_step(); it writes
only the named rail actuator and never steps physics or resets model state.
"""
from __future__ import annotations

import numpy as np


class RailMotion:
    """Rate-limited position commands, shared by keyboard and seeded random mode."""
    def __init__(self, model, data, *, speed=.15, seed=0,
                 joint='blue_base_slide', actuator='blue_base_drive'):
        self.model, self.data = model, data
        jid = model.joint(joint).id
        self.actuator = model.actuator(actuator).id
        # mjJNT_SLIDE = 2; mjTRN_JOINT = 0. Avoid importing a runtime or viewer.
        if (model.jnt_type[jid] != 2 or not model.jnt_limited[jid] or
                model.actuator_trntype[self.actuator] != 0 or
                model.actuator_trnid[self.actuator, 0] != jid):
            raise ValueError('Expected a limited slide joint with its own joint actuator')
        self.qadr = int(model.jnt_qposadr[jid])
        self.dadr = int(model.jnt_dofadr[jid])
        self.limits = model.jnt_range[jid].copy()
        self.speed = speed
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('Rail speed must be finite and positive')
        self._speed = float(speed)

    def reset(self, *, seed=None):
        """Resync after the host resets MjData; optionally restart random draws."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.command = self.goal = float(np.clip(self.data.qpos[self.qadr], *self.limits))
        self.axis = 0.
        self.data.ctrl[self.actuator] = self.command

    def set_goal(self, goal):
        if not np.isfinite(goal):
            raise ValueError('Target must be finite')
        self.axis = 0.
        self.goal = float(np.clip(goal, *self.limits))
        return self.goal

    def random_goal(self):
        return self.set_goal(self.rng.uniform(*self.limits))

    def stop(self):
        self.axis = 0.
        # Hold the current physical location after focus loss or link timeout.
        self.command = self.goal = float(np.clip(self.data.qpos[self.qadr], *self.limits))

    def tick(self):
        dt = self.model.opt.timestep
        if self.axis:
            self.goal = float(np.clip(self.command+self.axis*self.speed*dt, *self.limits))
        delta = np.clip(self.goal-self.command, -self.speed*dt, self.speed*dt)
        self.command = float(np.clip(self.command+delta, *self.limits))
        self.data.ctrl[self.actuator] = self.command

    def reached(self):
        return (abs(self.data.qpos[self.qadr]-self.goal) < .001 and
                abs(self.data.qvel[self.dadr]) < .005)
