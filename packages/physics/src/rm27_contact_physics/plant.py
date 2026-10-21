"""Reusable MuJoCo plant; callers supply configuration and asset paths."""
from __future__ import annotations
import copy
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .calibration import load_profile, verify_compiled, with_profile
from .control import motor_mixer
from .model import build_model, scenario, truth_estimate
from .teacher_interface import RateThrust, apply_proxy


class RM27Plant:
    """One environment owns one model and its complete mutable physics state."""

    def __init__(self, cfg, sources):
        self.cfg = copy.deepcopy(cfg)
        self.marker_config = json.loads(sources["target_geometry"].read_text())
        self.physical = scenario(self.cfg, seed=None)
        self.profile = load_profile(sources["dynamics_profile"])
        vehicle = with_profile(json.loads(sources["vehicle_config"].read_text()),
                               sources["dynamics_profile"])
        self.vehicle_config = copy.deepcopy(vehicle)
        self.model, self.data, self.info, self.xml = build_model(
            self.physical, self.cfg, vehicle_config=vehicle)
        motors = [self.model.actuator(f"uav_motor_{i}").id for i in range(4)]
        verify_compiled(self.model, self.model.body("uav").id, motors, self.profile)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.inertia = np.asarray(self.info["inertia_kg_m2"], dtype=float)
        self.mixer = motor_mixer(self.info["rotor_arm_m"], self.info["yaw_torque_per_thrust_m"])
        self.inverse_mixer = np.linalg.inv(self.mixer)
        controller = self.cfg["controller"]
        self.rate_gain = 2 * controller["attitude_damping"] * controller["attitude_frequency_rad_s"]
        self.maximum = float(self.info["max_rotor_thrust_N_assumed"])
        self.hover_force = float(self.info["mass_kg"] * -self.model.opt.gravity[2])

    @property
    def state(self):
        return truth_estimate(self.model, self.data)

    def reset(self, *, position=None, velocity=None, rotation=None, omega_body=None):
        """Optional explicit initial state; no-argument Contact reset is unchanged."""
        values = {"position": position, "velocity": velocity, "rotation": rotation,
                  "omega_body": omega_body}
        for name, value in values.items():
            if value is not None:
                expected = (3, 3) if name == "rotation" else (3,)
                if np.asarray(value).shape != expected or not np.isfinite(value).all():
                    raise ValueError(f"Invalid initial {name}")
        if rotation is not None and (not np.allclose(np.asarray(rotation).T @ rotation, np.eye(3), atol=1e-9)
                                     or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9)):
            raise ValueError("Initial rotation must belong to SO(3)")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        if position is not None:
            self.data.qpos[:3] = position
        if rotation is not None:
            q = Rotation.from_matrix(rotation).as_quat()
            self.data.qpos[3:7] = q[[3, 0, 1, 2]]
        if velocity is not None:
            self.data.qvel[:3] = velocity
        if omega_body is not None:
            self.data.qvel[3:6] = omega_body
        self.data.act[:] = self.hover_force / 4
        self.data.ctrl[:] = self.hover_force / 4
        mujoco.mj_forward(self.model, self.data)
        return self.state

    def apply_ctbr(self, clipped_ctbr):
        """Return actual motor *commands*, not actual lagged rotor thrust."""
        action = RateThrust(np.asarray(clipped_ctbr[1:], float), float(clipped_ctbr[0]))
        motors, saturated = apply_proxy(
            action, self.state.omega_body, self.inertia, self.inverse_mixer,
            self.rate_gain, self.maximum, return_saturation=True)
        self.data.ctrl[:] = motors
        return motors.copy(), saturated
