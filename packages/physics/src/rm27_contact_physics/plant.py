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
from .px4_rate import Px4Allocator, Px4RateController


class RM27Plant:
    """One environment owns one model and its complete mutable physics state."""

    def __init__(self, cfg, sources, *, physical_seed=None, physical_scale=1.0, physical_values=None):
        self.cfg = copy.deepcopy(cfg)
        self.marker_config = json.loads(sources["target_geometry"].read_text())
        # Default (seed None) keeps the nominal plant; a seed draws contact/actuator uncertainty.
        self.physical = scenario(self.cfg, seed=physical_seed, scale=physical_scale)
        self.profile = load_profile(sources["dynamics_profile"])
        vehicle = with_profile(json.loads(sources["vehicle_config"].read_text()),
                               sources["dynamics_profile"])
        if "vehicle_mjcf" in sources:  # import the runtime aircraft MJCF instead of the generated proxy
            vehicle["mjcf_path"] = str(sources["vehicle_mjcf"])
        self.vehicle_config = copy.deepcopy(vehicle)
        self.model, self.data, self.info, self.xml = build_model(
            self.physical, self.cfg, vehicle_config=vehicle)
        motors = [self.model.actuator(f"uav_motor_{i}").id for i in range(4)]
        verify_compiled(self.model, self.model.body("uav").id, motors, self.profile, self.physical["draw"])
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.inertia = np.asarray(self.info["inertia_kg_m2"], dtype=float)
        self.mixer = motor_mixer(self.info["rotor_arm_m"], self.info["yaw_torque_per_thrust_m"])
        self.inverse_mixer = np.linalg.inv(self.mixer)
        controller = self.cfg["controller"]
        self.rate_gain = 2 * controller["attitude_damping"] * controller["attitude_frequency_rad_s"]
        self.maximum = float(self.info["max_rotor_thrust_N_assumed"])
        self.hover_force = float(self.info["mass_kg"] * -self.model.opt.gravity[2])
        # Physical-only sensitivity: keep all controller references nominal.
        self.physical_values = dict(physical_values or {})
        allowed = {"thrust_to_weight", "inertia_scale", "yaw_torque_per_thrust_m"}
        if self.physical_values.keys() - allowed:
            raise ValueError("Unknown physical dynamics override")
        for value in self.physical_values.values():
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError("Physical dynamics values must be positive and finite")
        if self.physical_values:
            nominal_twr = 4 * self.maximum / self.hover_force
            thrust_scale = self.physical_values.get("thrust_to_weight", nominal_twr) / nominal_twr
            self.model.body_inertia[self.model.body("uav").id] *= self.physical_values.get("inertia_scale", 1.)
            self.model.actuator_gear[motors] *= thrust_scale
            yaw_scale = self.physical_values.get("yaw_torque_per_thrust_m", self.info["yaw_torque_per_thrust_m"])
            self.model.actuator_gear[motors, 5] = (
                np.array([1., -1., 1., -1.]) * yaw_scale
                * thrust_scale * self.physical["draw"]["thrust_scale"])
            mujoco.mj_setConst(self.model, self.data)
            self.data.qpos[:] = self.initial_qpos
            self.data.qvel[:] = self.initial_qvel
            mujoco.mj_forward(self.model, self.data)

    def actuator_capability(self, per_output=None):
        """Physical forces versus nominal command/output units; never controller inputs.

        per_output is the PX4 nominal-newtons per normalized motor output. Gear
        applies hidden effectiveness and the T/W sweep to both thrust and yaw.
        """
        ids = [self.model.actuator(f"uav_motor_{i}").id for i in range(4)]
        gain = self.model.actuator_gear[ids, 2]
        unit_output = self.maximum if per_output is None else float(per_output)
        mass = float(self.model.body_mass[self.model.body("uav").id])
        return dict(nominal_mass_kg=self.info["mass_kg"], physical_mass_kg=mass,
                    controller_inertia_kg_m2=self.inertia.tolist(),
                    physical_inertia_kg_m2=self.model.body_inertia[self.model.body("uav").id].tolist(),
                    nominal_N_per_unit_output=unit_output,
                    physical_N_per_unit_output=(unit_output * gain).tolist(),
                    physical_rotor_force_limit_N=(self.maximum * gain).tolist(),
                    reachable_thrust_to_weight=float(np.sum(min(unit_output, self.maximum) * gain) / (mass * 9.81)),
                    physical_yaw_torque_per_thrust_m=float(abs(self.model.actuator_gear[ids[0], 5] / gain[0])))

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


class RateActuation:
    """One simulation inner loop, shared by impact trials and free-flight checks."""

    def __init__(self, plant, inner_loop="proxy", rate_params=None):
        if inner_loop not in ("proxy", "px4"):
            raise ValueError("inner_loop must be 'proxy' or 'px4'")
        self.plant, self.inner_loop = plant, inner_loop
        self.per_output = None
        self.saturated = False
        if inner_loop == "px4":
            if rate_params is None:
                raise ValueError("The PX4 inner loop needs rate_params")
            if not np.isclose(float(rate_params["rate_loop_hz"]) * plant.model.opt.timestep, 1.):
                raise ValueError("The PX4 rate loop runs once per physics step")
            self.rate_loop = Px4RateController(rate_params)
            self.allocator = Px4Allocator(plant.mixer)
            self.per_output = (plant.hover_force / 4 / float(rate_params["hover_throttle"])
                               if "hover_throttle" in rate_params else plant.maximum)

    def apply(self, command, *, update_proxy=True):
        p = self.plant
        if self.inner_loop == "proxy":
            if update_proxy:
                _, self.saturated = p.apply_ctbr(command)
        else:
            torque = self.rate_loop.update(command[1:], p.data.qvel[3:6])
            outputs, self.rate_loop.saturated = self.allocator.allocate(command[0] / (4 * self.per_output), torque)
            p.data.ctrl[:] = np.minimum(outputs * self.per_output, p.maximum)
            self.saturated = bool(np.any(outputs <= 0.) or np.any(outputs >= 1.)
                                  or np.any(outputs * self.per_output > p.maximum))
        return self.saturated
