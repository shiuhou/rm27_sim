"""Endpoint-constrained septic trajectory and geometric feedback motor policy.

No MuJoCo imports: consumes estimates + a contact bit. Trajectory acceleration
sets thrust direction; terminal attitude is never an independent animation.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

from .model import frame, marker_center, unit


def oriented_frame(force, x_hint):
    """Build a thrust-aligned frame with a requested horizontal heading.

    The old controller used ``frame(force)`` which chooses a world-axis
    tangent.  That is physically valid, but it leaves the vehicle yaw
    unrelated to the direction in which it is travelling or looking.  The
    hint is projected into the plane normal to the thrust direction so this
    helper never treats yaw as an independent attitude animation.
    """
    z = unit(force)
    x = np.asarray(x_hint, dtype=float) - z * float(np.dot(z, x_hint))
    if np.linalg.norm(x) < 1e-8:
        return frame(z)
    x = unit(x)
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    return np.column_stack((x, y, z))


class Septic:
    """Unique degree-7 p/v/a/j endpoint interpolation in normalized time."""
    def __init__(self, start, end, duration):
        self.duration = float(duration)
        if self.duration <= 0:
            raise ValueError("Positive duration required")
        M, values = [], []
        for t, conditions in ((0., start), (1., end)):
            for d in range(4):
                M.append([0 if k < d else math.factorial(k)/math.factorial(k-d)*t**(k-d)
                          for k in range(8)])
                values.append(np.asarray(conditions[d]) * duration**d)
        self.c = np.linalg.solve(np.array(M), np.array(values))

    def sample(self, time):
        t = np.clip(time / self.duration, 0, 1)
        return tuple(np.array([0 if k < d else math.factorial(k)/math.factorial(k-d)*t**(k-d)
                               for k in range(8)]) @ self.c / self.duration**d for d in range(4))


def motor_mixer(rotor_arm_m, yaw_torque_per_thrust_m):
    """FLU wrench from rotor forces, ordered (+,+), (+,-), (-,-), (-,+)."""
    a, k = rotor_arm_m, yaw_torque_per_thrust_m
    return np.array([[1, 1, 1, 1], [a, -a, -a, a], [-a, -a, a, a], [k, -k, k, -k]], dtype=float)


class Skill:
    def __init__(self, cfg, parameters, initial, target, vehicle_info, *, marker_config):
        self.cfg, self.p, self.target = cfg, parameters, target
        self.start_time = initial.timestamp
        n, R = target.normal, frame(target.normal)
        rc = np.asarray(cfg["contact"]["offset_body_m"])
        endpoint = (target.position + n * (cfg["contact"]["radius_m"] - parameters["penetration_m"])
                    - R @ rc + R[:, 0]*parameters.get("aim_tangent_offset_m", 0.))
        end_v = -n*parameters["normal_speed_m_s"] + R[:, 0]*parameters["tangent_speed_m_s"]
        end_a = n*parameters["terminal_specific_thrust_m_s2"] - [0, 0, 9.81]
        self.trajectory = Septic((initial.position, initial.velocity, np.zeros(3), np.zeros(3)),
                                 (endpoint, end_v, end_a, np.zeros(3)), parameters["duration_s"])
        self.mass = vehicle_info["mass_kg"]
        self.inertia = np.array(vehicle_info["inertia_kg_m2"])
        self.max_motor = vehicle_info["max_rotor_thrust_N_assumed"]
        # Existing integration vehicle motor ordering: (+,+), (+,-), (-,-), (-,+).
        self.mixer = motor_mixer(vehicle_info["rotor_arm_m"], vehicle_info["yaw_torque_per_thrust_m"])
        self.inverse_mixer = np.linalg.inv(self.mixer)
        self.phase = "FREE_FLIGHT"
        self.contact_time = None
        self.recovery_anchor = None
        self.last_Rd = initial.rotation
        self.last_time = None
        self.last_reference = initial.position
        self.last_reference_state = np.zeros(16)
        self.invalid_target = not target.valid(initial.timestamp)
        self.recovery_stable_since = None
        self.motor_estimate = np.full(4, self.mass*9.81/4)
        self.previous_motor_command = self.motor_estimate.copy()
        self.previous_velocity = initial.velocity.copy()
        self.specific_thrust_ratio = 1.
        self.previous_rotation = initial.rotation.copy()
        # None retains the historical fixed-heading controller exactly.
        self.heading_yaw = None
        self.heading_rate = 0.
        target_pose = dict(center_m=np.asarray(target.position, float),
                           normal=np.asarray(target.normal, float),
                           in_plane_rotation_deg=cfg.get("target", {}).get("in_plane_rotation_deg", 0.))
        self.camera_target = marker_center(target_pose, marker_config)

    def set_heading_reference(self, yaw_rad, yaw_rate_rad_s=0.):
        """Track a planner's continuous yaw while preserving thrust direction.

        The caller owns rate/acceleration limits and phase handoff continuity.
        This API consumes reference values, not camera or simulator state.
        """
        if not np.isfinite([yaw_rad, yaw_rate_rad_s]).all():
            raise ValueError("Heading reference must be finite")
        self.heading_yaw, self.heading_rate = float(yaw_rad), float(yaw_rate_rad_s)

    def attitude_reference(self, force, time_offset_s=0.):
        if self.heading_yaw is None:
            return frame(force)
        yaw = self.heading_yaw + self.heading_rate * time_offset_s
        return oriented_frame(force, [np.cos(yaw), np.sin(yaw), 0.])

    def command(self, state, contact):
        t = state.timestamp
        elapsed = t-self.start_time
        # Keep the rigidly mounted forward/down camera aimed at the configured
        # guidance lamp during the approach. This is a planner-side geometric
        # heading reference; no image detection or estimator truth is consumed.
        if self.heading_yaw is None and self.cfg.get("controller", {}).get("look_at_marker", False):
            delta = self.camera_target-state.position
            if np.linalg.norm(delta[:2]) > 1e-6:
                self.set_heading_reference(float(np.arctan2(-delta[1], -delta[0])), 0.)
        if contact and self.contact_time is None:
            self.contact_time = t
            self.recovery_anchor = state.position + self.target.normal*self.p["retreat_m"]
            self.recovery_anchor[2] += self.p["recovery_lift_m"]
        if self.invalid_target:
            self.phase = "ABORT"
            desired = (self.trajectory.sample(0)[0], np.zeros(3), np.zeros(3), np.zeros(3))
        elif self.contact_time is not None:
            self.phase = "CONTACT" if contact else "SEPARATION" if t-self.contact_time < .3 else "RECOVERY"
            desired = (self.recovery_anchor, np.zeros(3), np.zeros(3), np.zeros(3))
            g = self.cfg["gates"]
            pc = state.position + state.rotation @ self.cfg["contact"]["offset_body_m"]
            stable = (not contact and (pc-self.target.position) @ self.target.normal > g["separation_m"]
                      + self.cfg["contact"]["radius_m"] and state.rotation[2, 2] > np.cos(np.deg2rad(g["max_recovery_tilt_deg"]))
                      and np.linalg.norm(state.velocity) < g["max_recovery_speed_m_s"]
                      and np.linalg.norm(state.omega_body) < g["max_recovery_rate_rad_s"])
            self.recovery_stable_since = (t if self.recovery_stable_since is None else self.recovery_stable_since) if stable else None
            if self.recovery_stable_since is not None and t-self.recovery_stable_since >= g["recovery_hold_s"]:
                self.phase = "NORMAL_FLIGHT"
        elif elapsed > self.p["duration_s"] + .25:
            self.phase = "ABORT"
            desired = (self.trajectory.sample(0)[0], np.zeros(3), np.zeros(3), np.zeros(3))
        else:
            self.phase = "PRE_CONTACT" if elapsed > .7*self.p["duration_s"] else "FREE_FLIGHT"
            desired = self.trajectory.sample(elapsed)
            if elapsed > self.p["duration_s"]:
                # Short bounded continuation, then event-independent miss abort.
                h = elapsed - self.p["duration_s"]
                p, v, a, j = desired
                desired = (p + v*h + .5*a*h*h, v+a*h, a, j)
        return self.track(state, desired)

    def track(self, state, desired, specific_thrust_ratio=None):
        """Track an external p/v/a/j reference using the same inner controller."""
        t = state.timestamp
        p, v, acceleration, jerk = desired
        self.last_reference = p
        c = self.cfg["controller"]
        dt = self.cfg["control_dt_s"]
        actuator_gain = c.get("actuator_feedback_gain", 0.)
        adaptive = c.get("adapt_specific_thrust", False)
        if actuator_gain or adaptive:
            decay = np.exp(-dt/c.get("actuator_predictor_tau_s", .04))
            self.motor_estimate = decay*self.motor_estimate + (1-decay)*self.previous_motor_command
        if specific_thrust_ratio is not None:
            self.specific_thrust_ratio = float(np.clip(specific_thrust_ratio, .7, 1.3))
        if adaptive and specific_thrust_ratio is None and self.last_time is not None and self.contact_time is None:
            # A bounded low-pass estimate of effective thrust/mass from ego
            # velocity increments. Never read the randomized gain, mass, or act.
            b3 = .5*(self.previous_rotation[:, 2]+state.rotation[:, 2])
            measured = float(((state.velocity-self.previous_velocity)/dt + [0, 0, 9.81]) @ b3)
            predicted = float(self.motor_estimate.sum()/self.mass)
            if predicted > 3.:
                ratio = np.clip(measured/predicted, .7, 1.3)
                beta = 1-np.exp(-dt/c.get("thrust_adaptation_tau_s", .12))
                self.specific_thrust_ratio += beta*(ratio-self.specific_thrust_ratio)
        self.previous_velocity = state.velocity.copy()
        self.previous_rotation = state.rotation.copy()
        force = self.mass*(acceleration + [0, 0, 9.81] + np.array(c["position_kp"])*(p-state.position)
                           + np.array(c["position_kd"])*(v-state.velocity))
        if adaptive:
            force = force/self.specific_thrust_ratio
        if np.linalg.norm(force) < 1e-6:
            force = np.array([0, 0, 1e-6])
        Rd = self.attitude_reference(force)
        wd = np.zeros(3) if self.last_time is None else Rotation.from_matrix(self.last_Rd.T @ Rd).as_rotvec()/dt
        if c.get('rate_reference_source') == 'analytic':
            # Differentiate the known p/v/a/j reference and nominal dynamics,
            # not finite differences of noisy feedback-adjusted attitudes.
            estimated_acc = (state.rotation[:, 2]*self.motor_estimate.sum()*self.specific_thrust_ratio/self.mass
                             - [0, 0, 9.81])
            force_dot = self.mass*(jerk + np.array(c['position_kp'])*(v-state.velocity)
                                  + np.array(c['position_kd'])*(acceleration-estimated_acc))
            if adaptive:force_dot = force_dot/self.specific_thrust_ratio
            epsilon = 1e-5
            next_frame = self.attitude_reference(force+epsilon*force_dot, epsilon)
            wd = Rotation.from_matrix(Rd.T @ next_frame).as_rotvec()/epsilon
        wd = np.clip(wd, -15, 15)
        # The contact reset changes the position anchor discontinuously. A finite
        # difference of Rd across that reset is not a physical angular-velocity
        # reference. The optional hybrid controller uses attitude/rate damping
        # without differentiating the impact/anchor jump throughout recovery.
        if self.contact_time is not None and (not c.get("recovery_rate_feedforward", True)
                or t-self.contact_time < c.get("recovery_rate_blank_s", 0.)):
            wd = np.zeros(3)
        qd = Rotation.from_matrix(Rd).as_quat()
        self.last_reference_state = np.r_[v, acceleration, qd[[3, 0, 1, 2]], wd, force]
        self.last_Rd, self.last_time = Rd, t
        R, omega = state.rotation, state.omega_body
        E = .5*(Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        ew = omega - R.T @ Rd @ wd
        w = c["attitude_frequency_rad_s"]
        torque = self.inertia*(-w*w*eR - 2*c["attitude_damping"]*w*ew) + np.cross(omega, self.inertia*omega)
        thrust = max(0., float(force @ R[:, 2]))
        raw = self.inverse_mixer @ np.r_[thrust, torque]
        if actuator_gain:
            # Internal command-driven motor observer, not privileged true data.act.
            raw = raw + actuator_gain*(raw-self.motor_estimate)
        command = np.clip(raw, 0, self.max_motor)
        self.previous_motor_command = command.copy()
        return command, bool(np.any(raw < 0) or np.any(raw > self.max_motor))
